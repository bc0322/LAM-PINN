from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lam_pinn.config import AdaptConfig
from lam_pinn.data.ingestion import build_task_properties
from lam_pinn.data.sampling import build_training_tensors
from lam_pinn.engine.checkpoints import load_checkpoint, save_checkpoint
from lam_pinn.evaluation.metrics import build_prediction_dataframe, evaluate_against_csv
from lam_pinn.evaluation.visualize import (
    plot_deformation_comparison,
    plot_gate_trajectory,
    plot_loss_curve,
)
from lam_pinn.physics.losses import task_loss
from lam_pinn.utils.device import select_device
from lam_pinn.utils.logging import setup_logger
from lam_pinn.utils.paths import make_run_dir, snapshot_file
from lam_pinn.utils.seed import set_seed


_CASE_LIBRARY = {
    1: {"E": 55.0, "f": 1.80, "k": 9.20},
    2: {"E": 60.0, "f": 1.90, "k": 9.00},
    3: {"E": 134.0, "f": 1.954, "k": 0.772},
    4: {"E": 55.0, "f": 1.70, "k": 9.80},
    5: {"E": 79.0, "f": 0.84, "k": 2.302},
    6: {"E": 82.0, "f": 1.847, "k": 6.759},
    7: {"E": 79.0, "f": 1.00, "k": 2.10},
    8: {"E": 80.0, "f": 1.91, "k": 7.00},
    9: {"E": 51.0, "f": 1.90, "k": 9.50},
    10: {"E": 79.0, "f": 1.00, "k": 2.40},
}


def _resolve_task_spec(task_config) -> dict:
    case_index = int(task_config.case_index)
    if case_index not in _CASE_LIBRARY:
        raise ValueError(
            f"Unsupported case_index={case_index}. "
            f"Available cases: {sorted(_CASE_LIBRARY.keys())}"
        )

    csv_path = Path(task_config.eval_dir) / task_config.eval_filename_pattern.format(idx=case_index)

    # legacy fallback for the current zip archive
    if not csv_path.exists() and case_index == 1:
        legacy_path = Path(task_config.eval_dir) / "sample_task.csv"
        if legacy_path.exists():
            csv_path = legacy_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found for case {case_index}: {csv_path}")

    case_params = _CASE_LIBRARY[case_index]
    return {
        "name": f"case_{case_index:02d}",
        "case_index": case_index,
        "E": float(case_params["E"]),
        "f": float(case_params["f"]),
        "k": float(case_params["k"]),
        "gt_csv_path": str(csv_path),
    }


def adapt_single_task(config: AdaptConfig, config_path: str | Path | None = None) -> dict:
    set_seed(config.seed)
    device = select_device(config.preferred_cuda_index)

    run_dir = make_run_dir(config.output_root, config.run_name)
    logger = setup_logger(f"lam_pinn_adapt_{run_dir.name}", run_dir / "logs" / "run.log")
    snapshot_file(config_path, run_dir / "artifacts" / "config.adapt.yaml")

    logger.info("device=%s", device)
    logger.info("checkpoint_path=%s", config.checkpoint_path)

    task_spec = _resolve_task_spec(config.task)
    config.task.E = float(task_spec["E"])
    config.task.f = float(task_spec["f"])
    config.task.k = float(task_spec["k"])
    config.task.gt_csv_path = str(task_spec["gt_csv_path"])

    properties = build_task_properties(config.task, config.physics)
    logger.info(
        "case_index=%d | E=%.6f | f=%.6f | k=%.6f | gt_csv=%s",
        int(task_spec["case_index"]),
        properties["E"],
        properties["f"],
        properties["k"],
        config.task.gt_csv_path,
    )

    model, payload = load_checkpoint(
        config.checkpoint_path,
        device=device,
        dropout_override=config.model.dropout_p,
    )
    model.reset_gates(config.model.gate_init)

    if config.optimization.freeze_basis:
        for parameter in model.basis.parameters():
            parameter.requires_grad = False

    training_tensors = build_training_tensors(config.sampling, seed=config.seed, device=device)

    trainable_params = list(model.backward.parameters())
    for forward_net in model.forward_nets:
        trainable_params.extend(list(forward_net.parameters()))

    optimizer = torch.optim.Adam(
        [
            {"params": trainable_params, "lr": config.optimization.lr_model},
            {"params": list(model.gates), "lr": config.optimization.lr_gate},
        ]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optimization.scheduler_factor,
        patience=config.optimization.scheduler_patience,
    )

    trace_records: list[dict] = []
    start_time = time.time()

    metrics0, coords, uv_true, uv_pred = evaluate_against_csv(
        model,
        properties,
        config.task.gt_csv_path,
        device=device,
    )
    best_mse = float(metrics0["deformation_mse"])
    best_epoch = 0

    trace_records.append(
        {
            "epoch": 0,
            "loss": None,
            "mse_u": float(metrics0["mse_u"]),
            "mse_v": float(metrics0["mse_v"]),
            "deformation_mse": float(metrics0["deformation_mse"]),
            "deformation_std": float(metrics0["deformation_std"]),
            **{f"gate_{index}": float(value) for index, value in enumerate(model.gate_values())},
        }
    )

    logger.info(
        "[adapt] epoch=%d/%d | deformation_mse=%.6e | gates=%s",
        0,
        config.optimization.epochs,
        float(metrics0["deformation_mse"]),
        [round(value, 4) for value in model.gate_values()],
    )

    domain_points, bc1_y0, bc2_x0, bc3_yb, bc4_xa = training_tensors
    for epoch in range(1, config.optimization.epochs + 1):
        optimizer.zero_grad()
        loss = task_loss(
            model,
            domain_points,
            bc1_y0,
            bc2_x0,
            bc3_yb,
            bc4_xa,
            config.optimization.bc_scale,
            properties,
        )
        loss.backward()
        optimizer.step()
        model.clamp_gates()
        scheduler.step(float(loss.item()))

        metrics, coords, uv_true, uv_pred = evaluate_against_csv(
            model,
            properties,
            config.task.gt_csv_path,
            device=device,
        )

        if float(metrics["deformation_mse"]) < best_mse:
            best_mse = float(metrics["deformation_mse"])
            best_epoch = int(epoch)

        record = {
            "epoch": int(epoch),
            "loss": float(loss.item()),
            "mse_u": float(metrics["mse_u"]),
            "mse_v": float(metrics["mse_v"]),
            "deformation_mse": float(metrics["deformation_mse"]),
            "deformation_std": float(metrics["deformation_std"]),
        }
        for gate_index, gate_value in enumerate(model.gate_values()):
            record[f"gate_{gate_index}"] = float(gate_value)
        trace_records.append(record)

        if epoch % config.optimization.print_every == 0 or epoch == config.optimization.epochs:
            logger.info(
                "[adapt] epoch=%d/%d | loss=%.6e | deformation_mse=%.6e | gates=%s",
                epoch,
                config.optimization.epochs,
                float(loss.item()),
                float(metrics["deformation_mse"]),
                [round(value, 4) for value in model.gate_values()],
            )

    adapt_minutes = (time.time() - start_time) / 60.0
    logger.info("===== adaptation done | %.2f minutes =====", adapt_minutes)

    trace_df = pd.DataFrame(trace_records)
    trace_df.to_csv(run_dir / "artifacts" / "adaptation_trace.csv", index=False)

    pred_df = build_prediction_dataframe(coords, uv_true, uv_pred)
    pred_df.to_csv(run_dir / "artifacts" / "prediction_comparison.csv", index=False)

    plot_deformation_comparison(
        coords,
        uv_true,
        uv_pred,
        output_path=run_dir / "figures" / "deformation_comparison.png",
        levels=config.visualization.levels,
        deformation_error_max=config.visualization.deformation_error_max,
    )
    plot_loss_curve(
        trace_df.dropna(subset=["loss"]).reset_index(drop=True),
        output_path=run_dir / "figures" / "loss_curve.png",
    )
    plot_gate_trajectory(
        trace_df,
        output_path=run_dir / "figures" / "gate_trajectory.png",
    )

    adapted_checkpoint_path = run_dir / "checkpoints" / "adapted_model.pt"
    save_checkpoint(
        adapted_checkpoint_path,
        model,
        payload={
            "num_clusters": int(payload.get("num_clusters", 3)),
            "cluster_ids": payload.get("cluster_ids", [0, 1, 2]),
            "cluster_to_index": payload.get("cluster_to_index", {0: 0, 1: 1, 2: 2}),
            "model_hparams": {
                "hidden_dim": int(config.model.hidden_dim),
                "dropout_p": float(config.model.dropout_p),
                "gate_init": float(config.model.gate_init),
            },
            "physics_defaults": asdict(config.physics),
            "sampling": asdict(config.sampling),
            "seed": int(config.seed),
            "case_index": int(task_spec["case_index"]),
        },
    )

    final_metrics = trace_records[-1]
    summary = {
        "run_dir": str(run_dir),
        "adapted_checkpoint_path": str(adapted_checkpoint_path),
        "device": str(device),
        "seed": int(config.seed),
        "case_index": int(task_spec["case_index"]),
        "E": float(properties["E"]),
        "f": float(properties["f"]),
        "k": float(properties["k"]),
        "gt_csv_path": str(config.task.gt_csv_path),
        "freeze_basis": bool(config.optimization.freeze_basis),
        "epochs": int(config.optimization.epochs),
        "lr_model": float(config.optimization.lr_model),
        "lr_gate": float(config.optimization.lr_gate),
        "best_mse": float(best_mse),
        "best_epoch": int(best_epoch),
        "final_mse": float(final_metrics["deformation_mse"]),
        "mse_epoch0": float(metrics0["deformation_mse"]),
        "mse_u": float(final_metrics["mse_u"]),
        "mse_v": float(final_metrics["mse_v"]),
        "deformation_std": float(final_metrics["deformation_std"]),
        "adapt_minutes": float(adapt_minutes),
    }
    with open(run_dir / "adaptation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    del optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return summary
