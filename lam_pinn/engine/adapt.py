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
    logger.info("loading checkpoint: %s", config.checkpoint_path)
    model, checkpoint_payload = load_checkpoint(
        config.checkpoint_path,
        device=device,
        dropout_override=config.model.dropout_p,
    )
    model.reset_gates(config.model.gate_init)

    if config.optimization.freeze_basis:
        for parameter in model.basis.parameters():
            parameter.requires_grad = False
        logger.info("basis network frozen during adaptation")

    training_tensors = build_training_tensors(config.sampling, seed=config.seed, device=device)
    resolved_task = _resolve_task_spec(config.task)

    task_properties = build_task_properties(
        resolved_task["E"],
        resolved_task["f"],
        resolved_task["k"],
        config.physics,
    )

    trainable_params = list(model.backward.parameters())
    for forward_net in model.forward_nets:
        trainable_params.extend(list(forward_net.parameters()))
        
    gate_lr = 3e-1

    optimizer = torch.optim.Adam(
        [
            {"params": trainable_params, "lr": config.optimization.lr_model},
            {"params": list(model.gates), "lr": gate_lr},
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
    model.train()

    logger.info(
        "starting adaptation | task=%s | E=%.6f | f=%.6f | k=%.6f",
        resolved_task["name"],
        resolved_task["E"],
        resolved_task["f"],
        resolved_task["k"],
    )
    domain_points, bc1_y0, bc2_x0, bc3_yb, bc4_xa = training_tensors
    for epoch in range(config.optimization.epochs):
        optimizer.zero_grad()
        loss = task_loss(
            model,
            domain_points,
            bc1_y0,
            bc2_x0,
            bc3_yb,
            bc4_xa,
            config.optimization.bc_scale,
            task_properties,
        )
        loss.backward()
        optimizer.step()
        model.clamp_gates()
        scheduler.step(float(loss.item()))

        record = {"epoch": epoch, "loss": float(loss.item())}
        for gate_index, gate_value in enumerate(model.gate_values()):
            record[f"gate_{gate_index}"] = gate_value
        trace_records.append(record)

        if epoch % config.optimization.print_every == 0 or epoch == config.optimization.epochs - 1:
            logger.info(
                "[adapt] epoch=%d/%d | loss=%.6e | gates=%s",
                epoch,
                config.optimization.epochs - 1,
                float(loss.item()),
                [round(value, 4) for value in model.gate_values()],
            )

    trace_df = pd.DataFrame(trace_records)
    trace_df.to_csv(run_dir / "artifacts" / "adaptation_trace.csv", index=False)

    metrics, coords, uv_true, uv_pred = evaluate_against_csv(
        model=model,
        properties=task_properties,
        csv_path=resolved_task["gt_csv_path"],
        device=device,
    )
    prediction_df = build_prediction_dataframe(coords, uv_true, uv_pred)
    prediction_df.to_csv(run_dir / "artifacts" / "prediction_comparison.csv", index=False)

    plot_deformation_comparison(
        coords=coords,
        uv_true=uv_true,
        uv_pred=uv_pred,
        output_path=run_dir / "figures" / "deformation_comparison.png",
        levels=config.visualization.levels,
        deformation_error_max=config.visualization.deformation_error_max,
    )
    plot_loss_curve(trace_df, run_dir / "figures" / "loss_curve.png")
    plot_gate_trajectory(trace_df, run_dir / "figures" / "gate_trajectory.png")

    checkpoint_path = run_dir / "checkpoints" / "adapted_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        payload={
            "num_clusters": int(checkpoint_payload["num_clusters"]),
            "cluster_ids": checkpoint_payload["cluster_ids"],
            "cluster_to_index": checkpoint_payload["cluster_to_index"],
            "model_hparams": checkpoint_payload["model_hparams"],
            "physics_defaults": asdict(config.physics),
            "sampling": asdict(config.sampling),
            "task": resolved_task,
            "seed": config.seed,
        },
    )

    elapsed_minutes = (time.time() - start_time) / 60.0
    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "base_checkpoint": str(config.checkpoint_path),
        "task": resolved_task,
        "metrics": metrics,
        "final_gates": model.gate_values(),
        "adaptation_minutes": float(elapsed_minutes),
        "device": str(device),
    }
    with open(run_dir / "adaptation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info("adaptation complete | deformation_mse=%.6e | minutes=%.2f", metrics["deformation_mse"], elapsed_minutes)

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return summary
