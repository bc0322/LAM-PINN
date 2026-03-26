from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lam_pinn.config import TrainConfig
from lam_pinn.data.ingestion import (
    add_affinity_distance,
    build_balanced_training_dataframe,
    get_cluster_ids,
    get_cluster_index_map,
    read_training_table,
    row_to_task_properties,
)
from lam_pinn.data.sampling import build_training_tensors
from lam_pinn.engine.checkpoints import save_checkpoint
from lam_pinn.models.serial_net import SerialNetwork
from lam_pinn.physics.losses import task_loss
from lam_pinn.utils.device import select_device
from lam_pinn.utils.logging import setup_logger
from lam_pinn.utils.paths import make_run_dir, snapshot_file
from lam_pinn.utils.seed import set_seed


def _train_single_task_stage(
    model: SerialNetwork,
    row: pd.Series,
    active_idx: int,
    training_tensors,
    config: TrainConfig,
    logger,
) -> tuple[float, list[float]]:
    domain_points, bc1_y0, bc2_x0, bc3_yb, bc4_xa = training_tensors
    properties = row_to_task_properties(row, config.physics)
    model.set_gate_pattern(active_idx=active_idx, active_value=1.0, inactive_value=0.1)

    trainable_params = []
    trainable_params.extend(list(model.basis.parameters()))
    trainable_params.extend(list(model.backward.parameters()))
    for forward_net in model.forward_nets:
        trainable_params.extend(list(forward_net.parameters()))

    gate_params = [gate for index, gate in enumerate(model.gates) if index != active_idx]

    defaults = _stage_lr_defaults()

    optimizer = torch.optim.Adam(
        [
            {"params": trainable_params, "lr": config.optimization.lr_model},
            {"params": gate_params, "lr": defaults["gate"]},
        ]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optimization.scheduler_factor,
        patience=config.optimization.scheduler_patience,
    )

    last_loss = 0.0
    for epoch in range(config.optimization.forward_stage_epochs):
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
        last_loss = float(loss.item())

        if epoch % 25 == 0 or epoch == config.optimization.forward_stage_epochs - 1:
            logger.info(
                "[task-stage] cluster=%s | E=%.6f | f=%.6f | k=%.6f | epoch=%d/%d | loss=%.6e | gates=%s",
                int(row["Cluster"]),
                properties["E"],
                properties["f"],
                properties["k"],
                epoch,
                config.optimization.forward_stage_epochs - 1,
                last_loss,
                [round(value, 4) for value in model.gate_values()],
            )

    return last_loss, model.gate_values()


def _train_backward_stage(
    model: SerialNetwork,
    cycle_df: pd.DataFrame,
    cluster_to_index: dict[int, int],
    training_tensors,
    config: TrainConfig,
    logger,
) -> float:
    domain_points, bc1_y0, bc2_x0, bc3_yb, bc4_xa = training_tensors

    defaults = _stage_lr_defaults()
    optimizer = torch.optim.Adam(list(model.backward.parameters()), lr=defaults["backward"])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optimization.backward_scheduler_factor,
        patience=config.optimization.backward_scheduler_patience,
    )

    last_total_loss = 0.0
    for epoch in range(config.optimization.backward_stage_epochs):
        optimizer.zero_grad()
        total_loss_value = 0.0

        for gate in model.gates:
            gate.grad = None

        for _, row in cycle_df.iterrows():
            active_idx = cluster_to_index[int(row["Cluster"])]
            model.set_gate_pattern(active_idx=active_idx, active_value=1.0, inactive_value=0.1)
            properties = row_to_task_properties(row, config.physics)
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
            total_loss_value += float(loss.item())
            loss.backward()

        optimizer.step()
        scheduler.step(total_loss_value)
        last_total_loss = total_loss_value

        if epoch % 25 == 0 or epoch == config.optimization.backward_stage_epochs - 1:
            logger.info(
                "[backward-stage] epoch=%d/%d | total_loss=%.6e",
                epoch,
                config.optimization.backward_stage_epochs - 1,
                last_total_loss,
            )

    return last_total_loss

def _stage_lr_defaults():
    return {
        "gate": 5e-3,
        "backward": 5e-5,
    }

def meta_train(config: TrainConfig, config_path: str | Path | None = None) -> dict:
    set_seed(config.seed)
    device = select_device(config.preferred_cuda_index)
    run_dir = make_run_dir(config.output_root, config.run_name)
    logger = setup_logger(f"lam_pinn_train_{run_dir.name}", run_dir / "logs" / "run.log")
    snapshot_file(config_path, run_dir / "artifacts" / "config.train.yaml")

    logger.info("device=%s", device)
    logger.info("loading training csv: %s", config.train_csv_path)

    raw_df = read_training_table(config.train_csv_path)
    cluster_ids = get_cluster_ids(raw_df)
    if config.expected_num_clusters is not None and len(cluster_ids) != int(config.expected_num_clusters):
        raise ValueError(
            f"expected_num_clusters={config.expected_num_clusters}, but the CSV contains {len(cluster_ids)} clusters: {cluster_ids}"
        )

    working_df = add_affinity_distance(raw_df, reference_row_index=config.reference_row_index)
    balanced_df = build_balanced_training_dataframe(working_df, cluster_ids, seed=config.seed)
    balanced_df.to_csv(run_dir / "artifacts" / "balanced_training_order.csv", index=False)

    cluster_to_index = get_cluster_index_map(cluster_ids)
    with open(run_dir / "artifacts" / "cluster_mapping.json", "w", encoding="utf-8") as file:
        json.dump({str(key): int(value) for key, value in cluster_to_index.items()}, file, indent=2)

    model = SerialNetwork(
        num_clusters=len(cluster_ids),
        hidden_dim=config.model.hidden_dim,
        dropout_p=config.model.dropout_p,
        gate_init=config.model.gate_init,
    ).to(device)

    training_tensors = build_training_tensors(config.sampling, seed=config.seed, device=device)
    trace_records: list[dict] = []
    start_time = time.time()
    cycle_size = len(cluster_ids)
    num_cycles = len(balanced_df) // cycle_size

    logger.info("num_clusters=%d | cluster_ids=%s | num_cycles=%d", len(cluster_ids), cluster_ids, num_cycles)
    logger.info("===== meta-training start =====")

    for cycle_idx in range(num_cycles):
        cycle_df = balanced_df.iloc[cycle_idx * cycle_size:(cycle_idx + 1) * cycle_size].reset_index(drop=True)
        logger.info("[cycle %d/%d]", cycle_idx + 1, num_cycles)

        for _, row in cycle_df.iterrows():
            active_idx = cluster_to_index[int(row["Cluster"])]
            final_loss, gates = _train_single_task_stage(
                model=model,
                row=row,
                active_idx=active_idx,
                training_tensors=training_tensors,
                config=config,
                logger=logger,
            )
            record = {
                "cycle": cycle_idx + 1,
                "stage": "task",
                "cluster": int(row["Cluster"]),
                "subnet_index": int(active_idx),
                "E": float(row["E_raw"]),
                "f": float(row["f_raw"]),
                "k": float(row["k_raw"]),
                "loss": float(final_loss),
            }
            for gate_index, gate_value in enumerate(gates):
                record[f"gate_{gate_index}"] = float(gate_value)
            trace_records.append(record)

        backward_loss = _train_backward_stage(
            model=model,
            cycle_df=cycle_df,
            cluster_to_index=cluster_to_index,
            training_tensors=training_tensors,
            config=config,
            logger=logger,
        )
        trace_records.append(
            {
                "cycle": cycle_idx + 1,
                "stage": "backward",
                "cluster": None,
                "subnet_index": None,
                "E": None,
                "f": None,
                "k": None,
                "loss": float(backward_loss),
            }
        )

    train_minutes = (time.time() - start_time) / 60.0
    logger.info("===== meta-training done | %.2f minutes =====", train_minutes)

    checkpoint_path = run_dir / "checkpoints" / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        payload={
            "num_clusters": len(cluster_ids),
            "cluster_ids": cluster_ids,
            "cluster_to_index": cluster_to_index,
            "model_hparams": {
                "hidden_dim": config.model.hidden_dim,
                "dropout_p": config.model.dropout_p,
                "gate_init": config.model.gate_init,
            },
            "physics_defaults": asdict(config.physics),
            "sampling": asdict(config.sampling),
            "seed": config.seed,
        },
    )

    trace_df = pd.DataFrame(trace_records)
    trace_df.to_csv(run_dir / "artifacts" / "training_trace.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_clusters": int(len(cluster_ids)),
        "cluster_ids": cluster_ids,
        "cluster_to_index": cluster_to_index,
        "num_cycles": int(num_cycles),
        "num_training_rows": int(len(balanced_df)),
        "train_minutes": float(train_minutes),
        "device": str(device),
    }
    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
