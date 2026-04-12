from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    dropout_p: float = 0.2
    gate_init: float = 0.5


@dataclass
class PhysicsConfig:
    nu: float = 0.3
    a: float = 10.0
    b: float = 10.0
    h: float = 1.0


@dataclass
class SamplingConfig:
    domain_range: list[list[float]] = field(default_factory=lambda: [[0.0, 10.0], [0.0, 10.0]])
    num_domain_points: int = 10000
    num_bc_points: int = 500


@dataclass
class TrainOptimizationConfig:
    forward_stage_epochs: int = 200
    backward_stage_epochs: int = 100
    lr_model: float = 0.002
    bc_scale: float = 50.0
    scheduler_factor: float = 0.7
    scheduler_patience: int = 150
    backward_scheduler_factor: float = 0.7
    backward_scheduler_patience: int = 50


@dataclass
class AdaptOptimizationConfig:
    epochs: int = 1000
    lr_model: float = 0.003
    lr_gate: float = 3e-1
    bc_scale: float = 50.0
    scheduler_factor: float = 0.7
    scheduler_patience: int = 150
    freeze_basis: bool = True
    print_every: int = 100


@dataclass
class VisualizationConfig:
    grid_size: int = 100
    levels: int = 20
    deformation_error_max: float | None = None


@dataclass
class TaskConfig:
    name: str = "demo_task"
    case_index: int = 1
    eval_dir: str = "data/eval"
    eval_filename_pattern: str = "sample_task{idx}.csv"
    E: float = 55.0
    f: float = 1.8
    k: float = 9.2
    gt_csv_path: str = "data/eval/sample_task.csv"


@dataclass
class TrainConfig:
    seed: int = 42
    preferred_cuda_index: int = 0
    output_root: str = "outputs/train"
    run_name: str = "lam_pinn_train"
    train_csv_path: str = "data/train/task_metadata.csv"
    expected_num_clusters: int | None = None
    reference_row_index: int = 13
    warm_start_checkpoint_path: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimization: TrainOptimizationConfig = field(default_factory=TrainOptimizationConfig)


@dataclass
class AdaptConfig:
    seed: int = 42
    preferred_cuda_index: int = 0
    checkpoint_path: str = "outputs/train/lam_pinn_train/checkpoints/model.pt"
    output_root: str = "outputs/adapt"
    run_name: str = "lam_pinn_adapt_demo"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(dropout_p=0.0))
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    optimization: AdaptOptimizationConfig = field(default_factory=AdaptOptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_train_config(path: str | Path) -> TrainConfig:
    data = _read_yaml(path)
    return TrainConfig(
        seed=data.get("seed", 42),
        preferred_cuda_index=data.get("preferred_cuda_index", 0),
        output_root=data.get("output_root", "outputs/train"),
        run_name=data.get("run_name", "lam_pinn_train"),
        train_csv_path=data.get("train_csv_path", "data/train/task_metadata.csv"),
        expected_num_clusters=data.get("expected_num_clusters"),
        reference_row_index=data.get("reference_row_index", 13),
        warm_start_checkpoint_path=data.get("warm_start_checkpoint_path"),
        model=ModelConfig(**data.get("model", {})),
        physics=PhysicsConfig(**data.get("physics", {})),
        sampling=SamplingConfig(**data.get("sampling", {})),
        optimization=TrainOptimizationConfig(**data.get("optimization", {})),
    )


def load_adapt_config(path: str | Path) -> AdaptConfig:
    data = _read_yaml(path)
    return AdaptConfig(
        seed=data.get("seed", 42),
        preferred_cuda_index=data.get("preferred_cuda_index", 0),
        checkpoint_path=data.get("checkpoint_path", "outputs/train/lam_pinn_train/checkpoints/model.pt"),
        output_root=data.get("output_root", "outputs/adapt"),
        run_name=data.get("run_name", "lam_pinn_adapt_demo"),
        model=ModelConfig(**data.get("model", {})),
        physics=PhysicsConfig(**data.get("physics", {})),
        sampling=SamplingConfig(**data.get("sampling", {})),
        task=TaskConfig(**data.get("task", {})),
        optimization=AdaptOptimizationConfig(**data.get("optimization", {})),
        visualization=VisualizationConfig(**data.get("visualization", {})),
    )
