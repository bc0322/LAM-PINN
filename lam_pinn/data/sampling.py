from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import qmc

from lam_pinn.config import SamplingConfig


PointSet = List[np.ndarray]
TensorPointSet = List[torch.Tensor]


def sample_training_points(
    config: SamplingConfig,
    seed: Optional[int] = None,
) -> Tuple[PointSet, PointSet, PointSet, PointSet, PointSet]:
    xmin, xmax = config.domain_range[0]
    ymin, ymax = config.domain_range[1]
    lb = np.array([xmin, ymin], dtype=np.float32)
    ub = np.array([xmax, ymax], dtype=np.float32)

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    domain = sampler.random(n=config.num_domain_points) * (ub - lb) + lb
    domain_x = domain[:, 0:1].astype(np.float32)
    domain_y = domain[:, 1:2].astype(np.float32)

    y0_x = np.linspace(xmin, xmax, config.num_bc_points, dtype=np.float32).reshape(-1, 1)
    y0_y = np.full((config.num_bc_points, 1), ymin, dtype=np.float32)

    x0_x = np.full((config.num_bc_points, 1), xmin, dtype=np.float32)
    x0_y = np.linspace(ymin, ymax, config.num_bc_points, dtype=np.float32).reshape(-1, 1)

    yb_x = np.linspace(xmin, xmax, config.num_bc_points, dtype=np.float32).reshape(-1, 1)
    yb_y = np.full((config.num_bc_points, 1), ymax, dtype=np.float32)

    xa_x = np.full((config.num_bc_points, 1), xmax, dtype=np.float32)
    xa_y = np.linspace(ymin, ymax, config.num_bc_points, dtype=np.float32).reshape(-1, 1)

    return [domain_x, domain_y], [y0_x, y0_y], [x0_x, x0_y], [yb_x, yb_y], [xa_x, xa_y]


def to_torch_points(points: Sequence[np.ndarray], device: torch.device) -> TensorPointSet:
    return [
        torch.tensor(array, dtype=torch.float32, requires_grad=True, device=device)
        for array in points
    ]


def build_training_tensors(
    config: SamplingConfig,
    seed: int,
    device: torch.device,
) -> Tuple[TensorPointSet, TensorPointSet, TensorPointSet, TensorPointSet, TensorPointSet]:
    sampled = sample_training_points(config, seed=seed)
    return tuple(to_torch_points(points, device=device) for points in sampled)
