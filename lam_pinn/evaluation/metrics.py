from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lam_pinn.data.ingestion import load_coords_uv_from_csv
from lam_pinn.physics.operators import pde_cal


def deformation_magnitude(uv: np.ndarray) -> np.ndarray:
    return np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)


def predict_uv(model: nn.Module, coords: np.ndarray, properties: dict[str, float], device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(coords[:, 0:1], dtype=torch.float32, device=device)
        y_tensor = torch.tensor(coords[:, 1:2], dtype=torch.float32, device=device)
        u_pred, v_pred = pde_cal(
            [x_tensor, y_tensor],
            model,
            properties["E"],
            properties["nu"],
            properties["f"],
            properties["h"],
            properties["b"],
            properties["k"],
            direct=True,
            bc=False,
        )
        return np.column_stack(
            [
                u_pred.detach().cpu().numpy().ravel(),
                v_pred.detach().cpu().numpy().ravel(),
            ]
        )


def compute_metrics(uv_true: np.ndarray, uv_pred: np.ndarray) -> dict[str, float]:
    err_u = (uv_true[:, 0] - uv_pred[:, 0]) ** 2
    err_v = (uv_true[:, 1] - uv_pred[:, 1]) ** 2
    def_true = deformation_magnitude(uv_true)
    def_pred = deformation_magnitude(uv_pred)
    err_def = (def_true - def_pred) ** 2

    return {
        "mse_u": float(np.mean(err_u)),
        "mse_v": float(np.mean(err_v)),
        "deformation_mse": float(np.mean(err_def)),
        "deformation_std": float(np.std(err_def)),
        "num_points": int(len(uv_true)),
    }


def evaluate_against_csv(
    model: nn.Module,
    properties: dict[str, float],
    csv_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    coords, uv_true = load_coords_uv_from_csv(csv_path)
    valid_mask = np.isfinite(coords).all(axis=1) & np.isfinite(uv_true).all(axis=1)
    coords = coords[valid_mask]
    uv_true = uv_true[valid_mask]
    uv_pred = predict_uv(model, coords, properties, device=device)
    metrics = compute_metrics(uv_true, uv_pred)
    return metrics, coords, uv_true, uv_pred


def build_prediction_dataframe(coords: np.ndarray, uv_true: np.ndarray, uv_pred: np.ndarray) -> pd.DataFrame:
    deformation_true = deformation_magnitude(uv_true)
    deformation_pred = deformation_magnitude(uv_pred)
    deformation_error_sq = (deformation_true - deformation_pred) ** 2

    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "u_true": uv_true[:, 0],
            "v_true": uv_true[:, 1],
            "u_pred": uv_pred[:, 0],
            "v_pred": uv_pred[:, 1],
            "deformation_true": deformation_true,
            "deformation_pred": deformation_pred,
            "deformation_error_sq": deformation_error_sq,
        }
    )
