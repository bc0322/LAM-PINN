from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from lam_pinn.evaluation.metrics import deformation_magnitude


def _grid_from_points(coords: np.ndarray, values: np.ndarray):
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": values})
    pivot = df.pivot_table(index="y", columns="x", values="value", aggfunc="first")
    if pivot.isnull().any().any():
        return None
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    if len(x_values) * len(y_values) != len(coords):
        return None
    X, Y = np.meshgrid(x_values, y_values)
    Z = pivot.to_numpy(dtype=float)
    return X, Y, Z


def _filled_contour(ax, coords: np.ndarray, values: np.ndarray, cmap: str, levels: np.ndarray):
    grid = _grid_from_points(coords, values)
    if grid is not None:
        X, Y, Z = grid
        return ax.contourf(X, Y, Z, cmap=cmap, levels=levels)

    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])
    return ax.tricontourf(triangulation, values, cmap=cmap, levels=levels)


def plot_deformation_comparison(
    coords: np.ndarray,
    uv_true: np.ndarray,
    uv_pred: np.ndarray,
    output_path: str | Path,
    levels: int = 20,
    deformation_error_max: float | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    deformation_true = deformation_magnitude(uv_true)
    deformation_pred = deformation_magnitude(uv_pred)
    error = (deformation_true - deformation_pred) ** 2

    d_min = float(min(deformation_true.min(), deformation_pred.min()))
    d_max = float(max(deformation_true.max(), deformation_pred.max()))
    deformation_levels = np.linspace(d_min, d_max, levels)

    error_max = float(error.max()) if deformation_error_max is None else float(deformation_error_max)
    error_levels = np.linspace(0.0, max(error_max, 1e-12), levels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    contour_true = _filled_contour(axes[0], coords, deformation_true, cmap="viridis", levels=deformation_levels)
    colorbar_true = fig.colorbar(contour_true, ax=axes[0], format="%.2f")
    colorbar_true.ax.tick_params(labelsize=11)
    axes[0].set_title("Ground Truth Deformation")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    contour_pred = _filled_contour(axes[1], coords, deformation_pred, cmap="viridis", levels=deformation_levels)
    colorbar_pred = fig.colorbar(contour_pred, ax=axes[1], format="%.2f")
    colorbar_pred.ax.tick_params(labelsize=11)
    axes[1].set_title("Predicted Deformation")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    contour_error = _filled_contour(axes[2], coords, error, cmap="Reds", levels=error_levels)
    colorbar_error = fig.colorbar(contour_error, ax=axes[2], format="%.3f")
    colorbar_error.ax.tick_params(labelsize=11)
    axes[2].set_title("Error in Deformation")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    axes[2].text(
        0.5,
        1.05,
        f"MSE: {float(np.mean(error)):.6e}",
        transform=axes[2].transAxes,
        fontsize=12,
        ha="center",
    )

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(trace_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trace_df["epoch"], trace_df["loss"], linewidth=2)
    ax.set_title("Adaptation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gate_trajectory(trace_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_columns = [column for column in trace_df.columns if column.startswith("gate_")]

    fig, ax = plt.subplots(figsize=(7, 4))
    for gate_column in gate_columns:
        ax.plot(trace_df["epoch"], trace_df[gate_column], linewidth=2, label=gate_column)
    ax.set_title("Gate Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gate value")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if gate_columns:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
