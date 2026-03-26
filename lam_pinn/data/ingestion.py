from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lam_pinn.config import PhysicsConfig


REQUIRED_TRAIN_COLUMNS = ["E_raw", "f_raw", "k_raw", "Initial_L1", "Final_L2", "Avg_L3", "Cluster"]


def read_training_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Training CSV is empty: {path}")
    missing = [column for column in REQUIRED_TRAIN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    cluster_values = pd.to_numeric(df["Cluster"], errors="coerce")
    if cluster_values.isna().any():
        raise ValueError("Column 'Cluster' must contain numeric values only.")

    df = df.copy()
    df["Cluster"] = np.round(cluster_values).astype(int)
    return df


def get_cluster_ids(df: pd.DataFrame) -> list[int]:
    return sorted(df["Cluster"].unique().tolist())


def get_cluster_index_map(cluster_ids: list[int]) -> dict[int, int]:
    return {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}


def add_affinity_distance(df: pd.DataFrame, reference_row_index: int = 13) -> pd.DataFrame:
    distance_cols = ["Initial_L1", "Final_L2", "Avg_L3"]
    ref_index = reference_row_index if len(df) > reference_row_index else len(df) // 2
    origin = df.loc[ref_index, distance_cols].to_numpy(dtype=np.float32)

    def compute_distance(row: pd.Series) -> float:
        vector = row[distance_cols].to_numpy(dtype=np.float32)
        return float(np.sqrt(np.sum((vector - origin) ** 2)))

    out = df.copy()
    out["Euclidean_Distance"] = out.apply(compute_distance, axis=1)
    return out


def build_balanced_training_dataframe(df: pd.DataFrame, cluster_ids: list[int], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters: dict[int, pd.DataFrame] = {}
    max_len = 0

    for cluster_id in cluster_ids:
        cluster_df = (
            df[df["Cluster"] == cluster_id]
            .copy()
            .sort_values("Euclidean_Distance")
            .reset_index(drop=True)
        )
        if len(cluster_df) == 0:
            raise ValueError(f"Cluster {cluster_id} has no rows.")
        clusters[cluster_id] = cluster_df
        max_len = max(max_len, len(cluster_df))

    balanced: dict[int, pd.DataFrame] = {}
    for cluster_id, cluster_df in clusters.items():
        if len(cluster_df) < max_len:
            deficit = max_len - len(cluster_df)
            distances = cluster_df["Euclidean_Distance"].to_numpy(dtype=np.float64)
            weights = 1.0 / np.clip(distances + 1e-8, 1e-8, None)
            weights = weights / weights.sum()
            sampled_indices = rng.choice(len(cluster_df), size=deficit, replace=True, p=weights)
            sampled_rows = cluster_df.iloc[sampled_indices].copy().reset_index(drop=True)
            cluster_df = pd.concat([cluster_df, sampled_rows], ignore_index=True)
        balanced[cluster_id] = cluster_df.reset_index(drop=True)

    rows = []
    for row_index in range(max_len):
        for cluster_id in cluster_ids:
            rows.append(balanced[cluster_id].iloc[[row_index]])
    return pd.concat(rows, ignore_index=True)


def row_to_task_properties(row: pd.Series, physics: PhysicsConfig) -> dict[str, float]:
    return {
        "E": float(row["E_raw"]),
        "nu": float(physics.nu),
        "a": float(physics.a),
        "b": float(physics.b),
        "f": float(row["f_raw"]),
        "k": float(row["k_raw"]),
        "h": float(physics.h),
    }


def build_task_properties(E: float, f: float, k: float, physics: PhysicsConfig) -> dict[str, float]:
    return {
        "E": float(E),
        "nu": float(physics.nu),
        "a": float(physics.a),
        "b": float(physics.b),
        "f": float(f),
        "k": float(k),
        "h": float(physics.h),
    }


def _find_alias(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_coords_uv_from_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Evaluation CSV is empty: {path}")

    columns = list(df.columns)
    x_col = _find_alias(columns, ["x", "xs", "coord_x"])
    y_col = _find_alias(columns, ["y", "ys", "coord_y"])
    u_col = _find_alias(columns, ["u", "ux", "disp_x"])
    v_col = _find_alias(columns, ["v", "uy", "disp_y"])

    if not all([x_col, y_col, u_col, v_col]):
        raise ValueError(
            "Evaluation CSV must contain coordinate/displacement columns such as x, y, u, v. "
            f"Found columns: {columns}"
        )

    first_row = df.iloc[0]

    def looks_like_list_string(value: Any) -> bool:
        return isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]")

    if (
        len(df) == 1
        and looks_like_list_string(first_row[x_col])
        and looks_like_list_string(first_row[y_col])
        and looks_like_list_string(first_row[u_col])
        and looks_like_list_string(first_row[v_col])
    ):
        x = np.asarray(ast.literal_eval(first_row[x_col]), dtype=np.float32)
        y = np.asarray(ast.literal_eval(first_row[y_col]), dtype=np.float32)
        u = np.asarray(ast.literal_eval(first_row[u_col]), dtype=np.float32)
        v = np.asarray(ast.literal_eval(first_row[v_col]), dtype=np.float32)
        if not (len(x) == len(y) == len(u) == len(v)):
            raise ValueError(f"List lengths mismatch in {path}")
        coords = np.column_stack([x, y])
        uv = np.column_stack([u, v])
        return coords, uv

    coords = df[[x_col, y_col]].to_numpy(dtype=np.float32)
    uv = df[[u_col, v_col]].to_numpy(dtype=np.float32)
    return coords, uv
