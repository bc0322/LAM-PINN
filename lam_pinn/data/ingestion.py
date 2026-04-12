from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from lam_pinn.config import PhysicsConfig


TRAIN_COLUMN_ALIASES = {
    "E_raw": ["E_raw", "E"],
    "f_raw": ["f_raw", "f"],
    "k_raw": ["k_raw", "k"],
    "Initial_L1": ["Initial_L1"],
    "Final_L2": ["Final_L2"],
    "Avg_L3": ["Avg_L3"],
    "Cluster": ["Cluster"],
}
REQUIRED_TRAIN_COLUMNS = list(TRAIN_COLUMN_ALIASES.keys())

_COORD_ALIASES_X = {"x", "coord_x", "x_coord"}
_COORD_ALIASES_Y = {"y", "coord_y", "y_coord"}
_UV_ALIASES_U = {"u", "ux", "disp_x", "u_x"}
_UV_ALIASES_V = {"v", "uy", "disp_y", "u_y"}


def _resolve_train_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical_name, aliases in TRAIN_COLUMN_ALIASES.items():
        for candidate in aliases:
            if candidate in df.columns:
                resolved[canonical_name] = candidate
                break
    missing = [column for column in REQUIRED_TRAIN_COLUMNS if column not in resolved]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")
    return resolved


def read_training_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Training CSV is empty: {path}")

    column_map = _resolve_train_columns(df)
    rename_map = {
        source_name: canonical_name
        for canonical_name, source_name in column_map.items()
        if source_name != canonical_name
    }
    df = df.rename(columns=rename_map).copy()

    cluster_values = pd.to_numeric(df["Cluster"], errors="coerce")
    if cluster_values.isna().any():
        raise ValueError("Column 'Cluster' must contain numeric values only.")

    df["Cluster"] = np.round(cluster_values).astype(int)
    return df


def get_cluster_ids(df: pd.DataFrame) -> list[int]:
    return sorted(int(value) for value in pd.unique(df["Cluster"]))


def get_cluster_index_map(cluster_ids: list[int]) -> dict[int, int]:
    return {int(cluster_id): index for index, cluster_id in enumerate(sorted(cluster_ids))}


def add_affinity_distance(df: pd.DataFrame, reference_row_index: int = 13) -> pd.DataFrame:
    out = df.copy()
    cols = ["Initial_L1", "Final_L2", "Avg_L3"]
    if reference_row_index < 0 or reference_row_index >= len(out):
        raise IndexError(
            f"reference_row_index={reference_row_index} is out of bounds for training CSV with {len(out)} rows."
        )
    origin = out.loc[reference_row_index, cols].to_numpy(dtype=np.float64)
    points = out[cols].to_numpy(dtype=np.float64)
    out["Euclidean_Distance"] = np.linalg.norm(points - origin[None, :], axis=1)
    return out


def build_balanced_training_dataframe(
    df: pd.DataFrame,
    cluster_ids: list[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Legacy solid-notebook ordering:
    1) fixed cluster order: 2 -> 1 -> 0
    2) within each cluster, consume the smallest Euclidean_Distance first
    3) when a cluster is exhausted, draw one row from the original cluster table without replacement;
       if that source is also exhausted, fall back to the smallest-distance row already present.
    """
    np.random.seed(seed)

    cluster_order = [2, 1, 0]
    missing_clusters = [cluster_id for cluster_id in cluster_order if cluster_id not in cluster_ids]
    if missing_clusters:
        raise ValueError(f"Training CSV is missing expected clusters for legacy ordering: {missing_clusters}")

    data2 = df.copy()
    cluster_frames = {cluster_id: data2[data2["Cluster"] == cluster_id].copy() for cluster_id in cluster_order}
    if any(frame.empty for frame in cluster_frames.values()):
        empty_ids = [cluster_id for cluster_id, frame in cluster_frames.items() if frame.empty]
        raise ValueError(f"Clusters have no rows: {empty_ids}")

    min_length = min(len(frame) for frame in cluster_frames.values())
    copy_frames = {cluster_id: frame.copy() for cluster_id, frame in cluster_frames.items()}

    extracted_rows: list[pd.DataFrame] = []
    for _ in range(min_length):
        for cluster_id in cluster_order:
            row = copy_frames[cluster_id].sort_values(by="Euclidean_Distance").head(1)
            extracted_rows.append(row)
            copy_frames[cluster_id].drop(row.index, inplace=True)

    data_new = pd.concat(extracted_rows, ignore_index=True)

    longest_length = max(len(frame) for frame in cluster_frames.values())
    original_lengths = {cluster_id: len(frame) for cluster_id, frame in cluster_frames.items()}
    mutable_source = data2.copy()

    for _ in range(longest_length - min_length):
        for cluster_id in cluster_order:
            current_cluster_count = len(data_new[data_new["Cluster"] == cluster_id])

            if original_lengths[cluster_id] > current_cluster_count:
                if not copy_frames[cluster_id].empty:
                    row = copy_frames[cluster_id].sort_values(by="Euclidean_Distance").head(1)
                    data_new = pd.concat([data_new, row], ignore_index=True)
                    copy_frames[cluster_id].drop(row.index, inplace=True)
            else:
                source_rows = mutable_source[mutable_source["Cluster"] == cluster_id]
                if not source_rows.empty:
                    row = source_rows.sample(n=1)
                    data_new = pd.concat([data_new, row], ignore_index=True)
                    mutable_source.drop(row.index, inplace=True)
                else:
                    refill = data_new[data_new["Cluster"] == cluster_id].copy()
                    row = refill.sort_values(by="Euclidean_Distance").head(1)
                    data_new = pd.concat([data_new, row], ignore_index=True)

    return data_new.reset_index(drop=True)


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


def build_task_properties(task, physics: PhysicsConfig) -> dict[str, float]:
    return {
        "E": float(task.E),
        "nu": float(physics.nu),
        "a": float(physics.a),
        "b": float(physics.b),
        "f": float(task.f),
        "k": float(task.k),
        "h": float(physics.h),
    }


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def _find_first_column(columns: list[str], aliases: set[str]) -> str | None:
    normalized = {_normalize_header(column): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _parse_cell_to_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return np.asarray(parsed, dtype=np.float64).reshape(-1)
    return np.asarray([value], dtype=np.float64).reshape(-1)


def load_coords_uv_from_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Ground-truth CSV is empty: {path}")

    columns = list(df.columns)
    x_col = _find_first_column(columns, _COORD_ALIASES_X)
    y_col = _find_first_column(columns, _COORD_ALIASES_Y)
    u_col = _find_first_column(columns, _UV_ALIASES_U)
    v_col = _find_first_column(columns, _UV_ALIASES_V)

    if any(column is None for column in [x_col, y_col, u_col, v_col]):
        raise ValueError(
            "Ground-truth CSV must contain coordinate/displacement columns. "
            f"Found columns: {columns}"
        )

    if len(df) == 1:
        x = _parse_cell_to_array(df.iloc[0][x_col])
        y = _parse_cell_to_array(df.iloc[0][y_col])
        u = _parse_cell_to_array(df.iloc[0][u_col])
        v = _parse_cell_to_array(df.iloc[0][v_col])
    else:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
        u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=np.float64)
        v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=np.float64)

    if not (len(x) == len(y) == len(u) == len(v)):
        raise ValueError(
            f"Ground-truth CSV columns have mismatched lengths: x={len(x)}, y={len(y)}, u={len(u)}, v={len(v)}"
        )

    coords = np.column_stack([x, y]).astype(np.float64)
    uv = np.column_stack([u, v]).astype(np.float64)
    return coords, uv
