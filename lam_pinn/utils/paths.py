from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def make_run_dir(output_root: str | Path, run_name: str) -> Path:
    output_root = Path(output_root)
    candidate = output_root / run_name
    if candidate.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = output_root / f"{run_name}_{suffix}"

    for subdir in ["checkpoints", "logs", "artifacts", "figures"]:
        (candidate / subdir).mkdir(parents=True, exist_ok=True)
    return candidate


def snapshot_file(src: str | Path | None, dst: str | Path) -> None:
    if src is None:
        return
    src_path = Path(src)
    if src_path.exists():
        shutil.copy2(src_path, dst)
