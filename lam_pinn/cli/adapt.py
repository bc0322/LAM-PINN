from __future__ import annotations

import argparse

from lam_pinn.config import load_adapt_config
from lam_pinn.engine.adapt import adapt_single_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-task LAM-PINN adaptation")
    parser.add_argument("--config", required=True, help="Path to the adaptation YAML config")
    parser.add_argument("--case-index", type=int, default=None, help="Optional case index override")
    args = parser.parse_args()

    config = load_adapt_config(args.config)
    if args.case_index is not None:
        config.task.case_index = int(args.case_index)

    summary = adapt_single_task(config, config_path=args.config)

if __name__ == "__main__":
    main()
