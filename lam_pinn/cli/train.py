from __future__ import annotations

import argparse

from lam_pinn.config import load_train_config
from lam_pinn.engine.meta_train import meta_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAM-PINN meta-training")
    parser.add_argument("--config", required=True, help="Path to the training YAML config")
    args = parser.parse_args()

    config = load_train_config(args.config)
    summary = meta_train(config, config_path=args.config)
    print(f"Training completed. Run directory: {summary['run_dir']}")
    print(f"Checkpoint: {summary['checkpoint_path']}")


if __name__ == "__main__":
    main()
