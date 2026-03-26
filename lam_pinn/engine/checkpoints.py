from __future__ import annotations

from pathlib import Path

import torch

from lam_pinn.models.serial_net import SerialNetwork


def save_checkpoint(path: str | Path, model: SerialNetwork, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    dropout_override: float | None = None,
) -> tuple[SerialNetwork, dict]:
    path = Path(path)
    payload = torch.load(path, map_location=device)
    model_hparams = payload["model_hparams"]
    model = SerialNetwork(
        num_clusters=int(payload["num_clusters"]),
        hidden_dim=int(model_hparams.get("hidden_dim", 64)),
        dropout_p=float(model_hparams.get("dropout_p", 0.2) if dropout_override is None else dropout_override),
        gate_init=float(model_hparams.get("gate_init", 0.5)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
