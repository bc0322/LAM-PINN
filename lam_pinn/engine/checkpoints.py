from __future__ import annotations

from pathlib import Path

import torch

from lam_pinn.models.serial_net import SerialNetwork


def save_checkpoint(path: str | Path, model: SerialNetwork, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)


def _legacy_state_dict_to_current(state_dict: dict) -> dict:
    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("forward1."):
            new_key = new_key.replace("forward1.", "forward_nets.0.")
        elif new_key.startswith("forward2."):
            new_key = new_key.replace("forward2.", "forward_nets.1.")
        elif new_key.startswith("forward3."):
            new_key = new_key.replace("forward3.", "forward_nets.2.")
        elif new_key == "n0":
            new_key = "gates.0"
        elif new_key == "n1":
            new_key = "gates.1"
        elif new_key == "n2":
            new_key = "gates.2"

        if new_key.startswith("gates.") and getattr(value, "ndim", None) == 0:
            value = value.view(1)

        mapped[new_key] = value
    return mapped


def load_legacy_warm_start(model: SerialNetwork, path: str | Path, device: torch.device) -> None:
    path = Path(path)
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    model.load_state_dict(_legacy_state_dict_to_current(payload), strict=True)


def save_legacy_split_weights(path: str | Path, model: SerialNetwork) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.basis.state_dict(), path / "basis_net.pth")
    torch.save(model.forward_nets[0].state_dict(), path / "forward_net_0.pth")
    torch.save(model.forward_nets[1].state_dict(), path / "forward_net_1.pth")
    torch.save(model.forward_nets[2].state_dict(), path / "forward_net_2.pth")
    torch.save(model.backward.state_dict(), path / "backward_net.pth")


def load_split_weight_directory(
    path: str | Path,
    device: torch.device,
    hidden_dim: int = 64,
    dropout_p: float = 0.0,
    gate_init: float = 0.5,
) -> tuple[SerialNetwork, dict]:
    path = Path(path)
    model = SerialNetwork(
        num_clusters=3,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        gate_init=gate_init,
    ).to(device)

    model.basis.load_state_dict(torch.load(path / "basis_net.pth", map_location=device))
    model.forward_nets[0].load_state_dict(torch.load(path / "forward_net_0.pth", map_location=device))
    model.forward_nets[1].load_state_dict(torch.load(path / "forward_net_1.pth", map_location=device))
    model.forward_nets[2].load_state_dict(torch.load(path / "forward_net_2.pth", map_location=device))
    model.backward.load_state_dict(torch.load(path / "backward_net.pth", map_location=device))

    payload = {
        "num_clusters": 3,
        "cluster_ids": [0, 1, 2],
        "cluster_to_index": {0: 0, 1: 1, 2: 2},
        "model_hparams": {
            "hidden_dim": hidden_dim,
            "dropout_p": dropout_p,
            "gate_init": gate_init,
        },
    }
    return model, payload


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    dropout_override: float | None = None,
) -> tuple[SerialNetwork, dict]:
    path = Path(path)

    if path.is_dir():
        effective_dropout = 0.2 if dropout_override is None else float(dropout_override)
        return load_split_weight_directory(
            path,
            device=device,
            hidden_dim=64,
            dropout_p=effective_dropout,
            gate_init=0.5,
        )

    payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload and "model_hparams" in payload:
        model_hparams = payload["model_hparams"]
        model = SerialNetwork(
            num_clusters=int(payload["num_clusters"]),
            hidden_dim=int(model_hparams.get("hidden_dim", 64)),
            dropout_p=float(
                model_hparams.get("dropout_p", 0.2)
                if dropout_override is None else dropout_override
            ),
            gate_init=float(model_hparams.get("gate_init", 0.5)),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        return model, payload

    legacy_state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    effective_dropout = 0.2 if dropout_override is None else float(dropout_override)
    model = SerialNetwork(num_clusters=3, hidden_dim=64, dropout_p=effective_dropout, gate_init=0.5).to(device)
    model.load_state_dict(_legacy_state_dict_to_current(legacy_state_dict), strict=True)
    return model, {
        "num_clusters": 3,
        "cluster_ids": [0, 1, 2],
        "cluster_to_index": {0: 0, 1: 1, 2: 2},
        "model_hparams": {"hidden_dim": 64, "dropout_p": effective_dropout, "gate_init": 0.5},
    }
