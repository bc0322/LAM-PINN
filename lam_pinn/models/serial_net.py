from __future__ import annotations

import torch
from torch import nn


def _make_hidden_stack(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    dropout_p: float = 0.0,
    use_dropout_layers: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.Tanh())
        if use_dropout_layers:
            layers.append(nn.Dropout(dropout_p))
        in_features = hidden_dim
    return nn.Sequential(*layers)


class BasisNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = _make_hidden_stack(
            input_dim=2,
            hidden_dim=hidden_dim,
            depth=2,
            dropout_p=0.0,
            use_dropout_layers=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.net = _make_hidden_stack(
            input_dim=2,
            hidden_dim=hidden_dim,
            depth=2,
            dropout_p=dropout_p,
            use_dropout_layers=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackwardNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SerialNetwork(nn.Module):
    def __init__(self, num_clusters: int, hidden_dim: int = 64, dropout_p: float = 0.2, gate_init: float = 0.5) -> None:
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.basis = BasisNetwork(hidden_dim=hidden_dim)
        self.forward_nets = nn.ModuleList(
            [ForwardNetwork(hidden_dim=hidden_dim, dropout_p=dropout_p) for _ in range(self.num_clusters)]
        )
        self.backward = BackwardNetwork(hidden_dim=hidden_dim)
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.tensor([gate_init], dtype=torch.float32)) for _ in range(self.num_clusters)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis_out = self.basis(x)
        combined = basis_out
        for gate, forward_net in zip(self.gates, self.forward_nets):
            combined = combined + gate.clamp(0.0, 1.0) * forward_net(x)
        return self.backward(combined)

    def set_gate_pattern(self, active_idx: int | None = None, active_value: float = 1.0, inactive_value: float = 0.1) -> None:
        if active_idx is None:
            for gate in self.gates:
                gate.data.fill_(inactive_value)
        else:
            for index, gate in enumerate(self.gates):
                gate.data.fill_(active_value if index == active_idx else inactive_value)

    def reset_gates(self, value: float = 0.5) -> None:
        with torch.no_grad():
            for gate in self.gates:
                gate.fill_(value)

    def clamp_gates(self) -> None:
        with torch.no_grad():
            for gate in self.gates:
                gate.clamp_(0.0, 1.0)

    def gate_values(self) -> list[float]:
        return [float(gate.detach().cpu().item()) for gate in self.gates]
