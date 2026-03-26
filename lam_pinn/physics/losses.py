from __future__ import annotations

import torch
from torch import nn

from lam_pinn.physics.operators import pde_cal


def task_loss(
    net: nn.Module,
    domain_points: list[torch.Tensor],
    bc1_y0: list[torch.Tensor],
    bc2_x0: list[torch.Tensor],
    bc3_yb: list[torch.Tensor],
    bc4_xa: list[torch.Tensor],
    scale: float,
    properties: dict[str, float],
) -> torch.Tensor:
    E = properties["E"]
    nu = properties["nu"]
    f = properties["f"]
    h = properties["h"]
    b = properties["b"]
    k = properties["k"]

    px, py = pde_cal(domain_points, net, E, nu, f, h, b, k, direct=False, bc=False)
    _, v_on_y0 = pde_cal(bc1_y0, net, E, nu, f, h, b, k, direct=True, bc=False)
    u_on_x0, _ = pde_cal(bc2_x0, net, E, nu, f, h, b, k, direct=True, bc=False)
    b3px, b3py, _, _, sigma_yx, sigma_yy = pde_cal(bc3_yb, net, E, nu, f, h, b, k, direct=False, bc=True)
    b4px, b4py, sigma_xx, sigma_ex, sigma_xy, _ = pde_cal(bc4_xa, net, E, nu, f, h, b, k, direct=False, bc=True)

    pde_residual = torch.mean(px**2) + torch.mean(py**2)
    bc1_loss = torch.mean(v_on_y0**2)
    bc2_loss = torch.mean(u_on_x0**2)
    bc3_loss = torch.mean(sigma_yy**2) + torch.mean(sigma_yx**2)
    bc4_loss = torch.mean((sigma_xx - sigma_ex) ** 2) + torch.mean(sigma_xy**2)
    bc_pde = torch.mean(b3px**2) + torch.mean(b3py**2) + torch.mean(b4px**2) + torch.mean(b4py**2)

    return pde_residual + scale * (bc1_loss + bc2_loss + bc3_loss + bc4_loss + bc_pde)
