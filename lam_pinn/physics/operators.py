from __future__ import annotations

import torch
from torch import nn


def pde_cal(
    xy_list: list[torch.Tensor],
    net: nn.Module,
    E: float,
    nu: float,
    f: float,
    h: float,
    b: float,
    k: float,
    direct: bool = False,
    bc: bool = False,
):
    x, y = xy_list
    xy = torch.cat([x, y], dim=1)

    uv_pred = net(xy)
    u = uv_pred[:, 0:1]
    v = uv_pred[:, 1:2]

    if direct:
        return u, v

    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_xy = torch.autograd.grad(u_x, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_xy = torch.autograd.grad(v_x, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    px = (E / (1.0 - nu**2)) * (u_xx + 0.5 * (1.0 - nu) * u_yy + 0.5 * (1.0 + nu) * v_xy)
    py = (E / (1.0 - nu**2)) * (v_yy + 0.5 * (1.0 - nu) * v_xx + 0.5 * (1.0 + nu) * u_xy)

    if bc:
        sigma_xx = (u_x + nu * v_y) * E / (1.0 - nu**2)
        sigma_yy = (v_y + nu * u_x) * E / (1.0 - nu**2)
        sigma_xy = (u_y + v_x) * E / (2.0 * (1.0 + nu))
        sigma_ex = (f * h * torch.cos((torch.pi * y) / (2.0 * b)) + k).reshape(-1, 1)
        return px, py, sigma_xx, sigma_ex, sigma_xy, sigma_yy

    return px, py
