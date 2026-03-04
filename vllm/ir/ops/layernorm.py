# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x


@register_op
def rms_norm_gated(
    x: Tensor,
    weight: Tensor,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> Tensor:
    """RMS normalization with optional SiLU gating.

    If z is not None:
      - norm_before_gate=True:  out = norm(x) * silu(z)
      - norm_before_gate=False: out = norm(x * silu(z))
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    z_f32 = z.to(torch.float32) if z is not None else None

    if z_f32 is not None and not norm_before_gate:
        x = x * F.silu(z_f32)

    if group_size is None:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
    else:
        from einops import rearrange

        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        variance = x_group.pow(2).mean(dim=-1, keepdim=True)
        x = rearrange(x_group * torch.rsqrt(variance + epsilon), "... g d -> ... (g d)")

    out = x.to(orig_dtype) * weight

    if z is not None and norm_before_gate:
        out = out * F.silu(z.to(orig_dtype))

    return out


@register_op(allow_inplace=True)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused add and weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)

    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight

    return x, x_residual
