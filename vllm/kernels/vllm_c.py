# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

rms_no_var_size = lambda x, w, e, var_size=None: var_size is None
"""vLLM Kernel does not support variance_size parameter."""


@ir.ops.rms_norm.register_impl(
    "vllm_c", supports_args=rms_no_var_size, supported=CUDA_ALIKE
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output


@ir.ops.rms_norm_gated.register_impl("triton", supported=CUDA_ALIKE)
def rms_norm_gated(
    x: Tensor,
    weight: Tensor,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> Tensor:
    from vllm.model_executor.layers.fla.ops.layernorm_guard import layer_norm_fwd

    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()

    out, _, _ = layer_norm_fwd(
        x, weight, None, epsilon,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )
    return out.reshape(x_shape_og)


rms_add_no_var_size = lambda x, x_res, w, e, var_size=None: var_size is None
"""vLLM Kernel does not support variance_size parameter."""


@ir.ops.fused_add_rms_norm.register_impl(
    "vllm_c",
    supports_args=rms_add_no_var_size,
    supported=CUDA_ALIKE,
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

    assert variance_size is None
    torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual
