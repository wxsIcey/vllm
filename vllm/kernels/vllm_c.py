# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

rms_no_var_size = lambda x, weight, epsilon, variance_size=None: variance_size is None
"""vLLM kernel does not support variance_size parameter."""


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


mixer2_rms_norm_gated_has_weight = (
    lambda x, gate, weight, epsilon, group_size=None: weight is not None
)
"""Triton gated RMSNorm kernel requires a weight tensor."""


@ir.ops.mixer2_rms_norm_gated.register_impl(
    "triton", supports_args=mixer2_rms_norm_gated_has_weight, supported=CUDA_ALIKE
)
def mixer2_rms_norm_gated(
    x: Tensor,
    gate: Tensor,
    weight: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
) -> Tensor:
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated

    assert weight is not None
    return rms_norm_gated(
        x,
        weight,
        bias=None,
        z=gate,
        eps=epsilon,
        group_size=group_size,
        norm_before_gate=False,
    )
