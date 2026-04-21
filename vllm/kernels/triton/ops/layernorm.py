# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()

GPGPU_DEVICE = CUDA_ALIKE or current_platform.is_xpu()

"""Most kernels in this file are supported on all CUDA-alike platforms."""


mixer2_rms_norm_gated_has_weight = (
    lambda x, gate, weight, epsilon, group_size=None: weight is not None
)
"""Triton gated RMSNorm kernel requires a weight tensor."""


def _mixer2_rms_norm_gated_triton_impl(
    result: Tensor,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = -1,
) -> None:
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
        rms_norm_gated as _rms_norm_gated,
    )

    out = _rms_norm_gated(
        x,
        weight,
        bias=None,
        z=gate,
        eps=epsilon,
        group_size=None if group_size == -1 else group_size,
        norm_before_gate=False,
    )
    result.copy_(out)


def _mixer2_rms_norm_gated_triton_fake(
    result: Tensor,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = -1,
) -> None:
    pass


direct_register_custom_op(
    op_name="mixer2_rms_norm_gated_triton",
    op_func=_mixer2_rms_norm_gated_triton_impl,
    mutates_args=["result"],
    fake_impl=_mixer2_rms_norm_gated_triton_fake,
)


@ir.ops.mixer2_rms_norm_gated.register_impl(
    "triton",
    supports_args=mixer2_rms_norm_gated_has_weight,
    supported=GPGPU_DEVICE,
)
def mixer2_rms_norm_gated(
    x: Tensor,
    gate: Tensor,
    weight: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
) -> Tensor:
    assert weight is not None
    result = torch.empty_like(x)
    torch.ops.vllm.mixer2_rms_norm_gated_triton(
        result,
        x,
        gate,
        weight,
        epsilon,
        group_size if group_size is not None else -1,
    )
    return result
