# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


def _apply_rotary_emb(
    x: Tensor, cos: Tensor, sin: Tensor, is_neox_style: bool
) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        output = torch.cat((o1, o2), dim=-1)
    else:
        output = torch.stack((o1, o2), dim=-1).flatten(-2)
    return output


@register_op
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor | None,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor | None]:
    """Apply rotary positional embeddings to query and key tensors."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    rotary_dim = cos_sin_cache.shape[-1]

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key
