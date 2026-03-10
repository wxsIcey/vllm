from torch import Tensor

from vllm import ir


@ir.ops.rms_norm_gated.register_impl("triton", supported=True)
def rms_norm_gated(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "",
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

    if bias is not None:
        bias = bias.contiguous()

    y, _, _ = layer_norm_fwd(
        x,
        weight,
        bias,
        epsilon,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
        activation=activation,
    )
    return y.reshape(x_shape_og)
