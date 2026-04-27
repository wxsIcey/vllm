# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from torch import fx as fx

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import set_env_var

from .base import PassManager, PassSpec
from .ir.lowering_pass import VllmIRLoweringPass
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

if rocm_aiter_ops.is_enabled():
    from .fusion.rocm_aiter_fusion import (
        MLADualRMSNormFusionPass,
        RocmAiterRMSNormQuantFusionPass,
        RocmAiterSiluMulFp8GroupQuantFusionPass,
        RocmAiterTritonAddRMSNormPadFusionPass,
    )

if current_platform.is_cuda_alike():
    from .fusion.act_quant_fusion import ActivationQuantFusionPass
    from .fusion.attn_quant_fusion import AttnQuantFusionPass
    from .fusion.mla_attn_quant_fusion import MLAAttnQuantFusionPass
    from .fusion.qk_norm_rope_fusion import QKNormRoPEFusionPass
    from .fusion.rms_quant_fusion import RMSNormQuantFusionPass
    from .fusion.rope_kvcache_fusion import RopeKVCacheFusionPass
    from .fusion.sequence_parallelism import SequenceParallelismPass
    from .utility.scatter_split_replace import ScatterSplitReplacementPass
    from .utility.split_coalescing import SplitCoalescingPass

if current_platform.is_cuda():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass
    from .fusion.collective_fusion import AsyncTPPass
    from .fusion.minimax_qk_norm_fusion import MiniMaxQKNormPass

from .inductor_pass import InductorPass, get_pass_context
from .utility.fix_functionalization import FixFunctionalizationPass
from .utility.noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_pattern_match_debug(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if (debug_val := envs.VLLM_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


class PostGradPassManager(PassManager):  # type: ignore[misc]
    """The default vLLM pass manager for post-grad passes."""

    @classmethod
    def compute_config_hash(
        cls, config: VllmConfig, pass_pipeline: list[PassSpec] | None
    ) -> str:
        # PassConfig can affect pass behavior even when the pass list is
        # explicitly provided, so include it until passes fully self-report
        # their config dependencies in uuid().
        return config.compilation_config.pass_config.compute_hash()

    @classmethod
    def default_pass_pipeline(cls, config: VllmConfig) -> list[InductorPass]:
        pass_config = config.compilation_config.pass_config
        pass_pipeline: list[PassSpec] = []

        if pass_config.eliminate_noops:
            pass_pipeline.append(NoOpEliminationPass)

        if pass_config.enable_sp and current_platform.is_cuda_alike():
            pass_pipeline.append(SequenceParallelismPass)
            if pass_config.fuse_gemm_comms and current_platform.is_cuda():
                pass_pipeline.append(AsyncTPPass)

        if pass_config.fuse_allreduce_rms and current_platform.is_cuda():
            pass_pipeline.append(AllReduceFusionPass)

        if pass_config.fuse_minimax_qk_norm and current_platform.is_cuda():
            pass_pipeline.append(MiniMaxQKNormPass)

        if pass_config.fuse_norm_quant and current_platform.is_cuda_alike():
            pass_pipeline.append(RMSNormQuantFusionPass)
            if rocm_aiter_ops.is_enabled():
                pass_pipeline.append(RocmAiterRMSNormQuantFusionPass)

        if pass_config.fuse_act_quant and current_platform.is_cuda_alike():
            pass_pipeline.append(ActivationQuantFusionPass)
            if rocm_aiter_ops.is_enabled():
                pass_pipeline.append(RocmAiterSiluMulFp8GroupQuantFusionPass)

        if pass_config.fuse_act_padding and rocm_aiter_ops.is_enabled():
            pass_pipeline.append(RocmAiterTritonAddRMSNormPadFusionPass)

        if pass_config.fuse_mla_dual_rms_norm and rocm_aiter_ops.is_enabled():
            pass_pipeline.append(MLADualRMSNormFusionPass)

        if pass_config.fuse_rope_kvcache and current_platform.is_cuda_alike():
            pass_pipeline.append(SplitCoalescingPass)
            pass_pipeline.append(ScatterSplitReplacementPass)
            pass_pipeline.append(RopeKVCacheFusionPass)

        if pass_config.fuse_attn_quant and current_platform.is_cuda_alike():
            pass_pipeline.append(AttnQuantFusionPass)
            pass_pipeline.append(MLAAttnQuantFusionPass)

        if pass_config.enable_qk_norm_rope_fusion and current_platform.is_cuda_alike():
            pass_pipeline.append(SplitCoalescingPass)
            pass_pipeline.append(QKNormRoPEFusionPass)

        pass_pipeline.append(PostCleanupPass)
        pass_pipeline.append(VllmIRLoweringPass)
        pass_pipeline.append(PostCleanupPass)
        pass_pipeline.append(FixFunctionalizationPass)
        return pass_pipeline

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph) -> None:
        VllmInductorPass.dump_prefix = 0
        compile_range = get_pass_context().compile_range
        try:
            for pass_id, pass_ in self.passes:
                if pass_.is_applicable_for_range(compile_range):
                    pass_(graph)
                    VllmInductorPass.dump_prefix += 1
                else:
                    logger.debug(
                        "Skipping %s (%s) with compile range %s",
                        pass_,
                        pass_id,
                        compile_range,
                    )
        finally:
            VllmInductorPass.dump_prefix = None
            VllmPatternMatcherPass.log_match_summary()
