# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from vllm.compilation.passes.base import PassManager
from vllm.compilation.passes.inductor_pass import (
    InductorPass,
    pass_context,
)
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import CompilationConfig, ModelConfig, VllmConfig
from vllm.config.utils import Range


# dummy custom pass that doesn't inherit
def simple_callable(graph: torch.fx.Graph):
    pass


def qualname(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


# Should fail to add directly to the pass manager
def test_bad_callable():
    pass_manager = PostGradPassManager.from_config(VllmConfig())

    with pytest.raises(AssertionError):
        pass_manager.add(simple_callable)  # type: ignore[arg-type]


# Pass that inherits from InductorPass
class ProperPass(InductorPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        pass


class OOTPassManager(PassManager):
    @classmethod
    def default_pass_pipeline(cls, config: VllmConfig):
        return [ProperPass]


def test_pass_manager_uuid():
    # Set the pass context as PassManager uuid uses it
    with pass_context(Range(start=1, end=8)):
        # Some passes need dtype to be set
        config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))

        pass_manager = PostGradPassManager.from_config(config)

        # Check that UUID is different if the same pass is added 2x
        pass_manager.add(ProperPass())
        uuid1 = pass_manager.uuid()
        pass_manager.add(ProperPass())
        uuid2 = pass_manager.uuid()
        assert uuid1 != uuid2

        # UUID should be the same as the original one,
        # as we constructed in the same way.
        pass_manager2 = PostGradPassManager.from_config(config)
        pass_manager2.add(ProperPass())
        assert uuid1 == pass_manager2.uuid()

        # UUID should be different due to config change
        config2 = copy.deepcopy(config)
        config2.compilation_config.pass_config.fuse_norm_quant = (
            not config2.compilation_config.pass_config.fuse_norm_quant
        )
        config2.compilation_config.pass_config.fuse_act_quant = (
            not config2.compilation_config.pass_config.fuse_act_quant
        )
        pass_manager3 = PostGradPassManager.from_config(config2)
        pass_manager3.add(ProperPass())
        assert uuid1 != pass_manager3.uuid()


def test_pass_manager_supports_explicit_pass_pipeline():
    config = VllmConfig(
        compilation_config=CompilationConfig(
            pass_pipeline=[
                ProperPass,
                PostCleanupPass,
                qualname(VllmIRLoweringPass),
                qualname(PostCleanupPass),
                FixFunctionalizationPass,
            ],
        )
    )

    pass_manager = PostGradPassManager.from_config(config)
    pass_ids = pass_manager.describe_pipeline()

    assert pass_ids == [
        qualname(ProperPass),
        qualname(PostCleanupPass),
        qualname(VllmIRLoweringPass),
        qualname(PostCleanupPass),
        qualname(FixFunctionalizationPass),
    ]


@pytest.mark.parametrize("pass_spec", [ProperPass(), simple_callable])
def test_pass_pipeline_rejects_unsupported_specs(pass_spec):
    with pytest.raises(ValidationError):
        CompilationConfig(pass_pipeline=[pass_spec])


def test_pass_pipeline_rejects_qualified_non_pass():
    config = VllmConfig(
        compilation_config=CompilationConfig(pass_pipeline=[qualname(simple_callable)])
    )

    with pytest.raises(TypeError):
        PostGradPassManager.from_config(config)


def test_base_pass_manager_supports_oot_default_pipeline():
    pass_manager = OOTPassManager.from_config(VllmConfig())

    assert pass_manager.describe_pipeline() == [qualname(ProperPass)]


def test_explicit_pass_pipeline_uuid_includes_pass_config():
    config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            pass_pipeline=[
                qualname(ProperPass),
                qualname(PostCleanupPass),
                qualname(VllmIRLoweringPass),
                qualname(PostCleanupPass),
                qualname(FixFunctionalizationPass),
            ],
        ),
    )
    config2 = copy.deepcopy(config)
    config2.compilation_config.pass_config.fuse_norm_quant = (
        not config2.compilation_config.pass_config.fuse_norm_quant
    )
    config2.compilation_config.pass_config.fuse_act_quant = (
        not config2.compilation_config.pass_config.fuse_act_quant
    )

    with pass_context(Range(start=1, end=8)):
        assert (
            PostGradPassManager.from_config(config).uuid()
            != PostGradPassManager.from_config(config2).uuid()
        )
