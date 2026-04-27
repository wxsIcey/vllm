# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from torch import fx

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.config import set_current_vllm_config
from vllm.utils.import_utils import resolve_obj_by_qualname

from .inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)


def _qualname(obj: type[Any]) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


PassSpec: TypeAlias = str | type[InductorPass]


def _resolve_pass(pass_like: PassSpec) -> tuple[str, InductorPass]:
    pass_cls = (
        resolve_obj_by_qualname(pass_like) if isinstance(pass_like, str) else pass_like
    )

    if not isinstance(pass_cls, type) or not issubclass(pass_cls, InductorPass):
        raise TypeError(
            "Pass pipeline entries must be fully-qualified InductorPass "
            f"class names or InductorPass classes, got {pass_like!r}."
        )

    pass_id = pass_like if isinstance(pass_like, str) else _qualname(pass_cls)
    return pass_id, pass_cls()


class PassManager(CustomGraphPass):  # type: ignore[misc]
    """A reusable ordered container for Inductor passes."""

    _SCHEMA_VERSION = 1

    def __init__(self, passes: list[tuple[str, InductorPass]] | None = None) -> None:
        self.passes: list[tuple[str, InductorPass]] = list(passes or [])
        self.hook: str = ""
        self._config_hash: str = ""

    def bind_to_hook(self, hook: str) -> PassManager:
        self.hook = hook
        return self

    @classmethod
    def from_config(cls, config: VllmConfig, hook: str | None = None) -> PassManager:
        """Build a configured pass manager from ``VllmConfig`` for a hook."""
        pass_pipeline = config.compilation_config.pass_pipeline
        manager = cls()
        if hook is not None:
            manager.bind_to_hook(hook)
        manager._config_hash = cls.compute_config_hash(config, pass_pipeline)

        with set_current_vllm_config(config, check_compile=False):
            if pass_pipeline is None:
                pass_pipeline = cls.default_pass_pipeline(config)
            for item in pass_pipeline:
                manager.passes.append(_resolve_pass(item))

        return manager

    @classmethod
    def compute_config_hash(
        cls, config: VllmConfig, pass_pipeline: list[PassSpec] | None
    ) -> str:
        """Return config state that should participate in this manager's uuid."""
        return ""

    @classmethod
    def default_pass_pipeline(cls, config: VllmConfig) -> list[PassSpec]:
        """Return the manager's default pipeline when no override is provided."""
        return []

    def describe_pipeline(self) -> list[str]:
        return [pass_id for pass_id, _ in self.passes]

    def __call__(self, graph: fx.Graph) -> None:
        compile_range = get_pass_context().compile_range
        for _, pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)

    def uuid(self) -> str:
        # The compile range is part of the cache key, so uuid() must be called
        # under pass_context(...), just like pass execution itself.
        state: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "manager": f"{type(self).__module__}.{type(self).__qualname__}",
            "hook": self.hook,
            "config_hash": self._config_hash,
            "compile_range": str(get_pass_context().compile_range),
            "passes": [
                {
                    "pass_id": pass_id,
                    "uuid": pass_.uuid(),
                }
                for pass_id, pass_ in self.passes
            ],
        }
        return InductorPass.hash_dict(state)
