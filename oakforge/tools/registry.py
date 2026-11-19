"""Named procedural tools available to recipes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from oakforge.core.context import GenerationContext

ToolFunction = Callable[[GenerationContext, dict[str, Any]], Any]


class ToolRegistry:
    """Resolve recipe tool names without hard-coding a growth sequence."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, function: ToolFunction) -> None:
        """Register a tool under the name used by YAML recipes."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = function

    def get(self, name: str) -> ToolFunction:
        """Return the requested tool or explain which name was not found."""
        try:
            return self._tools[name]
        except KeyError as error:
            available_names = ", ".join(sorted(self._tools))
            raise KeyError(f"Unknown tool '{name}'. Available tools: {available_names}") from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def create_default_registry() -> ToolRegistry:
    """Load the built-in tools while keeping each tool in its own source file."""
    from oakforge.tools.branching import spawn_branches
    from oakforge.tools.foliage import scatter_oak_leaves
    from oakforge.tools.twig import spawn_twigs
    from oakforge.tools.trunk import generate_trunk
    from oakforge.tools.roots import branch_surface_roots, spread_surface_roots
    from oakforge.tools.wood_surface import build_wood_surface

    registry = ToolRegistry()
    registry.register("trunk.generate", generate_trunk)
    registry.register("roots.spread", spread_surface_roots)
    registry.register("roots.branch", branch_surface_roots)
    registry.register("wood_surface.build", build_wood_surface)
    registry.register("branching.spawn", spawn_branches)
    registry.register("twig.spawn", spawn_twigs)
    registry.register("foliage.scatter_oak", scatter_oak_leaves)
    return registry
