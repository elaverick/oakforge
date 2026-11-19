"""Fine twig formation, kept separate from the larger branching tools."""

from __future__ import annotations

from typing import Any

from oakforge.core.context import GenerationContext
from oakforge.core.models import Branch
from oakforge.tools.branching import create_child_branches


def spawn_twigs(context: GenerationContext, arguments: dict[str, Any]) -> list[Branch]:
    """Create the small terminal shoots that support oak foliage."""
    parent = arguments.get("parent")
    if not isinstance(parent, Branch):
        raise TypeError("twig.spawn expects 'parent' to reference one Branch")
    twig_arguments = dict(arguments)
    role = str(twig_arguments.get("role", "twig"))
    twig_arguments.setdefault("radius_scale", 0.28)
    twig_arguments.setdefault("radius_tip_scale", 0.18)
    twig_arguments.setdefault("minimum_base_radius", 0.014)
    twig_arguments.setdefault("minimum_tip_radius", 0.006)
    twig_arguments.setdefault("junction_swell", 0.42)
    twig_arguments.setdefault("junction_decay", 0.16)
    twig_arguments.setdefault("segments", 6)
    return create_child_branches(context, parent, twig_arguments, role, default_depth=3)
