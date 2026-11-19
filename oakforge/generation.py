"""Load a growth recipe and produce a renderer-independent tree model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oakforge.core.context import GenerationContext
from oakforge.core.models import Branch, Leaf, TreeModel
from oakforge.core.recipe_engine import RecipeEngine
from oakforge.tools.registry import create_default_registry


def load_recipe(recipe_path: str | Path) -> dict[str, Any]:
    """Read and validate the top-level shape of a YAML tree recipe."""
    with open(recipe_path, "r", encoding="utf-8") as recipe_file:
        recipe = yaml.safe_load(recipe_file)
    if not isinstance(recipe, dict) or "tree" not in recipe or "pipeline" not in recipe:
        raise ValueError("A recipe must contain both 'tree' and 'pipeline' sections")
    return recipe


def generate_tree(recipe: dict[str, Any]) -> TreeModel:
    """Execute the recipe's growth graph and collect the resulting tree parts."""
    tree_settings = recipe["tree"]
    seed = int(tree_settings.get("seed", 1))
    initial_values = {
        "tree": tree_settings,
        "groups": {"trunk": [], "roots": [], "primary": [], "leaders": [], "secondary": [], "tertiary": [], "twigs": [], "sprays": [], "leaves": []},
    }
    context = GenerationContext(values=initial_values, random=np.random.default_rng(seed))
    engine = RecipeEngine(context, create_default_registry())
    engine.execute(recipe["pipeline"])

    groups = context.values["groups"]
    branches: list[Branch] = []
    leaves: list[Leaf] = []
    for group_name in ("trunk", "roots", "primary", "leaders", "secondary", "tertiary", "twigs", "sprays"):
        group_items = groups.get(group_name, [])
        branches.extend(item for item in group_items if isinstance(item, Branch))
    leaves.extend(item for item in groups.get("leaves", []) if isinstance(item, Leaf))

    if not branches or branches[0].role != "trunk":
        raise ValueError("The recipe did not generate a trunk. Check its trunk tool call.")

    return TreeModel(
        name=str(tree_settings.get("name", "Procedural Tree")),
        branches=branches,
        leaves=leaves,
        metadata={
            "species": tree_settings.get("species", "unspecified"),
            "seed": seed,
            "branch_groups": {name: len(groups.get(name, [])) for name in groups if name != "leaves"},
            "leaf_count": len(leaves),
            "wood_surface": context.values.get("wood_surface"),
        },
    )


def generate_from_file(recipe_path: str | Path) -> TreeModel:
    """Convenience entry point for tools that consume YAML recipes."""
    return generate_tree(load_recipe(recipe_path))
