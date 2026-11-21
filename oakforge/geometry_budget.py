"""Build and report the runtime geometry budgets for one procedural tree.

Run with:
    python -m oakforge.geometry_budget --instances 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oakforge.generation import generate_from_file
from oakforge.progressive import build_lod_representation, generate_coarse_tree


def build_geometry_budget_report(
    recipe_path: str | Path,
    instances: int = 1,
    triangle_budget_per_tree: int = 200_000,
) -> dict[str, Any]:
    """Build all four runtime LODs and report their measured buffer budgets."""
    if instances < 1:
        raise ValueError("instances must be at least one")
    if triangle_budget_per_tree < 1:
        raise ValueError("triangle_budget_per_tree must be positive")

    full_tree = generate_from_file(recipe_path)
    coarse_tree = generate_coarse_tree(recipe_path)

    representations = {
        3: build_lod_representation(coarse_tree, 3),
    }
    lod2 = build_lod_representation(full_tree, 2)
    representations[2] = lod2
    representations[1] = build_lod_representation(
        full_tree, 1, shared_wood_mesh=lod2.wood_mesh
    )
    representations[0] = build_lod_representation(
        full_tree, 0, shared_wood_mesh=lod2.wood_mesh
    )

    levels: list[dict[str, Any]] = []
    for level in (3, 2, 1, 0):
        representation = representations[level]
        wood_triangles = representation.wood_mesh.triangle_count
        foliage_triangles = representation.foliage_mesh.triangle_count
        total_triangles = wood_triangles + foliage_triangles
        if total_triangles > triangle_budget_per_tree:
            raise RuntimeError(
                f"LOD{level} uses {total_triangles:,} triangles, exceeding the "
                f"{triangle_budget_per_tree:,}-triangle per-tree budget"
            )
        levels.append({
            "lod": level,
            "name": representation.name,
            "wood_triangles": wood_triangles,
            "foliage_triangles": foliage_triangles,
            "leaf_cards": representation.foliage_mesh.leaf_count,
            "source_leaf_placements": representation.foliage_mesh.source_leaf_count,
            "triangles_per_tree": total_triangles,
            "triangles_for_instances": total_triangles * instances,
            "buffer_bytes_per_tree": representation.cpu_bytes,
            "buffer_mib_per_tree": round(representation.cpu_bytes / (1024 ** 2), 2),
            "mesh_build_seconds": round(representation.mesh_build_seconds, 3),
        })

    return {
        "recipe": str(recipe_path),
        "instances": instances,
        "canonical_source_leaves": len(full_tree.leaves),
        "levels": levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure OakForge runtime LOD triangle and buffer budgets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/english_oak.yaml"),
        help="Tree recipe YAML (default: configs/english_oak.yaml)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=16,
        help="Number of identical tree instances to aggregate (default: 16)",
    )
    parser.add_argument(
        "--max-triangles-per-tree",
        type=int,
        default=200_000,
        help="Fail if any runtime LOD exceeds this per-tree triangle budget",
    )
    args = parser.parse_args()
    report = build_geometry_budget_report(
        args.config, args.instances, args.max_triangles_per_tree
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
