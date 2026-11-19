"""Command-line entry point for OakForge procedural tree generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from oakforge.core.mesh import build_tree_mesh, write_obj
from oakforge.generation import generate_from_file, generate_tree, load_recipe
from oakforge.render.opengl_viewer import OpenGLViewer


def parse_arguments() -> argparse.Namespace:
    """Read generation, export and viewing options from the command line."""
    argument_parser = argparse.ArgumentParser(
        description="Generate and inspect a YAML-described procedural English oak tree."
    )
    argument_parser.add_argument(
        "--config", default="configs/english_oak.yaml",
        help="YAML growth recipe to execute (default: configs/english_oak.yaml)",
    )
    argument_parser.add_argument(
        "--export-obj", metavar="PATH",
        help="Export the generated tree as an OBJ file and exit without opening a window.",
    )
    argument_parser.add_argument(
        "--stats", action="store_true",
        help="Print generated component counts and exit unless --export-obj is also supplied.",
    )
    argument_parser.add_argument(
        "--no-viewer", action="store_true",
        help="Generate and report the tree without starting the interactive viewer.",
    )
    return argument_parser.parse_args()


def main() -> int:
    """Generate one oak, then export it or pass it to the OpenGL viewer."""
    arguments = parse_arguments()
    recipe_path = Path(arguments.config)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")

    tree = generate_from_file(recipe_path)
    print(f"Generated: {tree.name} ({tree.metadata.get('species', 'unknown species')})")
    print(f"  Branches : {len(tree.branches)}")
    print(f"  Leaves   : {len(tree.leaves)}")
    print(f"  Groups   : {tree.metadata.get('branch_groups', {})}")

    # Count-only inspection should not allocate millions of render vertices.
    if arguments.stats and not arguments.export_obj:
        return 0

    mesh = build_tree_mesh(tree)
    print(f"  Triangles: {mesh.triangle_count}")

    if arguments.export_obj:
        export_path = Path(arguments.export_obj)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_obj(mesh, str(export_path))
        print(f"OBJ exported to: {export_path.resolve()}")
    if arguments.no_viewer or arguments.stats or arguments.export_obj:
        return 0

    generation_seed = int(tree.metadata.get("seed", 1))

    def regenerate_tree():
        nonlocal generation_seed
        generation_seed += 1
        recipe = load_recipe(recipe_path)
        recipe["tree"]["seed"] = generation_seed
        return generate_tree(recipe)

    OpenGLViewer(tree, regenerate_tree).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
