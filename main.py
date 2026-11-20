"""Command-line entry point for OakForge procedural tree generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from oakforge.core.mesh import build_tree_mesh, write_obj
from oakforge.generation import generate_from_file, generate_tree, load_recipe
from oakforge.render.impostor import ImpostorSettings, bake_depth_impostor
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
    argument_parser.add_argument(
        "--impostor-views", type=int, default=64,
        help="Number of canopy capture directions used by the interactive viewer (default: 64).",
    )
    argument_parser.add_argument(
        "--impostor-tile-size", type=int, default=256,
        help="Pixels per square canopy capture used by the interactive viewer (default: 256).",
    )
    return argument_parser.parse_args()


def _impostor_cache_directory(
    recipe_path: Path, seed: int, views: int, tile_size: int
) -> Path:
    """Create a deterministic cache path that changes when the recipe changes."""
    recipe_bytes = recipe_path.read_bytes()
    recipe_hash = hashlib.sha256(recipe_bytes).hexdigest()[:12]
    name = (
        f"{recipe_path.stem}_seed{seed}_views{views}_tile{tile_size}_{recipe_hash}"
    )
    return Path("build") / "canopy_impostors" / name


def _is_valid_impostor_cache(
    directory: Path, seed: int, views: int, tile_size: int, leaf_count: int
) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = manifest["settings"]
        textures = manifest["textures"]
        return (
            manifest.get("format") == "oakforge-canopy-depth-impostor-v2"
            and manifest.get("tree_seed") == seed
            and manifest.get("source_geometry", {}).get("leaf_count") == leaf_count
            and settings.get("views") == views
            and settings.get("tile_size") == tile_size
            and settings.get("include_twigs") is False
            and isinstance(manifest.get("depth_projection"), dict)
            and all((directory / textures[name]).is_file()
                    for name in ("colour", "normal", "depth"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _ensure_impostor_cache(
    tree,
    recipe_path: Path,
    views: int,
    tile_size: int,
) -> Path:
    """Bake the canopy once, reusing a valid local asset cache thereafter."""
    seed = int(tree.metadata.get("seed", 1))
    directory = _impostor_cache_directory(recipe_path, seed, views, tile_size)
    if _is_valid_impostor_cache(directory, seed, views, tile_size, len(tree.leaves)):
        print(f"Using cached canopy impostor: {directory}")
        return directory

    print(
        f"Baking canopy depth impostor ({views} views, {tile_size}px per view). "
        "The first run can take a little while..."
    )
    manifest = bake_depth_impostor(
        tree,
        directory,
        ImpostorSettings(views=views, tile_size=tile_size, atlas_columns=8),
    )
    print(
        f"Canopy atlas ready: {directory} "
        f"({len(manifest['views'])} views, {len(tree.leaves):,} leaves)"
    )
    return directory


def main() -> int:
    """Generate a tree, then export it or open the comparison viewer."""
    arguments = parse_arguments()
    recipe_path = Path(arguments.config)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")

    tree = generate_from_file(recipe_path)
    print(f"Generated: {tree.name} ({tree.metadata.get('species', 'unknown species')})")
    print(f"  Branches : {len(tree.branches)}")
    print(f"  Leaves   : {len(tree.leaves)}")
    print(f"  Groups   : {tree.metadata.get('branch_groups', {})}")

    if arguments.stats:
        return 0

    if arguments.export_obj or arguments.no_viewer:
        mesh = build_tree_mesh(tree)
        print(f"  Triangles: {mesh.triangle_count:,}")
        if arguments.export_obj:
            export_path = Path(arguments.export_obj)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            write_obj(mesh, str(export_path))
            print(f"OBJ exported to: {export_path.resolve()}")
        return 0

    if arguments.impostor_views < 8:
        raise ValueError("--impostor-views must be at least 8")
    if arguments.impostor_tile_size < 16 or (
        arguments.impostor_tile_size & (arguments.impostor_tile_size - 1)
    ):
        raise ValueError("--impostor-tile-size must be a power of two and at least 16")

    impostor_directory = _ensure_impostor_cache(
        tree, recipe_path, arguments.impostor_views, arguments.impostor_tile_size
    )
    generation_seed = int(tree.metadata.get("seed", 1))

    def regenerate_tree():
        nonlocal generation_seed
        generation_seed += 1
        recipe = load_recipe(recipe_path)
        recipe["tree"]["seed"] = generation_seed
        return generate_tree(recipe)

    def rebake_impostor(regenerated_tree):
        """Bake a regenerated seed in a child process so the viewer context survives."""
        seed = int(regenerated_tree.metadata.get("seed", 1))
        output_directory = _impostor_cache_directory(
            recipe_path, seed, arguments.impostor_views, arguments.impostor_tile_size
        )
        if _is_valid_impostor_cache(
            output_directory, seed, arguments.impostor_views,
            arguments.impostor_tile_size, len(regenerated_tree.leaves),
        ):
            return output_directory
        command = [
            sys.executable, "-m", "oakforge.render.impostor",
            "--config", str(recipe_path),
            "--output", str(output_directory),
            "--seed", str(seed),
            "--views", str(arguments.impostor_views),
            "--tile-size", str(arguments.impostor_tile_size),
            "--columns", "8",
        ]
        subprocess.run(command, check=True)
        return output_directory

    OpenGLViewer(
        tree,
        regenerate_tree,
        impostor_directory,
        rebake_callback=rebake_impostor,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
