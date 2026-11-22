"""Shared alpha-cutout atlas for runtime English oak foliage cards.

The atlas contains sixteen individually authored/generated English oak leaf
variants. It is a real RGBA texture asset shared by the OpenGL viewer and the
OBJ exporter, so the appearance does not depend on procedural shader geometry.
"""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image


TILE_WIDTH = 88
TILE_HEIGHT = 128
ATLAS_COLUMNS = 2
ATLAS_ROWS = 2
ATLAS_FILENAME = "oak_leaf_atlas.png"
ATLAS_PATH = Path(__file__).resolve().parents[1] / "assets" / ATLAS_FILENAME


def build_oak_leaf_atlas() -> Image.Image:
    """Load the shared 4x4 English-oak leaf atlas as an RGBA image."""
    if not ATLAS_PATH.is_file():
        raise FileNotFoundError(f"Oak leaf texture atlas is missing: {ATLAS_PATH}")
    with Image.open(ATLAS_PATH) as image:
        expected_size = (
            TILE_WIDTH * ATLAS_COLUMNS,
            TILE_HEIGHT * ATLAS_ROWS,
        )
        if image.size != expected_size:
            raise ValueError(
                f"Oak leaf atlas must be {expected_size[0]}x{expected_size[1]}, "
                f"got {image.width}x{image.height}"
            )
        return image.convert("RGBA").copy()


def build_oak_cluster_atlas() -> Image.Image:
    """Compose four compact, multi-leaf foliage sprays from the shared leaf atlas."""
    source = build_oak_leaf_atlas()
    tile_width = 128
    tile_height = 128
    result = Image.new("RGBA", (tile_width * 2, tile_height * 2), (0, 0, 0, 0))
    source_tiles = [
        source.crop((column * TILE_WIDTH, row * TILE_HEIGHT,
                     (column + 1) * TILE_WIDTH, (row + 1) * TILE_HEIGHT))
        for row in range(ATLAS_ROWS)
        for column in range(ATLAS_COLUMNS)
    ]
    layouts = (
        (24, 32, 0.56), (60, 20, 0.58), (99, 38, 0.58),
        (35, 68, 0.60), (72, 62, 0.62), (104, 83, 0.57),
        (54, 103, 0.52), (83, 108, 0.47),
    )
    for tile_index in range(4):
        rng = random.Random(20261122 + tile_index * 97)
        tile = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
        placements = list(layouts)
        rng.shuffle(placements)
        for leaf_index, (centre_x, centre_y, base_scale) in enumerate(placements):
            leaf = source_tiles[(tile_index + leaf_index + rng.randrange(4)) % 4]
            scale = base_scale * rng.uniform(0.88, 1.12)
            leaf = leaf.resize(
                (round(TILE_WIDTH * scale), round(TILE_HEIGHT * scale)),
                Image.Resampling.LANCZOS,
            )
            leaf = leaf.rotate(
                rng.uniform(-42.0, 42.0),
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            x = round(centre_x - leaf.width * 0.5 + rng.uniform(-5.0, 5.0))
            y = round(centre_y - leaf.height * 0.5 + rng.uniform(-4.0, 4.0))
            tile.alpha_composite(leaf, (x, y))
        result.alpha_composite(
            tile,
            ((tile_index % 2) * tile_width, (tile_index // 2) * tile_height),
        )
    return result


def save_oak_cluster_atlas(path: str | Path) -> Path:
    """Generate and save the four-tile transparent foliage-clump atlas."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_oak_cluster_atlas().save(output_path, format="PNG", optimize=True)
    return output_path


def save_oak_leaf_atlas(path: str | Path) -> Path:
    """Copy the shared RGBA leaf atlas into an exported runtime asset directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_oak_leaf_atlas().save(output_path, format="PNG", optimize=True)
    return output_path
