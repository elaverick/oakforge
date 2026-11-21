"""Shared alpha-cutout atlas for runtime English oak foliage cards.

The atlas contains sixteen individually authored/generated English oak leaf
variants. It is a real RGBA texture asset shared by the OpenGL viewer and the
OBJ exporter, so the appearance does not depend on procedural shader geometry.
"""

from __future__ import annotations

from pathlib import Path

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


def save_oak_leaf_atlas(path: str | Path) -> Path:
    """Copy the shared RGBA leaf atlas into an exported runtime asset directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_oak_leaf_atlas().save(output_path, format="PNG", optimize=True)
    return output_path
