"""Procedural alpha-cutout atlas for runtime oak leaf cards.

The atlas is generated deterministically from a lobed Quercus-style silhouette,
with a midrib, lateral veins and per-tile colour variation. It avoids external
image dependencies while remaining exportable as a normal PNG texture.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TILE_WIDTH = 64
TILE_HEIGHT = 128
ATLAS_COLUMNS = 4
ATLAS_ROWS = 4
ATLAS_FILENAME = "oak_leaf_atlas.png"


def _leaf_polygon(
    width: int,
    height: int,
    phase: float,
    asymmetry: float,
    scale: int,
) -> list[tuple[int, int]]:
    """Return a rounded, alternately lobed oak leaf outline."""
    samples = 96
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for index in range(samples + 1):
        t = 0.035 + 0.93 * index / samples
        envelope = math.sin(math.pi * t) ** 0.68
        lobe_wave = math.cos(math.tau * (4.25 * t) + phase)
        half_width = 0.405 * envelope * (0.68 + 0.32 * lobe_wave)
        centre = 0.5 + asymmetry * math.sin(math.pi * t * 1.7) * 0.018
        left.append((
            int((centre - half_width) * width * scale),
            int((1.0 - t) * height * scale),
        ))
        right.append((
            int((centre + half_width) * width * scale),
            int((1.0 - t) * height * scale),
        ))
    return left + right[::-1]


def build_oak_leaf_atlas(
    *,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
    columns: int = ATLAS_COLUMNS,
    rows: int = ATLAS_ROWS,
    seed: int = 4107,
    supersampling: int = 3,
) -> Image.Image:
    """Build a deterministic RGBA atlas containing sixteen distinct oak leaves."""
    if min(tile_width, tile_height, columns, rows, supersampling) < 1:
        raise ValueError("Atlas dimensions and supersampling must be positive")

    rng = np.random.default_rng(seed)
    atlas = Image.new(
        "RGBA",
        (tile_width * columns * supersampling, tile_height * rows * supersampling),
        (0, 0, 0, 0),
    )

    palette = [
        (47, 91, 28), (58, 104, 31), (69, 117, 37), (79, 126, 42),
        (42, 82, 27), (62, 108, 34), (76, 119, 38), (89, 132, 47),
        (51, 96, 31), (68, 112, 34), (83, 126, 42), (95, 137, 50),
        (45, 87, 29), (57, 101, 34), (73, 119, 41), (86, 128, 45),
    ]

    for tile_index in range(columns * rows):
        tile_column = tile_index % columns
        tile_row = tile_index // columns
        x0 = tile_column * tile_width * supersampling
        y0 = tile_row * tile_height * supersampling
        tw = tile_width * supersampling
        th = tile_height * supersampling
        tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)

        phase = float(rng.uniform(-0.45, 0.45))
        asymmetry = float(rng.uniform(-1.0, 1.0))
        polygon = _leaf_polygon(tile_width, tile_height, phase, asymmetry, supersampling)
        base_colour = np.asarray(palette[tile_index], dtype=np.int16)
        colour_jitter = rng.integers(-5, 6, size=3)
        leaf_colour = tuple(
            int(np.clip(value + jitter, 0, 255))
            for value, jitter in zip(base_colour, colour_jitter)
        )
        edge_colour = tuple(max(0, int(channel * 0.66)) for channel in leaf_colour)
        vein_colour = tuple(
            min(255, int(channel * 1.42 + 22)) for channel in leaf_colour
        )

        # Draw a soft dark margin and the leaf body as separate shapes.
        draw.polygon(polygon, fill=(*edge_colour, 255))
        inset = [
            (int(x0p * 0.965 + tw * 0.0175), int(y0p * 0.985 + th * 0.0075))
            for x0p, y0p in polygon
        ]
        draw.polygon(inset, fill=(*leaf_colour, 255))

        centre_x = tw * (0.5 + asymmetry * 0.012)
        base_y = th * 0.97
        tip_y = th * 0.035
        midrib_width = max(1, supersampling)
        draw.line(
            (centre_x, base_y, centre_x + asymmetry * tw * 0.018, tip_y),
            fill=(*vein_colour, 230),
            width=midrib_width,
        )

        # Alternate veins follow the lobe rhythm and fade before the leaf edge.
        for vein_index in range(1, 8):
            t = vein_index / 8.0
            y = base_y * (1.0 - t) + tip_y * t
            envelope = math.sin(math.pi * t) ** 0.68
            wave = math.cos(math.tau * (4.25 * t) + phase)
            half_width = tw * 0.405 * envelope * (0.68 + 0.32 * wave)
            vein_start_x = centre_x + asymmetry * tw * 0.01 * math.sin(math.pi * t)
            vein_end_y = y - th * 0.045
            for sign in (-1.0, 1.0):
                vein_end_x = vein_start_x + sign * half_width * 0.84
                draw.line(
                    (vein_start_x, y, vein_end_x, vein_end_y),
                    fill=(*vein_colour, 170),
                    width=max(1, supersampling - 1),
                )

        # A short petiole gives the base a recognisable attachment point.
        draw.line(
            (centre_x, base_y - th * 0.005, centre_x + asymmetry * tw * 0.012, th * 0.995),
            fill=(*edge_colour, 255),
            width=max(2, supersampling * 2),
        )

        # Sparse fine flecks break up the flat fill without looking mottled.
        for _ in range(18):
            t = float(rng.uniform(0.14, 0.88))
            y = base_y * (1.0 - t) + tip_y * t
            envelope = math.sin(math.pi * t) ** 0.68
            half_width = tw * 0.34 * envelope
            x = centre_x + float(rng.uniform(-half_width, half_width))
            radius = float(rng.uniform(0.55, 1.0)) * supersampling
            fleck = tuple(
                int(np.clip(channel + rng.integers(-10, 11), 0, 255))
                for channel in leaf_colour
            )
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(*fleck, 150),
            )

        atlas.alpha_composite(tile, (x0, y0))

    return atlas.resize(
        (tile_width * columns, tile_height * rows),
        Image.Resampling.LANCZOS,
    )


def save_oak_leaf_atlas(path: str | Path) -> Path:
    """Generate and save the runtime leaf atlas as a transparent PNG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_oak_leaf_atlas().save(output_path, format="PNG", optimize=True)
    return output_path
