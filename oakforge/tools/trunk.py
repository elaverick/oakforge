"""Procedural main-stem generation for broad-crowned trees such as English oak."""

from __future__ import annotations

from typing import Any

import numpy as np

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import cubic_bezier_points
from oakforge.core.models import Branch


def _smoothstep(value: float) -> float:
    """Blend the root flare into the regular stem without a visible shoulder."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _build_trunk_radius_profile(
    height: float,
    base_radius: float,
    stem_radius: float,
    top_radius: float,
    flare_height: float,
    taper_power: float,
    segment_count: int,
) -> list[float]:
    """Give the oak a substantial root flare and a long, organic stem taper."""
    radii: list[float] = []
    flare_fraction = float(np.clip(flare_height / height, 0.04, 0.45))

    for sample_index in range(segment_count + 1):
        height_fraction = sample_index / segment_count
        if height_fraction <= flare_fraction:
            flare_progress = _smoothstep(height_fraction / flare_fraction)
            radius = base_radius + (stem_radius - base_radius) * flare_progress
        else:
            stem_progress = (height_fraction - flare_fraction) / (1.0 - flare_fraction)
            radius = top_radius + (stem_radius - top_radius) * ((1.0 - stem_progress) ** taper_power)
        radii.append(float(radius))

    return radii


def generate_trunk(context: GenerationContext, arguments: dict[str, Any]) -> Branch:
    """Create a broad-based oak stem with natural taper, lean and basal buttressing."""
    height = float(arguments.get("height", 14.0))
    base_radius = float(arguments.get("base_radius", 0.92))
    stem_radius = float(arguments.get("stem_radius", 0.52))
    top_radius = float(arguments.get("top_radius", 0.12))
    flare_height = float(arguments.get("flare_height", 1.7))
    curve_x = float(arguments.get("curve_x", 0.48))
    curve_z = float(arguments.get("curve_z", -0.32))
    sweep_x = float(arguments.get("sweep_x", 0.22))
    sweep_z = float(arguments.get("sweep_z", 0.16))
    stem_bulge_strength = float(arguments.get("stem_bulge_strength", 0.10))
    curvature_strength = float(arguments.get("curvature_strength", 0.16))
    root_spread = float(arguments.get("root_spread", 0.48))
    root_count = max(3, int(arguments.get("root_count", 6)))
    limb_knuckles = arguments.get("limb_knuckles", [
        {"height": 0.68, "angle": 0.35, "strength": 0.16, "width": 0.075},
        {"height": 0.79, "angle": 2.45, "strength": 0.13, "width": 0.065},
        {"height": 0.87, "angle": 4.35, "strength": 0.12, "width": 0.06},
    ])
    segment_count = max(12, int(arguments.get("segments", 36)))
    taper_power = float(arguments.get("taper_power", 1.15))

    if height <= 0.0:
        raise ValueError("Trunk height must be greater than zero")
    if not (base_radius >= stem_radius >= top_radius > 0.0):
        raise ValueError("Trunk radii must satisfy base_radius >= stem_radius >= top_radius > 0")

    # A restrained S-curve prevents the main stem from looking mechanically straight.
    start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    end = np.array([curve_x, height, curve_z], dtype=np.float64)
    control_one = np.array([curve_x * 0.04, height * 0.30, curve_z * 0.04])
    control_two = np.array([curve_x * 1.12, height * 0.72, curve_z * 1.12])
    centreline = cubic_bezier_points(start, control_one, control_two, end, segment_count)

    # Mature oak stems wander around obstacles and grow unevenly rather than following a perfect arc.
    for sample_index, centre_point in enumerate(centreline):
        height_fraction = sample_index / segment_count
        bend_envelope = np.sin(np.pi * height_fraction)
        centre_point[0] += sweep_x * bend_envelope * np.sin(2.2 * np.pi * height_fraction)
        centre_point[2] += sweep_z * bend_envelope * np.sin(1.6 * np.pi * height_fraction + 0.7)
        # Two broad, offset bends give the mature stem a slow organic wander.
        centre_point[0] += curvature_strength * (
            0.62 * np.sin(np.pi * height_fraction) - 0.38 * np.sin(2.0 * np.pi * height_fraction)
        )
        centre_point[2] += curvature_strength * 0.42 * np.sin(1.7 * np.pi * height_fraction + 0.5)

    radius_profile = _build_trunk_radius_profile(
        height, base_radius, stem_radius, top_radius, flare_height, taper_power, segment_count
    )
    for sample_index, radius in enumerate(radius_profile):
        height_fraction = sample_index / segment_count
        # Broad, low-frequency swelling gives the trunk a grown, non-machined profile.
        bulge_envelope = np.sin(np.pi * height_fraction) ** 0.7
        bulge = 1.0 + stem_bulge_strength * bulge_envelope * np.sin(
            2.0 * np.pi * height_fraction * 1.35 + 0.4
        )
        # Several restrained swellings break the uniform, manufactured taper.
        natural_swells = 1.0 + 0.035 * np.exp(-((height_fraction - 0.34) / 0.12) ** 2)
        natural_swells += 0.025 * np.exp(-((height_fraction - 0.73) / 0.09) ** 2)
        radius_profile[sample_index] = radius * bulge * natural_swells

    return Branch(
        branch_id=context.new_identifier("trunk"),
        parent_id=None,
        role="trunk",
        depth=0,
        points=centreline,
        radii=radius_profile,
        metadata={
            "species": "Quercus robur",
            "growth_axis": "main_stem",
            # The mesh builder uses these values to break the circular symmetry of the base.
            "buttress_count": int(arguments.get("buttress_count", 5)),
            "buttress_strength": float(arguments.get("buttress_strength", 0.15)),
            "buttress_height": flare_height,
            "trunk_height": height,
            "trunk_irregularity": float(arguments.get("trunk_irregularity", 0.065)),
            "surface_flute_strength": float(arguments.get("surface_flute_strength", 0.13)),
            "bark_form_strength": float(arguments.get("bark_form_strength", 0.025)),
            "root_spread": root_spread,
            "root_count": root_count,
            "root_phase": float(arguments.get("root_phase", 0.28)),
            "root_lobe_power": float(arguments.get("root_lobe_power", 8.0)),
            "root_valley_depth": float(arguments.get("root_valley_depth", 0.12)),
            "bark_ridge_count": int(arguments.get("bark_ridge_count", 9)),
            "bark_ridge_strength": float(arguments.get("bark_ridge_strength", 0.018)),
            "limb_knuckles": limb_knuckles,
            "knuckle_strength": float(arguments.get("knuckle_strength", 1.0)),
        },
    )
