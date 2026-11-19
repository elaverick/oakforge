"""Oak-specific foliage placement on terminal twig structures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import normalize, perpendicular_frame
from oakforge.core.models import Branch, Leaf


def scatter_oak_leaves(context: GenerationContext, arguments: dict[str, Any]) -> list[Leaf]:
    """Place lobed leaves along a twig with alternating, outward-facing growth."""
    twig = arguments.get("twig")
    if not isinstance(twig, Branch):
        raise TypeError("foliage.scatter_oak expects 'twig' to reference one Branch")

    leaf_count = max(0, int(arguments.get("count", 4)))
    pass_index = int(arguments.get("pass_index", 0))
    leaf_length = float(arguments.get("leaf_length", 0.48))
    width_ratio = float(arguments.get("width_ratio", 0.58))
    cluster_count = max(1, int(arguments.get("cluster_count", 4)))
    cluster_spread = max(0.0, float(arguments.get("cluster_spread", 0.055)))
    radial_spread = max(0.0, float(arguments.get("radial_spread", 0.12)))
    colour_variation = max(0.0, float(arguments.get("colour_variation", 0.055)))
    placement_range = arguments.get("placement_range", [0.12, 0.96])
    minimum_position, maximum_position = float(placement_range[0]), float(placement_range[1])
    base_direction = twig.tangent_at(0.72)
    side_axis, second_axis = perpendicular_frame(base_direction)
    world_up = np.array([0.0, 1.0, 0.0])
    generated_leaves: list[Leaf] = []

    for leaf_index in range(leaf_count):
        leaves_per_cluster = max(1, math.ceil(leaf_count / cluster_count))
        cluster_index = leaf_index % cluster_count
        cluster_slot = leaf_index // cluster_count
        cluster_phase = (cluster_index + 0.5 + pass_index * 0.22) / cluster_count
        cluster_phase %= 1.0
        within_cluster = (cluster_slot - (leaves_per_cluster - 1) * 0.5) * cluster_spread * 0.28
        cluster_jitter = float(context.random.normal(0.0, cluster_spread))
        distribution_position = float(np.clip(
            cluster_phase + within_cluster + cluster_jitter, 0.015, 0.985
        ))
        twig_position = minimum_position + (maximum_position - minimum_position) * distribution_position
        leaf_base = twig.point_at(twig_position)
        twig_direction = twig.tangent_at(twig_position)
        radial_angle = (leaf_index * 2.399963229728653) + pass_index * 1.17
        radial_angle += float(context.random.normal(0.0, 0.28))
        radial_direction = normalize(
            side_axis * math.cos(radial_angle) + second_axis * math.sin(radial_angle)
        )
        alternating_side = radial_direction

        # Leaves emerge in small, overlapping whorls around each shoot, filling
        # gaps between adjacent twigs without flattening the whole crown into a sheet.
        leaf_base = leaf_base + radial_direction * float(context.random.uniform(0.035, radial_spread))
        leaf_base = leaf_base + world_up * float(context.random.uniform(-0.025, 0.055))
        leaf_direction = normalize(
            twig_direction * 0.34 + radial_direction * 0.46 + world_up * 0.24
            + context.random.normal(0.0, 0.22, size=3)
        )
        width_hint = radial_direction + world_up * float(context.random.normal(0.0, 0.50))
        width_hint += context.random.normal(0.0, 0.27, size=3)
        width_direction = width_hint - leaf_direction * float(np.dot(width_hint, leaf_direction))
        leaf_width_direction = normalize(width_direction, alternating_side)
        local_scale = float(context.random.uniform(0.86, 1.14))
        local_colour_variation = float(context.random.uniform(-colour_variation, colour_variation))
        cooler_green = float(context.random.uniform(-0.018, 0.018))

        generated_leaves.append(Leaf(
            leaf_id=context.new_identifier("leaf"),
            base_position=leaf_base,
            growth_direction=leaf_direction,
            width_direction=leaf_width_direction,
            length=leaf_length * local_scale,
            width=leaf_length * width_ratio * local_scale,
            colour_variation=np.array([
                local_colour_variation * 0.58,
                local_colour_variation,
                cooler_green,
            ]),
        ))

    return generated_leaves
