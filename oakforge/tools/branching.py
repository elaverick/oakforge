"""Branch spawning tool for the primary and secondary oak framework."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import cubic_bezier_points, normalize, perpendicular_frame
from oakforge.core.models import Branch


def _fit_to_crown_envelope(
    centreline: list[np.ndarray],
    envelope: dict[str, Any],
    boundary_inset: float = 0.0,
) -> list[np.ndarray]:
    """Keep grown branch paths inside a broad, asymmetric crown envelope.

    The horizontal ellipsoid controls mature crown width while a height that
    falls towards the perimeter forms the characteristic domed English-oak top.
    The branch origin is preserved so junctions remain attached to their parent.
    """
    centre_x = float(envelope.get("centre_x", 0.7))
    centre_z = float(envelope.get("centre_z", -0.7))
    # Fine woody shoots must finish inside the foliage envelope. The leaves can
    # extend beyond their supporting wood, concealing terminal cuts naturally.
    boundary_inset = max(0.0, float(boundary_inset))
    radius_x = max(float(envelope.get("radius_x", 10.0)) - boundary_inset, 1e-5)
    radius_z = max(float(envelope.get("radius_z", 9.2)) - boundary_inset, 1e-5)
    top_height = float(envelope.get("top_height", 15.0)) - boundary_inset * 0.55
    bottom_height = float(envelope.get("bottom_height", 2.65)) + boundary_inset * 0.18
    bottom_rise = max(0.0, float(envelope.get("bottom_rise", 0.35)))
    dome_drop = max(0.0, float(envelope.get("dome_drop", 2.2)))
    dome_power = max(0.2, float(envelope.get("dome_power", 2.0)))
    strength = float(np.clip(envelope.get("strength", 1.0), 0.0, 1.0))

    shaped_points = [np.asarray(point, dtype=np.float64).copy() for point in centreline]
    last_index = max(1, len(shaped_points) - 1)
    for point_index in range(1, len(shaped_points)):
        point = shaped_points[point_index]
        progress = point_index / last_index
        horizontal_offset = np.array([point[0] - centre_x, point[2] - centre_z])
        normalized_radius = math.sqrt(
            (horizontal_offset[0] / radius_x) ** 2
            + (horizontal_offset[1] / radius_z) ** 2
        )
        if normalized_radius > 1.0:
            # A gradual blend avoids a hard kink when a branch just reaches the
            # outer silhouette; the final samples are held to the crown radius.
            horizontal_scale = 1.0 - strength * (1.0 - 1.0 / normalized_radius)
            point[0] = centre_x + horizontal_offset[0] * horizontal_scale
            point[2] = centre_z + horizontal_offset[1] * horizontal_scale
        profile_radius = min(normalized_radius, 1.0)
        local_top = top_height - dome_drop * (profile_radius ** dome_power)
        envelope_blend = strength * min(1.0, progress * 2.5)
        excess_height = max(0.0, float(point[1]) - local_top)
        point[1] -= excess_height * envelope_blend
        local_bottom = bottom_height + bottom_rise * (profile_radius ** 2.0)
        missing_height = max(0.0, local_bottom - float(point[1]))
        point[1] += missing_height * envelope_blend
        shaped_points[point_index] = point
    return shaped_points


def _child_base_radius(parent: Branch, position: float, radius_scale: float) -> float:
    """Estimate child calibre without compounding the parent's distal taper.

    A parent branch's local radius can be tiny near its tip. Using only that
    value for every descendant multiplies taper at each branching generation
    and can make a major fork appear to collapse. Blend the local value with
    a robust upper-quartile radius from the parent profile. The local value
    still wins on thicker sections; the floor only prevents accidental
    collapse caused by an attachment on a sharply tapered section.
    """
    local_radius = parent.radius_at(position)
    structural_radius = float(np.quantile(parent.radii, 0.75))
    reference_radius = max(local_radius, structural_radius * 0.55)
    return max(0.0, reference_radius * radius_scale)


def create_child_branches(
    context: GenerationContext,
    parent: Branch,
    arguments: dict[str, Any],
    role: str,
    default_depth: int,
) -> list[Branch]:
    """Grow radial child branches from selected positions on a parent branch."""
    branch_count = max(0, int(arguments.get("count", 4)))
    start_range = arguments.get("start_range", [0.2, 0.85])
    start_minimum, start_maximum = float(start_range[0]), float(start_range[1])
    start_minimum = float(np.clip(start_minimum, 0.0, 1.0))
    start_maximum = float(np.clip(start_maximum, start_minimum, 1.0))

    fixed_length = arguments.get("length")
    length_factor = float(arguments.get("length_factor", 0.5))
    nominal_length = float(fixed_length) if fixed_length is not None else parent.length * length_factor
    length_variation = float(arguments.get("length_variation", 0.16))
    direction_variation = float(arguments.get("direction_variation", 0.12))
    outward_bias = float(arguments.get("outward_bias", 1.0))
    upward_bias = float(arguments.get("upward_bias", 0.28))
    forward_bias = float(arguments.get("forward_bias", 0.0))
    droop = float(arguments.get("droop", 0.04))
    bend_strength = float(arguments.get("bend_strength", 0.10))
    taper_power = float(arguments.get("taper_power", 0.78))
    angular_variation = float(arguments.get("angular_variation", 0.06))
    placement_variation = float(arguments.get("placement_variation", 0.0))
    outward_inheritance = float(np.clip(arguments.get("outward_inheritance", 0.62), 0.0, 1.0))
    vertical_inheritance = float(np.clip(arguments.get("vertical_inheritance", 0.24), 0.0, 1.0))
    crown_mid_height = arguments.get("crown_mid_height")
    height_adaptive_strength = float(arguments.get("height_adaptive_strength", 0.0))
    height_adaptive_scale = max(float(arguments.get("height_adaptive_scale", 3.5)), 1e-5)
    crown_envelope = arguments.get("crown_envelope")
    crown_convergence = max(0.0, float(arguments.get("crown_convergence", 0.0)))
    crown_target_height = arguments.get("crown_target_height")
    crown_inset = max(0.0, float(arguments.get("crown_inset", 0.0)))
    radius_scale = float(arguments.get("radius_scale", 0.34))
    radius_tip_scale = float(arguments.get("radius_tip_scale", 0.12))
    minimum_base_radius = max(0.0, float(arguments.get("minimum_base_radius", 0.0)))
    minimum_tip_radius = max(0.0, float(arguments.get("minimum_tip_radius", 0.003)))
    junction_swell = max(0.0, float(arguments.get("junction_swell", 0.0)))
    junction_decay = max(1e-4, float(arguments.get("junction_decay", 0.14)))
    junction_embed = max(0.0, float(arguments.get("junction_embed", 0.0)))
    segment_count = max(3, int(arguments.get("segments", 7)))
    depth = int(arguments.get("depth", default_depth))
    azimuth_offset = float(arguments.get("azimuth_offset", 0.0))
    phase = math.radians(azimuth_offset)

    generated_branches: list[Branch] = []
    golden_angle = math.radians(137.507764)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for branch_index in range(branch_count):
        if branch_count == 1:
            branch_position = (start_minimum + start_maximum) * 0.5
        else:
            # Stratified placement avoids clumping while allowing each limb to
            # occupy a different height instead of forming a regular ladder.
            placement_slot = (start_maximum - start_minimum) / branch_count
            placement_jitter = float(context.random.uniform(-0.5, 0.5))
            branch_position = start_minimum + placement_slot * (
                branch_index + 0.5 + placement_jitter * placement_variation
            )
            branch_position = float(np.clip(branch_position, start_minimum, start_maximum))
        parent_attachment = parent.point_at(branch_position)
        parent_tangent = parent.tangent_at(branch_position)
        parent_radius = parent.radius_at(branch_position)
        # Start the child slightly inside the parent instead of exactly on its
        # centreline. Keep the offset based on the actual attachment-section
        # radius even though child calibre uses a more stable structural radius.
        parent_point = parent_attachment - parent_tangent * (parent_radius * junction_embed)
        radial_axis, _ = perpendicular_frame(parent_tangent)
        adaptive_upward_bias = upward_bias
        if crown_mid_height is not None and height_adaptive_strength > 0.0:
            height_delta = (float(crown_mid_height) - float(parent_point[1])) / height_adaptive_scale
            adaptive_upward_bias += height_adaptive_strength * math.tanh(height_delta)
        # Child limbs inherit the parent's horizontal sweep strongly, but only
        # a fraction of its vertical pitch. This stops successive tiers climbing
        # into a tall, narrow spire instead of filling the mature crown.
        forward_direction = parent_tangent.copy()
        forward_direction[1] *= vertical_inheritance
        azimuth = phase + branch_index * golden_angle
        azimuth += float(context.random.normal(0.0, angular_variation))
        local_radial_direction = normalize(
            radial_axis * math.cos(azimuth)
            + np.cross(parent_tangent, radial_axis) * math.sin(azimuth)
        )

        # Descendants inherit the crown's outward heading instead of scattering
        # equally towards the trunk centre and the outside of the canopy.
        inherited_outward = parent.metadata.get("crown_outward")
        if inherited_outward is None:
            outward_direction = local_radial_direction
            desired_direction = (
                local_radial_direction * outward_bias
                + world_up * adaptive_upward_bias
                + forward_direction * forward_bias
            )
        else:
            outward_direction = normalize(np.asarray(inherited_outward, dtype=np.float64))
            desired_direction = (
                outward_direction * outward_bias * outward_inheritance
                + local_radial_direction * outward_bias * (1.0 - outward_inheritance + 0.22)
                + world_up * adaptive_upward_bias
                + forward_direction * forward_bias
            )
        # High branches turn gently back towards the crown's upper middle.
        # This creates overlapping forks across the apex instead of letting every
        # successive tier continue outward and leave a hollow centre or exposed pole.
        if isinstance(crown_envelope, dict) and crown_convergence > 0.0:
            crown_top = float(crown_envelope.get("top_height", 14.5))
            crown_mid = float(crown_envelope.get("mid_height", (crown_top + 3.5) * 0.5))
            target_height = float(
                crown_target_height if crown_target_height is not None
                else crown_top - float(crown_envelope.get("dome_drop", 1.8)) * 0.45
            )
            vertical_span = max(crown_top - crown_mid, 1e-5)
            height_progress = float(np.clip((parent_point[1] - crown_mid) / vertical_span, 0.0, 1.0))
            height_gate = height_progress * height_progress * (3.0 - 2.0 * height_progress)
            crown_target_spread = max(
                0.0, float(crown_envelope.get("convergence_spread", 0.0))
            )
            crown_target = np.array([
                float(crown_envelope.get("centre_x", 0.0)),
                target_height,
                float(crown_envelope.get("centre_z", 0.0)),
            ], dtype=np.float64)
            if crown_target_spread > 0.0:
                target_offset = context.random.normal(0.0, crown_target_spread, size=3)
                target_offset[1] *= 0.45
                crown_target += target_offset
            convergence_direction = normalize(crown_target - parent_point)
            desired_direction = normalize(desired_direction) + (
                convergence_direction * crown_convergence * height_gate
            )

        random_offset = context.random.normal(0.0, direction_variation, size=3)
        branch_direction = normalize(desired_direction + random_offset)
        branch_length = nominal_length * float(context.random.uniform(
            max(0.05, 1.0 - length_variation), 1.0 + length_variation
        ))
        # A coherent sideways bend gives each large limb its own grown, asymmetrical sweep.
        bend_axis, second_bend_axis = perpendicular_frame(branch_direction)
        bend_direction = normalize(
            bend_axis * context.random.normal(0.0, 1.0)
            + second_bend_axis * context.random.normal(0.0, 1.0)
        )
        # Let each limb sweep along a broad arc rather than extending as a straight spoke.
        individual_bend = bend_strength * float(context.random.uniform(0.72, 1.32))
        bend_offset = bend_direction * branch_length * individual_bend
        branch_end = parent_point + branch_direction * branch_length + bend_offset * 0.38
        branch_end += world_up * (branch_length * droop * -1.0)

        control_one = parent_point + branch_direction * branch_length * 0.30
        control_lift = float(np.clip(adaptive_upward_bias * 0.15, -0.055, 0.11))
        control_one += world_up * branch_length * control_lift + bend_offset * 0.14
        control_two = branch_end - branch_direction * branch_length * 0.24
        control_two += world_up * branch_length * (0.025 - droop * 0.52) + bend_offset * 0.92
        centreline = cubic_bezier_points(
            parent_point, control_one, control_two, branch_end, segment_count
        )
        if isinstance(crown_envelope, dict):
            centreline = _fit_to_crown_envelope(centreline, crown_envelope, crown_inset)

        starting_radius = max(
            _child_base_radius(parent, branch_position, radius_scale),
            minimum_base_radius,
        )
        ending_radius = max(starting_radius * radius_tip_scale, minimum_tip_radius)
        radius_profile = []
        for sample_index in range(segment_count + 1):
            progress = sample_index / segment_count
            tapered_radius = max(
                ending_radius,
                starting_radius * ((1.0 - progress) ** taper_power),
            )
            # A short basal flare keeps a child shoot embedded in its parent.
            # It fades quickly so the rest of the branch retains its natural taper.
            collar = 1.0 + junction_swell * math.exp(-((progress / junction_decay) ** 2))
            radius_profile.append(max(ending_radius, tapered_radius * collar))

        # Pass the broad crown heading down the hierarchy so fine shoots fill
        # the outer crown instead of repeatedly aiming back through its centre.
        horizontal_heading = branch_direction.copy()
        horizontal_heading[1] = 0.0
        if np.linalg.norm(horizontal_heading) <= 1e-8:
            horizontal_heading = outward_direction.copy()
        if inherited_outward is not None:
            horizontal_heading = normalize(
                outward_direction * outward_inheritance
                + normalize(horizontal_heading) * (1.0 - outward_inheritance)
            )
        else:
            horizontal_heading = normalize(horizontal_heading)

        generated_branches.append(Branch(
            branch_id=context.new_identifier(role),
            parent_id=parent.branch_id,
            role=role,
            depth=depth,
            points=centreline,
            radii=radius_profile,
            metadata={
                "parent_position": branch_position,
                "azimuth": azimuth,
                "crown_outward": horizontal_heading.tolist(),
                "adaptive_upward_bias": adaptive_upward_bias,
                "crown_clipped": isinstance(crown_envelope, dict),
                "crown_inset": crown_inset,
                "junction_swell": junction_swell,
                "junction_embed": junction_embed,
                "minimum_base_radius": minimum_base_radius,
                "minimum_tip_radius": minimum_tip_radius,
            },
        ))

    return generated_branches


def spawn_branches(context: GenerationContext, arguments: dict[str, Any]) -> list[Branch]:
    """Create the configured tier of primary or secondary branches."""
    parent = arguments.get("parent")
    if not isinstance(parent, Branch):
        raise TypeError("branching.spawn expects 'parent' to reference one Branch")
    role = str(arguments.get("role", "primary"))
    return create_child_branches(context, parent, arguments, role, default_depth=1)
