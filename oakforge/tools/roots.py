"""Generate configurable structural roots for mature broadleaf trees."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import cubic_bezier_points, normalize
from oakforge.core.models import Branch


def _trunk_cross_section_at_height(trunk: Branch, requested_height: float) -> tuple[np.ndarray, float]:
    """Find the bole centre and radius where a root collar meets the requested height."""
    points = np.asarray(trunk.points, dtype=np.float64)
    radii = np.asarray(trunk.radii, dtype=np.float64)
    if len(points) < 2 or len(radii) != len(points):
        raise ValueError("The trunk must have matching centreline points and radii")

    lower_height = float(np.min(points[:, 1]))
    upper_height = float(np.max(points[:, 1]))
    if requested_height < lower_height or requested_height > upper_height:
        raise ValueError(
            f"Root attachment height {requested_height:.3f} m is outside the trunk's "
            f"vertical range ({lower_height:.3f}–{upper_height:.3f} m)"
        )

    for point_index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        first_height = float(first[1])
        second_height = float(second[1])
        if min(first_height, second_height) <= requested_height <= max(first_height, second_height):
            height_span = second_height - first_height
            blend = 0.0 if abs(height_span) < 1e-9 else (
                (requested_height - first_height) / height_span
            )
            centre = first * (1.0 - blend) + second * blend
            radius = radii[point_index] * (1.0 - blend) + radii[point_index + 1] * blend
            return centre, float(radius)

    nearest_index = int(np.argmin(np.abs(points[:, 1] - requested_height)))
    return points[nearest_index].copy(), float(radii[nearest_index])


def _procedural_angles(context: GenerationContext, arguments: dict[str, Any], count: int) -> list[float]:
    """Distribute roots around the bole with controlled natural variation."""
    distribution = str(arguments.get("distribution", "radial_jittered")).lower()
    phase = math.radians(float(arguments.get("phase_degrees", 16.0)))
    if "phase" in arguments:
        # Older recipes expressed their phase in radians; retain that contract.
        phase = float(arguments["phase"])

    if distribution == "radial":
        return [phase + 2.0 * math.pi * index / count for index in range(count)]
    if distribution == "random":
        angles = np.sort(context.random.uniform(0.0, 2.0 * math.pi, count))
        return [float(angle + phase) for angle in angles]
    if distribution != "radial_jittered":
        raise ValueError(
            f"Unknown root distribution '{distribution}'. Use radial, radial_jittered, "
            "random, or provide explicit placements."
        )

    if "angular_variation" in arguments:
        jitter_limit = abs(float(arguments["angular_variation"]))
    else:
        jitter_limit = math.radians(float(arguments.get("angular_jitter_degrees", 8.0)))
    angles = []
    for index in range(count):
        nominal_angle = phase + 2.0 * math.pi * index / count
        jitter = float(context.random.uniform(-jitter_limit, jitter_limit))
        angles.append(nominal_angle + jitter)
    return angles


def _make_root(
    context: GenerationContext,
    trunk: Branch,
    placement: Mapping[str, Any],
    root_index: int,
    defaults: Mapping[str, Any],
    placement_mode: str,
    distribution: str | None,
) -> Branch:
    """Construct one descending root from a precise bole attachment."""
    angle = math.radians(float(placement["angle_degrees"]))
    attachment_height = float(placement.get("attachment_height", defaults.get("collar_height", 1.25)))
    if "attachment_height_fraction" in placement:
        trunk_height = float(trunk.metadata.get("trunk_height", trunk.points[-1][1]))
        attachment_height = float(placement["attachment_height_fraction"]) * trunk_height

    centre, trunk_radius = _trunk_cross_section_at_height(trunk, attachment_height)
    radial_direction = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float64)
    sideways_direction = np.array([-radial_direction[2], 0.0, radial_direction[0]], dtype=np.float64)

    root_length = float(placement.get("length", defaults.get("length", 2.8)))
    base_radius = float(placement.get("base_radius", defaults.get("base_radius", 0.48)))
    tip_radius = float(placement.get("tip_radius", defaults.get("tip_radius", base_radius * 0.18)))
    tip_height = float(placement.get("tip_height", defaults.get("tip_height", 0.045)))
    attachment_inset = float(placement.get("attachment_inset", defaults.get("attachment_inset", 0.16)))
    sweep_degrees = float(placement.get("sweep_degrees", 0.0))
    sweep_radians = math.radians(sweep_degrees)
    direction = radial_direction * math.cos(sweep_radians) + sideways_direction * math.sin(sweep_radians)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)

    numeric_inputs = (angle, attachment_height, root_length, base_radius, tip_radius, tip_height,
                      attachment_inset, sweep_radians)
    if not np.isfinite(numeric_inputs).all():
        raise ValueError(f"Root placement {root_index} contains a non-finite numeric value")
    if root_length <= 0.0:
        raise ValueError(f"Root placement {root_index} has a non-positive length")
    if base_radius <= 0.0 or tip_radius < 0.0 or tip_radius >= base_radius:
        raise ValueError(f"Root placement {root_index} has an invalid radius profile")
    if tip_height < -0.5:
        raise ValueError(f"Root placement {root_index} cannot extend more than 0.5 m below ground")
    if not 0.0 <= attachment_inset <= 1.0:
        raise ValueError("Root attachment_inset must be between zero and one")

    # The collar starts just inside the bole surface so the fused surface has real overlap.
    start = centre + radial_direction * (trunk_radius - base_radius * attachment_inset)
    start[1] = attachment_height
    end = start + direction * root_length
    end[1] = tip_height

    first_control = start + direction * (root_length * 0.22)
    first_control[1] = start[1] * float(placement.get("first_control_height", 0.58))
    second_control = end - direction * (root_length * 0.24)
    second_control[1] = tip_height + (start[1] - tip_height) * float(
        placement.get("second_control_height", 0.035)
    )

    segment_count = max(6, int(placement.get("segments", defaults.get("segments", 14))))
    centreline = cubic_bezier_points(start, first_control, second_control, end, segment_count)
    curve_strength = float(placement.get("curve_strength", defaults.get("curve_strength", 0.0)))
    curve_phase = float(placement.get("curve_phase", 0.0))
    if not np.isfinite((curve_strength, curve_phase)).all() or not 0.0 <= curve_strength <= 0.5:
        raise ValueError("Root curve_strength must be a finite value between zero and 0.5")
    curve_axis = normalize(np.array([-direction[2], 0.0, direction[0]], dtype=np.float64))
    for sample_index, point in enumerate(centreline):
        progress = sample_index / segment_count
        lateral_offset = math.sin(math.pi * progress) * math.sin(2.0 * math.pi * progress + curve_phase)
        point += curve_axis * (root_length * curve_strength * lateral_offset)
    taper_power = float(placement.get("taper_power", defaults.get("taper_power", 1.18)))
    radii = []
    collar_swell = float(placement.get("collar_swell", defaults.get("collar_swell", 0.18)))
    belly_swell = float(placement.get("belly_swell", defaults.get("belly_swell", 0.10)))
    collar_width = max(float(placement.get("collar_width", 0.20)), 0.05)
    for sample_index in range(segment_count + 1):
        progress = sample_index / segment_count
        taper = (1.0 - progress) ** taper_power
        radius = tip_radius + (base_radius - tip_radius) * taper
        # A broad root collar and a subtle mid-span belly avoid a flat, ribbon-like profile.
        radius *= 1.0 + collar_swell * math.exp(-((progress / collar_width) ** 2))
        radius *= 1.0 + belly_swell * math.sin(math.pi * progress) ** 2
        radii.append(float(radius))

    root_name = str(placement.get("id", f"root_{root_index + 1:02d}"))
    branch = Branch(
        branch_id=context.new_identifier(root_name),
        parent_id=trunk.branch_id,
        role="root",
        depth=1,
        points=centreline,
        radii=radii,
        metadata={
            "root_index": root_index,
            "placement_id": root_name,
            "placement_mode": placement_mode,
            "distribution": distribution,
            "azimuth_degrees": math.degrees(angle) % 360.0,
            "attachment_height": attachment_height,
            "attachment_inset": attachment_inset,
            "length": root_length,
            "base_radius": base_radius,
            "tip_radius": tip_radius,
            "root_tier": "primary",
            "curve_strength": curve_strength,
            "curve_phase": curve_phase,
        },
    )
    for scale_name in ("cross_section_vertical_scale", "cross_section_thickness_scale"):
        if scale_name in placement:
            scale_value = float(placement[scale_name])
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError(f"{scale_name} must be a positive finite number")
            branch.metadata[scale_name] = scale_value
    return branch


def spread_surface_roots(context: GenerationContext, arguments: dict[str, Any]) -> list[Branch]:
    """Grow separately configurable surface roots around or explicitly on the lower bole.

    Supplying ``placements`` selects manual mode. Supplying ``distribution`` selects
    deterministic, seed-driven procedural placement. A recipe must choose one mode.
    """
    trunk = arguments.get("trunk")
    if not isinstance(trunk, Branch) or trunk.role != "trunk":
        raise TypeError("roots.spread expects 'trunk' to reference the generated trunk")

    placements_value = arguments.get("placements")
    defaults = {
        "length": float(arguments.get("length", 2.8)),
        "base_radius": float(arguments.get("base_radius", 0.48)),
        "tip_radius": arguments.get("tip_radius"),
        "tip_height": float(arguments.get("tip_height", -0.12)),
        "collar_height": float(arguments.get("collar_height", 1.25)),
        "attachment_inset": float(arguments.get("attachment_inset", 0.16)),
        "segments": int(arguments.get("segments", 14)),
        "taper_power": float(arguments.get("taper_power", 1.18)),
        "collar_swell": float(arguments.get("collar_swell", 0.18)),
        "belly_swell": float(arguments.get("belly_swell", 0.10)),
        "curve_strength": float(arguments.get("curve_strength", 0.0)),
    }
    if defaults["tip_radius"] is None:
        defaults["tip_radius"] = defaults["base_radius"] * 0.18

    for variation_name in ("length_variation", "width_variation", "collar_height_variation"):
        variation = float(arguments.get(variation_name, {
            "length_variation": 0.22,
            "width_variation": 0.20,
            "collar_height_variation": 0.12,
        }[variation_name]))
        if not 0.0 <= variation < 1.0:
            raise ValueError(f"{variation_name} must be in the range [0, 1)")
    direction_variation = float(arguments.get("direction_variation_degrees", 5.0))
    if direction_variation < 0.0:
        raise ValueError("direction_variation_degrees cannot be negative")

    if placements_value is not None:
        if not isinstance(placements_value, Sequence) or isinstance(placements_value, (str, bytes)):
            raise TypeError("roots.spread placements must be a YAML list of root mappings")
        if not placements_value:
            raise ValueError("Manual root placement requires at least one placement")
        placements: list[Mapping[str, Any]] = []
        for index, item in enumerate(placements_value):
            if not isinstance(item, Mapping):
                raise TypeError(f"Root placement {index} must be a mapping")
            if "angle_degrees" not in item:
                raise ValueError(f"Root placement {index} must define angle_degrees")
            placements.append(item)
        placement_mode = "manual"
        distribution = None
    else:
        if "distribution" not in arguments:
            raise ValueError(
                "Root placement must be explicit: provide 'placements' for manual mode "
                "or 'distribution' for procedural mode."
            )
        root_count = int(arguments.get("count", 7))
        if root_count < 1:
            raise ValueError("Procedural root count must be at least one")
        distribution = str(arguments["distribution"]).lower()
        angles = _procedural_angles(context, arguments, root_count)
        placements = []
        for root_index, angle in enumerate(angles):
            angular_position_degrees = math.degrees(angle)
            collar_height = defaults["collar_height"] * float(
                context.random.uniform(
                    1.0 - float(arguments.get("collar_height_variation", 0.12)),
                    1.0 + float(arguments.get("collar_height_variation", 0.12)),
                )
            )
            length = defaults["length"] * float(
                context.random.uniform(
                    1.0 - float(arguments.get("length_variation", 0.22)),
                    1.0 + float(arguments.get("length_variation", 0.22)),
                )
            )
            base_radius = defaults["base_radius"] * float(
                context.random.uniform(
                    1.0 - float(arguments.get("width_variation", 0.20)),
                    1.0 + float(arguments.get("width_variation", 0.20)),
                )
            )
            sweep = float(context.random.normal(0.0, direction_variation))
            placements.append({
                "id": f"procedural_root_{root_index + 1:02d}",
                "angle_degrees": angular_position_degrees,
                "attachment_height": collar_height,
                "length": length,
                "base_radius": base_radius,
                "sweep_degrees": sweep,
                "curve_strength": float(context.random.uniform(0.035, 0.085)),
                "curve_phase": float(context.random.uniform(0.0, 2.0 * math.pi)),
            })
        placement_mode = "procedural"

    roots = [
        _make_root(context, trunk, placement, root_index, defaults, placement_mode, distribution)
        for root_index, placement in enumerate(placements)
    ]
    return roots


def _make_lateral_root(
    context: GenerationContext,
    parent_root: Branch,
    placement: Mapping[str, Any],
    root_index: int,
    defaults: Mapping[str, Any],
    placement_mode: str,
    distribution: str | None,
) -> Branch:
    """Grow a low surface-root offshoot from the side of an existing buttress root."""
    parent_position = float(placement.get("parent_position", defaults.get("parent_position", 0.56)))
    if not 0.15 <= parent_position <= 0.92:
        raise ValueError("Lateral-root parent_position must be between 0.15 and 0.92")

    start = parent_root.point_at(parent_position).astype(np.float64)
    tangent = parent_root.tangent_at(parent_position).astype(np.float64)
    tangent[1] = 0.0
    if float(np.linalg.norm(tangent)) < 1e-8:
        tangent = parent_root.points[-1] - parent_root.points[0]
        tangent[1] = 0.0
    outward = normalize(tangent, (1.0, 0.0, 0.0))
    lateral = normalize(np.array([-outward[2], 0.0, outward[0]], dtype=np.float64))

    angle_degrees = float(placement.get("angle_degrees", defaults.get("angle_degrees", 58.0)))
    if not np.isfinite(angle_degrees) or not 0.0 <= abs(angle_degrees) <= 90.0:
        raise ValueError("Lateral-root angle_degrees must be between -90 and 90 degrees")
    signed_angle = math.radians(angle_degrees)
    side = float(placement.get("side", 1.0))
    if side not in {-1.0, 1.0}:
        raise ValueError("Lateral-root side must be either -1 or 1")
    direction = normalize(outward * math.cos(signed_angle) + lateral * side * math.sin(signed_angle))

    root_length = float(placement.get("length", defaults.get("length", 1.35)))
    parent_radius = parent_root.radius_at(parent_position)
    base_radius = float(placement.get(
        "base_radius", max(0.075, parent_radius * float(defaults.get("base_radius_scale", 0.82)))
    ))
    tip_radius = float(placement.get("tip_radius", base_radius * float(defaults.get("tip_radius_ratio", 0.20))))
    tip_height = float(placement.get("tip_height", defaults.get("tip_height", -0.12)))
    segments = max(6, int(placement.get("segments", defaults.get("segments", 10))))
    if not np.isfinite((signed_angle, root_length, base_radius, tip_radius, tip_height)).all():
        raise ValueError("Lateral-root placement contains a non-finite numeric value")
    if root_length <= 0.0 or base_radius <= 0.0 or not 0.0 <= tip_radius < base_radius:
        raise ValueError("Lateral-root length and radii must define a positive tapered volume")
    if float(placement.get("taper_power", defaults.get("taper_power", 1.05))) <= 0.0:
        raise ValueError("Lateral-root taper_power must be positive")
    if tip_height < -0.5:
        raise ValueError("Lateral roots cannot extend more than 0.5 m below ground")

    end = start + direction * root_length
    end[1] = tip_height
    first_control = start + direction * (root_length * 0.30)
    first_control[1] = max(tip_height + 0.07, start[1] * 0.88)
    second_control = end - direction * (root_length * 0.26)
    second_control[1] = tip_height + max(0.035, (start[1] - tip_height) * 0.12)
    points = cubic_bezier_points(start, first_control, second_control, end, segments)
    curve_strength = float(placement.get("curve_strength", defaults.get("curve_strength", 0.0)))
    curve_phase = float(placement.get("curve_phase", 0.0))
    if not np.isfinite((curve_strength, curve_phase)).all() or not 0.0 <= curve_strength <= 0.5:
        raise ValueError("Lateral-root curve_strength must be a finite value between zero and 0.5")
    for sample_index, point in enumerate(points):
        progress = sample_index / segments
        lateral_offset = math.sin(math.pi * progress) * math.sin(2.0 * math.pi * progress + curve_phase)
        point += lateral * (root_length * curve_strength * lateral_offset)

    taper_power = float(placement.get("taper_power", defaults.get("taper_power", 1.05)))
    radii = []
    for sample_index in range(segments + 1):
        progress = sample_index / segments
        taper = (1.0 - progress) ** taper_power
        radius = tip_radius + (base_radius - tip_radius) * taper
        radius *= 1.0 + 0.13 * math.exp(-((progress / 0.22) ** 2))
        radius *= 1.0 + 0.08 * math.sin(math.pi * progress) ** 2
        radii.append(float(radius))

    placement_name = str(placement.get("id", f"lateral_root_{root_index + 1:02d}"))
    branch = Branch(
        branch_id=context.new_identifier(placement_name),
        parent_id=parent_root.branch_id,
        role="root",
        depth=2 if str(defaults.get("child_tier", "lateral")) == "lateral" else 3,
        points=points,
        radii=radii,
        metadata={
            "root_index": root_index,
            "placement_id": placement_name,
            "root_tier": str(defaults.get("child_tier", "lateral")),
            "curve_strength": curve_strength,
            "curve_phase": curve_phase,
            "placement_mode": placement_mode,
            "distribution": distribution,
            "parent_placement_id": parent_root.metadata.get("placement_id"),
            "parent_position": parent_position,
            "side": side,
            "branch_angle_degrees": math.degrees(signed_angle),
            "length": root_length,
            "base_radius": base_radius,
            "tip_radius": tip_radius,
        },
    )
    for scale_name in ("cross_section_vertical_scale", "cross_section_thickness_scale"):
        if scale_name in placement:
            scale_value = float(placement[scale_name])
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError(f"{scale_name} must be a positive finite number")
            branch.metadata[scale_name] = scale_value
    return branch


def branch_surface_roots(context: GenerationContext, arguments: dict[str, Any]) -> list[Branch]:
    """Add seeded lateral roots or manually positioned offshoots to main surface roots.

    As with bole-root placement, recipes must explicitly choose procedural distribution
    or provide manual placements. The resulting branches remain children of their host root.
    """
    parents_value = arguments.get("parents")
    if not isinstance(parents_value, Sequence) or isinstance(parents_value, (str, bytes)):
        raise TypeError("roots.branch expects 'parents' to be a list of generated root branches")
    parent_tiers_value = arguments.get("parent_tiers", ["primary"])
    if isinstance(parent_tiers_value, str):
        parent_tiers = {parent_tiers_value}
    else:
        parent_tiers = {str(tier) for tier in parent_tiers_value}
    parents = [root for root in parents_value if isinstance(root, Branch)
               and root.role == "root" and root.metadata.get("root_tier", "primary") in parent_tiers]
    if not parents:
        tiers = ", ".join(sorted(parent_tiers))
        raise ValueError(f"roots.branch requires at least one parent root in tier(s): {tiers}")

    defaults = {
        "parent_position": 0.56,
        "angle_degrees": 58.0,
        "length": float(arguments.get("length", 1.35)),
        "tip_height": float(arguments.get("tip_height", -0.12)),
        "segments": int(arguments.get("segments", 10)),
        "base_radius_scale": float(arguments.get("base_radius_scale", 0.82)),
        "tip_radius_ratio": float(arguments.get("tip_radius_ratio", 0.20)),
        "taper_power": float(arguments.get("taper_power", 1.05)),
        "child_tier": str(arguments.get("child_tier", "lateral")),
        "curve_strength": float(arguments.get("curve_strength", 0.0)),
    }
    placements_value = arguments.get("placements")
    if placements_value is not None:
        if not isinstance(placements_value, Sequence) or isinstance(placements_value, (str, bytes)):
            raise TypeError("Lateral-root placements must be a YAML list of mappings")
        if not placements_value:
            raise ValueError("Manual lateral-root placement requires at least one placement")
        parent_lookup = {str(root.metadata.get("placement_id")): root for root in parents}
        placements: list[tuple[Branch, Mapping[str, Any]]] = []
        for index, item in enumerate(placements_value):
            if not isinstance(item, Mapping):
                raise TypeError(f"Lateral-root placement {index} must be a mapping")
            parent_name = str(item.get("parent_placement_id", ""))
            if parent_name not in parent_lookup:
                raise ValueError(
                    f"Lateral-root placement {index} references unknown parent root '{parent_name}'"
                )
            if "parent_position" not in item or "angle_degrees" not in item:
                raise ValueError(
                    f"Lateral-root placement {index} must define parent_position and angle_degrees"
                )
            placements.append((parent_lookup[parent_name], item))
        placement_mode = "manual"
        distribution = None
    else:
        if "distribution" not in arguments:
            raise ValueError(
                "Lateral-root placement must be explicit: provide 'placements' for manual mode "
                "or 'distribution' for procedural mode."
            )
        distribution = str(arguments["distribution"]).lower()
        if distribution not in {"staggered", "random"}:
            raise ValueError("Lateral-root distribution must be 'staggered' or 'random'")
        count_per_root = int(arguments.get("count_per_root", 1))
        if count_per_root < 1:
            raise ValueError("count_per_root must be at least one")
        probability = float(arguments.get("probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Lateral-root probability must be between zero and one")
        parent_range = arguments.get("parent_position_range", [0.40, 0.76])
        angle_range = arguments.get("angle_range_degrees", [44.0, 72.0])
        if (not isinstance(parent_range, Sequence) or len(parent_range) != 2
                or not 0.15 <= float(parent_range[0]) < float(parent_range[1]) <= 0.92):
            raise ValueError("parent_position_range must be an increasing pair inside [0.15, 0.92]")
        if (not isinstance(angle_range, Sequence) or len(angle_range) != 2
                or not 5.0 <= float(angle_range[0]) <= float(angle_range[1]) <= 85.0):
            raise ValueError("angle_range_degrees must be an increasing pair within 5 to 85 degrees")
        length_variation = float(arguments.get("length_variation", 0.28))
        if not 0.0 <= length_variation < 1.0:
            raise ValueError("length_variation must be in the range [0, 1)")
        placements = []
        for parent_index, parent in enumerate(parents):
            for lateral_index in range(count_per_root):
                if float(context.random.random()) > probability:
                    continue
                if distribution == "staggered":
                    # Alternate sides and stagger attachment positions to break radial symmetry.
                    side = -1.0 if (parent_index + lateral_index) % 2 else 1.0
                    parent_position = float(context.random.uniform(*parent_range))
                    angle = float(context.random.uniform(*angle_range))
                    angle += float(context.random.normal(0.0, 4.5))
                    side = side if float(context.random.random()) > 0.18 else -side
                else:
                    side = -1.0 if float(context.random.random()) < 0.5 else 1.0
                    parent_position = float(context.random.uniform(*parent_range))
                    angle = float(context.random.uniform(*angle_range))
                length = defaults["length"] * float(
                    context.random.uniform(1.0 - length_variation, 1.0 + length_variation)
                )
                placements.append((parent, {
                    "id": f"{parent.metadata.get('placement_id', 'root')}_lateral_{lateral_index + 1:02d}",
                    "parent_position": parent_position,
                    "angle_degrees": angle,
                    "side": side,
                    "length": length,
                    "curve_strength": float(context.random.uniform(0.045, 0.13)),
                    "curve_phase": float(context.random.uniform(0.0, 2.0 * math.pi)),
                }))
        placement_mode = "procedural"

    return [
        _make_lateral_root(context, parent, placement, index, defaults, placement_mode, distribution)
        for index, (parent, placement) in enumerate(placements)
    ]
