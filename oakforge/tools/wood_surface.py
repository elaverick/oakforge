"""Build unified woody surfaces from the tree's coarse structural skeleton."""

from __future__ import annotations

from array import array
from typing import Any, Iterable

import numpy as np
from skimage.measure import marching_cubes

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import normalize, transported_frame
from oakforge.core.mesh import BRANCH_COLOURS, MeshData
from oakforge.core.models import Branch


def _smooth_minimum(first: np.ndarray, second: np.ndarray, blend_width: float) -> np.ndarray:
    """Blend intersecting wood volumes into continuous branch and root unions."""
    difference = np.abs(first - second)
    blend = np.maximum(blend_width - difference, 0.0) / blend_width
    return np.minimum(first, second) - (blend * blend * blend_width * 0.25)


def _segment_bounds(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    padding: float,
) -> tuple[slice, slice, slice] | None:
    """Limit each swept-segment calculation to the small region it can affect."""
    bounds = []
    for axis, lower, upper in zip(axes, np.minimum(start, end), np.maximum(start, end)):
        first = max(0, int(np.searchsorted(axis, lower - radius - padding, side="left")))
        last = min(len(axis), int(np.searchsorted(axis, upper + radius + padding, side="right")))
        if first >= last:
            return None
        bounds.append(slice(first, last))
    return tuple(bounds)  # type: ignore[return-value]


def _segment_distance(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    start_radius: float,
    end_radius: float,
    root_vertical_scale: float,
    root_thickness_scale: float,
    cross_section_frame: tuple[np.ndarray, np.ndarray] | None,
    trunk_metadata: dict[str, Any] | None,
) -> np.ndarray:
    """Evaluate a tapered swept segment, with broad flutes on the basal bole."""
    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 1e-12:
        amount = np.zeros_like(coordinates[0])
        nearest = [np.full_like(amount, component) for component in start]
    else:
        offset_x = coordinates[0] - start[0]
        offset_y = coordinates[1] - start[1]
        offset_z = coordinates[2] - start[2]
        amount = np.clip(
            (offset_x * segment[0] + offset_y * segment[1] + offset_z * segment[2])
            / segment_length_squared,
            0.0,
            1.0,
        )
        nearest = [start[axis] + segment[axis] * amount for axis in range(3)]

    delta_x = coordinates[0] - nearest[0]
    delta_y = coordinates[1] - nearest[1]
    delta_z = coordinates[2] - nearest[2]
    radius = start_radius + (end_radius - start_radius) * amount

    # Root profiles stay softly oval; extreme flattening makes the buttresses read as blades.
    if root_vertical_scale > 0.0 and root_thickness_scale > 0.0 and cross_section_frame is not None:
        vertical_axis, lateral_axis = cross_section_frame
        vertical_distance = delta_x * vertical_axis[0] + delta_y * vertical_axis[1] + delta_z * vertical_axis[2]
        lateral_distance = delta_x * lateral_axis[0] + delta_y * lateral_axis[1] + delta_z * lateral_axis[2]
        radial_distance = np.sqrt(
            (vertical_distance / root_vertical_scale) ** 2
            + (lateral_distance / root_thickness_scale) ** 2
        )
    else:
        radial_distance = np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

    if cross_section_frame is not None and trunk_metadata is not None:
        side_axis, normal_axis = cross_section_frame
        radial_side = delta_x * side_axis[0] + delta_y * side_axis[1] + delta_z * side_axis[2]
        radial_normal = delta_x * normal_axis[0] + delta_y * normal_axis[1] + delta_z * normal_axis[2]
        angle = np.arctan2(radial_normal, radial_side)
        height = np.maximum(coordinates[1], 0.0)
        flare_height = max(float(trunk_metadata.get("buttress_height", 2.5)), 0.01)
        flare_fade = np.exp(-((height / flare_height) ** 1.5))

        # A few broad, uneven lobes suggest the buttress ridges of old English oaks.
        lobe_count = max(3, int(trunk_metadata.get("root_count", 7)))
        phase = float(trunk_metadata.get("root_phase", 0.0))
        lobe_wave = np.cos(angle * lobe_count + phase)
        lobe_strength = float(trunk_metadata.get("surface_flute_strength", 0.13))
        radius *= 1.0 + lobe_strength * lobe_wave * flare_fade

        # Lower-frequency section changes make the bole less like a turned wooden column.
        irregularity = float(trunk_metadata.get("trunk_irregularity", 0.04))
        asymmetry = (
            0.55 * np.cos(angle * 2.0 + height * 0.17)
            + 0.30 * np.sin(angle * 3.0 - height * 0.11)
            + 0.15 * np.cos(angle * 5.0 + height * 0.21)
        )
        radius *= 1.0 + irregularity * asymmetry

        # Narrow vertical ridges remain subtle so the silhouette stays broad and rounded.
        ridge_count = max(3, int(trunk_metadata.get("bark_ridge_count", 11)))
        ridge_strength = float(trunk_metadata.get("bark_ridge_strength", 0.012))
        radius *= 1.0 + ridge_strength * np.cos(angle * ridge_count + phase) * (0.5 + 0.5 * flare_fade)

        # Broad, irregular bark-form undulations interrupt the perfectly smooth tube silhouette.
        bark_form = (
            0.48 * np.sin(angle * 3.7 + height * 1.45 + phase)
            + 0.32 * np.sin(angle * 8.1 - height * 2.15 + 0.7)
            + 0.20 * np.cos(angle * 13.3 + height * 0.85 - 0.4)
        )
        bark_form_strength = float(trunk_metadata.get("bark_form_strength", 0.025))
        radius *= 1.0 + bark_form_strength * bark_form * (0.45 + 0.55 * flare_fade)

        # Branch collars swell locally around the actual limb origins.
        trunk_height = max(float(trunk_metadata.get("trunk_height", 8.0)), 1e-4)
        for collar in trunk_metadata.get("limb_knuckles", []):
            collar_y = float(collar.get("height", 0.7)) * trunk_height
            vertical_width = max(float(collar.get("width", 0.07)) * trunk_height, 0.05)
            angular_width = max(float(collar.get("angular_width", 0.62)), 0.1)
            angular_delta = np.arctan2(
                np.sin(angle - float(collar.get("angle", 0.0))),
                np.cos(angle - float(collar.get("angle", 0.0))),
            )
            collar_strength = float(collar.get("strength", 0.10))
            collar_influence = np.exp(-(((height - collar_y) / vertical_width) ** 2))
            collar_influence *= np.exp(-((angular_delta / angular_width) ** 2))
            radius *= 1.0 + collar_strength * collar_influence

    return radial_distance - radius




def _fit_limb_collars_to_branch_origins(branches: list[Branch]) -> None:
    """Place subtle trunk swellings exactly where the recipe's major limbs attach."""
    trunk = next((branch for branch in branches if branch.role == "trunk"), None)
    if trunk is None:
        return
    major_limbs = [
        branch for branch in branches
        if branch.role in {"primary", "leader"} and branch.parent_id == trunk.branch_id
    ]
    if not major_limbs:
        return

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    previous_side_axis = None
    for point_index in range(len(trunk.points)):
        previous_point = trunk.points[max(0, point_index - 1)]
        next_point = trunk.points[min(len(trunk.points) - 1, point_index + 1)]
        frame = transported_frame(normalize(next_point - previous_point), previous_side_axis)
        frames.append(frame)
        previous_side_axis = frame[0]

    collars = []
    trunk_height = max(float(trunk.points[-1][1]), 1e-5)
    for limb in major_limbs:
        parent_position = float(limb.metadata.get("parent_position", 0.5))
        point_index = min(int(round(parent_position * (len(trunk.points) - 1))), len(trunk.points) - 1)
        centre = trunk.points[point_index]
        tangent = normalize(trunk.points[min(point_index + 1, len(trunk.points) - 1)]
                            - trunk.points[max(point_index - 1, 0)])
        radial_direction = limb.points[0] - centre
        radial_direction -= tangent * float(np.dot(radial_direction, tangent))
        radial_direction = normalize(radial_direction)
        side_axis, normal_axis = frames[point_index]
        angle = float(np.arctan2(np.dot(radial_direction, normal_axis), np.dot(radial_direction, side_axis)))
        collars.append({
            "height": float(limb.points[0][1] / trunk_height),
            "angle": angle,
            "strength": 0.16 if limb.role == "leader" else 0.14,
            "width": 0.10 if limb.role == "leader" else 0.09,
            "angular_width": 0.82,
            "branch_role": limb.role,
        })
    trunk.metadata["limb_knuckles"] = collars

def _build_implicit_wood_surface(
    branches: list[Branch],
    voxel_size: float,
    blend_width: float,
    root_vertical_scale: float,
    root_thickness_scale: float,
    padding: float,
    root_tip_radius_minimum: float,
) -> MeshData:
    """Convert connected trunk, root and limb skeletons into one smooth polygon surface."""
    if not branches:
        return MeshData()

    _fit_limb_collars_to_branch_origins(branches)
    all_points = np.concatenate([np.asarray(branch.points, dtype=np.float64) for branch in branches])
    maximum_radius = max(max(branch.radii) for branch in branches)
    radius_scale_for_bounds = max(1.0, root_vertical_scale, root_thickness_scale)
    margin = maximum_radius * radius_scale_for_bounds + padding + blend_width * 2.0
    minimum = all_points.min(axis=0) - margin
    maximum = all_points.max(axis=0) + margin

    axes = tuple(
        np.arange(minimum[index], maximum[index] + voxel_size, voxel_size, dtype=np.float32)
        for index in range(3)
    )
    grid_shape = tuple(len(axis) for axis in axes)
    distance_field = np.full(grid_shape, 1.0e3, dtype=np.float32)

    # Evaluate each complete branch as one swept shape before blending it with other wood.
    # This avoids the bead-like joints produced by repeatedly blending every short segment.
    for branch in branches:
        branch_points = [np.asarray(point, dtype=np.float64) for point in branch.points]
        branch_radii = [float(radius) for radius in branch.radii]
        if branch.role == "root" and root_tip_radius_minimum > 0.0:
            # A very fine natural root tip can fall below one voxel in a distant LOD.
            # Add thickness progressively towards the endpoint instead of hard-clamping
            # the whole distal section, preserving the root's tapered silhouette.
            missing_tip_radius = max(0.0, root_tip_radius_minimum - branch_radii[-1])
            last_sample = max(1, len(branch_radii) - 1)
            branch_radii = [
                radius + missing_tip_radius * (sample_index / last_sample) ** 2
                for sample_index, radius in enumerate(branch_radii)
            ]
        branch_maximum_radius = max(branch_radii)
        branch_vertical_scale = float(branch.metadata.get("cross_section_vertical_scale", root_vertical_scale))
        branch_thickness_scale = float(branch.metadata.get("cross_section_thickness_scale", root_thickness_scale))
        if branch.role == "root":
            branch_maximum_radius *= max(branch_vertical_scale, branch_thickness_scale, 1.0)

        branch_minimum = np.min(np.asarray(branch_points), axis=0)
        branch_maximum = np.max(np.asarray(branch_points), axis=0)
        branch_bounds = _segment_bounds(
            axes, branch_minimum, branch_maximum, branch_maximum_radius, padding + blend_width
        )
        if branch_bounds is None:
            continue

        local_axes = tuple(axes[axis_index][branch_bounds[axis_index]] for axis_index in range(3))
        branch_distance = np.full(
            tuple(len(axis) for axis in local_axes), np.inf, dtype=np.float32
        )
        previous_side_axis = None
        segment_frames: list[tuple[np.ndarray, np.ndarray] | None] = [None] * (len(branch_points) - 1)
        if branch.role in {"trunk", "root"}:
            for segment_index in range(len(branch_points) - 1):
                tangent = normalize(branch_points[segment_index + 1] - branch_points[segment_index])
                frame = transported_frame(tangent, previous_side_axis)
                if branch.role == "trunk":
                    segment_frames[segment_index] = frame
                else:
                    vertical_axis = np.array([0.0, 1.0, 0.0]) - tangent * tangent[1]
                    vertical_axis = normalize(vertical_axis, frame[0])
                    lateral_axis = normalize(np.cross(tangent, vertical_axis), frame[1])
                    segment_frames[segment_index] = (vertical_axis, lateral_axis)
                previous_side_axis = frame[0]

        for segment_index, (start, end) in enumerate(zip(branch_points[:-1], branch_points[1:])):
            start_radius = branch_radii[segment_index]
            end_radius = branch_radii[segment_index + 1]
            maximum_segment_radius = max(start_radius, end_radius)
            if branch.role == "root":
                maximum_segment_radius *= max(branch_vertical_scale, branch_thickness_scale, 1.0)

            segment_bounds = _segment_bounds(
                local_axes, start, end, maximum_segment_radius, blend_width * 1.5
            )
            if segment_bounds is None:
                continue

            segment_coordinates = np.meshgrid(
                *(local_axes[axis_index][segment_bounds[axis_index]] for axis_index in range(3)),
                indexing="ij", sparse=False,
            )
            vertical_scale = branch_vertical_scale if branch.role == "root" else 0.0
            thickness_scale = branch_thickness_scale if branch.role == "root" else 0.0
            metadata = branch.metadata if branch.role == "trunk" else None
            segment_distance = _segment_distance(
                segment_coordinates, start, end, start_radius, end_radius,
                vertical_scale, thickness_scale, segment_frames[segment_index], metadata,
            ).astype(np.float32)
            branch_region = branch_distance[segment_bounds]
            branch_distance[segment_bounds] = np.minimum(branch_region, segment_distance)

        # Adjacent segments use a true minimum within one limb; smooth blending is
        # reserved for intersections between different limbs and the main stem.
        distance_field[branch_bounds] = _smooth_minimum(
            distance_field[branch_bounds], branch_distance, blend_width
        )

    # The generated tree rests on the terrain plane; roots may touch it but never extend below it.
    below_ground = axes[1] < 0.0
    if np.any(below_ground):
        ground_distance = -axes[1][below_ground][None, :, None]
        distance_field[:, below_ground, :] = np.maximum(
            distance_field[:, below_ground, :], ground_distance
        )

    if not np.isfinite(distance_field).all():
        raise ValueError("Implicit wood surface contains non-finite distance samples")
    if float(distance_field.min()) >= 0.0:
        raise ValueError("Implicit wood surface did not enclose any volume")

    vertices, faces, normals, _ = marching_cubes(
        distance_field,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
        gradient_direction="descent",
        allow_degenerate=False,
    )
    vertices += minimum.astype(np.float32)

    # Ensure outward-facing triangles so OpenGL back-face culling and lighting behave correctly.
    signed_volume = np.einsum(
        "ij,ij->i",
        vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
    ).sum() / 6.0
    if signed_volume < 0.0:
        faces = faces[:, [0, 2, 1]]
        normals = -normals

    mesh = MeshData()
    bark_colour = BRANCH_COLOURS["trunk"]
    for face in faces:
        first_index, second_index, third_index = (int(face[0]), int(face[1]), int(face[2]))
        first_normal = normalize(normals[first_index].astype(np.float64))
        second_normal = normalize(normals[second_index].astype(np.float64))
        third_normal = normalize(normals[third_index].astype(np.float64))
        first_point = vertices[first_index].astype(np.float64)
        second_point = vertices[second_index].astype(np.float64)
        third_point = vertices[third_index].astype(np.float64)

        # Keep polygon winding consistent with the outward marching-cubes normals.
        geometric_normal = np.cross(second_point - first_point, third_point - first_point)
        average_normal = first_normal + second_normal + third_normal
        if float(np.dot(geometric_normal, average_normal)) < 0.0:
            second_point, third_point = third_point, second_point
            second_normal, third_normal = third_normal, second_normal
        mesh.add_triangle(
            first_point,
            second_point,
            third_point,
            bark_colour,
            (first_normal, second_normal, third_normal),
        )

    # Correct all face winding from the signed volume of the emitted closed surface.
    triangle_positions = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, 3, 3)
    signed_mesh_volume = np.einsum(
        "ij,ij->i",
        triangle_positions[:, 0],
        np.cross(triangle_positions[:, 1], triangle_positions[:, 2]),
    ).sum() / 6.0
    if signed_mesh_volume < 0.0:
        triangle_positions = triangle_positions[:, [0, 2, 1], :]
        triangle_normals = np.asarray(mesh.normals, dtype=np.float64).reshape(-1, 3, 3)[:, [0, 2, 1], :] * -1.0
        mesh.positions = array("f", triangle_positions.reshape(-1))
        mesh.normals = array("f", triangle_normals.reshape(-1))
        triangle_colours = np.asarray(mesh.colours, dtype=np.float64).reshape(-1, 3, 3)[:, [0, 2, 1], :]
        mesh.colours = array("f", triangle_colours.reshape(-1))

    return mesh


def build_wood_surface(context: GenerationContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create a watertight smooth union for the coarse oak framework requested by YAML."""
    branch_groups = arguments.get("branches", [])
    if not isinstance(branch_groups, list):
        raise TypeError("wood_surface.build expects 'branches' to be a list of branch groups")
    branches = [branch for group in branch_groups for branch in group]
    if any(not isinstance(branch, Branch) for branch in branches):
        raise TypeError("wood_surface.build accepts only generated Branch objects")
    if not branches:
        raise ValueError("wood_surface.build requires at least one branch")

    # A caller can trade mesh density for generation time without changing the growth recipe.
    voxel_size = float(arguments.get("voxel_size", 0.075))
    blend_width = float(arguments.get("blend_width", 0.18))
    root_vertical_scale = float(arguments.get("root_vertical_scale", 1.08))
    root_thickness_scale = float(arguments.get("root_thickness_scale", 0.94))
    padding = float(arguments.get("padding", 0.35))
    root_tip_radius_minimum = float(arguments.get("root_tip_radius_minimum", 0.0))
    if (voxel_size <= 0.0 or blend_width <= 0.0 or padding < 0.0
            or root_tip_radius_minimum < 0.0):
        raise ValueError(
            "Wood-surface voxel_size and blend_width must be positive; padding and "
            "root_tip_radius_minimum cannot be negative"
        )

    mesh = _build_implicit_wood_surface(
        branches, voxel_size, blend_width, root_vertical_scale, root_thickness_scale,
        padding, root_tip_radius_minimum,
    )
    return {
        "mesh": mesh,
        "branch_ids": [branch.branch_id for branch in branches],
        "voxel_size": voxel_size,
        "triangle_count": mesh.triangle_count,
        "root_vertical_scale": root_vertical_scale,
        "root_thickness_scale": root_thickness_scale,
        "root_tip_radius_minimum": root_tip_radius_minimum,
    }
