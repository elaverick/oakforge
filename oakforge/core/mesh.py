"""Triangle mesh construction for the generated tree."""

from __future__ import annotations

from array import array
import math
from dataclasses import dataclass, field
import zlib
from typing import Iterable
from pathlib import Path

import numpy as np

from oakforge.core.geometry import normalize, perpendicular_frame, transported_frame
from oakforge.core.models import Branch, Leaf, TreeModel


@dataclass
class MeshData:
    """A non-indexed triangle mesh ready for OpenGL or OBJ export."""

    # Packed float buffers avoid the large per-tuple memory cost of Python lists.
    positions: array = field(default_factory=lambda: array("f"))
    normals: array = field(default_factory=lambda: array("f"))
    colours: array = field(default_factory=lambda: array("f"))
    texcoords: array = field(default_factory=lambda: array("f"))
    # Source and retained leaf counts make geometry-budget reports explicit.
    leaf_count: int = 0
    source_leaf_count: int = 0
    cluster_count: int = 0
    uses_leaf_texture: bool = False
    canopy_kind: str = "none"
    texture_kind: str = "none"

    @property
    def triangle_count(self) -> int:
        return len(self.positions) // 9

    def add_triangle(
        self,
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
        colour: Iterable[float],
        vertex_normals: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Append one shaded triangle to the renderable mesh."""
        face_normal = normalize(np.cross(second - first, third - first), (0.0, 1.0, 0.0))
        chosen_normals = vertex_normals or (face_normal, face_normal, face_normal)
        colour_tuple = tuple(float(component) for component in colour)
        for position, normal in zip((first, second, third), chosen_normals):
            self.positions.extend(float(component) for component in position)
            self.normals.extend(float(component) for component in normal)
            self.colours.extend(colour_tuple)
            self.texcoords.extend((0.0, 0.0))




def _append_float_buffer(target: array, values: np.ndarray) -> None:
    """Copy a contiguous float buffer without iterating over millions of scalars."""
    contiguous = np.asarray(values, dtype=np.float32, order="C")
    if not contiguous.flags.c_contiguous:
        contiguous = np.ascontiguousarray(contiguous, dtype=np.float32)
    target.frombytes(memoryview(contiguous).cast("B"))


BRANCH_COLOURS = {
    "trunk": (0.34, 0.20, 0.11),
    "root": (0.32, 0.18, 0.095),
    "primary": (0.39, 0.24, 0.13),
    "secondary": (0.43, 0.28, 0.16),
    "twig": (0.48, 0.32, 0.19),
    "spray": (0.46, 0.31, 0.18),
}
LEAF_BASE_COLOUR = np.array([0.42, 0.66, 0.25], dtype=np.float64)


# Each pair describes the half-width of an oak leaf at a position along its midrib.
# Repeated broad sections and narrow notches create the characteristic lobed edge.
FAR_LEAF_PROFILE = [
    (0.00, 0.018), (0.25, 0.20), (0.50, 0.29),
    (0.75, 0.20), (1.00, 0.0),
]
MID_LEAF_PROFILE = [
    (0.00, 0.012), (0.16, 0.19), (0.34, 0.11),
    (0.52, 0.30), (0.70, 0.13), (0.88, 0.17), (1.00, 0.0),
]
NEAR_LEAF_PROFILE = [
    (0.00, 0.012), (0.12, 0.18), (0.25, 0.095),
    (0.38, 0.29), (0.50, 0.14), (0.63, 0.31),
    (0.76, 0.15), (0.90, 0.09), (1.00, 0.0),
]
OAK_LEAF_PROFILE = [
    (0.00, 0.012), (0.10, 0.18), (0.19, 0.095),
    (0.31, 0.29), (0.41, 0.14), (0.55, 0.31),
    (0.65, 0.15), (0.79, 0.25), (0.90, 0.09),
    (0.97, 0.035), (1.00, 0.0),
]


def _make_branch_tube(
    branch: Branch,
    mesh: MeshData,
    radial_sides: int,
    max_segments: int | None = None,
) -> None:
    """Build a tapered tube, optionally resampling it for a cheaper LOD."""
    if branch.role == "trunk":
        side_count = max(12, radial_sides)
    elif branch.role in {"tertiary", "twig", "spray"}:
        side_count = max(4, radial_sides * 3 // 4)
    else:
        side_count = max(6, radial_sides // 2)

    points = np.asarray(branch.points, dtype=np.float64)
    radii = np.asarray(branch.radii, dtype=np.float64)
    if max_segments is not None and max_segments > 0 and len(points) - 1 > max_segments:
        source_positions = np.linspace(0.0, 1.0, len(points))
        target_positions = np.linspace(0.0, 1.0, max_segments + 1)
        points = np.column_stack([
            np.interp(target_positions, source_positions, points[:, axis])
            for axis in range(3)
        ])
        radii = np.interp(target_positions, source_positions, radii)
    branch_colour = np.asarray(
        BRANCH_COLOURS.get(branch.role, BRANCH_COLOURS["secondary"]), dtype=np.float32
    )
    previous_side_axis = None
    ring_vertices: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []
    side_angles = np.arange(side_count, dtype=np.float64) * (2.0 * np.pi / side_count)
    side_cosines = np.cos(side_angles)[:, None]
    side_sines = np.sin(side_angles)[:, None]

    for point_index, centre in enumerate(points):
        previous_point = points[max(0, point_index - 1)]
        next_point = points[min(len(points) - 1, point_index + 1)]
        tangent = normalize(next_point - previous_point)
        side_axis, normal_axis = transported_frame(tangent, previous_side_axis)
        previous_side_axis = side_axis
        radial_directions = normalize(
            side_cosines * side_axis[None, :] + side_sines * normal_axis[None, :]
        )
        local_radius = float(radii[point_index])
        height_above_ground = max(float(centre[1]), 0.0)
        trunk_height = max(float(points[-1][1]), 1e-5)
        height_fraction = float(np.clip(height_above_ground / trunk_height, 0.0, 1.0))

        if branch.role == "trunk":
            flare_height = max(float(branch.metadata.get("buttress_height", 1.7)), 1e-5)
            flare_fade = max(0.0, 1.0 - height_above_ground / flare_height) ** 1.45
            root_count = max(3, int(branch.metadata.get("root_count", 6)))
            root_phase = float(branch.metadata.get("root_phase", 0.0))
            root_spread = float(branch.metadata.get("root_spread", 0.35))
            root_lobe_power = max(2.0, float(branch.metadata.get("root_lobe_power", 8.0)))
            root_valley_depth = float(branch.metadata.get("root_valley_depth", 0.12))
            root_alignment = np.cos(side_angles * root_count + root_phase)
            root_lobe = np.maximum(0.0, root_alignment) ** root_lobe_power
            root_valley = np.maximum(0.0, -root_alignment) ** 3.0
            root_envelope = np.exp(-((height_above_ground / max(flare_height * 0.8, 1e-4)) ** 1.35))
            local_radius *= 1.0 + root_spread * root_lobe * root_envelope
            local_radius *= 1.0 - root_valley_depth * root_valley * root_envelope

            buttress_count = max(1, int(branch.metadata.get("buttress_count", root_count)))
            buttress_strength = float(branch.metadata.get("buttress_strength", 0.15))
            buttress_wave = (0.5 + 0.5 * np.cos(side_angles * buttress_count + root_phase)) ** 2.0
            local_radius *= 1.0 + buttress_strength * flare_fade * buttress_wave
            irregularity = float(branch.metadata.get("trunk_irregularity", 0.04))
            asymmetry = (
                0.52 * np.cos(side_angles * 2.0 + height_fraction * 1.7)
                + 0.31 * np.sin(side_angles * 3.0 - height_fraction * 2.3)
                + 0.17 * np.cos(side_angles * 5.0 + height_fraction * 3.1)
            )
            local_radius *= 1.0 + irregularity * asymmetry
            ridge_count = max(3, int(branch.metadata.get("bark_ridge_count", 9)))
            ridge_strength = float(branch.metadata.get("bark_ridge_strength", 0.018))
            ridge_fade = 0.35 + 0.65 * (1.0 - height_fraction)
            bark_ridges = np.cos(side_angles * ridge_count + root_phase * 0.7)
            local_radius *= 1.0 + ridge_strength * ridge_fade * bark_ridges

            for knuckle in branch.metadata.get("limb_knuckles", []):
                junction_height = float(knuckle.get("height", 0.7)) * trunk_height
                vertical_width = max(float(knuckle.get("width", 0.06)) * trunk_height, 1e-4)
                height_influence = np.exp(-(((height_above_ground - junction_height) / vertical_width) ** 2))
                angular_distance = np.arctan2(
                    np.sin(side_angles - float(knuckle.get("angle", 0.0))),
                    np.cos(side_angles - float(knuckle.get("angle", 0.0))),
                )
                angular_width = float(knuckle.get("angular_width", 0.55))
                angular_influence = np.exp(-((angular_distance / angular_width) ** 2))
                local_radius *= 1.0 + float(knuckle.get("strength", 0.12)) * float(
                    branch.metadata.get("knuckle_strength", 1.0)
                ) * height_influence * angular_influence

        ring_vertices.append(centre[None, :] + radial_directions * np.asarray(local_radius)[..., None])
        ring_normals.append(radial_directions)

    vertices_by_ring = np.asarray(ring_vertices, dtype=np.float32)
    normals_by_ring = np.asarray(ring_normals, dtype=np.float32)
    current_vertices = vertices_by_ring[:-1]
    next_vertices = vertices_by_ring[1:]
    current_normals = normals_by_ring[:-1]
    next_normals = normals_by_ring[1:]

    # Two triangles form each quad between adjacent rings. Keeping the whole
    # repeated operation in NumPy avoids hundreds of thousands of Python calls
    # when a mature oak has several thousand fine branch segments.
    triangle_positions = np.stack((
        np.stack((current_vertices, next_vertices, np.roll(next_vertices, -1, axis=1)), axis=2),
        np.stack((current_vertices, np.roll(next_vertices, -1, axis=1),
                  np.roll(current_vertices, -1, axis=1)), axis=2),
    ), axis=2)
    triangle_normals = np.stack((
        np.stack((current_normals, next_normals, np.roll(next_normals, -1, axis=1)), axis=2),
        np.stack((current_normals, np.roll(next_normals, -1, axis=1),
                  np.roll(current_normals, -1, axis=1)), axis=2),
    ), axis=2)
    _append_float_buffer(mesh.positions, triangle_positions)
    _append_float_buffer(mesh.normals, triangle_normals)
    _append_float_buffer(mesh.colours, np.broadcast_to(branch_colour, triangle_positions.shape))

    # Every tube is a closed piece of geometry. Child bases normally sit inside
    # their parent branch, so the basal cap is hidden; the distal cap prevents
    # camera-visible holes and broken silhouettes when leaves are disabled.
    start_tangent = normalize(points[1] - points[0])
    end_tangent = normalize(points[-1] - points[-2])
    cap_specs = (
        (ring_vertices[0], np.asarray(points[0]), -start_tangent),
        (ring_vertices[-1], np.asarray(points[-1]), end_tangent),
    )
    for ring, centre, desired_normal in cap_specs:
        ring_array = np.asarray(ring, dtype=np.float32)
        centre_array = np.asarray(centre, dtype=np.float32)
        first_ring = ring_array
        second_ring = np.roll(ring_array, -1, axis=0)
        orientation = np.cross(first_ring - centre_array, second_ring - centre_array)
        if float(np.mean(orientation @ desired_normal)) < 0.0:
            first_ring, second_ring = second_ring, first_ring
        cap_triangles = np.stack((
            np.broadcast_to(centre_array, first_ring.shape),
            first_ring,
            second_ring,
        ), axis=1)
        cap_normals = np.broadcast_to(
            np.asarray(desired_normal, dtype=np.float32), cap_triangles.shape
        )
        cap_colours = np.broadcast_to(branch_colour, cap_triangles.shape)
        _append_float_buffer(mesh.positions, cap_triangles)
        _append_float_buffer(mesh.normals, cap_normals)
        _append_float_buffer(mesh.colours, cap_colours)


def _make_oak_leaves(
    leaves: list[Leaf],
    mesh: MeshData,
    batch_size: int = 16384,
    profile: list[tuple[float, float]] | None = None,
    size_scale: float = 1.0,
    max_leaf_count: int | None = None,
    card_mode: bool = False,
) -> None:
    """Triangulate a deterministic leaf subset using full or two-triangle leaves."""
    source_leaf_count = len(leaves)
    mesh.source_leaf_count = source_leaf_count
    if not leaves:
        mesh.leaf_count = 0
        return
    if size_scale <= 0.0:
        raise ValueError("Leaf size_scale must be greater than zero")

    if max_leaf_count is not None:
        if max_leaf_count < 0:
            raise ValueError("max_leaf_count cannot be negative")
        if max_leaf_count == 0:
            mesh.leaf_count = 0
            return
        if max_leaf_count < source_leaf_count:
            # Evenly spaced selection spans the entire recipe order (twig and pass
            # order), avoiding the crown bias caused by taking the first N leaves.
            selected_indices = np.linspace(
                0, source_leaf_count - 1, max_leaf_count, dtype=np.int64
            )
            leaves = [leaves[int(index)] for index in selected_indices]
    mesh.leaf_count = len(leaves)

    base_positions = np.asarray([leaf.base_position for leaf in leaves], dtype=np.float32)
    growth_directions = np.asarray(
        [normalize(leaf.growth_direction) for leaf in leaves], dtype=np.float32
    )
    width_directions = np.asarray(
        [normalize(leaf.width_direction) for leaf in leaves], dtype=np.float32
    )
    leaf_lengths = np.asarray([leaf.length for leaf in leaves], dtype=np.float32) * size_scale
    leaf_widths = np.asarray([leaf.width for leaf in leaves], dtype=np.float32) * size_scale
    colour_variations = np.asarray([leaf.colour_variation for leaf in leaves], dtype=np.float32)

    if card_mode:
        # A rectangular two-triangle card carries an alpha-cutout oak-leaf
        # silhouette. The texture, rather than a diamond-shaped polygon, defines
        # the lobed margin. Cards remain independently oriented in 3D.
        base_centres = base_positions - growth_directions * (leaf_lengths[:, None] * 0.10)
        tip_centres = base_positions + growth_directions * (leaf_lengths[:, None] * 0.90)
        half_widths = width_directions * (leaf_widths[:, None] * 0.56)
        base_left = base_centres - half_widths
        base_right = base_centres + half_widths
        tip_left = tip_centres - half_widths
        tip_right = tip_centres + half_widths
        triangle_positions = np.stack((
            np.stack((base_left, tip_left, tip_right), axis=1),
            np.stack((base_left, tip_right, base_right), axis=1),
        ), axis=1)

        leaf_normals = np.cross(growth_directions, width_directions)
        normal_lengths = np.linalg.norm(leaf_normals, axis=1, keepdims=True)
        leaf_normals = leaf_normals / np.maximum(normal_lengths, 1e-8)
        triangle_normals = np.broadcast_to(
            leaf_normals[:, None, None, :], triangle_positions.shape
        )

        # The atlas has four 88x128 oak-leaf variants in a 2x2 layout.
        # A stable ID hash chooses a tile without introducing run-to-run flicker.
        tile_indices = np.fromiter(
            (zlib.crc32(leaf.leaf_id.encode("utf-8")) % 4 for leaf in leaves),
            dtype=np.int32,
            count=len(leaves),
        )
        tile_columns = tile_indices % 2
        tile_rows_from_top = tile_indices // 2
        atlas_width = 2 * 88
        atlas_height = 2 * 128
        padding_x = 2.0 / atlas_width
        padding_y = 2.0 / atlas_height
        u_min = tile_columns.astype(np.float32) / 2.0 + padding_x
        u_max = (tile_columns.astype(np.float32) + 1.0) / 2.0 - padding_x
        gl_rows_from_bottom = (1 - tile_rows_from_top).astype(np.float32)
        v_min = gl_rows_from_bottom / 2.0 + padding_y
        v_max = (gl_rows_from_bottom + 1.0) / 2.0 - padding_y

        triangle_uvs = np.stack((
            np.stack((
                np.stack((u_min, v_min), axis=1),
                np.stack((u_min, v_max), axis=1),
                np.stack((u_max, v_max), axis=1),
            ), axis=1),
            np.stack((
                np.stack((u_min, v_min), axis=1),
                np.stack((u_max, v_max), axis=1),
                np.stack((u_max, v_min), axis=1),
            ), axis=1),
        ), axis=1)
        # Texture colour supplies the leaf's botanical green; vertex colour
        # introduces only subtle per-leaf tint variation.
        leaf_tints = np.clip(0.96 + colour_variations * 1.8, 0.78, 1.12)
        triangle_colours = np.broadcast_to(
            leaf_tints[:, None, None, :], triangle_positions.shape
        )
        _append_float_buffer(mesh.positions, triangle_positions)
        _append_float_buffer(mesh.normals, triangle_normals)
        _append_float_buffer(mesh.colours, triangle_colours)
        _append_float_buffer(mesh.texcoords, triangle_uvs)
        mesh.uses_leaf_texture = True
        mesh.canopy_kind = "leaves"
        mesh.texture_kind = "leaf"
        return

    # The detailed profile remains available to the offline impostor baker and
    # close-up reference renders. It deliberately does not use the runtime budget.
    left_edge = np.asarray(profile or OAK_LEAF_PROFILE, dtype=np.float32)
    right_edge = left_edge[:-1].copy()
    right_edge[:, 1] *= -1.0
    outline = np.concatenate((left_edge, right_edge[::-1]), axis=0)
    longitudinal = outline[:, 0][None, :, None]
    lateral = outline[:, 1][None, :, None]

    for batch_start in range(0, len(leaves), max(1, batch_size)):
        batch_end = min(len(leaves), batch_start + max(1, batch_size))
        batch_positions = base_positions[batch_start:batch_end]
        batch_growth = growth_directions[batch_start:batch_end]
        batch_width = width_directions[batch_start:batch_end]
        batch_lengths = leaf_lengths[batch_start:batch_end]
        batch_widths = leaf_widths[batch_start:batch_end]
        batch_variations = colour_variations[batch_start:batch_end]

        outline_positions = (
            batch_positions[:, None, :]
            + batch_growth[:, None, :] * (longitudinal * batch_lengths[:, None, None])
            + batch_width[:, None, :] * (lateral * batch_widths[:, None, None])
        )
        centre_positions = batch_positions + batch_growth * (batch_lengths[:, None] * 0.48)
        next_outline_positions = np.roll(outline_positions, -1, axis=1)

        triangle_positions = np.stack((
            np.broadcast_to(centre_positions[:, None, :], outline_positions.shape),
            outline_positions,
            next_outline_positions,
        ), axis=2)
        leaf_normals = np.cross(batch_growth, batch_width)
        normal_lengths = np.linalg.norm(leaf_normals, axis=1, keepdims=True)
        leaf_normals = leaf_normals / np.maximum(normal_lengths, 1e-8)
        triangle_normals = np.broadcast_to(
            leaf_normals[:, None, None, :], triangle_positions.shape
        )
        leaf_colours = np.clip(LEAF_BASE_COLOUR + batch_variations, 0.06, 0.85)
        triangle_colours = np.broadcast_to(
            leaf_colours[:, None, None, :], triangle_positions.shape
        )
        _append_float_buffer(mesh.positions, triangle_positions)
        _append_float_buffer(mesh.normals, triangle_normals)
        _append_float_buffer(mesh.colours, triangle_colours)




def build_crown_proxy_mesh(
    tree: TreeModel,
    *,
    azimuth_segments: int = 36,
    polar_rings: int = 18,
    seed_offset: int = 0,
) -> MeshData:
    """Build one coarse, irregular crown volume instead of rendering leaf cards.

    The shell is fitted to the canopy's leaf-bearing point cloud, then displaced
    with low-frequency harmonics to create a broad, uneven oak crown silhouette.
    At far distances the canopy is intentionally read as a mass, not as leaves.
    """
    mesh = MeshData()
    if tree.leaves:
        leaf_bases = np.asarray(
            [leaf.base_position for leaf in tree.leaves], dtype=np.float64
        )
        leaf_tips = np.asarray([
            leaf.base_position + normalize(leaf.growth_direction) * leaf.length
            for leaf in tree.leaves
        ], dtype=np.float64)
        canopy_points = np.vstack((leaf_bases, leaf_tips))
    else:
        branch_points = [
            point
            for branch in tree.branches
            if branch.role in {"secondary", "tertiary", "spray", "twig"}
            for point in branch.points
        ]
        if not branch_points:
            return mesh
        canopy_points = np.asarray(branch_points, dtype=np.float64)

    lower = np.quantile(canopy_points, 0.015, axis=0)
    upper = np.quantile(canopy_points, 0.985, axis=0)
    centre = (lower + upper) * 0.5
    radii = np.maximum((upper - lower) * 0.55, np.asarray([0.35, 0.30, 0.35]))

    phase = (int(tree.metadata.get("seed", 1)) + seed_offset) * 0.61803398875
    vertices: list[list[np.ndarray]] = []
    vertex_normals: list[list[np.ndarray]] = []
    vertex_colours: list[list[tuple[float, float, float]]] = []
    for ring_index in range(polar_rings + 1):
        theta = math.pi * ring_index / polar_rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        ring_vertices: list[np.ndarray] = []
        ring_normals: list[np.ndarray] = []
        ring_colours: list[tuple[float, float, float]] = []
        for segment_index in range(azimuth_segments):
            phi = math.tau * segment_index / azimuth_segments
            direction = np.asarray([
                sin_theta * math.cos(phi),
                cos_theta,
                sin_theta * math.sin(phi),
            ], dtype=np.float64)
            # Broad harmonics create shallow lobes; the smaller term breaks up
            # the profile without turning the far canopy back into leaf noise.
            displacement = (
                0.075 * math.cos(3.0 * phi + phase) * sin_theta ** 2
                + 0.045 * math.sin(5.0 * phi - 1.7 * theta + phase * 0.7) * sin_theta ** 1.5
                + 0.035 * math.cos(3.0 * theta + 2.0 * phi + phase)
            )
            radial_scale = max(0.78, 1.0 + displacement)
            position = centre + direction * radii * radial_scale
            normal = normalize(direction / radii, direction)
            shade = (
                0.92
                + 0.075 * math.sin(4.0 * phi + 2.0 * theta + phase)
                + 0.035 * math.cos(7.0 * phi - 3.0 * theta + phase)
            )
            colour = tuple(
                float(np.clip(component * shade, 0.035, 0.85))
                for component in (0.15, 0.31, 0.075)
            )
            ring_vertices.append(position)
            ring_normals.append(normal)
            ring_colours.append(colour)
        vertices.append(ring_vertices)
        vertex_normals.append(ring_normals)
        vertex_colours.append(ring_colours)

    for ring_index in range(polar_rings):
        next_ring = ring_index + 1
        for segment_index in range(azimuth_segments):
            next_segment = (segment_index + 1) % azimuth_segments
            faces = (
                (
                    (ring_index, segment_index),
                    (next_ring, segment_index),
                    (next_ring, next_segment),
                ),
                (
                    (ring_index, segment_index),
                    (next_ring, next_segment),
                    (ring_index, next_segment),
                ),
            )
            for face in faces:
                positions = [vertices[row][column] for row, column in face]
                normals = [vertex_normals[row][column] for row, column in face]
                colours = [vertex_colours[row][column] for row, column in face]
                geometric = np.cross(positions[1] - positions[0], positions[2] - positions[0])
                if float(np.dot(geometric, np.mean(normals, axis=0))) < 0.0:
                    positions[1], positions[2] = positions[2], positions[1]
                    normals[1], normals[2] = normals[2], normals[1]
                    colours[1], colours[2] = colours[2], colours[1]
                for position, normal, colour in zip(positions, normals, colours):
                    mesh.positions.extend(float(value) for value in position)
                    mesh.normals.extend(float(value) for value in normal)
                    mesh.colours.extend(colour)
                    mesh.texcoords.extend((0.0, 0.0))

    mesh.source_leaf_count = len(tree.leaves)
    mesh.canopy_kind = "crown"
    mesh.texture_kind = "none"
    return mesh


def build_foliage_cluster_mesh(
    tree: TreeModel,
    *,
    cluster_count: int,
    cluster_scale: float = 0.42,
    seed_offset: int = 0,
    branch_roles: tuple[str, ...] = ("spray", "twig", "tertiary", "secondary"),
    planes_per_cluster: int = 3,
) -> MeshData:
    """Build crossed, branch-attached leaf-clump cards for middle LODs.

    Each clump card samples a multi-leaf atlas tile. Three intersecting planes
    provide volume at oblique camera angles without exposing individual leaves
    or carrying the cost of a fully modelled canopy.
    """
    mesh = MeshData()
    if cluster_count <= 0:
        return mesh
    if cluster_scale <= 0.0 or planes_per_cluster < 2:
        raise ValueError("Cluster scale must be positive and at least two planes are required")

    candidates = [
        branch for branch in tree.branches
        if branch.role in branch_roles and branch.length > 0.06
    ]
    if not candidates:
        return mesh

    role_weights = {
        "spray": 2.4,
        "twig": 0.72,
        "tertiary": 0.58,
        "secondary": 0.22,
    }
    weights = np.asarray([
        max(branch.length, 0.04)
        * math.sqrt(max(branch.radii[0], 0.002))
        * role_weights.get(branch.role, 0.5)
        for branch in candidates
    ], dtype=np.float64)
    probabilities = weights / max(float(np.sum(weights)), 1e-12)
    rng = np.random.default_rng(int(tree.metadata.get("seed", 1)) + seed_offset + 901)
    selected = rng.choice(len(candidates), size=cluster_count, replace=True, p=probabilities)

    atlas_width = 256.0
    atlas_height = 256.0
    padding_u = 3.0 / atlas_width
    padding_v = 3.0 / atlas_height
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    for cluster_index, candidate_index in enumerate(selected):
        branch = candidates[int(candidate_index)]
        proportion = float(rng.uniform(0.34, 0.88))
        tangent = normalize(branch.tangent_at(proportion), (0.0, 1.0, 0.0))
        side_axis, second_axis = perpendicular_frame(tangent)
        radial_angle = float(rng.uniform(0.0, math.tau))
        radial = normalize(
            side_axis * math.cos(radial_angle) + second_axis * math.sin(radial_angle),
            side_axis,
        )
        centre = branch.point_at(proportion) + radial * float(
            rng.uniform(0.015, 0.075)
        )
        growth = normalize(
            tangent * float(rng.uniform(0.18, 0.38))
            + world_up * float(rng.uniform(0.48, 0.78))
            + radial * float(rng.uniform(-0.18, 0.18)),
            world_up,
        )
        plane_axis_a, plane_axis_b = perpendicular_frame(growth)
        scale = cluster_scale * float(rng.uniform(0.82, 1.22))
        half_width = scale * float(rng.uniform(0.62, 0.78))
        half_height = scale * float(rng.uniform(0.72, 0.90))
        tile_index = int(rng.integers(0, 4))
        tile_column = tile_index % 2
        tile_row_from_top = tile_index // 2
        u_min = tile_column * 0.5 + padding_u
        u_max = (tile_column + 1) * 0.5 - padding_u
        gl_row = 1 - tile_row_from_top
        v_min = gl_row * 0.5 + padding_v
        v_max = (gl_row + 1) * 0.5 - padding_v
        tint = float(rng.uniform(0.87, 1.10))
        vertex_colour = (
            tint,
            tint * float(rng.uniform(0.94, 1.04)),
            tint * float(rng.uniform(0.84, 0.98)),
        )

        # Rotating three cards around the local growth axis creates a compact
        # 3D spray; each texture tile itself contains several oak leaves.
        phase = float(rng.uniform(0.0, math.tau))
        for plane_index in range(planes_per_cluster):
            angle = phase + math.pi * plane_index / planes_per_cluster
            width_axis = normalize(
                plane_axis_a * math.cos(angle) + plane_axis_b * math.sin(angle),
                plane_axis_a,
            )
            normal = normalize(np.cross(growth, width_axis), (0.0, 0.0, 1.0))
            bottom_left = centre - width_axis * half_width - growth * half_height
            top_left = centre - width_axis * half_width + growth * half_height
            top_right = centre + width_axis * half_width + growth * half_height
            bottom_right = centre + width_axis * half_width - growth * half_height
            triangles = (
                (bottom_left, top_left, top_right),
                (bottom_left, top_right, bottom_right),
            )
            uvs = (
                ((u_min, v_min), (u_min, v_max), (u_max, v_max)),
                ((u_min, v_min), (u_max, v_max), (u_max, v_min)),
            )
            for triangle, triangle_uvs in zip(triangles, uvs):
                for position, uv in zip(triangle, triangle_uvs):
                    mesh.positions.extend(float(value) for value in position)
                    mesh.normals.extend(float(value) for value in normal)
                    mesh.colours.extend(vertex_colour)
                    mesh.texcoords.extend(uv)

    mesh.cluster_count = cluster_count
    mesh.source_leaf_count = int(tree.metadata.get("source_leaf_count", len(tree.leaves)))
    mesh.uses_leaf_texture = True
    mesh.canopy_kind = "clusters"
    mesh.texture_kind = "cluster"
    return mesh


def build_wood_mesh(
    tree: TreeModel,
    radial_sides: int = 16,
    include_terminal_twigs: bool = True,
    max_branch_depth: int | None = None,
    max_branch_segments: int | None = None,
) -> MeshData:
    """Build trunk, roots and branch geometry without leaf polygons.

    Set `include_terminal_twigs=False` when terminal twig/spray structure is represented
    by runtime foliage cards or a canopy impostor. Optional depth and segment limits
    allow the progressive renderer to build cheaper branch LODs.
    """
    mesh = MeshData()
    fused_surface = tree.metadata.get("wood_surface")
    fused_branch_ids: set[str] = set()
    if isinstance(fused_surface, dict) and isinstance(fused_surface.get("mesh"), MeshData):
        unified_mesh = fused_surface["mesh"]
        mesh.positions.extend(unified_mesh.positions)
        mesh.normals.extend(unified_mesh.normals)
        mesh.colours.extend(unified_mesh.colours)
        fused_branch_ids = set(fused_surface.get("branch_ids", []))

    for branch in tree.branches:
        if branch.branch_id in fused_branch_ids:
            continue
        if max_branch_depth is not None and branch.depth > max_branch_depth:
            continue
        if not include_terminal_twigs and branch.role in {"twig", "spray"}:
            continue
        _make_branch_tube(branch, mesh, radial_sides, max_branch_segments)
    return mesh


def build_foliage_mesh(
    tree: TreeModel,
    include_twigs: bool = False,
    radial_sides: int = 16,
    leaf_profile: list[tuple[float, float]] | None = None,
    leaf_size_scale: float = 1.0,
    max_leaf_count: int | None = None,
    card_mode: bool = False,
) -> MeshData:
    """Build only canopy geometry for depth-impostor baking.

    Leaves are always included. Terminal twig and spray tubes are optional so
    their contribution can be assessed at the target atlas resolution. Trunk,
    roots, scaffold limbs and all other branch tiers are deliberately excluded
    and remain available through :func:`build_wood_mesh` for polygon LOD.
    """
    mesh = MeshData()
    if include_twigs:
        for branch in tree.branches:
            if branch.role in {"twig", "spray"}:
                _make_branch_tube(branch, mesh, radial_sides)
    if tree.leaves:
        _make_oak_leaves(
            tree.leaves,
            mesh,
            profile=leaf_profile,
            size_scale=leaf_size_scale,
            max_leaf_count=max_leaf_count,
            card_mode=card_mode,
        )
    return mesh


def build_tree_mesh(tree: TreeModel, radial_sides: int = 16) -> MeshData:
    """Assemble the complete legacy tree mesh from wood and leaf geometry."""
    mesh = build_wood_mesh(tree, radial_sides)
    if tree.leaves:
        leaf_mesh = MeshData()
        _make_oak_leaves(tree.leaves, leaf_mesh)
        mesh.positions.frombytes(memoryview(leaf_mesh.positions).cast("B"))
        mesh.normals.frombytes(memoryview(leaf_mesh.normals).cast("B"))
        mesh.colours.frombytes(memoryview(leaf_mesh.colours).cast("B"))
    return mesh

def write_obj(mesh: MeshData, file_path: str) -> None:
    """Write generated triangles, normals and optional atlas UVs to OBJ."""
    positions = np.asarray(mesh.positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
    colours = np.asarray(mesh.colours, dtype=np.float32).reshape(-1, 3)
    has_uvs = len(mesh.texcoords) == len(positions) * 2
    texcoords = (
        np.asarray(mesh.texcoords, dtype=np.float32).reshape(-1, 2)
        if has_uvs else None
    )
    material_name = Path(file_path).stem + ".mtl"
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write("# OakForge generated oak tree\n")
        material_variant = {
            "leaf": ("OakLeaf", "oak_leaf_atlas.png"),
            "cluster": ("OakCluster", "oak_foliage_cluster_atlas.png"),
        }.get(mesh.texture_kind)
        if material_variant is not None and has_uvs:
            output_file.write(
                f"mtllib {material_name}\nusemtl {material_variant[0]}\n"
            )
        for position, colour in zip(positions, colours):
            output_file.write("v %.6f %.6f %.6f %.4f %.4f %.4f\n" % (*position, *colour))
        if texcoords is not None:
            for uv in texcoords:
                output_file.write("vt %.6f %.6f\n" % tuple(uv))
        for normal in normals:
            output_file.write("vn %.6f %.6f %.6f\n" % tuple(normal))
        for triangle_start in range(0, len(positions), 3):
            first = triangle_start + 1
            second = triangle_start + 2
            third = triangle_start + 3
            if texcoords is not None:
                output_file.write(
                    f"f {first}/{first}/{first} {second}/{second}/{second} {third}/{third}/{third}\n"
                )
            else:
                output_file.write(f"f {first}//{first} {second}//{second} {third}//{third}\n")

    material_variant = {
        "leaf": ("OakLeaf", "oak_leaf_atlas.png"),
        "cluster": ("OakCluster", "oak_foliage_cluster_atlas.png"),
    }.get(mesh.texture_kind)
    if material_variant is not None and has_uvs:
        material_path = Path(file_path).with_suffix(".mtl")
        material_path.write_text(
            f"newmtl {material_variant[0]}\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "d 1.0\n"
            "illum 1\n"
            f"map_Kd {material_variant[1]}\n"
            f"map_d {material_variant[1]}\n",
            encoding="utf-8",
        )
