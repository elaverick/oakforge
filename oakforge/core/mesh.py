"""Triangle mesh construction for the generated tree."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Iterable

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
OAK_LEAF_PROFILE = [
    (0.00, 0.012), (0.10, 0.18), (0.19, 0.095),
    (0.31, 0.29), (0.41, 0.14), (0.55, 0.31),
    (0.65, 0.15), (0.79, 0.25), (0.90, 0.09),
    (0.97, 0.035), (1.00, 0.0),
]


def _make_branch_tube(branch: Branch, mesh: MeshData, radial_sides: int) -> None:
    """Build a tapered tube using batched ring and face construction."""
    if branch.role == "trunk":
        side_count = max(16, radial_sides)
    elif branch.role in {"tertiary", "twig", "spray"}:
        side_count = max(12, radial_sides * 3 // 4)
    else:
        side_count = max(10, radial_sides // 2)
    branch_colour = np.asarray(
        BRANCH_COLOURS.get(branch.role, BRANCH_COLOURS["secondary"]), dtype=np.float32
    )
    previous_side_axis = None
    ring_vertices: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []
    side_angles = np.arange(side_count, dtype=np.float64) * (2.0 * np.pi / side_count)
    side_cosines = np.cos(side_angles)[:, None]
    side_sines = np.sin(side_angles)[:, None]

    for point_index, centre in enumerate(branch.points):
        previous_point = branch.points[max(0, point_index - 1)]
        next_point = branch.points[min(len(branch.points) - 1, point_index + 1)]
        tangent = normalize(next_point - previous_point)
        side_axis, normal_axis = transported_frame(tangent, previous_side_axis)
        previous_side_axis = side_axis
        radial_directions = normalize(
            side_cosines * side_axis[None, :] + side_sines * normal_axis[None, :]
        )
        local_radius = float(branch.radii[point_index])
        height_above_ground = max(float(centre[1]), 0.0)
        trunk_height = max(float(branch.points[-1][1]), 1e-5)
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
    start_tangent = normalize(branch.points[1] - branch.points[0])
    end_tangent = normalize(branch.points[-1] - branch.points[-2])
    cap_specs = (
        (ring_vertices[0], np.asarray(branch.points[0]), -start_tangent),
        (ring_vertices[-1], np.asarray(branch.points[-1]), end_tangent),
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


def _make_oak_leaves(leaves: list[Leaf], mesh: MeshData, batch_size: int = 16384) -> None:
    """Triangulate foliage in bounded batches to keep large crowns responsive."""
    if not leaves:
        return

    base_positions = np.asarray([leaf.base_position for leaf in leaves], dtype=np.float32)
    growth_directions = np.asarray(
        [normalize(leaf.growth_direction) for leaf in leaves], dtype=np.float32
    )
    width_directions = np.asarray(
        [normalize(leaf.width_direction) for leaf in leaves], dtype=np.float32
    )
    leaf_lengths = np.asarray([leaf.length for leaf in leaves], dtype=np.float32)
    leaf_widths = np.asarray([leaf.width for leaf in leaves], dtype=np.float32)
    colour_variations = np.asarray([leaf.colour_variation for leaf in leaves], dtype=np.float32)

    # The fan outline keeps the oak's alternating lobes while remaining a lightweight polygon.
    left_edge = np.asarray(OAK_LEAF_PROFILE, dtype=np.float32)
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


def build_foliage_mesh(
    tree: TreeModel,
    include_twigs: bool = False,
    radial_sides: int = 16,
) -> MeshData:
    """Build only canopy geometry for depth-impostor baking.

    Leaves are always included. Terminal twig and spray tubes are optional so
    their contribution can be assessed at the target atlas resolution. Trunk,
    roots, scaffold limbs and all other branch tiers are deliberately excluded
    and remain available through :func:`build_tree_mesh` for polygon LOD.
    """
    mesh = MeshData()
    if include_twigs:
        for branch in tree.branches:
            if branch.role in {"twig", "spray"}:
                _make_branch_tube(branch, mesh, radial_sides)
    if tree.leaves:
        _make_oak_leaves(tree.leaves, mesh)
    return mesh


def build_tree_mesh(tree: TreeModel, radial_sides: int = 16) -> MeshData:
    """Assemble YAML-built fused wood, fine branch tubes and foliage into one mesh."""
    mesh = MeshData()
    fused_surface = tree.metadata.get("wood_surface")
    fused_branch_ids: set[str] = set()
    if isinstance(fused_surface, dict) and isinstance(fused_surface.get("mesh"), MeshData):
        unified_mesh = fused_surface["mesh"]
        mesh.positions.extend(unified_mesh.positions)
        mesh.normals.extend(unified_mesh.normals)
        mesh.colours.extend(unified_mesh.colours)
        fused_branch_ids = set(fused_surface.get("branch_ids", []))

    # Fine branches remain lightweight tubes; coarse limbs share one continuous surface.
    for branch in tree.branches:
        if branch.branch_id not in fused_branch_ids:
            _make_branch_tube(branch, mesh, radial_sides)

    # Build the large foliage buffer separately. Repeatedly growing a populated
    # branch array while appending hundreds of thousands of leaves causes costly
    # reallocations on some Python runtimes; one final array-to-array copy avoids it.
    if tree.leaves:
        leaf_mesh = MeshData()
        _make_oak_leaves(tree.leaves, leaf_mesh)
        mesh.positions.frombytes(memoryview(leaf_mesh.positions).cast("B"))
        mesh.normals.frombytes(memoryview(leaf_mesh.normals).cast("B"))
        mesh.colours.frombytes(memoryview(leaf_mesh.colours).cast("B"))
    return mesh


def write_obj(mesh: MeshData, file_path: str) -> None:
    """Write the generated triangles to an OBJ file for external inspection."""
    positions = np.asarray(mesh.positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
    colours = np.asarray(mesh.colours, dtype=np.float32).reshape(-1, 3)
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write("# OakForge generated oak tree\n")
        for position, colour in zip(positions, colours):
            output_file.write("v %.6f %.6f %.6f %.4f %.4f %.4f\n" % (*position, *colour))
        for normal in normals:
            output_file.write("vn %.6f %.6f %.6f\n" % tuple(normal))
        for triangle_start in range(0, len(positions), 3):
            first = triangle_start + 1
            second = triangle_start + 2
            third = triangle_start + 3
            output_file.write(f"f {first}//{first} {second}//{second} {third}//{third}\n")
