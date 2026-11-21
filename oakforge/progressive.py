"""Background generation and CPU-side LOD construction for OakForge.

The worker in this module never calls OpenGL. It produces renderer-independent
models and packed meshes which the viewer's context-owning thread can upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any
import math

import numpy as np

from oakforge.core.geometry import normalize, perpendicular_frame
from oakforge.core.mesh import MeshData, build_foliage_mesh, build_wood_mesh
from oakforge.core.models import Leaf, TreeModel
from oakforge.generation import generate_from_file, generate_tree, load_recipe


@dataclass(frozen=True)
class ProgressiveLOD:
    """A complete CPU-side representation ready for render-thread upload."""

    level: int
    name: str
    tree: TreeModel
    wood_mesh: MeshData
    foliage_mesh: MeshData

    @property
    def triangle_count(self) -> int:
        return self.wood_mesh.triangle_count + self.foliage_mesh.triangle_count

    @property
    def cpu_bytes(self) -> int:
        return sum(
            len(buffer)
            for mesh in (self.wood_mesh, self.foliage_mesh)
            for buffer in (mesh.positions, mesh.normals, mesh.colours)
        )


def build_cluster_leaves(
    tree: TreeModel,
    *,
    target_count: int,
    branch_roles: tuple[str, ...] = ("primary", "leader", "secondary", "tertiary"),
    seed_offset: int = 0,
) -> list[Leaf]:
    """Create deterministic lobed-leaf clusters distributed along scaffold limbs.

    These proxy leaves are anchored to the actual generated branches. They are
    a low-cost distant representation, not a replacement for botanical foliage
    placement in the full model.
    """
    candidates = [branch for branch in tree.branches if branch.role in branch_roles]
    if not candidates or target_count <= 0:
        return []

    rng = np.random.default_rng(int(tree.metadata.get("seed", 1)) + seed_offset)
    # Give each eligible branch several cluster anchors. The stride and cap keep
    # proxy density bounded for large species configurations.
    anchors_per_branch = 3
    leaves_per_anchor = 4
    maximum_anchors = max(1, math.ceil(target_count / leaves_per_anchor))
    stride = max(1, math.ceil(len(candidates) * anchors_per_branch / maximum_anchors))
    leaves: list[Leaf] = []
    for branch_index, branch in enumerate(candidates):
        if branch_index % stride:
            continue
        tangent_frame = None
        for anchor_index, proportion in enumerate((0.24, 0.56, 0.86)):
            if len(leaves) >= target_count:
                return leaves
            tangent = normalize(branch.tangent_at(proportion), (0.0, 1.0, 0.0))
            side_axis, second_axis = perpendicular_frame(tangent)
            centre = branch.point_at(proportion)
            base_angle = float(rng.uniform(0.0, math.tau))
            for cluster_index in range(leaves_per_anchor):
                if len(leaves) >= target_count:
                    return leaves
                angle = base_angle + cluster_index * (math.tau / leaves_per_anchor)
                radial = normalize(
                    side_axis * math.cos(angle) + second_axis * math.sin(angle)
                )
                radial += rng.normal(0.0, 0.16, size=3)
                radial = normalize(radial, side_axis)
                base = centre + radial * float(rng.uniform(0.035, 0.13))
                base += np.array([0.0, float(rng.uniform(-0.035, 0.045)), 0.0])
                growth = normalize(
                    tangent * 0.34 + radial * 0.48
                    + np.array([0.0, 0.28, 0.0])
                    + rng.normal(0.0, 0.18, size=3)
                )
                width_hint = radial + rng.normal(0.0, 0.22, size=3)
                width_hint -= growth * float(np.dot(width_hint, growth))
                width = normalize(width_hint, second_axis)
                scale = float(rng.uniform(0.78, 1.18))
                variation = float(rng.uniform(-0.045, 0.045))
                leaves.append(Leaf(
                    leaf_id=f"cluster_{branch.branch_id}_{anchor_index}_{cluster_index}",
                    base_position=base,
                    growth_direction=growth,
                    width_direction=width,
                    length=0.21 * scale,
                    width=0.21 * 0.70 * scale,
                    colour_variation=np.array([variation * 0.58, variation, rng.uniform(-0.012, 0.012)]),
                ))
    return leaves


def generate_coarse_tree(recipe_path: str | Path) -> TreeModel:
    """Generate a quick scaffold tree without fine twigs, sprays or leaf work.

    The recipe's first growth loop is limited to its secondary branching pass.
    Its original seed and botanical parameters are retained.
    """
    recipe = load_recipe(recipe_path)
    recipe["pipeline"] = []
    for action in load_recipe(recipe_path)["pipeline"]:
        if "while" in action:
            progressive_action = dict(action["while"])
            progressive_action["condition"] = {"lt": ["$growth_level", 2]}
            progressive_action["max_iterations"] = 1
            recipe["pipeline"].append({"while": progressive_action})
        elif "call" in action:
            if action["call"].get("tool") != "wood_surface.build":
                recipe["pipeline"].append(action)
        elif "set" in action:
            recipe["pipeline"].append(action)
        # The top-level for/if actions after the growth loop create sprays and
        # full leaf populations. Both are deliberately deferred.
    tree = generate_tree(recipe)
    tree.leaves = build_cluster_leaves(tree, target_count=1800, seed_offset=3)
    tree.metadata = dict(tree.metadata)
    tree.metadata.update({
        "progressive_level": 3,
        "progressive_status": "initial scaffold and clustered canopy",
        "source_leaf_count": 0,
        "proxy_leaf_count": len(tree.leaves),
    })
    return tree


def _tree_for_level(source: TreeModel, level: int) -> TreeModel:
    """Select a structural and foliage subset for one runtime representation."""
    metadata = dict(source.metadata)
    if level == 2:
        branches = [
            branch for branch in source.branches
            if branch.role in {"trunk", "root", "primary", "leader", "secondary", "tertiary"}
        ]
        tree = TreeModel(source.name, branches, [], metadata)
        tree.leaves = build_cluster_leaves(tree, target_count=7200, seed_offset=12)
        metadata.update({
            "progressive_level": 2,
            "progressive_status": "tertiary scaffold and dense branch-anchored clusters",
            "source_leaf_count": len(source.leaves),
            "proxy_leaf_count": len(tree.leaves),
        })
    elif level == 1:
        # One in every eight leaves preserves the original placements and colours
        # while substantially reducing close/mid-range geometry and upload cost.
        tree = TreeModel(source.name, list(source.branches), source.leaves[::8], metadata)
        metadata.update({
            "progressive_level": 1,
            "progressive_status": "full shoots and one-eighth leaf sampling",
            "source_leaf_count": len(source.leaves),
            "proxy_leaf_count": len(tree.leaves),
        })
    elif level == 0:
        tree = TreeModel(source.name, list(source.branches), list(source.leaves), metadata)
        metadata.update({
            "progressive_level": 0,
            "progressive_status": "complete detailed foliage",
            "source_leaf_count": len(source.leaves),
            "proxy_leaf_count": 0,
        })
    else:
        raise ValueError(f"Unsupported progressive LOD level: {level}")
    tree.metadata = metadata
    return tree


def build_lod_representation(tree: TreeModel, level: int) -> ProgressiveLOD:
    """Build one LOD's CPU geometry without creating or touching GPU resources."""
    lod_tree = _tree_for_level(tree, level) if level < 3 else tree
    wood_mesh = build_wood_mesh(lod_tree)
    foliage_mesh = build_foliage_mesh(lod_tree)
    names = {
        0: "Detailed foliage",
        1: "Full shoots · sampled leaves",
        2: "Tertiary branches · foliage clusters",
        3: "Primary scaffold · foliage clusters",
    }
    return ProgressiveLOD(level, names[level], lod_tree, wood_mesh, foliage_mesh)


class ProgressiveTreeWorker:
    """Generate the complete tree and publish successively finer CPU LODs."""

    def __init__(self, recipe_path: str | Path) -> None:
        self.recipe_path = Path(recipe_path)
        self._queue: Queue[ProgressiveLOD | BaseException | None] = Queue(maxsize=1)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="OakForgeTreeGeneration", daemon=True)
        self._thread.start()

    def _publish(self, item: ProgressiveLOD | BaseException | None) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except Exception:
                continue
        return False

    def _run(self) -> None:
        try:
            tree = generate_from_file(self.recipe_path)
            for level in (2, 1, 0):
                if self._stop.is_set():
                    return
                representation = build_lod_representation(tree, level)
                if not self._publish(representation):
                    return
        except BaseException as error:
            self._publish(error)
        finally:
            self._publish(None)

    def poll(self) -> list[ProgressiveLOD | BaseException | None]:
        """Return completed work without blocking the render thread."""
        items: list[ProgressiveLOD | BaseException | None] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                return items

    def close(self, timeout: float = 2.0) -> None:
        """Stop publishing and allow the daemon worker to exit cleanly."""
        self._stop.set()
        self._thread.join(timeout)
