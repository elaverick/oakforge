"""Background generation and CPU-side LOD construction for OakForge.

The worker in this module never calls OpenGL. It produces renderer-independent
models and packed meshes which the viewer's context-owning thread can upload.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
import math

import numpy as np

from oakforge.core.context import GenerationContext
from oakforge.core.geometry import normalize, perpendicular_frame
from oakforge.core.models import Branch
from oakforge.core.recipe_engine import RecipeEngine
from oakforge.tools.registry import create_default_registry
from oakforge.core.mesh import (
    FAR_LEAF_PROFILE,
    MID_LEAF_PROFILE,
    NEAR_LEAF_PROFILE,
    MeshData,
    build_foliage_mesh,
    build_wood_mesh,
)
from oakforge.core.models import Leaf, TreeModel
from oakforge.generation import generate_tree, load_recipe


@dataclass(frozen=True)
class ProgressiveLOD:
    """A complete CPU-side representation ready for render-thread upload."""

    level: int
    name: str
    tree: TreeModel
    wood_mesh: MeshData
    foliage_mesh: MeshData
    mesh_build_seconds: float = 0.0
    generation_elapsed_seconds: float = 0.0
    generation_complete: bool = False

    @property
    def triangle_count(self) -> int:
        return self.wood_mesh.triangle_count + self.foliage_mesh.triangle_count

    @property
    def cpu_bytes(self) -> int:
        return sum(
            len(buffer) * buffer.itemsize
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
    # Spread a bounded number of anchors across the entire branch set rather
    # than exhausting an early section of the hierarchy and leaving crown gaps.
    # Six compact whorls along each eligible scaffold limb fill the crown
    # volume more evenly than three widely separated anchor points.
    anchors_per_branch = 6
    leaves_per_anchor = 8
    maximum_branches = max(1, math.ceil(target_count / (anchors_per_branch * leaves_per_anchor)))
    selected_indices = set(np.linspace(
        0, len(candidates) - 1, min(len(candidates), maximum_branches), dtype=int
    ).tolist())
    leaves: list[Leaf] = []
    for branch_index, branch in enumerate(candidates):
        if branch_index not in selected_indices:
            continue
        for anchor_index, proportion in enumerate((0.08, 0.24, 0.40, 0.60, 0.76, 0.92)):
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
                base = centre + radial * float(rng.uniform(0.06, 0.20))
                base += np.array([0.0, float(rng.uniform(-0.05, 0.06)), 0.0])
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
                    length=0.25 * scale,
                    width=0.25 * 0.70 * scale,
                    colour_variation=np.array([variation * 0.58, variation, rng.uniform(-0.012, 0.012)]),
                ))
    return leaves


def generate_coarse_tree(recipe_path: str | Path, seed_override: int | None = None) -> TreeModel:
    """Generate the mature far-LOD scaffold with the shared fused major-wood surface.

    The two early growth iterations preserve the primary, leader, secondary and
    tertiary crown framework while omitting terminal shoots, sprays and original
    leaf placements. The trunk, roots and major scaffold limbs use the same fused
    surface as the refined LODs, so their apparent calibre remains continuous
    across the LOD3-to-LOD2 transition. Tertiary branches remain swept tubes.
    """
    recipe = load_recipe(recipe_path)
    if seed_override is not None:
        recipe["tree"]["seed"] = int(seed_override)
    wood_parameters = recipe.get("tree", {}).get("parameters", {}).get("wood_surface", {})
    coarse_root_tip_minimum = max(
        0.12, float(wood_parameters.get("voxel_size", 0.075)) * 1.35
    )
    source_pipeline = recipe["pipeline"]
    recipe["pipeline"] = []
    for action in source_pipeline:
        if "while" in action:
            progressive_action = dict(action["while"])
            progressive_action["condition"] = {"lt": ["$growth_level", 3]}
            progressive_action["max_iterations"] = 2
            recipe["pipeline"].append({"while": progressive_action})
        elif "call" in action:
            call = action["call"]
            if call.get("tool") == "wood_surface.build":
                # Keep the major structural limbs in the same fused surface
                # at every LOD. If they are tubes only at LOD3, their apparent
                # calibre changes abruptly when LOD2 replaces them with the
                # smooth-unioned wood surface used by the refined representations.
                coarse_action = dict(action)
                coarse_call = dict(call)
                coarse_arguments = dict(call.get("args", {}))
                coarse_arguments["branches"] = [
                    "$groups.trunk", "$groups.roots", "$groups.primary",
                    "$groups.leaders", "$groups.secondary",
                ]
                coarse_arguments["root_tip_radius_minimum"] = coarse_root_tip_minimum
                coarse_call["args"] = coarse_arguments
                coarse_action["call"] = coarse_call
                recipe["pipeline"].append(coarse_action)
            else:
                recipe["pipeline"].append(action)
        elif "set" in action:
            recipe["pipeline"].append(action)
        # The top-level for/if actions after the growth loop create sprays and
        # full leaf populations. Both are deliberately deferred.
    tree = generate_tree(recipe)
    tree.leaves = build_cluster_leaves(tree, target_count=35520, seed_offset=3)
    tree.metadata = dict(tree.metadata)
    tree.metadata.update({
        "progressive_level": 3,
        "progressive_status": "initial scaffold and clustered canopy",
        "source_leaf_count": 0,
        "proxy_leaf_count": len(tree.leaves),
        "leaf_count": len(tree.leaves),
        "representation_leaf_count": len(tree.leaves),
        "representation_branch_count": len(tree.branches),
    })
    return tree


class ProgressiveGenerationSession:
    """Execute one tree's structural recipe once, then add foliage pass by pass.

    The recipe's branch/twig/spray and wood-surface stages are completed first.
    The foliage loop is retained as executable recipe actions and advanced one
    pass at a time, so LOD2/LOD1 can be published before the hero foliage exists.
    """

    BRANCH_GROUPS = (
        "trunk", "roots", "primary", "leaders",
        "secondary", "tertiary", "twigs", "sprays",
    )

    def __init__(self, recipe_path: str | Path, seed_override: int | None = None) -> None:
        self.recipe = load_recipe(recipe_path)
        self.tree_settings = self.recipe["tree"]
        if seed_override is not None:
            self.tree_settings["seed"] = int(seed_override)
        seed = int(self.tree_settings.get("seed", 1))
        initial_values = {
            "tree": self.tree_settings,
            "groups": {
                "trunk": [], "roots": [], "primary": [], "leaders": [],
                "secondary": [], "tertiary": [], "twigs": [], "sprays": [],
                "leaves": [],
            },
        }
        self.context = GenerationContext(
            values=initial_values, random=np.random.default_rng(seed)
        )
        self.engine = RecipeEngine(self.context, create_default_registry())
        self.structural_actions: list[dict[str, Any]] = []
        self.trailing_actions: list[dict[str, Any]] = []
        self.foliage_prefix_actions: list[dict[str, Any]] = []
        self.foliage_actions: list[dict[str, Any]] = []
        self.foliage_suffix_actions: list[dict[str, Any]] = []
        self.foliage_disabled_actions: list[dict[str, Any]] = []
        self.foliage_enabled = False
        self.total_leaf_passes = 0
        self.current_leaf_pass = 0
        self._prepared = False
        self._finished = False

        pipeline = list(self.recipe["pipeline"])
        foliage_index = None
        foliage_if: dict[str, Any] | None = None
        foliage_loop: dict[str, Any] | None = None
        for action_index, action in enumerate(pipeline):
            if not isinstance(action, dict) or "if" not in action:
                continue
            candidate = action["if"]
            if not isinstance(candidate, dict):
                continue
            then_actions = candidate.get("then", [])
            loop_index = next(
                (
                    index for index, item in enumerate(then_actions)
                    if isinstance(item, dict) and "do_while" in item
                ),
                None,
            )
            if loop_index is not None:
                if sum(
                    1 for item in then_actions
                    if isinstance(item, dict) and "do_while" in item
                ) != 1:
                    raise ValueError("A progressive foliage branch must contain exactly one do_while")
                foliage_index = action_index
                foliage_if = candidate
                foliage_loop = then_actions[loop_index]["do_while"]
                self.foliage_prefix_actions = list(then_actions[:loop_index])
                self.foliage_suffix_actions = list(then_actions[loop_index + 1:])
                break

        if foliage_index is None or foliage_if is None or foliage_loop is None:
            self.structural_actions = pipeline
        else:
            self.structural_actions = pipeline[:foliage_index]
            self.trailing_actions = pipeline[foliage_index + 1:]
            self.foliage_enabled = bool(
                self.engine.evaluate(foliage_if.get("condition", False))
            )
            self.foliage_disabled_actions = list(foliage_if.get("else", []))
            if self.foliage_enabled:
                self.foliage_actions = list(foliage_loop.get("do", []))
                condition = foliage_loop.get("condition", {})
                operands = condition.get("lt") if isinstance(condition, dict) else None
                if not isinstance(operands, (list, tuple)) or len(operands) != 2:
                    raise ValueError(
                        "Progressive foliage do_while must use an 'lt' pass-count condition"
                    )
                self.total_leaf_passes = int(self.engine.evaluate(operands[1]))
                if self.total_leaf_passes < 0:
                    raise ValueError("Foliage pass count must be non-negative")
                maximum_iterations = foliage_loop.get("max_iterations")
                if maximum_iterations is not None:
                    maximum_iterations = int(self.engine.evaluate(maximum_iterations))
                    if maximum_iterations < self.total_leaf_passes:
                        raise ValueError(
                            "Foliage loop max_iterations is smaller than its declared pass count"
                        )
            else:
                self.context.values["foliage_status"] = "disabled"

    def prepare_structure(self) -> float:
        """Run trunk through wood-surface generation, but not individual leaves."""
        if self._prepared:
            raise RuntimeError("Progressive generation structure has already been prepared")
        started = perf_counter()
        self.engine.execute(self.structural_actions)
        if self.foliage_enabled:
            self.engine.execute(self.foliage_prefix_actions)
            self.context.values["leaf_pass"] = 0
        else:
            self.engine.execute(self.foliage_disabled_actions)
            self.engine.execute(self.trailing_actions)
            self._finished = True
        self._prepared = True
        return perf_counter() - started

    def advance_leaf_pass(self) -> bool:
        """Execute exactly one recipe-declared foliage pass."""
        if not self._prepared:
            raise RuntimeError("Call prepare_structure before advancing foliage")
        if self._finished or self.current_leaf_pass >= self.total_leaf_passes:
            return False
        self.engine.execute(self.foliage_actions)
        self.current_leaf_pass = int(self.context.values["leaf_pass"])
        if self.current_leaf_pass >= self.total_leaf_passes:
            self.finish()
        return True

    def finish(self) -> None:
        """Execute recipe actions after the foliage stage exactly once."""
        if self._finished:
            return
        if self.foliage_enabled and self.foliage_suffix_actions:
            self.engine.execute(self.foliage_suffix_actions)
        if self.trailing_actions:
            self.engine.execute(self.trailing_actions)
        self._finished = True

    def build_tree_model(self) -> TreeModel:
        """Snapshot the current recipe context as a renderer-independent tree."""
        if not self._prepared:
            raise RuntimeError("Call prepare_structure before building a tree snapshot")
        groups = self.context.values["groups"]
        branches: list[Branch] = []
        for group_name in self.BRANCH_GROUPS:
            branches.extend(
                item for item in groups.get(group_name, [])
                if isinstance(item, Branch)
            )
        leaves = [
            item for item in groups.get("leaves", [])
            if isinstance(item, Leaf)
        ]
        if not branches or branches[0].role != "trunk":
            raise ValueError("The progressive recipe did not generate a trunk")

        expected_leaf_count = 0
        if self.current_leaf_pass > 0 and self.total_leaf_passes > 0:
            expected_leaf_count = int(
                round(len(leaves) * self.total_leaf_passes / self.current_leaf_pass)
            )
        seed = int(self.tree_settings.get("seed", 1))
        metadata = {
            "species": self.tree_settings.get("species", "unspecified"),
            "seed": seed,
            "branch_groups": {
                name: len(groups.get(name, []))
                for name in groups if name != "leaves"
            },
            "leaf_count": len(leaves),
            "source_leaf_count": expected_leaf_count,
            "progressive_leaf_pass": self.current_leaf_pass,
            "progressive_leaf_passes": self.total_leaf_passes,
            "wood_surface": self.context.values.get("wood_surface"),
        }
        return TreeModel(
            name=str(self.tree_settings.get("name", "Procedural Tree")),
            branches=branches,
            leaves=leaves,
            metadata=metadata,
        )


def _tree_for_level(
    source: TreeModel,
    level: int,
    *,
    leaf_population_pre_sampled: bool = False,
) -> TreeModel:
    """Select a structural and foliage subset for one runtime representation."""
    metadata = dict(source.metadata)
    source_leaf_count = int(metadata.get("source_leaf_count", len(source.leaves)))
    if level == 2:
        # A staged source already contains the first quarter of the recipe's
        # foliage passes. A full model is sampled from the same leading passes.
        leaves = (
            list(source.leaves) if leaf_population_pre_sampled
            else list(source.leaves[:(len(source.leaves) + 3) // 4])
        )
        tree = TreeModel(source.name, list(source.branches), leaves, metadata)
        metadata.update({
            "progressive_level": 2,
            "progressive_status": "complete branch hierarchy and one-quarter enlarged actual foliage",
            "source_leaf_count": source_leaf_count,
            "proxy_leaf_count": 0,
        })
    elif level == 1:
        # Two of four recipe passes preserve a balanced intermediate density.
        leaves = (
            list(source.leaves) if leaf_population_pre_sampled
            else list(source.leaves[:(len(source.leaves) + 1) // 2])
        )
        tree = TreeModel(source.name, list(source.branches), leaves, metadata)
        metadata.update({
            "progressive_level": 1,
            "progressive_status": "complete branch hierarchy and one-half actual foliage",
            "source_leaf_count": source_leaf_count,
            "proxy_leaf_count": 0,
        })
    elif level == 0:
        tree = TreeModel(source.name, list(source.branches), list(source.leaves), metadata)
        metadata.update({
            "progressive_level": 0,
            "progressive_status": "complete detailed foliage",
            "source_leaf_count": source_leaf_count,
            "proxy_leaf_count": 0,
        })
    else:
        raise ValueError(f"Unsupported progressive LOD level: {level}")
    metadata["leaf_count"] = len(tree.leaves)
    metadata["representation_leaf_count"] = len(tree.leaves)
    metadata["representation_branch_count"] = len(tree.branches)
    tree.metadata = metadata
    return tree

def build_lod_representation(
    tree: TreeModel,
    level: int,
    *,
    leaf_population_pre_sampled: bool = False,
    shared_wood_mesh: MeshData | None = None,
) -> ProgressiveLOD:
    """Build one LOD's CPU geometry without creating or touching GPU resources.

    Pass a previously built wood mesh when successive LODs share the same
    branch hierarchy and fused wood surface. In that case only foliage differs.
    """
    build_started = perf_counter()
    lod_tree = (
        _tree_for_level(
            tree, level, leaf_population_pre_sampled=leaf_population_pre_sampled
        ) if level < 3 else tree
    )
    wood_mesh = shared_wood_mesh if shared_wood_mesh is not None else build_wood_mesh(lod_tree)
    leaf_profiles = {
        3: FAR_LEAF_PROFILE,
        2: MID_LEAF_PROFILE,
        1: NEAR_LEAF_PROFILE,
    }
    leaf_size_scales = {2: 1.20}
    foliage_mesh = build_foliage_mesh(
        lod_tree,
        leaf_profile=leaf_profiles.get(level),
        leaf_size_scale=leaf_size_scales.get(level, 1.0),
    )
    mesh_build_seconds = perf_counter() - build_started
    names = {
        0: "Detailed foliage",
        1: "Full shoots · one-half lobed foliage",
        2: "Fine shoots · one-quarter enlarged foliage",
        3: "Tertiary scaffold · clustered foliage",
    }
    return ProgressiveLOD(
        level, names[level], lod_tree, wood_mesh, foliage_mesh,
        mesh_build_seconds=mesh_build_seconds,
    )


class ProgressiveTreeWorker:
    """Build the structural tree once, then publish foliage passes incrementally."""

    def __init__(self, recipe_path: str | Path, seed_override: int | None = None) -> None:
        self.recipe_path = Path(recipe_path)
        self.seed_override = seed_override
        self._queue: Queue[ProgressiveLOD | BaseException | None] = Queue(maxsize=1)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="OakForgeTreeGeneration", daemon=True)
        self._thread.start()

    def _publish(self, item: ProgressiveLOD | BaseException | None) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _publish_level(
        self,
        session: ProgressiveGenerationSession,
        level: int,
        generation_started: float,
        shared_wood_mesh: MeshData | None = None,
    ) -> MeshData | None:
        tree = session.build_tree_model()
        representation = build_lod_representation(
            tree,
            level,
            leaf_population_pre_sampled=True,
            shared_wood_mesh=shared_wood_mesh,
        )
        representation = replace(
            representation,
            generation_elapsed_seconds=perf_counter() - generation_started,
            generation_complete=(level == 0),
        )
        if not self._publish(representation):
            return None
        return representation.wood_mesh

    def _run(self) -> None:
        try:
            generation_started = perf_counter()
            session = ProgressiveGenerationSession(
                self.recipe_path, seed_override=self.seed_override
            )
            session.prepare_structure()
            total_passes = session.total_leaf_passes

            if total_passes <= 0:
                session.finish()
                shared_wood_mesh = None
                for level in (2, 1, 0):
                    if self._stop.is_set():
                        return
                    shared_wood_mesh = self._publish_level(
                        session, level, generation_started, shared_wood_mesh
                    )
                    if shared_wood_mesh is None:
                        return
                return

            # The first recipe pass supplies approximately one quarter of the
            # actual leaf population; the second supplies half. Remaining
            # passes are only needed to complete the hero-detail representation.
            lod2_pass = max(1, int(math.ceil(total_passes * 0.25)))
            lod1_pass = min(
                total_passes,
                max(lod2_pass + 1, int(math.ceil(total_passes * 0.5))),
            )
            shared_wood_mesh: MeshData | None = None
            for pass_number in range(1, total_passes + 1):
                if self._stop.is_set():
                    return
                if not session.advance_leaf_pass():
                    break
                if pass_number == lod2_pass:
                    shared_wood_mesh = self._publish_level(
                        session, 2, generation_started
                    )
                    if shared_wood_mesh is None:
                        return
                if pass_number == lod1_pass:
                    shared_wood_mesh = self._publish_level(
                        session, 1, generation_started, shared_wood_mesh
                    )
                    if shared_wood_mesh is None:
                        return
                if pass_number == total_passes:
                    session.finish()
                    shared_wood_mesh = self._publish_level(
                        session, 0, generation_started, shared_wood_mesh
                    )
                    if shared_wood_mesh is None:
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
