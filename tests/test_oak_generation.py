"""Regression tests for the English oak growth recipe and mesh output."""

import unittest
from copy import copy

import numpy as np
from pathlib import Path

from oakforge.core.context import GenerationContext
from oakforge.core.mesh import MeshData, _make_branch_tube, build_tree_mesh
from oakforge.generation import generate_from_file, generate_tree, load_recipe
from oakforge.core.models import Branch
from oakforge.tools.roots import branch_surface_roots, spread_surface_roots


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OakGenerationTests(unittest.TestCase):
    """Check that the default oak produces every expected structural tier."""

    @classmethod
    def setUpClass(cls):
        cls.tree = generate_from_file(PROJECT_ROOT / "configs" / "english_oak.yaml")
        cls._trunk_study = None
        cls._trunk_study_with_surface = None

    @classmethod
    def get_trunk_study(cls, include_surface=False):
        # Root-placement tests inspect structure only. A separate coarse-grid
        # fixture exercises the fused mesher without retaining a second full asset.
        if include_surface:
            if cls._trunk_study_with_surface is None:
                recipe = load_recipe(PROJECT_ROOT / "configs" / "oak_trunk_only.yaml")
                for step in recipe["pipeline"]:
                    if isinstance(step, dict) and isinstance(step.get("call"), dict):
                        if step["call"].get("tool") == "wood_surface.build":
                            step["call"]["args"]["voxel_size"] = 0.085
                cls._trunk_study_with_surface = generate_tree(recipe)
            return cls._trunk_study_with_surface
        if cls._trunk_study is None:
            recipe = load_recipe(PROJECT_ROOT / "configs" / "oak_trunk_only.yaml")
            recipe["pipeline"] = [
                step for step in recipe["pipeline"]
                if not (isinstance(step, dict) and isinstance(step.get("call"), dict)
                        and step["call"].get("tool") == "wood_surface.build")
            ]
            cls._trunk_study = generate_tree(recipe)
        return cls._trunk_study

    def test_tree_has_a_single_trunk_and_multiple_branch_tiers(self):
        roles = [branch.role for branch in self.tree.branches]
        self.assertEqual(roles.count("trunk"), 1)
        self.assertGreaterEqual(roles.count("primary"), 6)
        self.assertGreaterEqual(roles.count("leader"), 4)
        self.assertGreaterEqual(roles.count("secondary"), 40)
        self.assertGreaterEqual(roles.count("tertiary"), 100)
        self.assertGreaterEqual(roles.count("twig"), 500)


    def test_oak_has_a_broad_crown_with_three_fine_branching_tiers(self):
        roles = {role: [branch for branch in self.tree.branches if branch.role == role]
                 for role in ("primary", "leader", "secondary", "tertiary", "twig")}
        self.assertEqual(len(roles["primary"]), 8)
        self.assertEqual(len(roles["leader"]), 6)
        self.assertEqual(len(roles["secondary"]), 104)
        self.assertEqual(len(roles["tertiary"]), 624)
        self.assertEqual(len(roles["twig"]), 4992)
        twig_points = np.concatenate([np.asarray(branch.points) for branch in roles["twig"]])
        self.assertGreater(float(twig_points[:, 0].max() - twig_points[:, 0].min()), 14.0)
        self.assertGreater(float(twig_points[:, 2].max() - twig_points[:, 2].min()), 13.0)
        self.assertLess(float(twig_points[:, 1].max()), 15.2)
        self.assertGreater(float(twig_points[:, 1].max()), 13.4)

    def test_every_child_branch_references_a_generated_parent(self):
        branch_ids = {branch.branch_id for branch in self.tree.branches}
        for branch in self.tree.branches:
            if branch.parent_id is not None:
                self.assertIn(branch.parent_id, branch_ids)

    def test_oak_foliage_and_mesh_are_generated(self):
        self.assertGreater(len(self.tree.leaves), 10000)
        self.assertEqual(len(self.tree.leaves), 311136)
        self.assertTrue(all(0.10 <= leaf.length <= 0.16 for leaf in self.tree.leaves[:100]))
        # A representative fine-branch and leaf subset exercises rendering without
        # turning every regression run into a full multi-million-triangle asset bake.
        mesh_input = copy(self.tree)
        mesh_input.leaves = self.tree.leaves[:2000]
        coarse_roles = {"trunk", "root", "primary", "leader", "secondary"}
        fine_branches = [branch for branch in self.tree.branches if branch.role in {"tertiary", "twig", "spray"}]
        mesh_input.branches = [
            branch for branch in self.tree.branches if branch.role in coarse_roles
        ] + fine_branches[:100]
        mesh = build_tree_mesh(mesh_input)
        self.assertGreater(mesh.triangle_count, 1000)
        self.assertEqual(len(mesh.positions), len(mesh.normals))
        self.assertEqual(len(mesh.positions), len(mesh.colours))


    def test_terminal_branches_have_visible_bases_and_finite_tip_radii(self):
        twigs = [branch for branch in self.tree.branches if branch.role == "twig"]
        self.assertTrue(twigs)
        branch_by_id = {branch.branch_id: branch for branch in self.tree.branches}
        for twig in twigs:
            self.assertGreaterEqual(twig.radii[0], 0.0139)
            self.assertGreaterEqual(twig.radii[-1], 0.0059)
            self.assertGreater(twig.radii[0], twig.radii[1])
            self.assertGreater(twig.metadata.get("junction_swell", 0.0), 0.0)
            parent = branch_by_id[twig.parent_id]
            attachment = parent.point_at(float(twig.metadata["parent_position"]))
            embedded_distance = float(np.linalg.norm(attachment - twig.points[0]))
            expected_embed = parent.radius_at(float(twig.metadata["parent_position"])) * 0.90
            self.assertAlmostEqual(embedded_distance, expected_embed, delta=0.025)

    def test_terminal_wood_finishes_inside_the_leaf_envelope(self):
        twigs = [branch for branch in self.tree.branches if branch.role == "twig"]
        terminal_points = np.asarray([branch.points[-1] for branch in twigs])
        centre_x, centre_z = 0.82, -0.68
        radius_x, radius_z = 9.2 - 0.86, 8.6 - 0.86
        normalized_radius = np.sqrt(
            ((terminal_points[:, 0] - centre_x) / radius_x) ** 2
            + ((terminal_points[:, 2] - centre_z) / radius_z) ** 2
        )
        self.assertLessEqual(float(normalized_radius.max()), 1.002)
        terminal_cap = 14.5 - 0.86 * 0.55 - 1.85 * np.minimum(normalized_radius, 1.0) ** 2.0
        self.assertLessEqual(float((terminal_points[:, 1] - terminal_cap).max()), 0.04)

    def test_branch_tubes_are_closed_at_attachment_and_terminal_ends(self):
        branch = Branch(
            branch_id="test-twig",
            parent_id="test-parent",
            role="twig",
            depth=4,
            points=[np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            radii=[0.02, 0.008],
        )
        mesh = MeshData()
        _make_branch_tube(branch, mesh, radial_sides=16)
        # Twelve radial sides produce 24 tube-wall triangles and 24 cap triangles.
        self.assertEqual(mesh.triangle_count, 48)
        self.assertEqual(len(mesh.positions), len(mesh.normals))
        self.assertEqual(len(mesh.positions), len(mesh.colours))

    def test_oak_trunk_has_a_flared_base_and_tapered_crown(self):
        trunk = next(branch for branch in self.tree.branches if branch.role == "trunk")
        self.assertGreater(max(trunk.radii[:8]), trunk.radii[12])
        self.assertGreater(trunk.radii[12], trunk.radii[-1])
        self.assertGreater(trunk.metadata["buttress_strength"], 0.0)
        self.assertGreater(len(trunk.points), 30)

    def test_crown_envelope_shapes_the_mature_canopy(self):
        twig_branches = [branch for branch in self.tree.branches if branch.role == "twig"]
        twig_points = np.concatenate([np.asarray(branch.points) for branch in twig_branches])
        centre_x = 0.82
        centre_z = -0.68
        normalized_radius = np.sqrt(
            ((twig_points[:, 0] - centre_x) / 9.2) ** 2
            + ((twig_points[:, 2] - centre_z) / 8.6) ** 2
        )
        self.assertLessEqual(float(normalized_radius.max()), 1.001)
        local_cap = 14.5 - 1.85 * np.minimum(normalized_radius, 1.0) ** 2.0
        self.assertLessEqual(float((twig_points[:, 1] - local_cap).max()), 0.03)
        self.assertGreaterEqual(float(twig_points[:, 1].min()), 3.35)
        self.assertTrue(all(branch.metadata.get("crown_clipped") for branch in twig_branches))

    def test_crown_leaders_converge_into_the_upper_canopy(self):
        trunk = next(branch for branch in self.tree.branches if branch.role == "trunk")
        leaders = [branch for branch in self.tree.branches if branch.role == "leader"]
        leader_secondary = [branch for branch in self.tree.branches if branch.role == "secondary"
                            and branch.parent_id in {leader.branch_id for leader in leaders}]
        self.assertEqual(len(leaders), 6)
        self.assertLess(trunk.points[-1][1], 9.0)
        self.assertGreaterEqual(len(leader_secondary), 40)
        self.assertTrue(all(branch.parent_id == trunk.branch_id for branch in leaders))
        # All upper leaders have branching children, so no single central spear is left unbranched.
        self.assertTrue(all(any(child.parent_id == leader.branch_id for child in leader_secondary)
                            for leader in leaders))
        leaf_positions = np.asarray([leaf.base_position for leaf in self.tree.leaves])
        top_leaf_mask = leaf_positions[:, 1] >= 12.0
        top_leaf_count = int(np.count_nonzero(top_leaf_mask))
        self.assertGreater(top_leaf_count, 20000)
        crown_radius = np.sqrt(
            ((leaf_positions[:, 0] - 0.82) / 9.2) ** 2
            + ((leaf_positions[:, 2] + 0.68) / 8.6) ** 2
        )
        upper_centre_fill = int(np.count_nonzero(top_leaf_mask & (crown_radius < 0.45)))
        self.assertGreater(upper_centre_fill, 15000)

    def test_inner_canopy_sprays_fill_all_scaffold_levels(self):
        sprays = [branch for branch in self.tree.branches if branch.role == "spray"]
        branch_by_id = {branch.branch_id: branch for branch in self.tree.branches}
        self.assertEqual(len(sprays), 1490)
        self.assertEqual(self.tree.metadata["branch_groups"]["sprays"], 1490)
        self.assertEqual(self.tree.metadata["branch_groups"]["twigs"], 4992)
        for spray in sprays:
            self.assertIn(spray.parent_id, branch_by_id)
            parent_role = branch_by_id[spray.parent_id].role
            self.assertIn(parent_role, {"primary", "leader", "secondary", "tertiary"})
            expected_depth = {"primary": 2, "leader": 2, "secondary": 3, "tertiary": 4}[parent_role]
            self.assertEqual(spray.depth, expected_depth)
            self.assertLess(spray.length, branch_by_id[spray.parent_id].length * 0.70)

    def test_foliage_uses_grouped_whorls_with_radial_volume(self):
        leaves = self.tree.leaves[:240]
        base_positions = np.asarray([leaf.base_position for leaf in leaves])
        self.assertGreater(float(np.std(base_positions[:, 0])), 0.015)
        self.assertGreater(float(np.std(base_positions[:, 1])), 0.015)
        self.assertGreater(float(np.std(base_positions[:, 2])), 0.015)
        colour_variations = np.asarray([leaf.colour_variation for leaf in leaves])
        self.assertGreater(float(np.std(colour_variations[:, 1])), 0.015)

    def test_primary_limbs_have_irregular_attachment_spacing_and_sweeps(self):
        primary_branches = [branch for branch in self.tree.branches if branch.role == "primary"]
        attachment_positions = sorted(
            float(branch.metadata["parent_position"]) for branch in primary_branches
        )
        spacing = np.diff(attachment_positions)
        self.assertGreater(float(np.std(spacing)), 0.005)
        branch_headings = [
            (branch.points[-1] - branch.points[0]) / branch.length
            for branch in primary_branches
        ]
        self.assertGreater(float(np.std(np.asarray(branch_headings)[:, 1])), 0.04)


    def test_coarse_wood_fuses_secondary_limbs_into_the_trunk_network(self):
        fused_surface = self.tree.metadata.get("wood_surface")
        self.assertIsInstance(fused_surface, dict)
        fused_ids = set(fused_surface["branch_ids"])
        fused_roles = {branch.role for branch in self.tree.branches
                       if branch.branch_id in fused_ids}
        self.assertEqual(fused_roles, {"trunk", "root", "primary", "leader", "secondary"})
        secondary_ids = {branch.branch_id for branch in self.tree.branches
                         if branch.role == "secondary"}
        self.assertTrue(secondary_ids.issubset(fused_ids))

    def test_full_oak_contains_a_fused_three_tier_surface_root_network(self):
        root_tiers = {
            tier: [branch for branch in self.tree.branches
                   if branch.role == "root" and branch.metadata.get("root_tier") == tier]
            for tier in ("primary", "lateral", "feeder")
        }
        self.assertEqual(len(root_tiers["primary"]), 7)
        self.assertGreaterEqual(len(root_tiers["lateral"]), 8)
        self.assertGreaterEqual(len(root_tiers["feeder"]), 3)
        root_ids = {branch.branch_id for branches in root_tiers.values() for branch in branches}
        fused_ids = set(self.tree.metadata["wood_surface"]["branch_ids"])
        self.assertTrue(root_ids.issubset(fused_ids))
        for branch in root_tiers["lateral"]:
            self.assertIn(branch.parent_id, {root.branch_id for root in root_tiers["primary"]})
        for branch in root_tiers["feeder"]:
            self.assertIn(branch.parent_id, {root.branch_id for root in root_tiers["lateral"]})

    def test_seed_is_reproducible(self):
        # A reduced recipe keeps this determinism check focused on procedural
        # placement rather than rebuilding the full high-density foliage mesh.
        small_recipe = load_recipe(PROJECT_ROOT / "configs" / "english_oak.yaml")
        parameters = small_recipe["tree"]["parameters"]
        parameters["primary"]["count"] = 3
        parameters["leaders"]["count"] = 2
        parameters["leader_secondary"]["count"] = 2
        parameters["secondary"]["count"] = 2
        parameters["tertiary"]["count"] = 2
        parameters["twigs"]["count"] = 2
        parameters["foliage"]["enabled"] = False
        small_recipe["pipeline"] = [
            step for step in small_recipe["pipeline"]
            if not (isinstance(step, dict) and isinstance(step.get("call"), dict)
                    and step["call"].get("tool") == "wood_surface.build")
        ]
        first_tree = generate_tree(small_recipe)
        second_tree = generate_tree(small_recipe)
        self.assertEqual(len(first_tree.branches), len(second_tree.branches))
        self.assertEqual(len(first_tree.leaves), len(second_tree.leaves))
        for first_branch, second_branch in zip(first_tree.branches, second_tree.branches):
            np.testing.assert_allclose(first_branch.points, second_branch.points)

    def test_trunk_has_root_spread_and_limb_junction_swellings(self):
        trunk = next(branch for branch in self.tree.branches if branch.role == "trunk")
        self.assertGreater(trunk.metadata["root_spread"], 0.0)
        self.assertGreaterEqual(trunk.metadata["root_count"], 3)
        self.assertEqual(len(trunk.metadata["limb_knuckles"]), 14)
        collar_roles = [collar["branch_role"] for collar in trunk.metadata["limb_knuckles"]]
        self.assertEqual(collar_roles.count("primary"), 8)
        self.assertEqual(collar_roles.count("leader"), 6)
        # The centreline should wander in both horizontal axes rather than remain a pole.
        horizontal_offsets = [point[[0, 2]] for point in trunk.points]
        self.assertGreater(float(max(np.linalg.norm(offset) for offset in horizontal_offsets)), 0.1)

    def test_trunk_recipe_grows_separate_surface_roots(self):
        trunk_study = self.get_trunk_study()
        trunk = next(branch for branch in trunk_study.branches if branch.role == "trunk")
        roots = [branch for branch in trunk_study.branches
                 if branch.role == "root" and branch.metadata.get("root_tier") == "primary"]
        lateral_roots = [branch for branch in trunk_study.branches
                         if branch.role == "root" and branch.metadata.get("root_tier") == "lateral"]
        feeder_roots = [branch for branch in trunk_study.branches
                        if branch.role == "root" and branch.metadata.get("root_tier") == "feeder"]
        self.assertEqual(len(roots), 7)
        self.assertGreaterEqual(len(lateral_roots), 8)
        self.assertGreaterEqual(len(feeder_roots), 3)
        self.assertTrue(all(root.parent_id == trunk.branch_id for root in roots))
        self.assertTrue(all(root.metadata["placement_mode"] == "procedural" for root in roots))
        self.assertTrue(all(root.metadata["curve_strength"] > 0.0 for root in roots + lateral_roots + feeder_roots))
        self.assertTrue(all(root.points[-1][1] < 0.0 for root in roots))
        self.assertTrue(all(root.length > 1.5 for root in roots))
        primary_ids = {root.branch_id for root in roots}
        self.assertTrue(all(root.parent_id in primary_ids for root in lateral_roots))
        self.assertTrue(all(root.points[-1][1] < 0.0 for root in lateral_roots))
        lateral_ids = {root.branch_id for root in lateral_roots}
        self.assertTrue(all(root.parent_id in lateral_ids for root in feeder_roots))
        self.assertTrue(all(root.depth == 3 for root in feeder_roots))

    def test_trunk_study_declares_a_single_fused_wood_surface(self):
        recipe = load_recipe(PROJECT_ROOT / "configs" / "oak_trunk_only.yaml")
        surface_calls = [
            step["call"] for step in recipe["pipeline"]
            if isinstance(step, dict) and isinstance(step.get("call"), dict)
            and step["call"].get("tool") == "wood_surface.build"
        ]
        self.assertEqual(len(surface_calls), 1)
        branch_inputs = surface_calls[0]["args"]["branches"]
        self.assertIn("$groups.trunk", branch_inputs)
        self.assertIn("$groups.roots", branch_inputs)
        self.assertNotIn("$groups.primary", branch_inputs)

    def test_manual_root_recipe_uses_explicit_angles_and_attachment_heights(self):
        manual_recipe = load_recipe(PROJECT_ROOT / "configs" / "oak_roots_manual.yaml")
        # This test is about authored root placement. The manual recipe still
        # declares the fused-surface call, but expensive voxel meshing is already
        # covered by the separate closed-surface tests.
        wood_surface_calls = [
            step for step in manual_recipe["pipeline"]
            if isinstance(step, dict) and isinstance(step.get("call"), dict)
            and step["call"].get("tool") == "wood_surface.build"
        ]
        self.assertEqual(len(wood_surface_calls), 1)
        manual_recipe["pipeline"] = [
            step for step in manual_recipe["pipeline"] if step not in wood_surface_calls
        ]
        manual_tree = generate_tree(manual_recipe)
        roots = [branch for branch in manual_tree.branches
                 if branch.role == "root" and branch.metadata.get("root_tier") == "primary"]
        manual_offshoots = [branch for branch in manual_tree.branches
                            if branch.role == "root" and branch.metadata.get("root_tier") == "lateral"]
        self.assertEqual(len(roots), 6)
        self.assertEqual(len(manual_offshoots), 3)
        self.assertTrue(all(branch.metadata["placement_mode"] == "manual" for branch in manual_offshoots))
        self.assertTrue(all(branch.metadata["curve_strength"] == 0.0 for branch in manual_offshoots))
        self.assertTrue(all(root.metadata["placement_mode"] == "manual" for root in roots))
        self.assertEqual(
            [root.metadata["placement_id"] for root in roots],
            ["north_west", "north", "north_east", "south_east", "south", "south_west"],
        )
        self.assertEqual(
            [round(root.metadata["azimuth_degrees"]) for root in roots],
            [145, 82, 24, 312, 238, 176],
        )
        for root in roots:
            self.assertAlmostEqual(float(root.points[0][1]), root.metadata["attachment_height"], places=5)
            self.assertLess(float(root.points[-1][1]), 0.1)
        configured_fusion_inputs = wood_surface_calls[0]["call"]["args"]["branches"]
        self.assertIn("$groups.roots", configured_fusion_inputs)


    def test_radial_and_random_root_distributions_are_seeded(self):
        trunk = next(branch for branch in self.get_trunk_study().branches if branch.role == "trunk")
        radial_context = GenerationContext({}, np.random.default_rng(18))
        radial_roots = spread_surface_roots(radial_context, {
            "trunk": trunk,
            "count": 4,
            "distribution": "radial",
            "phase_degrees": 0.0,
            "collar_height": 1.0,
            "length": 1.5,
            "base_radius": 0.3,
        })
        self.assertEqual(
            [round(root.metadata["azimuth_degrees"]) for root in radial_roots],
            [0, 90, 180, 270],
        )
        self.assertTrue(all(root.points[-1][1] < 0.0 for root in radial_roots))

        with self.assertRaisesRegex(ValueError, "must be explicit"):
            spread_surface_roots(GenerationContext({}, np.random.default_rng(7)), {
                "trunk": trunk,
                "count": 4,
            })

        first_context = GenerationContext({}, np.random.default_rng(42))
        second_context = GenerationContext({}, np.random.default_rng(42))
        first_random = spread_surface_roots(first_context, {
            "trunk": trunk, "count": 5, "distribution": "random", "collar_height": 1.0,
        })
        second_random = spread_surface_roots(second_context, {
            "trunk": trunk, "count": 5, "distribution": "random", "collar_height": 1.0,
        })
        self.assertEqual(
            [root.metadata["azimuth_degrees"] for root in first_random],
            [root.metadata["azimuth_degrees"] for root in second_random],
        )


    def test_surface_roots_use_rounded_sections_and_lateral_offshoots_are_reproducible(self):
        roots = [branch for branch in self.get_trunk_study().branches
                 if branch.role == "root" and branch.metadata.get("root_tier") == "primary"]
        lateral_roots = [branch for branch in self.get_trunk_study().branches
                         if branch.role == "root" and branch.metadata.get("root_tier") == "lateral"]
        root_recipe = load_recipe(PROJECT_ROOT / "configs" / "oak_trunk_only.yaml")
        wood_surface_call = next(
            step["call"]["args"] for step in root_recipe["pipeline"]
            if isinstance(step, dict) and isinstance(step.get("call"), dict)
            and step["call"].get("tool") == "wood_surface.build"
        )
        wood_surface = wood_surface_call
        vertical_scale = wood_surface["root_vertical_scale"]
        thickness_scale = wood_surface["root_thickness_scale"]
        self.assertGreater(vertical_scale / thickness_scale, 0.85)
        self.assertLess(vertical_scale / thickness_scale, 1.25)
        self.assertGreater(len(lateral_roots), 0)
        for lateral_root in lateral_roots:
            parent = next(root for root in roots if root.branch_id == lateral_root.parent_id)
            expected_attachment = parent.point_at(lateral_root.metadata["parent_position"])
            np.testing.assert_allclose(lateral_root.points[0], expected_attachment, atol=1e-6)

    def test_manual_lateral_root_uses_named_parent_and_explicit_position(self):
        primary_roots = [branch for branch in self.get_trunk_study().branches
                         if branch.role == "root" and branch.metadata.get("root_tier") == "primary"]
        parent = primary_roots[0]
        context = GenerationContext({}, np.random.default_rng(12))
        lateral_roots = branch_surface_roots(context, {
            "parents": primary_roots,
            "placements": [{
                "id": "manual_east_offshoot",
                "parent_placement_id": parent.metadata["placement_id"],
                "parent_position": 0.61,
                "angle_degrees": 63.0,
                "side": -1,
                "length": 1.2,
                "base_radius": 0.14,
                "tip_radius": 0.035,
            }],
        })
        self.assertEqual(len(lateral_roots), 1)
        lateral = lateral_roots[0]
        self.assertEqual(lateral.parent_id, parent.branch_id)
        self.assertEqual(lateral.metadata["placement_mode"], "manual")
        self.assertEqual(lateral.metadata["placement_id"], "manual_east_offshoot")
        np.testing.assert_allclose(lateral.points[0], parent.point_at(0.61), atol=1e-6)
        self.assertAlmostEqual(lateral.metadata["branch_angle_degrees"], 63.0)

    def test_fused_wood_surface_is_closed_and_outward_facing(self):
        # The topology contract is tested on the isolated bole and root network
        # at a coarser grid. The complete oak separately asserts that every
        # scaffold tier is included in the production fused surface.
        trunk_study = self.get_trunk_study(include_surface=True)
        fused_surface = trunk_study.metadata["wood_surface"]
        mesh = fused_surface["mesh"]
        positions = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, 3)
        _, inverse_indices = np.unique(positions, axis=0, return_inverse=True)
        indexed_faces = inverse_indices.reshape(-1, 3)
        edges = np.concatenate((indexed_faces[:, [0, 1]], indexed_faces[:, [1, 2]],
                                indexed_faces[:, [2, 0]]), axis=0)
        edges.sort(axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(edge_counts == 2))
        triangles = positions.reshape(-1, 3, 3)
        signed_volume = np.einsum(
            "ij,ij->i", triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2])
        ).sum() / 6.0
        self.assertGreater(signed_volume, 0.0)
        self.assertTrue(np.isfinite(positions).all())



if __name__ == "__main__":
    unittest.main()
