"""Thread lifecycle and staged foliage publication tests."""

from pathlib import Path
import tempfile
import time
import unittest

from oakforge.progressive import ProgressiveTreeWorker


MINIMAL_RECIPE = """tree:
  name: Worker Test Oak
  species: Quercus robur
  seed: 17
  parameters:
    foliage:
      enabled: true
      passes: 4
      leaves_per_pass: 4
      leaf_length: 0.135
      width_ratio: 0.70
      cluster_count: 2
      cluster_spread: 0.05
      radial_spread: 0.12
      colour_variation: 0.05
pipeline:
  - call:
      tool: trunk.generate
      args:
        height: 2.0
        base_radius: 0.30
        stem_radius: 0.22
        top_radius: 0.05
        flare_height: 0.4
        segments: 12
      save_as: trunk
      append_to: groups.trunk
  - if:
      condition: "$tree.parameters.foliage.enabled"
      then:
        - set:
            leaf_pass: 0
        - do_while:
            condition:
              lt: ["$leaf_pass", "$tree.parameters.foliage.passes"]
            max_iterations: 5
            do:
              - call:
                  tool: foliage.scatter_oak
                  args:
                    twig: "$trunk"
                    count: "$tree.parameters.foliage.leaves_per_pass"
                    pass_index: "$leaf_pass"
                    leaf_length: "$tree.parameters.foliage.leaf_length"
                    width_ratio: "$tree.parameters.foliage.width_ratio"
                    cluster_count: "$tree.parameters.foliage.cluster_count"
                    cluster_spread: "$tree.parameters.foliage.cluster_spread"
                    radial_spread: "$tree.parameters.foliage.radial_spread"
                    colour_variation: "$tree.parameters.foliage.colour_variation"
                  append_to: groups.leaves
              - set:
                  leaf_pass:
                    add: ["$leaf_pass", 1]
      else:
        - set:
            foliage_status: disabled
"""


class ProgressiveWorkerTests(unittest.TestCase):
    def test_worker_publishes_recipe_leaf_passes_as_successive_lods(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            recipe_path = Path(temporary_directory) / "minimal_tree.yaml"
            recipe_path.write_text(MINIMAL_RECIPE, encoding="utf-8")
            worker = ProgressiveTreeWorker(recipe_path, seed_override=23)
            messages = []
            deadline = time.monotonic() + 15.0
            try:
                while time.monotonic() < deadline:
                    messages.extend(worker.poll())
                    if None in messages:
                        break
                    time.sleep(0.01)
            finally:
                worker.close(timeout=1.0)

        self.assertIn(None, messages)
        errors = [message for message in messages if isinstance(message, BaseException)]
        self.assertEqual(errors, [])
        lods = [message for message in messages if message is not None]
        self.assertEqual([lod.level for lod in lods], [2, 1, 0])
        self.assertEqual([lod.tree.metadata["seed"] for lod in lods], [23, 23, 23])
        self.assertEqual([len(lod.tree.leaves) for lod in lods], [4, 8, 16])
        self.assertEqual(
            [lod.tree.metadata["source_leaf_count"] for lod in lods], [16, 16, 16]
        )
        self.assertTrue(all(lod.wood_mesh.triangle_count > 0 for lod in lods))
        self.assertTrue(all(lod.foliage_mesh.triangle_count > 0 for lod in lods))
        self.assertTrue(all(lod.mesh_build_seconds >= 0.0 for lod in lods))
        self.assertTrue(all(lod.generation_elapsed_seconds >= 0.0 for lod in lods))
        self.assertEqual([lod.generation_complete for lod in lods], [False, False, True])
        elapsed = [lod.generation_elapsed_seconds for lod in lods]
        self.assertEqual(elapsed, sorted(elapsed))


if __name__ == "__main__":
    unittest.main()
