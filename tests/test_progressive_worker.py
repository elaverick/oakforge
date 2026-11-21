"""Thread lifecycle and result ordering tests for progressive generation."""

from pathlib import Path
import tempfile
import time
import unittest

from oakforge.progressive import ProgressiveTreeWorker


MINIMAL_RECIPE = """tree:
  name: Worker Test Oak
  species: Quercus robur
  seed: 17
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
"""


class ProgressiveWorkerTests(unittest.TestCase):
    def test_worker_publishes_successively_finer_lods_without_open_gl(self):
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
        self.assertTrue(all(lod.wood_mesh.triangle_count > 0 for lod in lods))
        self.assertTrue(all(lod.foliage_mesh.triangle_count == 0 for lod in lods))
        self.assertTrue(all(lod.mesh_build_seconds >= 0.0 for lod in lods))
        self.assertTrue(all(lod.full_generation_seconds >= 0.0 for lod in lods))


if __name__ == "__main__":
    unittest.main()
