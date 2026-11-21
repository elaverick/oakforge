"""Headless OpenGL coverage for the progressive viewer's LOD uploads."""

from dataclasses import replace
import os
from pathlib import Path
import unittest

import numpy as np

from oakforge.progressive import build_lod_representation, generate_coarse_tree
from oakforge.render.opengl_viewer import OpenGLViewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("OAKFORGE_PROGRESSIVE_GL_SMOKE") == "1",
    "Set OAKFORGE_PROGRESSIVE_GL_SMOKE=1 under Xvfb to run the progressive GL smoke test",
)
class ProgressiveRuntimeOpenGLTests(unittest.TestCase):
    def test_viewer_uploads_and_draws_each_progressive_lod(self):
        tree = generate_coarse_tree(PROJECT_ROOT / "configs" / "english_oak.yaml")
        initial = build_lod_representation(tree, 3)
        packages = [
            replace(initial, level=2, name="LOD2 smoke"),
            replace(initial, level=1, name="LOD1 smoke"),
            replace(initial, level=0, name="LOD0 smoke"),
            None,
        ]

        class FakeWorker:
            recipe_path = PROJECT_ROOT / "configs" / "english_oak.yaml"

            def poll(self):
                nonlocal packages
                result, packages = packages, []
                return result

            def close(self, timeout=2.0):
                return None

        viewer = OpenGLViewer(
            tree,
            lambda: tree,
            Path("build") / "progressive-smoke",
            progressive_worker=FakeWorker(),
            initial_lod=initial,
        )
        try:
            viewer._poll_progressive_updates()
            self.assertTrue(viewer.progressive_finished)
            for level, distance in ((3, 60.0), (2, 40.0), (1, 20.0), (0, 8.0)):
                viewer.camera_distance = distance
                viewer._activate_progressive_lod(force=True)
                self.assertEqual(viewer.active_lod, level)
                self.assertGreater(viewer.wood_vertex_count, 0)
                self.assertGreater(viewer.canopy_vertex_count, 0)

                gl = viewer.gl
                gl.glUseProgram(viewer.program)
                for name, matrix in (
                    ("model_matrix", np.identity(4, dtype=np.float32)),
                    ("view_matrix", viewer._view_matrix()),
                    ("projection_matrix", viewer._projection_matrix(16.0 / 9.0)),
                ):
                    gl.glUniformMatrix4fv(viewer.tree_uniforms[name], 1, gl.GL_TRUE, matrix)
                gl.glBindVertexArray(viewer.wood_vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, viewer.wood_vertex_count)
                gl.glBindVertexArray(viewer.canopy_vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, viewer.canopy_vertex_count)
                gl.glBindVertexArray(0)
                gl.glFinish()
                self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

            viewer.glfw.set_window_should_close(viewer.window, True)
            viewer.run()
            viewer = None
        finally:
            if viewer is not None:
                viewer.glfw.set_window_should_close(viewer.window, True)
                viewer.run()


if __name__ == "__main__":
    unittest.main()
