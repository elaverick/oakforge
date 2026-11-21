"""Headless OpenGL verification for the actual progressive oak LOD meshes."""

import gc
import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from oakforge.generation import generate_from_file
from oakforge.progressive import build_lod_representation, generate_coarse_tree
from oakforge.render.opengl_viewer import OpenGLViewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "configs" / "english_oak.yaml"


@unittest.skipUnless(
    os.environ.get("OAKFORGE_PROGRESSIVE_GL_SMOKE") == "1",
    "Set OAKFORGE_PROGRESSIVE_GL_SMOKE=1 under Xvfb to run the progressive GL smoke test",
)
class ProgressiveRuntimeOpenGLTests(unittest.TestCase):
    def test_viewer_uploads_and_draws_actual_progressive_lod_meshes(self):
        coarse_tree = generate_coarse_tree(RECIPE)
        initial = build_lod_representation(coarse_tree, 3)
        full_tree = generate_from_file(RECIPE)
        lod2 = build_lod_representation(full_tree, 2)
        lod1 = build_lod_representation(full_tree, 1)
        packages = [lod2, lod1, None]

        class FakeWorker:
            recipe_path = RECIPE

            def poll(self):
                nonlocal packages
                result, packages = packages, []
                return result

            def close(self, timeout=2.0):
                return None

        viewer = OpenGLViewer(
            coarse_tree,
            lambda: coarse_tree,
            Path("build") / "progressive-smoke",
            progressive_worker=FakeWorker(),
            initial_lod=initial,
        )
        output_directory = Path("build") / "progressive-lod"
        output_directory.mkdir(parents=True, exist_ok=True)

        def draw_and_capture(level: int, distance: float) -> None:
            viewer.camera_distance = distance
            viewer._activate_progressive_lod(force=True)
            self.assertEqual(viewer.active_lod, level)
            self.assertGreater(viewer.wood_vertex_count, 0)
            self.assertGreater(viewer.canopy_vertex_count, 0)

            gl = viewer.gl
            width, height = viewer.glfw.get_framebuffer_size(viewer.window)
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.78, 0.82, 0.79, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glUseProgram(viewer.program)
            for name, matrix in (
                ("model_matrix", np.identity(4, dtype=np.float32)),
                ("view_matrix", viewer._view_matrix()),
                ("projection_matrix", viewer._projection_matrix(width / max(height, 1))),
            ):
                gl.glUniformMatrix4fv(viewer.tree_uniforms[name], 1, gl.GL_TRUE, matrix)

            gl.glBindVertexArray(viewer.grid_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, viewer.grid_vertex_count)
            gl.glBindVertexArray(viewer.wood_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, viewer.wood_vertex_count)
            gl.glBindVertexArray(viewer.canopy_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, viewer.canopy_vertex_count)
            gl.glBindVertexArray(0)
            gl.glFinish()
            self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            preview = Image.frombytes("RGB", (width, height), pixels)
            preview = preview.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            preview.save(output_directory / f"english_oak_lod{level}.png")

        try:
            viewer._poll_progressive_updates()
            self.assertTrue(viewer.progressive_finished)
            self.assertIn(2, viewer.progressive_lods)
            self.assertIn(1, viewer.progressive_lods)

            draw_and_capture(3, 60.0)
            draw_and_capture(2, 40.0)
            draw_and_capture(1, 20.0)

            # Release the first three CPU mesh packages before constructing
            # the very large hero-level canopy mesh.
            viewer.progressive_lods.clear()
            del initial, lod2, lod1, packages
            gc.collect()

            lod0 = build_lod_representation(full_tree, 0)
            viewer.progressive_lods = {0: lod0}
            draw_and_capture(0, 8.0)
            self.assertEqual(
                lod0.tree.metadata["source_leaf_count"],
                len(full_tree.leaves),
            )
            gl = viewer.gl
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
