"""Headless OpenGL verification for the actual progressive oak LOD meshes."""

import gc
import os
import time
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from oakforge.progressive import (
    ProgressiveTreeWorker,
    build_lod_representation,
    generate_coarse_tree,
)
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
        worker = ProgressiveTreeWorker(RECIPE)
        viewer = OpenGLViewer(
            coarse_tree,
            lambda: coarse_tree,
            Path("build") / "progressive-smoke",
            progressive_worker=worker,
            initial_lod=initial,
        )
        output_directory = Path("build") / "progressive-lod"
        output_directory.mkdir(parents=True, exist_ok=True)

        def draw_and_capture(
            level: int,
            distance: float = 30.0,
            suffix: str = "",
        ) -> None:
            # The default camera distance compares representation quality
            # directly; runtime captures use each LOD's intended distance band.
            viewer.camera_distance = distance
            viewer._activate_progressive_lod(force=True, requested_level=level)
            self.assertEqual(viewer.active_lod, level)
            self.assertGreater(viewer.wood_vertex_count, 0)
            self.assertGreater(viewer.canopy_vertex_count, 0)
            active_representation = viewer.progressive_lods[level]
            expected_texture_kind = {3: "none", 2: "cluster", 1: "cluster", 0: "leaf"}[level]
            self.assertEqual(active_representation.foliage_mesh.texture_kind, expected_texture_kind)
            self.assertEqual(viewer.canopy_texture_kind, expected_texture_kind)
            self.assertEqual(
                len(active_representation.foliage_mesh.texcoords),
                viewer.canopy_vertex_count * 2,
            )
            if expected_texture_kind == "leaf":
                self.assertNotEqual(viewer.leaf_texture, 0)
            elif expected_texture_kind == "cluster":
                self.assertNotEqual(viewer.cluster_texture, 0)

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

            if viewer.lod_transition_start is not None:
                # Exercise the outgoing/incoming complementary dither passes
                # halfway through the transition, then capture the settled LOD.
                viewer.lod_transition_start = time.perf_counter() - viewer.lod_transition_duration * 0.5
                viewer._draw_progressive_geometry()
                gl.glFinish()
                self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
                viewer.lod_transition_start = time.perf_counter() - viewer.lod_transition_duration * 2.0
            viewer._draw_progressive_geometry()
            gl.glFinish()
            self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            preview = Image.frombytes("RGB", (width, height), pixels)
            preview = preview.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            preview.save(output_directory / f"english_oak_lod{level}{suffix}.png")
            self.assertEqual(viewer.previous_wood_vao, 0)
            self.assertEqual(viewer.previous_canopy_vao, 0)
            self.assertNotIn("transition_wood_vbo", viewer.gpu_allocations)
            self.assertNotIn("transition_canopy_vbo", viewer.gpu_allocations)

        try:
            # Let the real recipe worker publish each foliage-pass milestone.
            deadline = time.monotonic() + 240.0
            while not viewer.progressive_finished and time.monotonic() < deadline:
                viewer._poll_progressive_updates()
                time.sleep(0.01)
            self.assertTrue(viewer.progressive_finished, viewer.status_message)
            self.assertFalse(viewer.progressive_error, viewer.status_message)
            self.assertEqual(sorted(viewer.progressive_lods), [0, 1, 2, 3])

            lod2 = viewer.progressive_lods[2]
            lod1 = viewer.progressive_lods[1]
            lod0 = viewer.progressive_lods[0]
            full_leaf_count = len(lod0.tree.leaves)
            self.assertEqual(len(lod2.tree.leaves), (full_leaf_count + 3) // 4)
            self.assertEqual(len(lod1.tree.leaves), (full_leaf_count + 1) // 2)
            self.assertEqual(lod2.tree.metadata["source_leaf_count"], full_leaf_count)
            self.assertEqual(lod1.tree.metadata["source_leaf_count"], full_leaf_count)
            self.assertEqual(lod0.tree.metadata["source_leaf_count"], full_leaf_count)
            self.assertEqual(lod2.generation_complete, False)
            self.assertEqual(lod1.generation_complete, False)
            self.assertEqual(lod0.generation_complete, True)

            # Same-distance comparison isolates representation quality.
            draw_and_capture(3)
            draw_and_capture(2)
            draw_and_capture(1)
            draw_and_capture(0)

            # Runtime-distance captures verify the expected visual scale and
            # crown silhouette in the camera bands used by automatic selection.
            draw_and_capture(3, distance=60.0, suffix="_runtime")
            draw_and_capture(2, distance=36.0, suffix="_runtime")
            draw_and_capture(1, distance=19.0, suffix="_runtime")
            draw_and_capture(0, distance=8.0, suffix="_runtime")
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
