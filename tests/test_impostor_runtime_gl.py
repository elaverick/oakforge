"""Optional Xvfb-backed runtime test for forward-projected depth splats."""

import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from oakforge.render.impostor_runtime import DepthImpostorRenderer


def _perspective_view_projection(eye, target, aspect=1.0):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up, eye)
    view[2, 3] = np.dot(forward, eye)
    near, far = 0.1, 250.0
    scale = 1.0 / np.tan(np.radians(48.0) * 0.5)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = scale / aspect
    projection[1, 1] = scale
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return projection @ view


@unittest.skipUnless(os.environ.get("OAKFORGE_GL_SMOKE") == "1",
                     "Set OAKFORGE_GL_SMOKE=1 after baking an atlas to run the GL smoke test")
class RuntimeImpostorOpenGLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import glfw
        from OpenGL import GL

        cls.glfw = glfw
        cls.gl = GL
        if not glfw.init():
            raise RuntimeError("GLFW could not initialise for the runtime impostor smoke test")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        cls.window = glfw.create_window(128, 128, "OakForge runtime impostor smoke test", None, None)
        if not cls.window:
            glfw.terminate()
            raise RuntimeError("Could not create hidden OpenGL 3.3 context")
        glfw.make_context_current(cls.window)

    @classmethod
    def tearDownClass(cls):
        cls.glfw.destroy_window(cls.window)
        cls.glfw.terminate()

    def test_bake_contains_multiple_real_depth_layers(self):
        asset_directory = Path("build/impostor")
        manifest = json.loads((asset_directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], "oakforge-canopy-layered-depth-impostor-v3")
        self.assertGreaterEqual(manifest["depth_layers"]["count"], 3)

        colour_files = manifest["textures"]["colour"]
        depth_files = manifest["textures"]["depth"]
        coverage = []
        for filename in colour_files:
            with Image.open(asset_directory / filename) as image:
                pixels = np.asarray(image.convert("RGBA"))
            coverage.append(int(np.count_nonzero(pixels[:, :, 3])))

        self.assertGreater(coverage[0], 100)
        self.assertGreater(coverage[1], 100)
        self.assertGreater(coverage[2], 20)

        with Image.open(asset_directory / depth_files[0]) as image:
            front_depth = np.asarray(image).astype(np.uint16)
        with Image.open(asset_directory / depth_files[1]) as image:
            second_depth = np.asarray(image).astype(np.uint16)
        self.assertGreater(int(np.unique(front_depth).size), 20)
        self.assertGreater(int(np.count_nonzero(front_depth != second_depth)), 100)

    def test_runtime_forward_projects_depth_layers_and_resolves_occlusion(self):
        asset_directory = Path("build/impostor")
        renderer = DepthImpostorRenderer(self.gl, asset_directory)
        try:
            self.assertGreaterEqual(renderer.layer_count, 3)
            self.assertGreater(renderer.allocated_bytes, 0)
            self.gl.glViewport(0, 0, 128, 128)
            self.gl.glEnable(self.gl.GL_DEPTH_TEST)
            self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)

            direction = np.asarray(renderer.views[0]["direction"], dtype=np.float32)
            camera_position = renderer.centre + direction * (renderer.far * 0.8)
            matrix = _perspective_view_projection(
                camera_position, renderer.centre, aspect=1.0
            )
            self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
            selected_views = renderer.draw(matrix, camera_position, (128, 128))
            self.gl.glFinish()

            pixels = np.frombuffer(
                self.gl.glReadPixels(0, 0, 128, 128, self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE),
                dtype=np.uint8,
            ).reshape(128, 128, 4)
            depth_pixels = np.frombuffer(
                self.gl.glReadPixels(0, 0, 128, 128,
                                     self.gl.GL_DEPTH_COMPONENT, self.gl.GL_FLOAT),
                dtype=np.float32,
            ).reshape(128, 128)
            Image.fromarray(pixels, mode="RGBA").save(
                Path("build/impostor-forward-splat-capture.png")
            )
            covered = np.any(pixels[:, :, :3] > 4, axis=2)
            covered_pixels = int(np.count_nonzero(covered))
            self.assertEqual(len(selected_views), 1)
            self.assertEqual(selected_views[0], 0)
            self.assertEqual(renderer.point_count, renderer.tile_size ** 2 * renderer.layer_count)
            self.assertGreater(covered_pixels, 10)
            self.assertTrue(np.all(depth_pixels[covered] < 1.0))
            self.assertEqual(self.gl.glGetError(), self.gl.GL_NO_ERROR)

            # A displaced camera changes the forward projection while retaining
            # one coherent source capture (no cross-view geometry averaging).
            second_camera = camera_position + np.asarray([renderer.extent * 0.25, 0.0, 0.0])
            second_matrix = _perspective_view_projection(
                second_camera, renderer.centre, aspect=1.0
            )
            self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
            second_selected = renderer.draw(second_matrix, second_camera, (128, 128))
            self.assertEqual(len(second_selected), 1)
            self.gl.glFinish()
            second_pixels = np.frombuffer(
                self.gl.glReadPixels(0, 0, 128, 128, self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE),
                dtype=np.uint8,
            ).reshape(128, 128, 4)
            Image.fromarray(second_pixels, mode="RGBA").save(
                Path("build/impostor-forward-splat-oblique.png")
            )
            second_covered = np.any(second_pixels[:, :, :3] > 4, axis=2)
            self.assertGreater(int(np.count_nonzero(second_covered)), 10)
            changed_pixels = int(np.count_nonzero(np.logical_xor(covered, second_covered)))
            self.assertGreater(changed_pixels, 0)
            self.assertEqual(self.gl.glGetError(), self.gl.GL_NO_ERROR)
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
