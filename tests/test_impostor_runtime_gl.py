"""Optional Xvfb-backed OpenGL smoke test for the runtime impostor shader."""

import os
import unittest
from pathlib import Path

import numpy as np

from oakforge.render.impostor import _orthographic_view_projection
from oakforge.render.impostor_runtime import DepthImpostorRenderer


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
        cls.window = glfw.create_window(64, 64, "OakForge runtime impostor smoke test", None, None)
        if not cls.window:
            glfw.terminate()
            raise RuntimeError("Could not create hidden OpenGL 3.3 context")
        glfw.make_context_current(cls.window)

    @classmethod
    def tearDownClass(cls):
        cls.glfw.destroy_window(cls.window)
        cls.glfw.terminate()

    def test_runtime_shader_uploads_atlases_and_draws_without_gl_errors(self):
        asset_directory = Path("build/impostor")
        self.assertTrue((asset_directory / "manifest.json").is_file())
        renderer = DepthImpostorRenderer(self.gl, asset_directory)
        try:
            self.gl.glViewport(0, 0, 64, 64)
            self.gl.glEnable(self.gl.GL_DEPTH_TEST)
            self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
            view = renderer.views[0]
            direction = np.asarray(view["direction"], dtype=np.float32)
            matrix = _orthographic_view_projection(
                direction,
                renderer.centre,
                renderer.extent,
                renderer.near,
                renderer.far,
            )
            view_index = renderer.draw(matrix, direction)
            self.gl.glFinish()
            pixels = np.frombuffer(
                self.gl.glReadPixels(0, 0, 64, 64, self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE),
                dtype=np.uint8,
            ).reshape(64, 64, 4)
            covered_pixels = int(np.count_nonzero(np.any(pixels[:, :, :3] > 4, axis=2)))
            self.assertEqual(view_index, 0)
            self.assertGreater(covered_pixels, 10)
            self.assertEqual(self.gl.glGetError(), self.gl.GL_NO_ERROR)
            self.assertGreater(renderer.allocated_bytes, 0)
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
