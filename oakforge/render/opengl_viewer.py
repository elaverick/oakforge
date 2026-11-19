"""Interactive OpenGL 3.3 viewer for inspecting procedural tree structure."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np

from oakforge.core.mesh import MeshData, build_tree_mesh
from oakforge.core.models import TreeModel


VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec3 vertex_colour;
uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
out vec3 world_normal;
out vec3 surface_colour;
void main()
{
    world_normal = mat3(transpose(inverse(model_matrix))) * vertex_normal;
    surface_colour = vertex_colour;
    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0);
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 world_normal;
in vec3 surface_colour;
out vec4 fragment_colour;
void main()
{
    vec3 normal_direction = normalize(world_normal);
    if (!gl_FrontFacing) normal_direction = -normal_direction;
    vec3 light_direction = normalize(vec3(-0.45, 0.82, 0.36));
    float diffuse_amount = max(dot(normal_direction, light_direction), 0.0);
    float ambient_amount = 0.34;
    vec3 shaded_colour = surface_colour * (ambient_amount + diffuse_amount * 0.72);
    fragment_colour = vec4(shaded_colour, 1.0);
}
"""


class OpenGLViewer:
    """Display a tree using a small shader-based OpenGL renderer."""

    def __init__(self, tree: TreeModel, regenerate_callback: Any) -> None:
        # Graphics imports are delayed so geometry generation and OBJ export do not need OpenGL installed.
        try:
            import glfw
            from OpenGL import GL
        except ImportError as error:
            raise RuntimeError(
                "The interactive viewer needs glfw and PyOpenGL. Install dependencies with "
                "'python -m pip install -r requirements.txt'. Geometry export does not need them."
            ) from error

        self.glfw = glfw
        self.gl = GL
        self.tree = tree
        self.regenerate_callback = regenerate_callback
        self.window = None
        self.program = 0
        self.tree_vao = 0
        self.tree_vbo = 0
        self.grid_vao = 0
        self.grid_vbo = 0
        self.tree_vertex_count = 0
        self.grid_vertex_count = 0
        self.max_depth = 4
        self.show_leaves = True
        self.wireframe = False
        self.camera_yaw = math.radians(38.0)
        self.camera_pitch = math.radians(17.0)
        tree_heights = [point[1] for branch in tree.branches for point in branch.points]
        tree_centre_height = (min(tree_heights) + max(tree_heights)) * 0.5
        self.camera_distance = max(26.0, (max(tree_heights) - min(tree_heights)) * 1.42)
        self.camera_target = np.array([0.0, tree_centre_height, 0.0], dtype=np.float64)
        self.default_camera_distance = self.camera_distance
        self.default_camera_target = self.camera_target.copy()
        self.dragging = False
        self.last_cursor_position: tuple[float, float] | None = None

        if not glfw.init():
            raise RuntimeError("GLFW could not initialise. A desktop OpenGL session is required.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.window = glfw.create_window(1280, 900, "OakForge | English Oak", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create an OpenGL 3.3 window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        print(f"OpenGL renderer: {self._decode_gl_string(self.gl.GL_VERSION)}")
        self._configure_callbacks()
        self._create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._create_buffers()
        self._upload_tree_mesh()
        self._upload_grid_mesh()
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glClearColor(0.78, 0.82, 0.79, 1.0)
        self._print_controls()

    @staticmethod
    def _decode_gl_string(gl_enum: int) -> str:
        """Return a printable OpenGL driver string."""
        from OpenGL.GL import glGetString
        value = glGetString(gl_enum)
        return value.decode("utf-8", errors="replace") if value else "unknown"

    def _configure_callbacks(self) -> None:
        """Connect keyboard and mouse input to camera and inspection controls."""
        glfw = self.glfw
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _create_shader_program(self, vertex_source: str, fragment_source: str) -> None:
        """Compile the simple directional-light shader used for all tree polygons."""
        gl = self.gl

        def compile_shader(shader_type: int, source: str) -> int:
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                message = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenGL shader compilation failed: {message}")
            return shader

        vertex_shader = compile_shader(gl.GL_VERTEX_SHADER, vertex_source)
        fragment_shader = compile_shader(gl.GL_FRAGMENT_SHADER, fragment_source)
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vertex_shader)
        gl.glAttachShader(self.program, fragment_shader)
        gl.glLinkProgram(self.program)
        if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            message = gl.glGetProgramInfoLog(self.program).decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenGL shader linking failed: {message}")
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

    def _create_buffers(self) -> None:
        """Allocate independent vertex buffers for the tree and its ground reference grid."""
        gl = self.gl
        self.tree_vao = gl.glGenVertexArrays(1)
        self.tree_vbo = gl.glGenBuffers(1)
        self.grid_vao = gl.glGenVertexArrays(1)
        self.grid_vbo = gl.glGenBuffers(1)

    def _upload_vertex_data(self, vertex_array: int, buffer: int, mesh: MeshData, primitive: int) -> int:
        """Upload interleaved position, normal and colour attributes to a GPU buffer."""
        gl = self.gl
        if not mesh.positions:
            return 0
        vertex_positions = np.asarray(mesh.positions, dtype=np.float32).reshape(-1, 3)
        vertex_normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
        vertex_colours = np.asarray(mesh.colours, dtype=np.float32).reshape(-1, 3)
        interleaved_vertices = np.column_stack((vertex_positions, vertex_normals, vertex_colours))
        gl.glBindVertexArray(vertex_array)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved_vertices.nbytes,
                        interleaved_vertices, gl.GL_STATIC_DRAW)
        attribute_stride = 9 * interleaved_vertices.itemsize
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, attribute_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, attribute_stride, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, attribute_stride, ctypes.c_void_p(6 * 4))
        gl.glBindVertexArray(0)
        return len(interleaved_vertices)

    def _visible_tree(self) -> TreeModel:
        """Select branch tiers and foliage for the current structural inspection view."""
        visible_branches = [branch for branch in self.tree.branches if branch.depth <= self.max_depth]
        visible_leaves = self.tree.leaves if self.show_leaves and self.max_depth >= 4 else []
        metadata = self.tree.metadata
        if self.max_depth < 2:
            # A fused trunk-and-limb surface cannot be partially hidden by depth,
            # so use the individual skeleton tubes while inspecting early stages.
            metadata = dict(self.tree.metadata)
            metadata["wood_surface"] = None
        return TreeModel(self.tree.name, visible_branches, visible_leaves, metadata)

    def _upload_tree_mesh(self) -> None:
        """Rebuild the display buffer after changing the visible branch hierarchy."""
        mesh = build_tree_mesh(self._visible_tree())
        self.tree_vertex_count = self._upload_vertex_data(self.tree_vao, self.tree_vbo, mesh, self.gl.GL_TRIANGLES)

    def _upload_grid_mesh(self) -> None:
        """Build a subdued ground grid to make the tree's scale and orientation legible."""
        mesh = MeshData()
        grid_extent = 14
        grid_colour = (0.43, 0.49, 0.46)
        normal = np.array([0.0, 1.0, 0.0])
        for grid_position in range(-grid_extent, grid_extent + 1):
            first_line = np.array([float(grid_position), 0.0, -float(grid_extent)])
            second_line = np.array([float(grid_position), 0.0, float(grid_extent)])
            third_line = np.array([-float(grid_extent), 0.0, float(grid_position)])
            fourth_line = np.array([float(grid_extent), 0.0, float(grid_position)])
            mesh.positions.extend(float(value) for point in (first_line, second_line, third_line, fourth_line) for value in point)
            mesh.normals.extend(float(value) for _ in range(4) for value in normal)
            mesh.colours.extend(float(value) for _ in range(4) for value in grid_colour)
        self.grid_vertex_count = self._upload_vertex_data(
            self.grid_vao, self.grid_vbo, mesh, self.gl.GL_LINES
        )

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        """Normalise a camera direction without relying on external math packages."""
        magnitude = np.linalg.norm(vector)
        return vector / magnitude if magnitude > 1e-8 else vector

    def _view_matrix(self) -> np.ndarray:
        """Build a look-at matrix from the orbit camera state."""
        horizontal_distance = self.camera_distance * math.cos(self.camera_pitch)
        eye_position = self.camera_target + np.array([
            horizontal_distance * math.sin(self.camera_yaw),
            self.camera_distance * math.sin(self.camera_pitch),
            horizontal_distance * math.cos(self.camera_yaw),
        ])
        forward_direction = self._normalise(self.camera_target - eye_position)
        side_direction = self._normalise(np.cross(forward_direction, np.array([0.0, 1.0, 0.0])))
        up_direction = np.cross(side_direction, forward_direction)
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[0, :3] = side_direction
        view_matrix[1, :3] = up_direction
        view_matrix[2, :3] = -forward_direction
        view_matrix[0, 3] = -np.dot(side_direction, eye_position)
        view_matrix[1, 3] = -np.dot(up_direction, eye_position)
        view_matrix[2, 3] = np.dot(forward_direction, eye_position)
        return view_matrix

    @staticmethod
    def _projection_matrix(aspect_ratio: float) -> np.ndarray:
        """Build a perspective projection suitable for a tall vegetation asset."""
        field_of_view = math.radians(48.0)
        near_plane = 0.1
        far_plane = 250.0
        scale = 1.0 / math.tan(field_of_view * 0.5)
        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        projection_matrix[0, 0] = scale / max(aspect_ratio, 1e-5)
        projection_matrix[1, 1] = scale
        projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
        projection_matrix[2, 3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
        projection_matrix[3, 2] = -1.0
        return projection_matrix

    def _on_key(self, window: Any, key: int, scan_code: int, action: int, modifiers: int) -> None:
        """Handle one-shot keyboard controls for structural inspection."""
        glfw = self.glfw
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_L:
            self.show_leaves = not self.show_leaves
            self._upload_tree_mesh()
        elif key == glfw.KEY_W:
            self.wireframe = not self.wireframe
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
            self.max_depth = key - glfw.KEY_1
            self._upload_tree_mesh()
        elif key == glfw.KEY_R:
            self.tree = self.regenerate_callback()
            self.max_depth = 4
            self.show_leaves = True
            self._upload_tree_mesh()
            print(f"Regenerated tree: {self.tree.metadata}")
        elif key == glfw.KEY_F:
            self.camera_yaw = math.radians(38.0)
            self.camera_pitch = math.radians(17.0)
            self.camera_distance = self.default_camera_distance
            self.camera_target = self.default_camera_target.copy()

    def _on_mouse_button(self, window: Any, button: int, action: int, modifiers: int) -> None:
        """Begin or end camera orbit when the user drags with the left mouse button."""
        if button == self.glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == self.glfw.PRESS
            self.last_cursor_position = None

    def _on_cursor_move(self, window: Any, cursor_x: float, cursor_y: float) -> None:
        """Orbit around the tree while the left mouse button is held."""
        if not self.dragging:
            return
        if self.last_cursor_position is not None:
            previous_x, previous_y = self.last_cursor_position
            self.camera_yaw += (cursor_x - previous_x) * 0.006
            self.camera_pitch += (cursor_y - previous_y) * 0.006
            self.camera_pitch = float(np.clip(self.camera_pitch, -0.15, 1.25))
        self.last_cursor_position = (cursor_x, cursor_y)

    def _on_scroll(self, window: Any, horizontal_offset: float, vertical_offset: float) -> None:
        """Zoom the orbit camera without moving its target point."""
        self.camera_distance *= math.exp(-vertical_offset * 0.10)
        self.camera_distance = float(np.clip(self.camera_distance, 3.5, 90.0))

    def _print_controls(self) -> None:
        """Print the available viewer controls where they remain visible to the user."""
        print("\nOakForge viewer controls:")
        print("  Drag with left mouse button : orbit camera")
        print("  Mouse wheel                 : zoom")
        print("  1 / 2 / 3 / 4 / 5         : trunk / primary / secondary / tertiary / full tree")
        print("  L                           : toggle leaves")
        print("  W                           : toggle wireframe")
        print("  R                           : regenerate with a new seed")
        print("  F                           : reset camera")
        print("  Escape                      : close viewer\n")

    def run(self) -> None:
        """Run the interactive render loop until the viewer window is closed."""
        gl = self.gl
        while not self.glfw.window_should_close(self.window):
            framebuffer_width, framebuffer_height = self.glfw.get_framebuffer_size(self.window)
            gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glUseProgram(self.program)

            model_matrix = np.identity(4, dtype=np.float32)
            view_matrix = self._view_matrix()
            projection_matrix = self._projection_matrix(
                framebuffer_width / max(framebuffer_height, 1)
            )
            for uniform_name, matrix in (
                ("model_matrix", model_matrix),
                ("view_matrix", view_matrix),
                ("projection_matrix", projection_matrix),
            ):
                uniform_location = gl.glGetUniformLocation(self.program, uniform_name)
                gl.glUniformMatrix4fv(uniform_location, 1, gl.GL_TRUE, matrix)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL)
            gl.glBindVertexArray(self.grid_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.grid_vertex_count)
            gl.glBindVertexArray(self.tree_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.tree_vertex_count)
            gl.glBindVertexArray(0)

            self.glfw.swap_buffers(self.window)
            self.glfw.poll_events()

        self.glfw.destroy_window(self.window)
        self.glfw.terminate()
