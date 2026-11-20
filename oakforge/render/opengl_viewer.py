"""Interactive OpenGL viewer with live polygon/impostor comparison and diagnostics."""

from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from oakforge.core.mesh import (
    OAK_LEAF_PROFILE,
    MeshData,
    build_foliage_mesh,
    build_wood_mesh,
)
from oakforge.core.models import TreeModel
from oakforge.render.gpu_metrics import GpuTelemetry
from oakforge.render.impostor_runtime import DepthImpostorRenderer


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

OVERLAY_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texture_coordinate;
out vec2 uv;
void main() {
    uv = texture_coordinate;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

OVERLAY_FRAGMENT_SHADER = """#version 330 core
in vec2 uv;
out vec4 fragment_colour;
uniform sampler2D overlay_texture;
void main() {
    fragment_colour = texture(overlay_texture, uv);
}
"""


class OpenGLViewer:
    """Inspect a generated tree and compare polygon canopy with depth impostors."""

    def __init__(
        self,
        tree: TreeModel,
        regenerate_callback: Callable[[], TreeModel],
        impostor_directory: str | Path,
        rebake_callback: Callable[[TreeModel], str | Path] | None = None,
    ) -> None:
        try:
            import glfw
            from OpenGL import GL
        except ImportError as error:
            raise RuntimeError(
                "The interactive viewer needs glfw and PyOpenGL. Install dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from error
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install Pillow to display the viewer diagnostics overlay") from error

        self.glfw = glfw
        self.gl = GL
        self.tree = tree
        self.regenerate_callback = regenerate_callback
        self.rebake_callback = rebake_callback
        self.impostor_directory = Path(impostor_directory)
        self.window = None
        self.program = 0

        self.wood_vao = 0
        self.wood_vbo = 0
        self.wood_vertex_count = 0
        self.wood_triangle_count = 0
        self.canopy_vao = 0
        self.canopy_vbo = 0
        self.canopy_vertex_count = 0
        self.canopy_triangle_count = 0
        self.canopy_mode = "polygons"
        self.canopy_renderer: DepthImpostorRenderer | None = None
        self.show_leaves = True
        self.wireframe = False
        self.vsync = False

        self.grid_vao = 0
        self.grid_vbo = 0
        self.grid_vertex_count = 0
        self.overlay_program = 0
        self.overlay_vao = 0
        self.overlay_vbo = 0
        self.overlay_texture = 0
        self.overlay_width = 760
        self.overlay_height = 320
        self.overlay_last_update = 0.0

        self.gpu_allocations: dict[str, int] = {}
        self.gpu_queries: list[int] = []
        self.gpu_query_pending: list[bool] = []
        self.gpu_query_write_index = 0
        self.gpu_query_active: int | None = None
        self.gpu_scene_ms: float | None = None
        self.frame_ms = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.previous_frame_time = time.perf_counter()
        self.status_message = ""
        self.last_selected_view = -1

        self.max_depth = 4
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
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.window = glfw.create_window(1440, 960, "OakForge | Canopy LOD comparison", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create an OpenGL 3.3 window.")

        glfw.make_context_current(self.window)
        self.vsync = False
        glfw.swap_interval(0)
        print(f"OpenGL version : {self._decode_gl_string(GL.GL_VERSION)}")
        print(f"OpenGL renderer: {self._decode_gl_string(GL.GL_RENDERER)}")
        self._configure_callbacks()
        self._create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._create_buffers()
        self._upload_grid_mesh()
        self._upload_wood_mesh()
        self._set_canopy_mode("polygons")
        self._create_overlay()
        self._create_gpu_queries()
        self.gpu_telemetry = GpuTelemetry(GL, self._decode_gl_string(GL.GL_RENDERER))
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.78, 0.82, 0.79, 1.0)
        self._print_controls()

    @staticmethod
    def _decode_gl_string(gl_enum: int) -> str:
        from OpenGL.GL import glGetString
        value = glGetString(gl_enum)
        return value.decode("utf-8", errors="replace") if value else "unknown"

    def _configure_callbacks(self) -> None:
        glfw = self.glfw
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _compile_program(self, vertex_source: str, fragment_source: str) -> int:
        gl = self.gl
        shaders: list[int] = []
        for shader_type, source in (
            (gl.GL_VERTEX_SHADER, vertex_source),
            (gl.GL_FRAGMENT_SHADER, fragment_source),
        ):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                message = gl.glGetShaderInfoLog(shader).decode("utf-8", "replace")
                gl.glDeleteShader(shader)
                raise RuntimeError(f"OpenGL shader compilation failed: {message}")
            shaders.append(shader)
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            message = gl.glGetProgramInfoLog(program).decode("utf-8", "replace")
            for shader in shaders:
                gl.glDeleteShader(shader)
            gl.glDeleteProgram(program)
            raise RuntimeError(f"OpenGL shader link failed: {message}")
        for shader in shaders:
            gl.glDetachShader(program, shader)
            gl.glDeleteShader(shader)
        return int(program)

    def _create_shader_program(self, vertex_source: str, fragment_source: str) -> None:
        self.program = self._compile_program(vertex_source, fragment_source)
        uniform_names = ("model_matrix", "view_matrix", "projection_matrix")
        self.tree_uniforms = {
            name: int(self.gl.glGetUniformLocation(self.program, name))
            for name in uniform_names
        }
        if any(location < 0 for location in self.tree_uniforms.values()):
            raise RuntimeError("The tree shader is missing a required matrix uniform")

    def _create_buffers(self) -> None:
        self.wood_vao = int(self.gl.glGenVertexArrays(1))
        self.wood_vbo = int(self.gl.glGenBuffers(1))
        self.grid_vao = int(self.gl.glGenVertexArrays(1))
        self.grid_vbo = int(self.gl.glGenBuffers(1))

    def _upload_vertex_data(
        self, vertex_array: int, buffer: int, mesh: MeshData, allocation_name: str
    ) -> tuple[int, int]:
        gl = self.gl
        vertex_count = len(mesh.positions) // 3
        if vertex_count == 0:
            self.gpu_allocations[allocation_name] = 0
            return 0, 0

        # Read the packed source arrays without copies, then interleave once for
        # a single compact GPU vertex buffer (position, normal, colour).
        positions = np.frombuffer(mesh.positions, dtype=np.float32).reshape(-1, 3)
        normals = np.frombuffer(mesh.normals, dtype=np.float32).reshape(-1, 3)
        colours = np.frombuffer(mesh.colours, dtype=np.float32).reshape(-1, 3)
        interleaved = np.empty((vertex_count, 9), dtype=np.float32)
        interleaved[:, 0:3] = positions
        interleaved[:, 3:6] = normals
        interleaved[:, 6:9] = colours

        gl.glBindVertexArray(vertex_array)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW)
        stride = 9 * 4
        for location, size, offset in ((0, 3, 0), (1, 3, 12), (2, 3, 24)):
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(
                location, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
            )
        gl.glBindVertexArray(0)
        allocated_bytes = int(interleaved.nbytes)
        self.gpu_allocations[allocation_name] = allocated_bytes
        return vertex_count, allocated_bytes

    def _visible_tree(self, include_leaves: bool = False) -> TreeModel:
        visible_branches = [
            branch for branch in self.tree.branches if branch.depth <= self.max_depth
        ]
        visible_leaves = self.tree.leaves if include_leaves and self.max_depth >= 4 else []
        metadata = self.tree.metadata
        if self.max_depth < 2:
            metadata = dict(self.tree.metadata)
            metadata["wood_surface"] = None
        return TreeModel(self.tree.name, visible_branches, visible_leaves, metadata)

    def _upload_wood_mesh(self) -> None:
        mesh = build_wood_mesh(self._visible_tree())
        self.wood_vertex_count, _ = self._upload_vertex_data(
            self.wood_vao, self.wood_vbo, mesh, "wood_vbo"
        )
        self.wood_triangle_count = self.wood_vertex_count // 3
        if not hasattr(self, "source_wood_triangles"):
            self.source_wood_triangles = self.wood_triangle_count
            self.model_leaf_triangles = len(self.tree.leaves) * (2 * len(OAK_LEAF_PROFILE) - 1)
            self.model_total_triangles = self.source_wood_triangles + self.model_leaf_triangles

    def _upload_canopy_polygons(self) -> None:
        gl = self.gl
        self.canopy_vao = int(gl.glGenVertexArrays(1))
        self.canopy_vbo = int(gl.glGenBuffers(1))
        if not self.show_leaves or self.max_depth < 4:
            self.canopy_vertex_count = 0
            self.canopy_triangle_count = 0
            self.gpu_allocations["canopy_vbo"] = 0
            return
        mesh = build_foliage_mesh(self._visible_tree(include_leaves=True))
        self.canopy_vertex_count, _ = self._upload_vertex_data(
            self.canopy_vao, self.canopy_vbo, mesh, "canopy_vbo"
        )
        self.canopy_triangle_count = self.canopy_vertex_count // 3

    def _release_canopy_representation(self) -> None:
        gl = self.gl
        if self.canopy_renderer is not None:
            self.canopy_renderer.close()
            self.canopy_renderer = None
        if self.canopy_vao:
            gl.glDeleteVertexArrays(1, [self.canopy_vao])
            self.canopy_vao = 0
        if self.canopy_vbo:
            gl.glDeleteBuffers(1, [self.canopy_vbo])
            self.canopy_vbo = 0
        self.gpu_allocations.pop("canopy_vbo", None)
        self.gpu_allocations.pop("canopy_impostor", None)
        self.canopy_vertex_count = 0
        self.canopy_triangle_count = 0

    def _set_canopy_mode(self, mode: str) -> None:
        if mode not in {"polygons", "impostor"}:
            raise ValueError(f"Unknown canopy representation: {mode}")
        self._release_canopy_representation()
        self.canopy_mode = mode
        if not self.show_leaves or self.max_depth < 4:
            return
        if mode == "polygons":
            self._upload_canopy_polygons()
        else:
            renderer = DepthImpostorRenderer(self.gl, self.impostor_directory)
            self.canopy_renderer = renderer
            self.gpu_allocations["canopy_impostor"] = renderer.allocated_bytes
        self.status_message = f"Canopy representation: {mode.upper()}"

    def _upload_grid_mesh(self) -> None:
        mesh = MeshData()
        grid_extent = 14
        grid_colour = (0.43, 0.49, 0.46)
        normal = np.array([0.0, 1.0, 0.0])
        for grid_position in range(-grid_extent, grid_extent + 1):
            first_line = np.array([float(grid_position), 0.0, -float(grid_extent)])
            second_line = np.array([float(grid_position), 0.0, float(grid_extent)])
            third_line = np.array([-float(grid_extent), 0.0, float(grid_position)])
            fourth_line = np.array([float(grid_extent), 0.0, float(grid_position)])
            mesh.positions.extend(
                float(value)
                for point in (first_line, second_line, third_line, fourth_line)
                for value in point
            )
            mesh.normals.extend(float(value) for _ in range(4) for value in normal)
            mesh.colours.extend(float(value) for _ in range(4) for value in grid_colour)
        self.grid_vertex_count, _ = self._upload_vertex_data(
            self.grid_vao, self.grid_vbo, mesh, "grid_vbo"
        )

    def _create_gpu_queries(self) -> None:
        gl = self.gl
        if not hasattr(gl, "GL_TIME_ELAPSED") or not hasattr(gl, "glGetQueryObjectui64v"):
            return
        try:
            generated = gl.glGenQueries(4)
            if isinstance(generated, (int, np.integer)):
                self.gpu_queries = [int(generated)]
            else:
                self.gpu_queries = [int(query) for query in generated]
            self.gpu_query_pending = [False] * len(self.gpu_queries)
        except Exception:
            self.gpu_queries = []
            self.gpu_query_pending = []

    def _poll_gpu_queries(self) -> None:
        gl = self.gl
        for index, query in enumerate(self.gpu_queries):
            if not self.gpu_query_pending[index]:
                continue
            try:
                available = gl.glGetQueryObjectiv(query, gl.GL_QUERY_RESULT_AVAILABLE)
                if available:
                    nanoseconds = gl.glGetQueryObjectui64v(query, gl.GL_QUERY_RESULT)
                    self.gpu_scene_ms = float(nanoseconds) / 1_000_000.0
                    self.gpu_query_pending[index] = False
            except Exception:
                self.gpu_queries = []
                self.gpu_query_pending = []
                self.gpu_scene_ms = None
                return

    def _begin_gpu_query(self) -> None:
        if not self.gpu_queries:
            return
        index = self.gpu_query_write_index
        if self.gpu_query_pending[index]:
            return
        try:
            self.gl.glBeginQuery(self.gl.GL_TIME_ELAPSED, self.gpu_queries[index])
            self.gpu_query_active = index
        except Exception:
            self.gpu_queries = []
            self.gpu_query_pending = []
            self.gpu_query_active = None

    def _end_gpu_query(self) -> None:
        if self.gpu_query_active is None:
            return
        self.gl.glEndQuery(self.gl.GL_TIME_ELAPSED)
        self.gpu_query_pending[self.gpu_query_active] = True
        self.gpu_query_write_index = (self.gpu_query_active + 1) % len(self.gpu_queries)
        self.gpu_query_active = None

    @staticmethod
    def _format_bytes(value: int | None) -> str:
        if value is None:
            return "n/a"
        amount = float(value)
        for unit in ("B", "KB", "MB", "GB"):
            if amount < 1024.0 or unit == "GB":
                return f"{amount:.1f} {unit}"
            amount /= 1024.0
        return f"{amount:.1f} GB"

    def _stats_lines(self) -> list[str]:
        telemetry = self.gpu_telemetry.sample()
        active_canopy_triangles = (
            self.canopy_triangle_count if self.canopy_mode == "polygons"
            else (2 if self.canopy_renderer is not None else 0)
        ) if self.show_leaves and self.max_depth >= 4 else 0
        active_triangles = self.wood_triangle_count + active_canopy_triangles
        active_name = "DEPTH IMPOSTOR" if self.canopy_mode == "impostor" else "POLYGON LEAVES"
        allocated = sum(self.gpu_allocations.values())
        canopy_allocated = self.gpu_allocations.get(
            "canopy_vbo", self.gpu_allocations.get("canopy_impostor", 0)
        )
        total = telemetry["total_bytes"]
        used = telemetry["used_bytes"]
        process = telemetry["process_bytes"]
        util = telemetry["utilization_percent"]
        memory_util = telemetry["memory_utilization_percent"]
        gpu_frame = f"{self.gpu_scene_ms:.2f} ms" if self.gpu_scene_ms is not None else "n/a"
        frame_ms = f"{self.frame_ms:.2f} ms" if self.frame_ms > 0 else "n/a"
        fps = f"{self.fps:.1f}" if self.fps > 0 else "warming up"
        lines = [
            "OAKFORGE | CANOPY REPRESENTATION BENCHMARK",
            f"Mode: {active_name}   [I] switch   [L] canopy on/off",
            f"Model triangles: {self.model_total_triangles:,} total",
            f"  Wood: {self.source_wood_triangles:,}   Leaf polygons: {self.model_leaf_triangles:,}",
            f"Active triangles: {active_triangles:,} (wood {self.wood_triangle_count:,} + canopy {active_canopy_triangles:,})",
            f"OakForge GPU allocations: {self._format_bytes(allocated)} (tracked buffers/textures)",
            f"Canopy allocation: {self._format_bytes(canopy_allocated)}",
            f"Device VRAM: {self._format_bytes(used)} used / {self._format_bytes(total)} total",
            f"Process VRAM: {self._format_bytes(process)}   GPU load: {str(util) + '%' if util is not None else 'n/a'}",
            f"GPU memory controller: {str(memory_util) + '%' if memory_util is not None else 'n/a'}",
            f"Performance: {fps} FPS   CPU frame: {frame_ms}   GPU scene: {gpu_frame}",
            f"Telemetry: {telemetry['backend']}   Renderer: {telemetry['gpu_name']}",
            f"Atlas view: {self.last_selected_view + 1 if self.last_selected_view >= 0 else 'n/a'} / {len(self.canopy_renderer.views) if self.canopy_renderer else 'n/a'}",
            "Keys: I representation | L canopy | 1-5 structure | W wireframe | V VSync | R regenerate",
        ]
        if self.status_message:
            lines.append(self.status_message)
        return lines

    def _create_overlay(self) -> None:
        gl = self.gl
        self.overlay_program = self._compile_program(OVERLAY_VERTEX_SHADER, OVERLAY_FRAGMENT_SHADER)
        self.overlay_vao = int(gl.glGenVertexArrays(1))
        self.overlay_vbo = int(gl.glGenBuffers(1))
        self.overlay_texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.overlay_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.overlay_width, self.overlay_height,
            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glUseProgram(self.overlay_program)
        self.overlay_sampler_location = int(
            gl.glGetUniformLocation(self.overlay_program, "overlay_texture")
        )
        gl.glUniform1i(self.overlay_sampler_location, 0)
        gl.glBindVertexArray(self.overlay_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.overlay_vbo)
        stride = 4 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)
        self.gpu_allocations["overlay_texture"] = self.overlay_width * self.overlay_height * 4
        self.gpu_allocations["overlay_vbo"] = 6 * 4 * 4

    def _render_overlay(self, framebuffer_width: int, framebuffer_height: int) -> None:
        now = time.perf_counter()
        if now - self.overlay_last_update >= 0.15:
            from PIL import Image, ImageDraw, ImageFont

            image = Image.new(
                "RGBA", (self.overlay_width, self.overlay_height), (15, 22, 18, 224)
            )
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (0, 0, self.overlay_width - 1, self.overlay_height - 1),
                radius=10, outline=(154, 185, 152, 255), width=2,
            )
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", 13)
            except OSError:
                font = ImageFont.load_default()
            for line_index, line in enumerate(self._stats_lines()):
                y = 12 + line_index * 19
                if y + 18 >= self.overlay_height:
                    break
                draw.text((14, y), line, font=font, fill=(232, 239, 228, 255))
            pixels = np.ascontiguousarray(np.flipud(np.asarray(image)))
            self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.overlay_texture)
            self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
            self.gl.glTexSubImage2D(
                self.gl.GL_TEXTURE_2D, 0, 0, 0, self.overlay_width, self.overlay_height,
                self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE, pixels,
            )
            self.overlay_last_update = now

        gl = self.gl
        left = -1.0 + 16.0 * 2.0 / max(framebuffer_width, 1)
        top = 1.0 - 16.0 * 2.0 / max(framebuffer_height, 1)
        right = left + self.overlay_width * 2.0 / max(framebuffer_width, 1)
        bottom = top - self.overlay_height * 2.0 / max(framebuffer_height, 1)
        vertices = np.asarray([
            [left, bottom, 0.0, 0.0], [right, bottom, 1.0, 0.0],
            [right, top, 1.0, 1.0], [left, bottom, 0.0, 0.0],
            [right, top, 1.0, 1.0], [left, top, 0.0, 1.0],
        ], dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.overlay_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.overlay_program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.overlay_texture)
        gl.glBindVertexArray(self.overlay_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _camera_position(self) -> np.ndarray:
        horizontal_distance = self.camera_distance * math.cos(self.camera_pitch)
        return self.camera_target + np.array([
            horizontal_distance * math.sin(self.camera_yaw),
            self.camera_distance * math.sin(self.camera_pitch),
            horizontal_distance * math.cos(self.camera_yaw),
        ])

    def _view_matrix(self) -> np.ndarray:
        eye_position = self._camera_position()
        forward_direction = self._normalise(self.camera_target - eye_position)
        side_direction = self._normalise(
            np.cross(forward_direction, np.array([0.0, 1.0, 0.0]))
        )
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
    def _normalise(vector: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(vector)
        return vector / magnitude if magnitude > 1e-8 else vector

    @staticmethod
    def _projection_matrix(aspect_ratio: float) -> np.ndarray:
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
        glfw = self.glfw
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_I:
            try:
                self._set_canopy_mode(
                    "impostor" if self.canopy_mode == "polygons" else "polygons"
                )
            except Exception as error:
                self.status_message = f"Impostor switch failed: {error}"
                print(self.status_message)
        elif key == glfw.KEY_L:
            self.show_leaves = not self.show_leaves
            self._set_canopy_mode(self.canopy_mode)
        elif key == glfw.KEY_W:
            self.wireframe = not self.wireframe
        elif key == glfw.KEY_V:
            self.vsync = not self.vsync
            glfw.swap_interval(1 if self.vsync else 0)
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
            self.max_depth = key - glfw.KEY_1
            self._upload_wood_mesh()
            self._set_canopy_mode(self.canopy_mode)
        elif key == glfw.KEY_R:
            previous_mode = self.canopy_mode
            self.tree = self.regenerate_callback()
            if self.rebake_callback is not None:
                self.status_message = f"Baking impostor for seed {self.tree.metadata.get('seed', '?')}..."
                self.impostor_directory = Path(self.rebake_callback(self.tree))
            self.max_depth = 4
            self.show_leaves = True
            self._upload_wood_mesh()
            self.source_wood_triangles = self.wood_triangle_count
            self.model_leaf_triangles = len(self.tree.leaves) * (2 * len(OAK_LEAF_PROFILE) - 1)
            self.model_total_triangles = self.source_wood_triangles + self.model_leaf_triangles
            self._set_canopy_mode(previous_mode)
            self.status_message = f"Regenerated seed {self.tree.metadata.get('seed', '?')}"
        elif key == glfw.KEY_F:
            self.camera_yaw = math.radians(38.0)
            self.camera_pitch = math.radians(17.0)
            self.camera_distance = self.default_camera_distance
            self.camera_target = self.default_camera_target.copy()

    def _on_mouse_button(self, window: Any, button: int, action: int, modifiers: int) -> None:
        if button == self.glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == self.glfw.PRESS
            self.last_cursor_position = None

    def _on_cursor_move(self, window: Any, cursor_x: float, cursor_y: float) -> None:
        if not self.dragging:
            return
        if self.last_cursor_position is not None:
            previous_x, previous_y = self.last_cursor_position
            self.camera_yaw += (cursor_x - previous_x) * 0.006
            self.camera_pitch += (cursor_y - previous_y) * 0.006
            self.camera_pitch = float(np.clip(self.camera_pitch, -0.15, 1.25))
        self.last_cursor_position = (cursor_x, cursor_y)

    def _on_scroll(self, window: Any, horizontal_offset: float, vertical_offset: float) -> None:
        self.camera_distance *= math.exp(-vertical_offset * 0.10)
        self.camera_distance = float(np.clip(self.camera_distance, 3.5, 90.0))

    def _print_controls(self) -> None:
        print("\nOakForge viewer controls:")
        print("  Drag with left mouse button : orbit camera")
        print("  Mouse wheel                 : zoom")
        print("  I                           : toggle polygon leaves / depth impostor")
        print("  L                           : toggle canopy visibility")
        print("  1 / 2 / 3 / 4 / 5           : trunk / primary / secondary / tertiary / full tree")
        print("  W                           : toggle polygon wireframe")
        print("  V                           : toggle VSync (off by default for benchmarking)")
        print("  R                           : regenerate tree and re-bake its canopy")
        print("  F                           : reset camera")
        print("  Escape                      : close viewer\n")

    def run(self) -> None:
        gl = self.gl
        # Exclude procedural generation, atlas baking and GPU uploads from the
        # first displayed frame-time/FPS sample.
        self.previous_frame_time = time.perf_counter()
        try:
            while not self.glfw.window_should_close(self.window):
                frame_start = time.perf_counter()
                delta = frame_start - self.previous_frame_time
                self.previous_frame_time = frame_start
                if delta > 0:
                    instantaneous_ms = delta * 1000.0
                    self.frame_ms = (
                        instantaneous_ms if self.frame_ms <= 0
                        else self.frame_ms * 0.88 + instantaneous_ms * 0.12
                    )
                    self.fps = 1000.0 / self.frame_ms
                self.frame_count += 1
                self._poll_gpu_queries()

                framebuffer_width, framebuffer_height = self.glfw.get_framebuffer_size(self.window)
                gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                model_matrix = np.identity(4, dtype=np.float32)
                view_matrix = self._view_matrix()
                projection_matrix = self._projection_matrix(
                    framebuffer_width / max(framebuffer_height, 1)
                )
                view_projection = projection_matrix @ view_matrix

                gl.glUseProgram(self.program)
                for uniform_name, matrix in (
                    ("model_matrix", model_matrix),
                    ("view_matrix", view_matrix),
                    ("projection_matrix", projection_matrix),
                ):
                    gl.glUniformMatrix4fv(
                        self.tree_uniforms[uniform_name], 1, gl.GL_TRUE, matrix
                    )

                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glBindVertexArray(self.grid_vao)
                gl.glDrawArrays(gl.GL_LINES, 0, self.grid_vertex_count)

                self._begin_gpu_query()
                gl.glPolygonMode(
                    gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL
                )
                gl.glUseProgram(self.program)
                gl.glBindVertexArray(self.wood_vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.wood_vertex_count)
                if self.show_leaves and self.max_depth >= 4:
                    if self.canopy_mode == "polygons":
                        gl.glBindVertexArray(self.canopy_vao)
                        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.canopy_vertex_count)
                    elif self.canopy_renderer is not None:
                        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                        direction = self._camera_position() - self.canopy_renderer.centre
                        self.last_selected_view = self.canopy_renderer.draw(
                            view_projection, direction
                        )
                gl.glBindVertexArray(0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                self._end_gpu_query()

                self._render_overlay(framebuffer_width, framebuffer_height)
                self.glfw.swap_buffers(self.window)
                self.glfw.poll_events()
        finally:
            self._release_canopy_representation()
            self.gpu_telemetry.close()
            if self.gpu_queries:
                gl.glDeleteQueries(len(self.gpu_queries), self.gpu_queries)
            for resource in (
                self.wood_vao, self.grid_vao, self.overlay_vao,
            ):
                if resource:
                    gl.glDeleteVertexArrays(1, [resource])
            for resource in (
                self.wood_vbo, self.grid_vbo, self.overlay_vbo,
            ):
                if resource:
                    gl.glDeleteBuffers(1, [resource])
            if self.overlay_texture:
                gl.glDeleteTextures(1, [self.overlay_texture])
            if self.program:
                gl.glDeleteProgram(self.program)
            if self.overlay_program:
                gl.glDeleteProgram(self.overlay_program)
            self.glfw.destroy_window(self.window)
            self.glfw.terminate()
