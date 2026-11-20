"""Runtime rendering for OakForge multi-view canopy depth impostors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oakforge.render.impostor import view_basis


IMPOSTOR_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texture_coordinate;
uniform mat4 view_projection;
uniform vec3 canopy_centre;
uniform vec3 view_right;
uniform vec3 view_up;
uniform float canopy_extent;
out vec2 local_uv;
void main() {
    local_uv = texture_coordinate;
    vec3 plane_position = canopy_centre
        + view_right * (position.x * canopy_extent)
        + view_up * (position.y * canopy_extent);
    gl_Position = view_projection * vec4(plane_position, 1.0);
}
"""

IMPOSTOR_FRAGMENT_SHADER = """#version 330 core
in vec2 local_uv;
layout(location = 0) out vec4 fragment_colour;
uniform sampler2D colour_atlas;
uniform sampler2D normal_atlas;
uniform sampler2D depth_atlas;
uniform vec4 atlas_rect;
uniform vec3 canopy_centre;
uniform vec3 view_right;
uniform vec3 view_up;
uniform vec3 view_forward;
uniform float canopy_extent;
uniform float depth_near;
uniform float depth_far;
uniform mat4 view_projection;

void main() {
    vec2 atlas_uv = atlas_rect.xy + local_uv * atlas_rect.zw;
    vec4 colour_sample = texture(colour_atlas, atlas_uv);
    if (colour_sample.a < 0.5) discard;

    vec3 packed_depth = texture(depth_atlas, atlas_uv).rgb;
    float depth_value = dot(packed_depth, vec3(65536.0, 256.0, 1.0)) / 16777215.0;
    float depth_offset = depth_far * 0.5
        - (depth_near + depth_value * (depth_far - depth_near));

    vec3 world_position = canopy_centre
        + view_right * ((local_uv.x - 0.5) * canopy_extent)
        + view_up * ((local_uv.y - 0.5) * canopy_extent)
        + view_forward * depth_offset;

    vec4 clip_position = view_projection * vec4(world_position, 1.0);
    if (clip_position.w <= 0.0) discard;
    gl_FragDepth = clamp((clip_position.z / clip_position.w) * 0.5 + 0.5, 0.0, 1.0);

    vec3 normal_direction = normalize(texture(normal_atlas, atlas_uv).rgb * 2.0 - 1.0);
    vec3 light_direction = normalize(vec3(-0.45, 0.82, 0.36));
    float diffuse_amount = max(dot(normal_direction, light_direction), 0.0);
    float ambient_amount = 0.34;
    vec3 shaded_colour = colour_sample.rgb * (ambient_amount + diffuse_amount * 0.72);
    fragment_colour = vec4(shaded_colour, colour_sample.a);
}
"""


def nearest_view_index(views: list[dict[str, Any]], direction: np.ndarray) -> int:
    """Return the baked view whose direction best matches the requested camera."""
    if not views:
        raise ValueError("The impostor manifest does not contain any views")
    normalised = np.asarray(direction, dtype=np.float64)
    normalised /= max(float(np.linalg.norm(normalised)), 1e-12)
    view_directions = np.asarray([view["direction"] for view in views], dtype=np.float64)
    view_lengths = np.linalg.norm(view_directions, axis=1, keepdims=True)
    view_directions /= np.maximum(view_lengths, 1e-12)
    return int(np.argmax(view_directions @ normalised))


def reconstruct_depth_offset(depth_value: float, near: float, far: float) -> float:
    """Convert the baker's OpenGL orthographic depth back to canopy-local depth."""
    return float(far * 0.5 - (near + float(depth_value) * (far - near)))


def _compile_program(gl, vertex_source: str, fragment_source: str) -> int:
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
            raise RuntimeError(f"Canopy impostor shader compilation failed: {message}")
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
        raise RuntimeError(f"Canopy impostor shader link failed: {message}")
    for shader in shaders:
        gl.glDetachShader(program, shader)
        gl.glDeleteShader(shader)
    return int(program)


class DepthImpostorRenderer:
    """Upload and render a canopy atlas using a depth-reconstructed view quad."""

    def __init__(self, gl, asset_directory: str | Path) -> None:
        self.gl = gl
        self.asset_directory = Path(asset_directory)
        manifest_path = self.asset_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Canopy impostor manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "oakforge-canopy-depth-impostor-v1":
            raise ValueError("Unsupported canopy impostor format")
        if not self.manifest.get("views"):
            raise ValueError("Canopy impostor has no capture views")

        bounds = self.manifest["bounds"]
        depth_projection = self.manifest.get("depth_projection")
        if not depth_projection:
            raise ValueError("Canopy impostor manifest is missing depth projection metadata")
        self.centre = np.asarray(bounds["centre"], dtype=np.float32)
        self.extent = float(bounds["extent"])
        self.near = float(depth_projection["near"])
        self.far = float(depth_projection["far"])
        self.views = self.manifest["views"]
        self.program = _compile_program(gl, IMPOSTOR_VERTEX_SHADER, IMPOSTOR_FRAGMENT_SHADER)
        self.textures: dict[str, int] = {}
        self.texture_bytes = 0
        self.vao = 0
        self.vbo = 0
        self._upload_atlases()
        self._upload_quad()

    def _load_pixels(self, filename: str, mode: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Install Pillow to load canopy impostor atlases") from error
        path = self.asset_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Canopy impostor texture not found: {path}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert(mode), dtype=np.uint8)
        return np.ascontiguousarray(np.flipud(pixels))

    def _upload_atlases(self) -> None:
        gl = self.gl
        specifications = (
            ("colour", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_LINEAR),
            ("normal", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_LINEAR),
            ("depth", "RGB", gl.GL_RGB8, gl.GL_RGB, gl.GL_NEAREST),
        )
        for name, mode, internal_format, pixel_format, filtering in specifications:
            pixels = self._load_pixels(self.manifest["textures"][name], mode)
            height, width, channels = pixels.shape
            texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                pixel_format, gl.GL_UNSIGNED_BYTE, pixels,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filtering)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filtering)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            self.textures[name] = int(texture)
            self.texture_bytes += width * height * channels
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _upload_quad(self) -> None:
        gl = self.gl
        # Local UV is bottom-left based; manifest UVs are PNG top-left based.
        vertices = np.asarray([
            [-0.5, -0.5, 0.0, 0.0],
            [ 0.5, -0.5, 1.0, 0.0],
            [ 0.5,  0.5, 1.0, 1.0],
            [-0.5, -0.5, 0.0, 0.0],
            [ 0.5,  0.5, 1.0, 1.0],
            [-0.5,  0.5, 0.0, 1.0],
        ], dtype=np.float32)
        self.vao = int(gl.glGenVertexArrays(1))
        self.vbo = int(gl.glGenBuffers(1))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        stride = 4 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(8))
        gl.glBindVertexArray(0)
        self.quad_bytes = int(vertices.nbytes)

    def draw(self, view_projection: np.ndarray, camera_direction: np.ndarray) -> int:
        """Render one depth-reconstructed quad and return the selected atlas view."""
        gl = self.gl
        view_index = nearest_view_index(self.views, camera_direction)
        view = self.views[view_index]
        direction = np.asarray(view["direction"], dtype=np.float32)
        right, up, forward = view_basis(direction)
        rect_x, rect_y_top, rect_width, rect_height = view["uv_rect"]
        rect_y_bottom = 1.0 - float(rect_y_top) - float(rect_height)

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.program, "view_projection"), 1, gl.GL_TRUE, view_projection
        )
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "canopy_centre"), *self.centre)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "view_right"), *right)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "view_up"), *up)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "view_forward"), *forward)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "canopy_extent"), self.extent)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "depth_near"), self.near)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "depth_far"), self.far)
        gl.glUniform4f(
            gl.glGetUniformLocation(self.program, "atlas_rect"),
            float(rect_x), rect_y_bottom, float(rect_width), float(rect_height),
        )
        for unit, name, uniform_name in (
            (0, "colour", "colour_atlas"),
            (1, "normal", "normal_atlas"),
            (2, "depth", "depth_atlas"),
        ):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[name])
            gl.glUniform1i(gl.glGetUniformLocation(self.program, uniform_name), unit)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        return view_index

    @property
    def allocated_bytes(self) -> int:
        return self.texture_bytes + self.quad_bytes

    def close(self) -> None:
        gl = self.gl
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.textures:
            gl.glDeleteTextures(list(self.textures.values()))
            self.textures.clear()
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = 0
