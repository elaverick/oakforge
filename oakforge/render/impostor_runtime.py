"""Forward-projected layered depth impostor rendering for tree canopies.

The runtime treats every valid captured texel as a real, depth-bearing sample.
It reconstructs that sample in canopy-local space, projects it into the active
camera, and rasterises it as a small point splat. The scene depth buffer resolves
occlusion between layers. No inverse texture search or cross-view position
averaging is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oakforge.render.impostor import view_basis


MAX_DEPTH_LAYERS = 4


IMPOSTOR_VERTEX_SHADER = """#version 330 core
uniform sampler2DArray colour_atlas;
uniform sampler2DArray normal_atlas;
uniform sampler2DArray depth_atlas;

uniform mat4 view_projection;
uniform vec2 viewport_size;
uniform vec3 canopy_centre;
uniform vec3 capture_right;
uniform vec3 capture_up;
uniform vec3 capture_forward;
uniform float canopy_extent;
uniform float depth_near;
uniform float depth_far;
uniform float splat_scale;
uniform int tile_size;
uniform int layer_count;
uniform int view_index;

out vec3 surface_colour;
out vec3 surface_normal;
flat out vec3 sample_local_position;
flat out vec3 sample_footprint_x;
flat out vec3 sample_footprint_y;
flat out vec3 sample_identity;

vec3 reconstruct_pixel(ivec2 pixel, int slice_index) {
    float depth_value = texelFetch(
        depth_atlas, ivec3(pixel, slice_index), 0
    ).r;
    float depth_offset = depth_far * 0.5
        - (depth_near + depth_value * (depth_far - depth_near));
    vec2 uv = (vec2(pixel) + vec2(0.5)) / float(tile_size);
    return canopy_centre
        + capture_right * ((uv.x - 0.5) * canopy_extent)
        + capture_up * ((uv.y - 0.5) * canopy_extent)
        + capture_forward * depth_offset;
}

void main() {
    ivec2 pixel = ivec2(
        gl_VertexID % tile_size,
        gl_VertexID / tile_size
    );
    int layer_index = gl_InstanceID;
    int slice_index = view_index * layer_count + layer_index;
    vec4 colour_sample = texelFetch(
        colour_atlas, ivec3(pixel, slice_index), 0
    );

    // Empty texels must stay empty. In particular, never fill them by
    // searching for an unrelated nearby leaf in the atlas.
    if (colour_sample.a < 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        surface_colour = vec3(0.0);
        surface_normal = vec3(0.0, 1.0, 0.0);
        sample_local_position = vec3(0.0);
        sample_footprint_x = vec3(0.0);
        sample_footprint_y = vec3(0.0);
        sample_identity = vec3(vec2(pixel), float(layer_index));
        return;
    }

    vec3 world_position = reconstruct_pixel(pixel, slice_index);
    vec4 clip_position = view_projection * vec4(world_position, 1.0);
    if (clip_position.w <= 0.00001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        surface_colour = vec3(0.0);
        surface_normal = vec3(0.0, 1.0, 0.0);
        sample_local_position = vec3(0.0);
        sample_footprint_x = vec3(0.0);
        sample_footprint_y = vec3(0.0);
        sample_identity = vec3(vec2(pixel), float(layer_index));
        return;
    }
    gl_Position = clip_position;
    surface_colour = colour_sample.rgb;
    surface_normal = normalize(
        texelFetch(normal_atlas, ivec3(pixel, slice_index), 0).rgb * 2.0 - 1.0
    );
    sample_local_position = world_position - canopy_centre;
    sample_identity = vec3(vec2(pixel), float(layer_index));

    // Estimate the projected footprint of this capture texel. At a grazing
    // angle, source pixels spread over more screen pixels; splatting that
    // footprint prevents the dotted, disconnected appearance of point samples.
    int step_x = (pixel.x + 1 < tile_size) ? 1 : -1;
    int step_y = (pixel.y + 1 < tile_size) ? 1 : -1;
    ivec2 neighbour_x = clamp(
        pixel + ivec2(step_x, 0), ivec2(0), ivec2(tile_size - 1)
    );
    ivec2 neighbour_y = clamp(
        pixel + ivec2(0, step_y), ivec2(0), ivec2(tile_size - 1)
    );

    vec3 world_x;
    vec3 world_y;
    if (texelFetch(colour_atlas, ivec3(neighbour_x, slice_index), 0).a >= 0.5) {
        world_x = reconstruct_pixel(neighbour_x, slice_index);
    } else {
        world_x = world_position
            + capture_right * (float(step_x) * canopy_extent / float(tile_size));
    }
    if (texelFetch(colour_atlas, ivec3(neighbour_y, slice_index), 0).a >= 0.5) {
        world_y = reconstruct_pixel(neighbour_y, slice_index);
    } else {
        world_y = world_position
            + capture_up * (float(step_y) * canopy_extent / float(tile_size));
    }

    sample_footprint_x = world_x - world_position;
    sample_footprint_y = world_y - world_position;
    vec4 clip_x = view_projection * vec4(world_x, 1.0);
    vec4 clip_y = view_projection * vec4(world_y, 1.0);
    float footprint_x = 1.0;
    float footprint_y = 1.0;
    if (clip_x.w > 0.00001) {
        footprint_x = length(
            (clip_x.xy / clip_x.w - clip_position.xy / clip_position.w)
            * viewport_size * 0.5
        );
    }
    if (clip_y.w > 0.00001) {
        footprint_y = length(
            (clip_y.xy / clip_y.w - clip_position.xy / clip_position.w)
            * viewport_size * 0.5
        );
    }
    gl_PointSize = clamp(
        max(footprint_x, footprint_y) * 1.15 * splat_scale, 1.0, 5.5
    );
}
"""


IMPOSTOR_FRAGMENT_SHADER = """#version 330 core
in vec3 surface_colour;
in vec3 surface_normal;
flat in vec3 sample_local_position;
flat in vec3 sample_footprint_x;
flat in vec3 sample_footprint_y;
flat in vec3 sample_identity;

uniform int shape_mode;
uniform int polygon_sides;
uniform float angularity;
uniform float noise_intensity;
uniform float noise_scale;

out vec4 fragment_colour;

float hash12(vec2 point) {
    vec3 value = fract(vec3(point.xyx) * 0.1031);
    value += dot(value, value.yzx + 33.33);
    return fract((value.x + value.y) * value.z);
}

float value_noise(vec2 point) {
    vec2 cell = floor(point);
    vec2 local = fract(point);
    local = local * local * (3.0 - 2.0 * local);
    float lower = mix(
        hash12(cell),
        hash12(cell + vec2(1.0, 0.0)),
        local.x
    );
    float upper = mix(
        hash12(cell + vec2(0.0, 1.0)),
        hash12(cell + vec2(1.0, 1.0)),
        local.x
    );
    return mix(lower, upper, local.y);
}

void main() {
    vec2 point = (gl_PointCoord - vec2(0.5)) * 2.0;
    float radius = length(point);

    // The square mode is the original point-splat renderer. Round and angular
    // modes shape the footprint; foliage mode additionally breaks up its edge
    // with object-space noise. None of these modes changes reconstructed depth.
    if (shape_mode == 1) {
        if (radius > 1.0) discard;
    } else if (shape_mode >= 2) {
        int side_offset = int(floor(
            hash12(sample_identity.xy + vec2(sample_identity.z * 3.1, 7.7)) * 3.0
        )) - 1;
        int sides = clamp(polygon_sides + side_offset, 4, 10);
        float sector = 6.28318530718 / float(sides);
        float angle = atan(point.y, point.x)
            + hash12(sample_identity.xy + vec2(19.1, sample_identity.z)) * 6.28318530718;
        float sector_angle = mod(angle + sector * 0.5, sector) - sector * 0.5;
        float polygon_limit = cos(sector * 0.5) / max(cos(sector_angle), 0.1);

        float wave_a = sin(angle * 3.0 + sample_identity.x * 0.071);
        float wave_b = sin(angle * 5.0 + sample_identity.y * 0.053
            + sample_identity.z * 1.7);
        float irregularity = angularity * (0.09 * wave_a + 0.045 * wave_b);
        polygon_limit *= 1.0 + irregularity;

        if (shape_mode == 3 && noise_intensity > 0.001) {
            vec3 local_position = sample_local_position
                + (gl_PointCoord.x - 0.5) * sample_footprint_x
                + (gl_PointCoord.y - 0.5) * sample_footprint_y;
            // An oblique 2-D projection of canopy-local XYZ keeps the pattern
            // attached to the tree while avoiding screen-space crawling.
            vec2 noise_position = vec2(
                dot(local_position, vec3(1.0, 0.31, 0.17)),
                dot(local_position, vec3(-0.23, 0.41, 1.0))
            );
            float broad_noise = value_noise(noise_position * noise_scale);
            float fine_noise = value_noise(
                noise_position * noise_scale * 2.35 + vec2(17.0, 9.0)
            );
            float foliage_noise = mix(broad_noise, fine_noise, 0.32);
            polygon_limit *= 1.0
                + (foliage_noise - 0.5) * noise_intensity * 0.72;
        }

        if (radius > polygon_limit) discard;
    }

    vec3 normal_direction = normalize(surface_normal);
    vec3 light_direction = normalize(vec3(-0.45, 0.82, 0.36));
    float diffuse_amount = max(dot(normal_direction, light_direction), 0.0);
    float ambient_amount = 0.34;
    vec3 shaded_colour = surface_colour
        * (ambient_amount + diffuse_amount * 0.72);
    fragment_colour = vec4(shaded_colour, 1.0);
}
"""


IMPOSTOR_STYLES = ("square", "round", "angular", "foliage")


def nearest_view_index(views: list[dict[str, Any]], direction: np.ndarray) -> int:
    """Return the capture direction most closely aligned with the camera."""
    if not views:
        raise ValueError("The impostor manifest does not contain any views")
    normalised = np.asarray(direction, dtype=np.float64)
    normalised /= max(float(np.linalg.norm(normalised)), 1e-12)
    directions = np.asarray([view["direction"] for view in views], dtype=np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return int(np.argmax(directions @ normalised))


def reconstruct_depth_offset(depth_value: float, near: float, far: float) -> float:
    """Convert normalised orthographic depth to displacement from canopy centre."""
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
            for compiled in shaders:
                gl.glDeleteShader(compiled)
            raise RuntimeError(f"Canopy point-splat shader compilation failed: {message}")
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
        raise RuntimeError(f"Canopy point-splat shader link failed: {message}")
    for shader in shaders:
        gl.glDetachShader(program, shader)
        gl.glDeleteShader(shader)
    return int(program)


class DepthImpostorRenderer:
    """Render one coherent layered capture as forward-projected depth splats."""

    def __init__(self, gl, asset_directory: str | Path) -> None:
        self.gl = gl
        self.style = "foliage"
        self.angularity = 0.65
        self.noise_intensity = 0.6
        self.noise_scale = 8.0
        self.polygon_sides = 6
        self.splat_scale = 1.3
        self.asset_directory = Path(asset_directory)
        manifest_path = self.asset_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Canopy impostor manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "oakforge-canopy-layered-depth-impostor-v3":
            raise ValueError(
                "Unsupported canopy impostor format. Re-bake the canopy using the layered v3 baker."
            )
        self.views = self.manifest.get("views", [])
        if len(self.views) < 1:
            raise ValueError("Canopy impostor has no capture views")

        bounds = self.manifest["bounds"]
        depth_projection = self.manifest.get("depth_projection")
        if not depth_projection:
            raise ValueError("Canopy impostor manifest is missing depth projection metadata")
        self.centre = np.asarray(bounds["centre"], dtype=np.float32)
        self.bounds_min = np.asarray(bounds["min"], dtype=np.float32)
        self.bounds_max = np.asarray(bounds["max"], dtype=np.float32)
        self.extent = float(bounds["extent"])
        self.near = float(depth_projection["near"])
        self.far = float(depth_projection["far"])
        self.view_directions = np.asarray(
            [view["direction"] for view in self.views], dtype=np.float64
        )
        self.view_directions /= np.maximum(
            np.linalg.norm(self.view_directions, axis=1, keepdims=True), 1e-12
        )
        self.view_bases = [
            view_basis(direction) for direction in self.view_directions
        ]

        atlas = self.manifest["atlas"]
        self.tile_size = int(atlas["tile_size"])
        self.atlas_columns = int(atlas["columns"])
        self.atlas_padding = int(atlas["padding"])
        self.tile_pitch = int(atlas["tile_pitch"])
        self.layer_count = int(self.manifest["depth_layers"]["count"])
        if not 1 <= self.layer_count <= MAX_DEPTH_LAYERS:
            raise ValueError("The runtime supports between one and four depth layers")
        max_array_layers = int(gl.glGetIntegerv(gl.GL_MAX_ARRAY_TEXTURE_LAYERS))
        if len(self.views) * self.layer_count > max_array_layers:
            raise RuntimeError(
                f"Layered canopy needs {len(self.views) * self.layer_count} texture-array "
                f"slices, but this GPU supports {max_array_layers}"
            )
        texture_files = self.manifest["textures"]
        for key in ("colour", "normal", "depth"):
            if len(texture_files.get(key, [])) != self.layer_count:
                raise ValueError(f"Layered impostor manifest has incomplete {key} textures")

        self.program = _compile_program(
            gl, IMPOSTOR_VERTEX_SHADER, IMPOSTOR_FRAGMENT_SHADER
        )
        uniform_names = (
            "colour_atlas", "normal_atlas", "depth_atlas",
            "view_projection", "viewport_size", "canopy_centre",
            "capture_right", "capture_up", "capture_forward",
            "canopy_extent", "depth_near", "depth_far",
            "tile_size", "layer_count", "view_index", "splat_scale", "shape_mode", "polygon_sides",
            "angularity", "noise_intensity", "noise_scale",
        )
        self.uniforms = {
            name: int(gl.glGetUniformLocation(self.program, name))
            for name in uniform_names
        }
        missing = [name for name, location in self.uniforms.items() if location < 0]
        if missing:
            gl.glDeleteProgram(self.program)
            raise RuntimeError("Canopy point-splat shader is missing uniforms: " + ", ".join(missing))

        self.textures: dict[str, int] = {}
        self.texture_bytes = 0
        self.vao = 0
        self.point_count = self.tile_size * self.tile_size * self.layer_count
        self._upload_texture_arrays()
        self._create_vertex_array()

    def set_style(self, style: str) -> None:
        """Select square, round, angular, or noise-broken foliage splats."""
        if style not in IMPOSTOR_STYLES:
            raise ValueError(f"Unknown impostor foliage style: {style}")
        self.style = style

    def adjust_noise_intensity(self, delta: float) -> float:
        self.noise_intensity = float(np.clip(self.noise_intensity + delta, 0.0, 1.0))
        return self.noise_intensity

    def adjust_noise_scale(self, delta: float) -> float:
        self.noise_scale = float(np.clip(self.noise_scale + delta, 0.5, 32.0))
        return self.noise_scale

    def adjust_angularity(self, delta: float) -> float:
        self.angularity = float(np.clip(self.angularity + delta, 0.0, 1.0))
        return self.angularity

    def _load_array_slices(self, filenames: list[str], mode: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Install Pillow to load canopy impostor atlases") from error

        slice_count = len(self.views) * self.layer_count
        if mode == "RGBA":
            result = np.empty(
                (slice_count, self.tile_size, self.tile_size, 4), dtype=np.uint8
            )
        else:
            result = np.empty(
                (slice_count, self.tile_size, self.tile_size), dtype=np.uint16
            )

        expected_height = int(self.manifest["atlas"]["height"])
        expected_width = int(self.manifest["atlas"]["width"])
        for layer_index, filename in enumerate(filenames):
            path = self.asset_directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Canopy impostor texture not found: {path}")
            with Image.open(path) as image:
                if mode == "RGBA":
                    atlas_pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                else:
                    atlas_pixels = np.asarray(image).astype(np.uint16, copy=False)
            if atlas_pixels.shape[0] != expected_height or atlas_pixels.shape[1] != expected_width:
                raise ValueError(f"Unexpected dimensions in canopy impostor texture: {path}")

            for view_index in range(len(self.views)):
                column = view_index % self.atlas_columns
                row = view_index // self.atlas_columns
                x = column * self.tile_pitch + self.atlas_padding
                y = row * self.tile_pitch + self.atlas_padding
                tile = np.flipud(
                    atlas_pixels[y:y + self.tile_size, x:x + self.tile_size]
                )
                result[view_index * self.layer_count + layer_index] = tile
        return np.ascontiguousarray(result)

    def _upload_texture_arrays(self) -> None:
        gl = self.gl
        specifications = (
            ("colour", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
            ("normal", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
            ("depth", "R16", gl.GL_R16, gl.GL_RED, gl.GL_UNSIGNED_SHORT),
        )
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        for name, mode, internal_format, pixel_format, pixel_type in specifications:
            slices = self._load_array_slices(self.manifest["textures"][name], mode)
            texture = int(gl.glGenTextures(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, texture)
            gl.glTexImage3D(
                gl.GL_TEXTURE_2D_ARRAY, 0, internal_format,
                self.tile_size, self.tile_size, len(slices), 0,
                pixel_format, pixel_type, slices,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_BASE_LEVEL, 0)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAX_LEVEL, 0)
            self.textures[name] = texture
            self.texture_bytes += int(slices.nbytes)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)

        gl.glUseProgram(self.program)
        gl.glUniform1i(self.uniforms["colour_atlas"], 0)
        gl.glUniform1i(self.uniforms["normal_atlas"], 1)
        gl.glUniform1i(self.uniforms["depth_atlas"], 2)
        gl.glUniform1i(self.uniforms["tile_size"], self.tile_size)
        gl.glUniform1i(self.uniforms["layer_count"], self.layer_count)
        gl.glUseProgram(0)

    def _create_vertex_array(self) -> None:
        # The vertex shader derives its source pixel from gl_VertexID and its
        # depth layer from gl_InstanceID, so no large per-vertex buffer is needed.
        self.vao = int(self.gl.glGenVertexArrays(1))
        self.vao_bytes = 0

    def draw(
        self,
        view_projection: np.ndarray,
        camera_position: np.ndarray,
        viewport_size: tuple[int, int],
    ) -> list[int]:
        """Forward-project all valid texels from the nearest coherent capture view."""
        gl = self.gl
        camera_direction = np.asarray(camera_position, dtype=np.float64) - self.centre
        camera_direction /= max(float(np.linalg.norm(camera_direction)), 1e-12)
        view_index = int(np.argmax(self.view_directions @ camera_direction))
        view_right, view_up, view_forward = self.view_bases[view_index]

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(
            self.uniforms["view_projection"], 1, gl.GL_TRUE, view_projection
        )
        gl.glUniform2f(
            self.uniforms["viewport_size"],
            float(max(viewport_size[0], 1)), float(max(viewport_size[1], 1)),
        )
        gl.glUniform3f(self.uniforms["canopy_centre"], *self.centre)
        gl.glUniform3f(self.uniforms["capture_right"], *view_right)
        gl.glUniform3f(self.uniforms["capture_up"], *view_up)
        gl.glUniform3f(self.uniforms["capture_forward"], *view_forward)
        gl.glUniform1f(self.uniforms["canopy_extent"], self.extent)
        gl.glUniform1f(self.uniforms["depth_near"], self.near)
        gl.glUniform1f(self.uniforms["depth_far"], self.far)
        gl.glUniform1i(self.uniforms["view_index"], view_index)
        gl.glUniform1f(self.uniforms["splat_scale"], self.splat_scale)
        gl.glUniform1i(self.uniforms["shape_mode"], IMPOSTOR_STYLES.index(self.style))
        gl.glUniform1i(self.uniforms["polygon_sides"], self.polygon_sides)
        gl.glUniform1f(self.uniforms["angularity"], self.angularity)
        gl.glUniform1f(self.uniforms["noise_intensity"], self.noise_intensity)
        gl.glUniform1f(self.uniforms["noise_scale"], self.noise_scale)

        for unit, name in enumerate(("colour", "normal", "depth")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.textures[name])

        gl.glBindVertexArray(self.vao)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDrawArraysInstanced(
            gl.GL_POINTS, 0, self.tile_size * self.tile_size, self.layer_count
        )
        gl.glBindVertexArray(0)

        for unit in range(3):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        return [view_index]

    @property
    def allocated_bytes(self) -> int:
        return self.texture_bytes + self.vao_bytes

    def close(self) -> None:
        gl = self.gl
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.textures:
            for unit in range(3):
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glDeleteTextures(len(self.textures), list(self.textures.values()))
            self.textures.clear()
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = 0
