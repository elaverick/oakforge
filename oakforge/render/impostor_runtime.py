"""Runtime rendering for layered, parallax-corrected canopy depth impostors."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np

from oakforge.render.impostor import view_basis


MAX_DEPTH_LAYERS = 4
BLENDED_CAPTURE_VIEWS = 3


IMPOSTOR_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 ndc_position;
void main() {
    gl_Position = vec4(ndc_position, 0.0, 1.0);
}
"""


IMPOSTOR_FRAGMENT_SHADER = """#version 330 core
out vec4 fragment_colour;

uniform sampler2DArray colour_atlas;
uniform sampler2DArray normal_atlas;
uniform sampler2DArray depth_atlas;

uniform mat4 view_projection;
uniform mat4 inverse_view_projection;
uniform vec2 viewport_size;
uniform vec3 camera_position;
uniform vec3 canopy_centre;
uniform float canopy_extent;
uniform float depth_near;
uniform float depth_far;
uniform float texel_size;
uniform int layer_count;

uniform vec3 capture_rights[3];
uniform vec3 capture_ups[3];
uniform vec3 capture_forwards[3];
uniform int capture_indices[3];
uniform float capture_weights[3];
uniform float capture_search_radii[3];

struct SurfaceSample {
    vec3 world_position;
    vec4 colour;
    vec3 normal;
    float ray_distance;
    float valid;
};

vec3 reconstruct_point(
    vec2 uv, float slice_index, vec3 right, vec3 up, vec3 forward
) {
    float depth_value = texture(depth_atlas, vec3(uv, slice_index)).r;
    float depth_offset = depth_far * 0.5
        - (depth_near + depth_value * (depth_far - depth_near));
    return canopy_centre
        + right * ((uv.x - 0.5) * canopy_extent)
        + up * ((uv.y - 0.5) * canopy_extent)
        + forward * depth_offset;
}

vec2 project_point(vec3 world_position, out float clip_w) {
    vec4 clip = view_projection * vec4(world_position, 1.0);
    clip_w = clip.w;
    if (clip.w <= 0.00001) return vec2(1e6);
    return clip.xy / clip.w;
}

SurfaceSample empty_sample() {
    SurfaceSample result;
    result.world_position = vec3(0.0);
    result.colour = vec4(0.0);
    result.normal = vec3(0.0, 1.0, 0.0);
    result.ray_distance = 1e30;
    result.valid = 0.0;
    return result;
}

SurfaceSample sample_capture_view(
    int slot, vec3 ray_origin, vec3 ray_direction, vec2 target_ndc
) {
    SurfaceSample best = empty_sample();
    vec3 right = capture_rights[slot];
    vec3 up = capture_ups[slot];
    vec3 forward = capture_forwards[slot];
    float denominator = dot(ray_direction, forward);
    if (abs(denominator) < 0.0001) return best;

    // Intersect the current camera ray with the capture's centre plane. This
    // supplies a good initial UV; the iterations below undo depth parallax.
    float plane_t = dot(canopy_centre - ray_origin, forward) / denominator;
    if (plane_t <= 0.0) return best;
    vec3 plane_hit = ray_origin + ray_direction * plane_t;
    vec2 initial_uv = vec2(
        dot(plane_hit - canopy_centre, right),
        dot(plane_hit - canopy_centre, up)
    ) / canopy_extent + vec2(0.5);

    int view_index = capture_indices[slot];
    float search_radius = capture_search_radii[slot];

    for (int layer = 0; layer < 4; ++layer) {
        if (layer >= layer_count) break;
        float slice_index = float(view_index * layer_count + layer);
        vec2 uv = initial_uv;
        bool valid = true;
        vec4 colour = texture(colour_atlas, vec3(uv, slice_index));

        // A parallax-shifted silhouette can start just outside its original
        // alpha coverage. Search a small direction-dependent neighbourhood
        // before abandoning this layer.
        if (colour.a < 0.5) {
            float nearest_offset = 1e30;
            bool found = false;
            for (int sy = -1; sy <= 1; ++sy) {
                for (int sx = -1; sx <= 1; ++sx) {
                    if (sx == 0 && sy == 0) continue;
                    vec2 candidate_uv = initial_uv
                        + vec2(float(sx), float(sy)) * search_radius;
                    if (any(lessThan(candidate_uv, vec2(0.001)))
                        || any(greaterThan(candidate_uv, vec2(0.999)))) continue;
                    vec4 candidate_colour = texture(
                        colour_atlas, vec3(candidate_uv, slice_index)
                    );
                    float offset_length = dot(
                        candidate_uv - initial_uv, candidate_uv - initial_uv
                    );
                    if (candidate_colour.a >= 0.5 && offset_length < nearest_offset) {
                        nearest_offset = offset_length;
                        uv = candidate_uv;
                        colour = candidate_colour;
                        found = true;
                    }
                }
            }
            if (!found) valid = false;
        }

        // Newton-style inverse projection: find the capture texel whose
        // reconstructed 3D point projects onto this exact output pixel.
        for (int iteration = 0; iteration < 4 && valid; ++iteration) {
            if (any(lessThan(uv, vec2(0.001)))
                || any(greaterThan(uv, vec2(0.999)))) {
                valid = false;
                break;
            }
            colour = texture(colour_atlas, vec3(uv, slice_index));
            if (colour.a < 0.5) {
                valid = false;
                break;
            }

            vec3 point = reconstruct_point(uv, slice_index, right, up, forward);
            float clip_w;
            vec2 mapped = project_point(point, clip_w);
            if (clip_w <= 0.00001) {
                valid = false;
                break;
            }
            vec2 error = mapped - target_ndc;
            if (dot(error, error) < 0.00000025) break;

            float du = (uv.x + texel_size < 0.999) ? texel_size : -texel_size;
            float dv = (uv.y + texel_size < 0.999) ? texel_size : -texel_size;
            vec2 uv_x = uv + vec2(du, 0.0);
            vec2 uv_y = uv + vec2(0.0, dv);
            vec3 point_x = reconstruct_point(uv_x, slice_index, right, up, forward);
            vec3 point_y = reconstruct_point(uv_y, slice_index, right, up, forward);
            float clip_x_w;
            float clip_y_w;
            vec2 mapped_x = project_point(point_x, clip_x_w);
            vec2 mapped_y = project_point(point_y, clip_y_w);
            if (clip_x_w <= 0.00001 || clip_y_w <= 0.00001) {
                valid = false;
                break;
            }
            vec2 derivative_u = (mapped_x - mapped) / du;
            vec2 derivative_v = (mapped_y - mapped) / dv;
            float determinant = derivative_u.x * derivative_v.y
                              - derivative_v.x * derivative_u.y;
            if (abs(determinant) < 0.0000001) {
                valid = false;
                break;
            }
            vec2 correction = vec2(
                (error.x * derivative_v.y - error.y * derivative_v.x) / determinant,
                (derivative_u.x * error.y - derivative_u.y * error.x) / determinant
            );
            uv -= clamp(correction, vec2(-0.12), vec2(0.12));
        }

        if (!valid) continue;
        if (any(lessThan(uv, vec2(0.001)))
            || any(greaterThan(uv, vec2(0.999)))) continue;

        colour = texture(colour_atlas, vec3(uv, slice_index));
        if (colour.a < 0.5) continue;
        vec3 world_position = reconstruct_point(
            uv, slice_index, right, up, forward
        );
        float final_clip_w;
        vec2 final_ndc = project_point(world_position, final_clip_w);
        if (final_clip_w <= 0.00001
            || distance(final_ndc, target_ndc) > 0.012) continue;

        float ray_distance = dot(world_position - ray_origin, ray_direction);
        if (ray_distance <= 0.0 || ray_distance >= best.ray_distance) continue;

        best.world_position = world_position;
        best.colour = colour;
        best.normal = normalize(
            texture(normal_atlas, vec3(uv, slice_index)).rgb * 2.0 - 1.0
        );
        best.ray_distance = ray_distance;
        best.valid = 1.0;
    }
    return best;
}

void main() {
    vec2 target_ndc = gl_FragCoord.xy / viewport_size * 2.0 - 1.0;
    vec4 far_homogeneous = inverse_view_projection * vec4(target_ndc, 1.0, 1.0);
    if (abs(far_homogeneous.w) < 0.000001) discard;
    vec3 far_world = far_homogeneous.xyz / far_homogeneous.w;
    vec3 ray_direction = normalize(far_world - camera_position);

    vec4 blended_colour = vec4(0.0);
    vec3 blended_normal = vec3(0.0);
    vec3 blended_position = vec3(0.0);
    float total_weight = 0.0;

    for (int slot = 0; slot < 3; ++slot) {
        float weight = capture_weights[slot];
        if (weight <= 0.00001) continue;
        SurfaceSample surface_sample = sample_capture_view(
            slot, camera_position, ray_direction, target_ndc
        );
        if (surface_sample.valid < 0.5) continue;
        blended_colour += surface_sample.colour * weight;
        blended_normal += surface_sample.normal * weight;
        blended_position += surface_sample.world_position * weight;
        total_weight += weight;
    }

    if (total_weight <= 0.00001) discard;
    blended_colour /= total_weight;
    blended_position /= total_weight;
    blended_normal = normalize(blended_normal / total_weight);

    vec4 clip_position = view_projection * vec4(blended_position, 1.0);
    if (clip_position.w <= 0.00001) discard;
    gl_FragDepth = clamp(
        (clip_position.z / clip_position.w) * 0.5 + 0.5, 0.0, 1.0
    );

    vec3 light_direction = normalize(vec3(-0.45, 0.82, 0.36));
    float diffuse_amount = max(dot(blended_normal, light_direction), 0.0);
    float ambient_amount = 0.34;
    vec3 shaded_colour = blended_colour.rgb
        * (ambient_amount + diffuse_amount * 0.72);
    fragment_colour = vec4(shaded_colour, blended_colour.a);
}
"""


def nearest_view_index(views: list[dict[str, Any]], direction: np.ndarray) -> int:
    """Return the capture closest to the requested direction."""
    indices, _ = select_view_blend(views, direction)
    return indices[0]


def select_view_blend(
    views: list[dict[str, Any]], direction: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Find the spherical triangle containing a direction and its barycentric weights.

    Convex-hull interpolation is continuous at capture boundaries: the weight
    for the vertex opposite a shared edge reaches zero on that edge.
    """
    if len(views) < 3:
        raise ValueError("At least three capture views are required for spherical blending")
    target = np.asarray(direction, dtype=np.float64)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    directions = np.asarray([view["direction"] for view in views], dtype=np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(directions)
        for simplex in hull.simplices:
            basis = directions[np.asarray(simplex)].T
            try:
                coefficients = np.linalg.solve(basis, target)
            except np.linalg.LinAlgError:
                continue
            if np.min(coefficients) >= -1e-7:
                weights = np.clip(coefficients, 0.0, None)
                total = float(np.sum(weights))
                if total > 1e-10:
                    return [int(index) for index in simplex], weights / total
    except (ImportError, ValueError):
        pass

    # Defensive fallback for an unavailable/degenerate convex hull.
    scores = directions @ target
    indices = np.argsort(scores)[-3:][::-1]
    weights = np.maximum(scores[indices], 0.0) ** 16
    if float(np.sum(weights)) <= 1e-12:
        weights[:] = 1.0
    weights /= np.sum(weights)
    return [int(index) for index in indices], weights


def reconstruct_depth_offset(depth_value: float, near: float, far: float) -> float:
    """Convert normalised orthographic depth to canopy-local displacement."""
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
    """Upload layered capture arrays and render them with per-pixel parallax."""

    def __init__(self, gl, asset_directory: str | Path) -> None:
        self.gl = gl
        self.asset_directory = Path(asset_directory)
        manifest_path = self.asset_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Canopy impostor manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "oakforge-canopy-layered-depth-impostor-v3":
            raise ValueError(
                "Unsupported canopy impostor format. Re-bake the canopy using the layered v3 baker."
            )
        if not self.manifest.get("views"):
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
        self.views = self.manifest["views"]
        atlas = self.manifest["atlas"]
        self.tile_size = int(atlas["tile_size"])
        self.atlas_columns = int(atlas["columns"])
        self.atlas_padding = int(atlas["padding"])
        self.tile_pitch = int(atlas["tile_pitch"])
        self.layer_count = int(self.manifest["depth_layers"]["count"])
        if not 1 <= self.layer_count <= MAX_DEPTH_LAYERS:
            raise ValueError("The runtime supports between one and four depth layers")
        texture_files = self.manifest["textures"]
        for key in ("colour", "normal", "depth"):
            if len(texture_files.get(key, [])) != self.layer_count:
                raise ValueError(f"Layered impostor manifest has incomplete {key} textures")

        self.program = _compile_program(gl, IMPOSTOR_VERTEX_SHADER, IMPOSTOR_FRAGMENT_SHADER)
        uniform_names = (
            "view_projection", "inverse_view_projection", "viewport_size",
            "camera_position", "canopy_centre", "canopy_extent",
            "depth_near", "depth_far", "texel_size", "layer_count",
            "capture_rights[0]", "capture_ups[0]", "capture_forwards[0]",
            "capture_indices[0]", "capture_weights[0]", "capture_search_radii[0]",
            "colour_atlas", "normal_atlas", "depth_atlas",
        )
        self.uniforms = {
            name: int(gl.glGetUniformLocation(self.program, name))
            for name in uniform_names
        }
        missing = [name for name, location in self.uniforms.items() if location < 0]
        if missing:
            gl.glDeleteProgram(self.program)
            raise RuntimeError("Canopy impostor shader is missing uniforms: " + ", ".join(missing))

        self.textures: dict[str, int] = {}
        self.texture_bytes = 0
        self.vao = 0
        self.vbo = 0
        self._upload_texture_arrays()
        self._upload_screen_quad()

    def _load_array_slices(self, filenames: list[str], mode: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Install Pillow to load canopy impostor atlases") from error

        view_count = len(self.views)
        slice_count = view_count * self.layer_count
        if mode == "RGBA":
            result = np.empty(
                (slice_count, self.tile_size, self.tile_size, 4), dtype=np.uint8
            )
        else:
            result = np.empty(
                (slice_count, self.tile_size, self.tile_size), dtype=np.uint16
            )

        for layer_index, filename in enumerate(filenames):
            path = self.asset_directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Canopy impostor texture not found: {path}")
            with Image.open(path) as image:
                if mode == "RGBA":
                    atlas_pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                else:
                    atlas_pixels = np.asarray(image).astype(np.uint16, copy=False)
            expected_height = int(self.manifest["atlas"]["height"])
            expected_width = int(self.manifest["atlas"]["width"])
            if atlas_pixels.shape[0] != expected_height or atlas_pixels.shape[1] != expected_width:
                raise ValueError(f"Unexpected dimensions in canopy impostor texture: {path}")

            for view_index in range(view_count):
                column = view_index % self.atlas_columns
                row = view_index // self.atlas_columns
                x = column * self.tile_pitch + self.atlas_padding
                y = row * self.tile_pitch + self.atlas_padding
                tile = atlas_pixels[y:y + self.tile_size, x:x + self.tile_size]
                # PNG rows start at the top; OpenGL array layers start at the bottom.
                tile = np.flipud(tile)
                result[view_index * self.layer_count + layer_index] = tile
        return np.ascontiguousarray(result)

    def _upload_texture_arrays(self) -> None:
        gl = self.gl
        specifications = (
            ("colour", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
            ("normal", "RGBA", gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
            ("depth", "R16", gl.GL_R16, gl.GL_RED, gl.GL_UNSIGNED_SHORT),
        )
        for name, mode, internal_format, pixel_format, pixel_type in specifications:
            slices = self._load_array_slices(
                self.manifest["textures"][name], mode
            )
            texture = int(gl.glGenTextures(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage3D(
                gl.GL_TEXTURE_2D_ARRAY, 0, internal_format,
                self.tile_size, self.tile_size, len(slices), 0,
                pixel_format, pixel_type, slices,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
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
        gl.glUseProgram(0)

    def _upload_screen_quad(self) -> None:
        gl = self.gl
        vertices = np.asarray([
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
            [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        ], dtype=np.float32)
        self.vao = int(gl.glGenVertexArrays(1))
        self.vbo = int(gl.glGenBuffers(1))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, None)
        gl.glBindVertexArray(0)
        self.quad_bytes = int(vertices.nbytes)

    def _projected_bounds(
        self, view_projection: np.ndarray
    ) -> tuple[float, float, float, float] | None:
        corners = np.asarray([
            [x, y, z, 1.0]
            for x in (self.bounds_min[0], self.bounds_max[0])
            for y in (self.bounds_min[1], self.bounds_max[1])
            for z in (self.bounds_min[2], self.bounds_max[2])
        ], dtype=np.float32)
        clip = (view_projection @ corners.T).T
        if np.any(clip[:, 3] <= 0.0001):
            return (-1.0, -1.0, 1.0, 1.0)
        ndc = clip[:, :2] / clip[:, 3:4]
        margin = 0.025
        left = max(-1.0, float(np.min(ndc[:, 0])) - margin)
        right = min(1.0, float(np.max(ndc[:, 0])) + margin)
        bottom = max(-1.0, float(np.min(ndc[:, 1])) - margin)
        top = min(1.0, float(np.max(ndc[:, 1])) + margin)
        if left >= right or bottom >= top:
            return None
        return left, bottom, right, top

    def draw(
        self,
        view_projection: np.ndarray,
        camera_position: np.ndarray,
        viewport_size: tuple[int, int],
    ) -> list[int]:
        """Draw layered depth surfaces, returning the three blended capture indices."""
        gl = self.gl
        camera_position = np.asarray(camera_position, dtype=np.float64)
        camera_direction = camera_position - self.centre
        view_indices, weights = select_view_blend(self.views, camera_direction)
        bounds = self._projected_bounds(view_projection)
        if bounds is None:
            return view_indices

        left, bottom, right, top = bounds
        vertices = np.asarray([
            [left, bottom], [right, bottom], [right, top],
            [left, bottom], [right, top], [left, top],
        ], dtype=np.float32)
        inverse_view_projection = np.linalg.inv(view_projection).astype(np.float32)

        capture_rights = []
        capture_ups = []
        capture_forwards = []
        search_radii = []
        camera_normal = camera_direction / max(np.linalg.norm(camera_direction), 1e-12)
        for view_index in view_indices:
            direction = np.asarray(self.views[view_index]["direction"], dtype=np.float32)
            view_right, view_up, view_forward = view_basis(direction)
            capture_rights.append(view_right)
            capture_ups.append(view_up)
            capture_forwards.append(view_forward)
            alignment = float(np.clip(np.dot(camera_normal, view_forward), -1.0, 1.0))
            angular_sine = float(np.sqrt(max(0.0, 1.0 - alignment * alignment)))
            search_radii.append(min(0.14, 0.52 * angular_sine + 2.0 / self.tile_size))

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(
            self.uniforms["view_projection"], 1, gl.GL_TRUE, view_projection
        )
        gl.glUniformMatrix4fv(
            self.uniforms["inverse_view_projection"], 1, gl.GL_TRUE,
            inverse_view_projection,
        )
        gl.glUniform2f(
            self.uniforms["viewport_size"],
            float(max(viewport_size[0], 1)), float(max(viewport_size[1], 1)),
        )
        gl.glUniform3f(self.uniforms["camera_position"], *camera_position)
        gl.glUniform3f(self.uniforms["canopy_centre"], *self.centre)
        gl.glUniform1f(self.uniforms["canopy_extent"], self.extent)
        gl.glUniform1f(self.uniforms["depth_near"], self.near)
        gl.glUniform1f(self.uniforms["depth_far"], self.far)
        gl.glUniform1f(self.uniforms["texel_size"], 1.0 / self.tile_size)
        gl.glUniform1i(self.uniforms["layer_count"], self.layer_count)
        gl.glUniform3fv(
            self.uniforms["capture_rights[0]"], len(view_indices),
            np.ascontiguousarray(capture_rights, dtype=np.float32),
        )
        gl.glUniform3fv(
            self.uniforms["capture_ups[0]"], len(view_indices),
            np.ascontiguousarray(capture_ups, dtype=np.float32),
        )
        gl.glUniform3fv(
            self.uniforms["capture_forwards[0]"], len(view_indices),
            np.ascontiguousarray(capture_forwards, dtype=np.float32),
        )
        gl.glUniform1iv(
            self.uniforms["capture_indices[0]"], len(view_indices),
            np.asarray(view_indices, dtype=np.int32),
        )
        gl.glUniform1fv(
            self.uniforms["capture_weights[0]"], len(view_indices),
            np.asarray(weights, dtype=np.float32),
        )
        gl.glUniform1fv(
            self.uniforms["capture_search_radii[0]"], len(view_indices),
            np.asarray(search_radii, dtype=np.float32),
        )

        for unit, name in enumerate(("colour", "normal", "depth")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.textures[name])
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        for unit in range(3):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        return view_indices

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
            for unit in range(3):
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glDeleteTextures(len(self.textures), list(self.textures.values()))
            self.textures.clear()
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = 0
