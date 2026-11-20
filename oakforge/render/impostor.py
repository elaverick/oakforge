"""Offline multi-view depth-impostor baking for game-ready tree canopies.

The source tree remains fully procedural. This module captures the leaf canopy
(and optionally terminal twigs/sprays) into colour, encoded-normal and depth
atlases. Structural wood remains polygon geometry for conventional LOD.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from oakforge.core.mesh import build_foliage_mesh
from oakforge.generation import generate_from_file


@dataclass(frozen=True)
class ImpostorSettings:
    """Layered capture quality. Modern GL supports non-power-of-two texture sizes."""

    views: int = 64
    tile_size: int = 512
    atlas_columns: int = 8
    padding: int = 2
    layers: int = 4
    elevation_bias: float = 0.0
    include_twigs: bool = False

    def validate(self) -> None:
        if self.views < 8:
            raise ValueError("At least 8 views are required for a useful full-rotation impostor")
        if self.tile_size < 32:
            raise ValueError("tile_size must be at least 32 pixels")
        if self.atlas_columns < 1:
            raise ValueError("atlas_columns must be positive")
        if not 1 <= self.layers <= 4:
            raise ValueError("layers must be between 1 and 4")
        if self.padding < 0 or self.padding * 2 >= self.tile_size:
            raise ValueError("padding must be non-negative and smaller than half a tile")

def fibonacci_directions(count: int, elevation_bias: float = 0.0) -> np.ndarray:
    """Return evenly distributed unit view directions around the complete tree."""
    if count < 1:
        raise ValueError("count must be positive")
    indices = np.arange(count, dtype=np.float64) + 0.5
    y = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = indices * golden_angle
    directions = np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))
    if elevation_bias:
        directions[:, 1] += float(elevation_bias)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions.astype(np.float32)


def atlas_dimensions(settings: ImpostorSettings) -> tuple[int, int]:
    """Return pixel dimensions for a row-major square-tile atlas."""
    settings.validate()
    rows = math.ceil(settings.views / settings.atlas_columns)
    pitch = settings.tile_size + settings.padding * 2
    return settings.atlas_columns * pitch, rows * pitch


def view_basis(direction: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a stable right/up/forward basis for a camera looking at the origin."""
    forward = np.asarray(direction, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    reference_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(forward, reference_up))) > 0.985:
        reference_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(reference_up, forward)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(forward, right)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_png(path: Path, pixels: np.ndarray) -> None:
    """Write RGB/RGBA uint8 images and single-channel uint16 depth images."""
    image = np.asarray(pixels)
    if image.ndim == 2 and image.dtype == np.uint16:
        height, width = image.shape
        bit_depth, colour_type = 16, 0
        scanlines = b"".join(
            b"\x00" + image[row].astype(">u2", copy=False).tobytes()
            for row in range(height)
        )
    elif image.ndim == 3 and image.shape[2] in (3, 4) and image.dtype == np.uint8:
        height, width, channels = image.shape
        bit_depth, colour_type = 8, 2 if channels == 3 else 6
        scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    else:
        raise ValueError("PNG pixels must be RGB/RGBA uint8 or greyscale uint16")
    header = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, 6))
        + _png_chunk(b"IEND", b"")
    )

VERTEX_SHADER = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 colour;
uniform mat4 view_projection;
out vec3 v_normal;
out vec3 v_colour;
void main() {
    v_normal = normalize(normal);
    v_colour = colour;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_normal;
in vec3 v_colour;
layout(location=0) out vec4 out_colour;
layout(location=1) out vec4 out_normal;
layout(location=2) out float out_depth;
uniform sampler2D previous_depth;
uniform int has_previous_depth;
uniform int tile_size;
uniform float peel_epsilon;
void main() {
    if (has_previous_depth != 0) {
        vec2 pixel_uv = gl_FragCoord.xy / float(tile_size);
        float previous = texture(previous_depth, pixel_uv).r;
        if (gl_FragCoord.z <= previous + peel_epsilon) discard;
    }
    vec3 n = normalize(gl_FrontFacing ? v_normal : -v_normal);
    out_colour = vec4(v_colour, 1.0);
    out_normal = vec4(n * 0.5 + 0.5, 1.0);
    out_depth = gl_FragCoord.z;
}
"""


def _compile_program(gl) -> int:
    shaders = []
    for shader_type, source in ((gl.GL_VERTEX_SHADER, VERTEX_SHADER),
                                (gl.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)):
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(shader).decode("utf-8", "replace"))
        shaders.append(shader)
    program = gl.glCreateProgram()
    for shader in shaders:
        gl.glAttachShader(program, shader)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program).decode("utf-8", "replace"))
    for shader in shaders:
        gl.glDeleteShader(shader)
    return program


def _orthographic_view_projection(direction: np.ndarray, centre: np.ndarray,
                                  extent: float, near: float, far: float) -> np.ndarray:
    right, up, forward = view_basis(direction)
    eye = centre + forward * (far * 0.5)
    # direction points from the tree towards the camera. OpenGL looks down -Z,
    # so points in front of the camera must have negative camera-space Z.
    view = np.identity(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = right, up, forward
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(up, eye))
    view[2, 3] = -float(np.dot(forward, eye))
    projection = np.identity(4, dtype=np.float32)
    projection[0, 0] = 2.0 / extent
    projection[1, 1] = 2.0 / extent
    projection[2, 2] = -2.0 / (far - near)
    projection[2, 3] = -(far + near) / (far - near)
    return projection @ view


def bake_depth_impostor(tree, output_directory: str | Path,
                        settings: ImpostorSettings = ImpostorSettings()) -> dict:
    """Bake several depth-peeling layers for every canopy capture direction.

    Only leaves and optional terminal twigs/sprays are captured. Each pixel can
    retain up to settings.layers ordered surfaces instead of just the front hit.
    """
    settings.validate()
    try:
        import glfw
        from OpenGL import GL as gl
    except ImportError as error:
        raise RuntimeError("Install glfw and PyOpenGL to bake impostors") from error

    mesh = build_foliage_mesh(tree, include_twigs=settings.include_twigs)
    positions = np.asarray(mesh.positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
    colours = np.asarray(mesh.colours, dtype=np.float32).reshape(-1, 3)
    if not len(positions):
        raise ValueError("Cannot bake an empty canopy mesh")

    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    centre = (bounds_min + bounds_max) * 0.5
    extent = float(np.max(bounds_max - bounds_min)) * 1.04
    near = 0.01
    far = extent * 3.0
    interleaved = np.ascontiguousarray(
        np.column_stack((positions, normals, colours)), dtype=np.float32
    )

    if not glfw.init():
        raise RuntimeError("GLFW could not initialise an off-screen OpenGL context")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(settings.tile_size, settings.tile_size,
                                "OakForge layered impostor bake", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 off-screen context")

    output = Path(output_directory)
    width, height = atlas_dimensions(settings)
    pitch = settings.tile_size + settings.padding * 2
    colour_atlases = [
        np.zeros((height, width, 4), dtype=np.uint8) for _ in range(settings.layers)
    ]
    normal_atlases = [
        np.zeros((height, width, 4), dtype=np.uint8) for _ in range(settings.layers)
    ]
    depth_atlases = [
        np.full((height, width), 65535, dtype=np.uint16)
        for _ in range(settings.layers)
    ]

    program = vao = vbo = framebuffer = colour_texture = normal_texture = depth_buffer = 0
    depth_textures: list[int] = []
    try:
        glfw.make_context_current(window)
        program = _compile_program(gl)
        vao = int(gl.glGenVertexArrays(1))
        vbo = int(gl.glGenBuffers(1))
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW)
        stride = 9 * 4
        for location, size, offset in ((0, 3, 0), (1, 3, 12), (2, 3, 24)):
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(location, size, gl.GL_FLOAT, gl.GL_FALSE,
                                     stride, ctypes.c_void_p(offset))

        framebuffer = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)

        def make_texture(internal_format: int, pixel_format: int, pixel_type: int) -> int:
            texture = int(gl.glGenTextures(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format,
                            settings.tile_size, settings.tile_size, 0,
                            pixel_format, pixel_type, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            return texture

        colour_texture = make_texture(gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        normal_texture = make_texture(gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        depth_textures = [
            make_texture(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT)
            for _ in range(settings.layers)
        ]
        depth_buffer = int(gl.glGenRenderbuffers(1))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
                                 settings.tile_size, settings.tile_size)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                     gl.GL_RENDERBUFFER, depth_buffer)
        gl.glDrawBuffers(3, [
            gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2
        ])

        gl.glViewport(0, 0, settings.tile_size, settings.tile_size)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glUseProgram(program)
        matrix_location = gl.glGetUniformLocation(program, "view_projection")
        previous_location = gl.glGetUniformLocation(program, "previous_depth")
        has_previous_location = gl.glGetUniformLocation(program, "has_previous_depth")
        tile_size_location = gl.glGetUniformLocation(program, "tile_size")
        epsilon_location = gl.glGetUniformLocation(program, "peel_epsilon")
        required = (matrix_location, previous_location, has_previous_location,
                    tile_size_location, epsilon_location)
        if any(location < 0 for location in required):
            raise RuntimeError("Layered impostor bake shader is missing a required uniform")
        gl.glUniform1i(previous_location, 0)
        gl.glUniform1i(tile_size_location, settings.tile_size)
        gl.glUniform1f(epsilon_location, 1.0 / 16777215.0 * 4.0)

        directions = fibonacci_directions(settings.views, settings.elevation_bias)
        print(f"Impostor GL: {gl.glGetString(gl.GL_VERSION).decode('utf-8', 'replace')}")
        print(
            f"Layered impostor mesh: {len(interleaved):,} vertices, "
            f"{settings.views} views x {settings.layers} depth layers at "
            f"{settings.tile_size}px"
        )

        for view_index, direction in enumerate(directions):
            matrix = _orthographic_view_projection(direction, centre, extent, near, far)
            gl.glUniformMatrix4fv(matrix_location, 1, gl.GL_TRUE, matrix)
            col = view_index % settings.atlas_columns
            row = view_index // settings.atlas_columns
            x = col * pitch + settings.padding
            y = row * pitch + settings.padding
            for layer_index in range(settings.layers):
                gl.glFramebufferTexture2D(
                    gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                    gl.GL_TEXTURE_2D, colour_texture, 0
                )
                gl.glFramebufferTexture2D(
                    gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1,
                    gl.GL_TEXTURE_2D, normal_texture, 0
                )
                gl.glFramebufferTexture2D(
                    gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2,
                    gl.GL_TEXTURE_2D, depth_textures[layer_index], 0
                )
                if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError("Layered impostor bake framebuffer is incomplete")

                gl.glClearBufferfv(
                    gl.GL_COLOR, 0, np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                )
                gl.glClearBufferfv(
                    gl.GL_COLOR, 1, np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                )
                gl.glClearBufferfv(
                    gl.GL_COLOR, 2, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                )
                gl.glClearDepth(1.0)
                gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

                gl.glActiveTexture(gl.GL_TEXTURE0)
                previous_texture = depth_textures[layer_index - 1] if layer_index else 0
                gl.glBindTexture(gl.GL_TEXTURE_2D, previous_texture)
                gl.glUniform1i(has_previous_location, 1 if layer_index else 0)
                gl.glBindVertexArray(vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(interleaved))
                gl.glFinish()
                draw_error = gl.glGetError()
                if draw_error != gl.GL_NO_ERROR:
                    raise RuntimeError(
                        f"OpenGL error in view {view_index}, depth layer {layer_index}: "
                        f"0x{draw_error:04x}"
                    )

                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
                colour = np.frombuffer(
                    gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE), dtype=np.uint8
                ).reshape(settings.tile_size, settings.tile_size, 4)[::-1]
                covered = int(np.count_nonzero(colour[:, :, 3]))
                if covered == 0:
                    break

                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
                normal = np.frombuffer(
                    gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE), dtype=np.uint8
                ).reshape(settings.tile_size, settings.tile_size, 4)[::-1]
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT2)
                depth = np.frombuffer(
                    gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                    gl.GL_RED, gl.GL_FLOAT), dtype=np.float32
                ).reshape(settings.tile_size, settings.tile_size)[::-1]
                depth_u16 = np.rint(np.clip(depth, 0.0, 1.0) * 65535.0).astype(np.uint16)

                pad = settings.padding
                x0, y0 = x - pad, y - pad
                colour_tile = np.pad(colour, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                normal_tile = np.pad(normal, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                depth_tile = np.pad(depth_u16, ((pad, pad), (pad, pad)), mode="edge")
                colour_atlases[layer_index][y0:y0 + pitch, x0:x0 + pitch] = colour_tile
                normal_atlases[layer_index][y0:y0 + pitch, x0:x0 + pitch] = normal_tile
                depth_atlases[layer_index][y0:y0 + pitch, x0:x0 + pitch] = depth_tile

                if view_index == 0:
                    print(
                        f"View 1 depth layer {layer_index + 1}: "
                        f"{covered:,} covered pixels"
                    )

        if not any(np.any(atlas[:, :, 3]) for atlas in colour_atlases):
            raise RuntimeError(
                "Impostor bake rendered no covered pixels; check the view-projection matrix and GL state"
            )

        colour_files = []
        normal_files = []
        depth_files = []
        for layer_index in range(settings.layers):
            colour_name = f"colour_layer_{layer_index}.png"
            normal_name = f"normal_layer_{layer_index}.png"
            depth_name = f"depth_layer_{layer_index}.png"
            write_png(output / colour_name, colour_atlases[layer_index])
            write_png(output / normal_name, normal_atlases[layer_index])
            write_png(output / depth_name, depth_atlases[layer_index])
            colour_files.append(colour_name)
            normal_files.append(normal_name)
            depth_files.append(depth_name)

        manifest = {
            "format": "oakforge-canopy-layered-depth-impostor-v3",
            "tree_name": tree.name,
            "tree_seed": tree.metadata.get("seed"),
            "source_geometry": {
                "scope": "canopy",
                "leaf_count": len(tree.leaves),
                "available_terminal_twig_branches": sum(
                    1 for branch in tree.branches if branch.role in {"twig", "spray"}
                ),
                "terminal_twig_branches_included": settings.include_twigs,
            },
            "settings": asdict(settings),
            "atlas": {
                "width": width, "height": height, "columns": settings.atlas_columns,
                "rows": math.ceil(settings.views / settings.atlas_columns),
                "tile_size": settings.tile_size, "padding": settings.padding,
                "tile_pitch": pitch,
            },
            "bounds": {
                "min": bounds_min.tolist(), "max": bounds_max.tolist(),
                "centre": centre.tolist(), "extent": extent,
            },
            "depth_projection": {"near": near, "far": far},
            "views": [
                {"index": index, "direction": direction.tolist()}
                for index, direction in enumerate(directions)
            ],
            "view_blending": "convex-hull spherical barycentric interpolation",
            "depth_layers": {
                "count": settings.layers,
                "ordering": "front-to-back per capture pixel using iterative depth peeling",
            },
            "runtime_geometry": {"vertices": 4, "triangles": 2},
            "textures": {
                "colour": colour_files, "normal": normal_files, "depth": depth_files,
            },
            "depth_encoding": "16-bit unsigned normalised OpenGL orthographic depth",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
    finally:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        if depth_textures:
            gl.glDeleteTextures(len(depth_textures), depth_textures)
        if colour_texture:
            gl.glDeleteTextures(1, [colour_texture])
        if normal_texture:
            gl.glDeleteTextures(1, [normal_texture])
        if depth_buffer:
            gl.glDeleteRenderbuffers(1, [depth_buffer])
        if framebuffer:
            gl.glDeleteFramebuffers(1, [framebuffer])
        if vao:
            gl.glDeleteVertexArrays(1, [vao])
        if vbo:
            gl.glDeleteBuffers(1, [vbo])
        if program:
            gl.glDeleteProgram(program)
        glfw.destroy_window(window)
        glfw.terminate()
def main() -> int:
    parser = argparse.ArgumentParser(description="Bake a game-ready multi-view depth impostor for an OakForge canopy")
    parser.add_argument("--config", default="configs/english_oak.yaml")
    parser.add_argument("--output", default="build/impostor")
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4, help="Depth-peeling layers per capture (1-4)")
    parser.add_argument("--seed", type=int, help="Override the recipe seed for this bake")
    parser.add_argument(
        "--include-twigs", action="store_true",
        help="Include terminal twig and spray tubes in the canopy impostor bake",
    )
    args = parser.parse_args()
    if args.seed is None:
        tree = generate_from_file(args.config)
    else:
        from oakforge.generation import generate_tree, load_recipe
        recipe = load_recipe(args.config)
        recipe["tree"]["seed"] = args.seed
        tree = generate_tree(recipe)
    manifest = bake_depth_impostor(
        tree, args.output,
        ImpostorSettings(
            views=args.views, tile_size=args.tile_size,
            atlas_columns=args.columns, layers=args.layers,
            include_twigs=args.include_twigs,
        ),
    )
    print(f"Baked {len(manifest['views'])} views to {Path(args.output).resolve()}")
    print(f"Runtime canopy representation: {manifest['runtime_geometry']['triangles']} triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
