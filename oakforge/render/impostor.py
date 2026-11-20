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
    """Bake quality and atlas layout. Defaults favour inexpensive game assets."""

    views: int = 64
    tile_size: int = 128
    atlas_columns: int = 8
    padding: int = 2
    elevation_bias: float = 0.0
    include_twigs: bool = False

    def validate(self) -> None:
        if self.views < 8:
            raise ValueError("At least 8 views are required for a useful full-rotation impostor")
        if self.tile_size < 16 or self.tile_size & (self.tile_size - 1):
            raise ValueError("tile_size must be a power of two and at least 16")
        if self.atlas_columns < 1:
            raise ValueError("atlas_columns must be positive")
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
    """Write an RGB or RGBA uint8 image without adding an imaging dependency."""
    image = np.asarray(pixels, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("PNG pixels must have shape (height, width, 3|4)")
    height, width, channels = image.shape
    colour_type = 2 if channels == 3 else 6
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
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
void main() {
    vec3 n = normalize(gl_FrontFacing ? v_normal : -v_normal);
    out_colour = vec4(v_colour, 1.0);
    out_normal = vec4(n * 0.5 + 0.5, 1.0);
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
    """Render the canopy into colour, normal and depth view atlases.

    Leaves are always captured. Terminal twig and spray geometry is optional;
    the trunk, roots and structural branches are excluded so they can use
    conventional polygon LOD independently. Requires desktop OpenGL 3.3.
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
        raise ValueError("Cannot bake an empty tree mesh")

    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    centre = (bounds_min + bounds_max) * 0.5
    extent = float(np.max(bounds_max - bounds_min)) * 1.04
    near = 0.01
    far = extent * 3.0
    interleaved = np.ascontiguousarray(np.column_stack((positions, normals, colours)), dtype=np.float32)

    if not glfw.init():
        raise RuntimeError("GLFW could not initialise an off-screen OpenGL context")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(settings.tile_size, settings.tile_size, "OakForge impostor bake", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 off-screen context")

    output = Path(output_directory)
    width, height = atlas_dimensions(settings)
    colour_atlas = np.zeros((height, width, 4), dtype=np.uint8)
    normal_atlas = np.zeros_like(colour_atlas)
    depth_atlas = np.full((height, width, 3), 255, dtype=np.uint8)

    try:
        glfw.make_context_current(window)
        program = _compile_program(gl)
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW)
        stride = 9 * 4
        for location, size, offset in ((0, 3, 0), (1, 3, 12), (2, 3, 24)):
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(location, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset))

        framebuffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
        attachments = []
        for attachment, internal_format in ((gl.GL_COLOR_ATTACHMENT0, gl.GL_RGBA8),
                                            (gl.GL_COLOR_ATTACHMENT1, gl.GL_RGBA8)):
            texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, settings.tile_size,
                            settings.tile_size, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, attachment, gl.GL_TEXTURE_2D, texture, 0)
            attachments.append(texture)
        depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
                                 settings.tile_size, settings.tile_size)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                     gl.GL_RENDERBUFFER, depth_buffer)
        gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Impostor bake framebuffer is incomplete")

        gl.glViewport(0, 0, settings.tile_size, settings.tile_size)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(program)
        matrix_location = gl.glGetUniformLocation(program, "view_projection")
        if matrix_location < 0:
            raise RuntimeError("The view_projection uniform was not found in the linked impostor shader")
        directions = fibonacci_directions(settings.views, settings.elevation_bias)
        print(f"Impostor GL: {gl.glGetString(gl.GL_VERSION).decode('utf-8', 'replace')}")
        print(f"Impostor mesh: {len(interleaved)} vertices, {matrix_location=}, framebuffer={gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)}")
        for index, direction in enumerate(directions):
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClearDepth(1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            matrix = _orthographic_view_projection(direction, centre, extent, near, far)
            # Match the established viewer upload path: NumPy row-major + GL transpose.
            gl.glUniformMatrix4fv(matrix_location, 1, gl.GL_TRUE, matrix)
            gl.glBindVertexArray(vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(interleaved))
            gl.glFinish()
            draw_error = gl.glGetError()
            if draw_error != gl.GL_NO_ERROR:
                raise RuntimeError(f"OpenGL error while drawing impostor view {index}: 0x{draw_error:04x}")
            col = index % settings.atlas_columns
            row = index // settings.atlas_columns
            pitch = settings.tile_size + settings.padding * 2
            x, y = col * pitch + settings.padding, row * pitch + settings.padding
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            colour = np.frombuffer(gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE), dtype=np.uint8)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
            normal = np.frombuffer(gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE), dtype=np.uint8)
            colour = colour.reshape(settings.tile_size, settings.tile_size, 4)[::-1]
            normal = normal.reshape(settings.tile_size, settings.tile_size, 4)[::-1]
            if index == 0:
                print(f"Impostor view 0: matrix={matrix.tolist()}, covered={int(np.count_nonzero(colour[:, :, 3]))}")
            pad = settings.padding
            x0, y0 = x - pad, y - pad
            colour_tile = np.pad(colour, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            normal_tile = np.pad(normal, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            colour_atlas[y0:y0 + pitch, x0:x0 + pitch] = colour_tile
            normal_atlas[y0:y0 + pitch, x0:x0 + pitch] = normal_tile
            depth = np.frombuffer(gl.glReadPixels(0, 0, settings.tile_size, settings.tile_size,
                                                  gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT), dtype=np.float32)
            depth = np.clip(depth.reshape(settings.tile_size, settings.tile_size)[::-1], 0.0, 1.0)
            depth24 = np.rint(depth * 16777215.0).astype(np.uint32)
            depth_rgb = np.stack(((depth24 >> 16) & 255, (depth24 >> 8) & 255, depth24 & 255), axis=-1).astype(np.uint8)
            depth_tile = np.pad(depth_rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            depth_atlas[y0:y0 + pitch, x0:x0 + pitch] = depth_tile

        write_png(output / "colour.png", colour_atlas)
        write_png(output / "normal.png", normal_atlas)
        write_png(output / "depth.png", depth_atlas)
        # Do not publish a superficially successful bake with empty atlases.
        if not np.any(colour_atlas[:, :, 3]):
            raise RuntimeError(
                "Impostor bake rendered no covered pixels; check the view-projection matrix and GL draw state"
            )
        manifest = {
            "format": "oakforge-canopy-depth-impostor-v1",
            "tree_name": tree.name,
            "source_geometry": {
                "scope": "canopy",
                "leaf_count": len(tree.leaves),
                "available_terminal_twig_branches": sum(
                    1 for branch in tree.branches if branch.role in {"twig", "spray"}
                ),
                "terminal_twig_branches_included": settings.include_twigs,
            },
            "settings": asdict(settings),
            "atlas": {"width": width, "height": height, "columns": settings.atlas_columns,
                      "rows": math.ceil(settings.views / settings.atlas_columns),
                      "tile_size": settings.tile_size, "padding": settings.padding,
                      "tile_pitch": settings.tile_size + settings.padding * 2},
            "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist(),
                       "centre": centre.tolist(), "extent": extent},
            "views": [{"index": i, "direction": direction.tolist(),
                       "uv_rect": [((i % settings.atlas_columns) * (settings.tile_size + settings.padding * 2) + settings.padding) / width,
                                   ((i // settings.atlas_columns) * (settings.tile_size + settings.padding * 2) + settings.padding) / height,
                                   settings.tile_size / width, settings.tile_size / height]}
                      for i, direction in enumerate(directions)],
            "runtime_geometry": {"vertices": 4, "triangles": 2},
            "textures": {"colour": "colour.png", "normal": "normal.png", "depth": "depth.png"},
            "depth_encoding": "OpenGL orthographic depth, linear over [0,1], packed big-endian into RGB24",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake a game-ready multi-view depth impostor for an OakForge tree")
    parser.add_argument("--config", default="configs/english_oak.yaml")
    parser.add_argument("--output", default="build/impostor")
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument(
        "--include-twigs", action="store_true",
        help="Include terminal twig and spray tubes in the canopy impostor bake",
    )
    args = parser.parse_args()
    tree = generate_from_file(args.config)
    manifest = bake_depth_impostor(
        tree, args.output,
        ImpostorSettings(
            views=args.views, tile_size=args.tile_size,
            atlas_columns=args.columns, include_twigs=args.include_twigs,
        ),
    )
    print(f"Baked {len(manifest['views'])} views to {Path(args.output).resolve()}")
    print(f"Runtime representation: {manifest['runtime_geometry']['triangles']} triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
