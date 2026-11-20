# Layered canopy depth impostors and LOD comparison

OakForge keeps the trunk, roots and structural branches as genuine polygon
geometry. Only the leaf canopy is replaced by the impostor. The runtime
representation is a view-dependent, layered depth surface rather than a flat
colour billboard.

## Bake the canopy

The interactive viewer defaults to 64 capture directions, 512 x 512 pixels per
direction and four depth-peeling layers per pixel:

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_canopy \
  --views 64 --tile-size 384 --layers 4 --columns 8
```

The tile size can be any value of at least 32; power-of-two dimensions are not
required by the OpenGL texture-array implementation. Increase it to 512 for
close-up assets, or reduce the view count/layer count for a smaller runtime
asset. The maximum supported layer count is four.

The default bake captures leaves only. Add `--include-twigs` to capture
terminal twig and spray tubes as well. Trunk, roots and structural branches
remain excluded from the canopy asset and continue to use polygon LOD.

## What is captured

For each view, the baker renders the canopy repeatedly using depth peeling:

1. The first pass records the nearest visible leaf surface.
2. Each later pass rejects fragments at or in front of the previous pass's
   depth, then records the next visible surface at that pixel.
3. Colour, world-space normal and 16-bit linear orthographic depth are written
   for every layer.

The resulting asset contains one colour, normal and depth atlas per peel layer,
plus a manifest with the canopy bounds, view directions and projection range.
Depth atlases are 16-bit greyscale images, not 8-bit colour-encoded depth.

## Runtime reconstruction

The viewer draws only a screen-space rectangle around the projected canopy
bounds. For each covered screen pixel, the fragment shader:

- Builds a ray from the active camera through that pixel.
- Intersects the ray with each capture view's centre plane to establish initial
  texture coordinates.
- Uses iterative inverse projection to find the source texel whose reconstructed
  3D point projects back to that exact screen pixel.
- Evaluates every available depth layer and keeps the closest valid surface
  along the current camera ray.
- Blends the three surrounding capture views using spherical convex-hull
  barycentric weights. The blend is continuous across the capture-triangle
  boundaries; it does not snap from one billboard to another.
- Reconstructs the final world position and writes it to the scene depth buffer,
  so the canopy depth-tests correctly against the polygon trunk and branches.

This is a layered depth impostor: it retains several visible surfaces along a
capture ray and uses those surfaces to produce camera-dependent parallax. It
does not preserve every leaf as independently addressable geometry, and it is
not a full volumetric or wind-deformable canopy.

## Interactive comparison

Run the viewer:

```bash
python main.py
```

Press **I** to switch between genuine polygon leaves and the layered depth
impostor. The trunk, roots and structural branches stay as polygons in both
modes. Other controls are **L** canopy visibility, **1-5** structural branch
levels, **W** wireframe, **V** VSync, **R** regenerate/re-bake and **B** reset
benchmark averages.

Quality controls:

- `--impostor-views`: capture directions (default 64)
- `--impostor-tile-size`: pixels per capture direction (default 384)
- `--impostor-layers`: depth layers per capture pixel (default 4)

The cache key includes recipe, seed, view count, tile size and layer count.
Assets from the earlier single-depth v2 format are not reused; the viewer
automatically creates a new layered v3 asset.

## Diagnostics panel

The panel reports:

- **Model triangles:** the full source polygon model's wood and leaf triangle
  counts, independent of the active canopy representation.
- **Active triangles:** wood triangles plus either all polygon canopy triangles
  or the two triangles used to rasterise the screen-space impostor region.
- **Tracked viewer allocations:** bytes uploaded for known vertex buffers and
  the three layered texture arrays. This is an exact sum of those requested
  resource sizes, not a driver-reported process total.
- **Device VRAM and process VRAM:** NVIDIA NVML telemetry where available.
- **GPU utilisation and memory-controller load:** NVIDIA telemetry where
  supported.
- **FPS and CPU frame time:** measured frame interval, with VSync off by default.
- **GPU scene time:** asynchronous OpenGL timer-query time for the wood and
  canopy draw, excluding the grid and diagnostics overlay.

The atlas arrays use 16-bit depth, RGBA8 colour and RGBA8 normals. With the
default 64 views, 384px tiles and four layers, the three texture arrays require
approximately 360 MiB of GPU storage before small GL overheads. Reduce tile
size, view count or layer count when targeting a smaller memory budget.

## Validation

The test suite checks depth encoding, spherical view interpolation and runtime
shader compilation. The Xvfb-backed smoke test also inspects the baked atlases
to verify that secondary depth layers contain geometry, then renders from the
capture direction and a displaced camera position to exercise parallax
reprojection.

Run the tests and smoke bake with:

```bash
python -m unittest discover -s tests -v
xvfb-run -a python -m oakforge.render.impostor --config configs/english_oak.yaml \
  --output build/impostor --views 16 --tile-size 192 --layers 4
xvfb-run -a env OAKFORGE_GL_SMOKE=1 python -m unittest discover -s tests \
  -p test_impostor_runtime_gl.py -v
```

## Current limitations

- Four depth-peeling layers provide a finite number of visible surfaces. Dense
  foliage can still contain more layers than the asset stores.
- Capture blending and parallax depend on the baked view density and source
  resolution. Extreme close-ups or camera positions substantially outside the
  capture coverage can reveal reconstruction gaps.
- The capture is static. Wind animation requires multiple baked states or a
  separate animated near-field representation.
- Fine twigs remain optional because their appearance is highly dependent on
  target screen size and atlas resolution.
- Validate the baked result at the actual game camera distances before using
  the asset as a production vegetation LOD.
