# Layered canopy depth impostors and LOD comparison

OakForge keeps the trunk, roots and structural branches as genuine polygon
geometry. Only the leaf canopy is replaced by the impostor. The runtime
representation is a layered, depth-bearing capture that is forward-projected
into the active camera as depth-tested point splats.

## Bake the canopy

The interactive viewer defaults to 64 capture directions, 512 x 512 pixels per
direction and four depth-peeling layers per pixel:

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_canopy \
  --views 64 --tile-size 512 --layers 4 --columns 8
```

The tile size can be any value of at least 32; power-of-two dimensions are not
required by the OpenGL texture-array implementation. Increase it for close-up
assets, or reduce the view count/layer count for a smaller runtime asset. The
maximum supported layer count is four.

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

The runtime deliberately avoids inverse texture-coordinate searches and
cross-view averaging. Those operations are unstable around foliage silhouettes
and can combine unrelated leaves into positions that do not exist.

Instead, for the capture direction closest to the active camera, the vertex
shader:

- Visits each texel in each of the captured depth layers.
- Leaves transparent texels empty; it never substitutes a nearby unrelated
  leaf to fill a hole.
- Reconstructs the texel's actual canopy-space position from its capture UV and
  stored depth.
- Projects that 3D position through the active camera matrix.
- Estimates the projected source-texel footprint and rasterises a small point
  splat at that position.

All layers are drawn into the normal scene depth buffer. The depth test resolves
occlusion between the front and rear leaf samples, and between the canopy and
the polygon trunk/branches. The rendered samples retain the real depth values;
there is no averaging of the positions of unrelated surfaces.

Only one capture view is used for a frame. This is intentional: adjacent
captures do not contain guaranteed pixel-to-pixel correspondences, so blending
their world positions created incoherent phantom geometry. The view is selected
by maximum directional alignment. As the camera crosses a capture boundary,
the source view changes; the reconstructed samples remain genuine surfaces,
although the view transition can still be visible.

This is a layered depth impostor, not a full volumetric canopy. It retains up to
four visible surfaces per capture pixel, not every leaf at every angle.

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

### Foliage footprint styles

The point-splat renderer provides four runtime footprint styles, selectable with
**C** while the depth impostor is active:

- **Square** preserves the original point-splat appearance as the baseline.
- **Round** clips each sample to a circular footprint.
- **Angular** uses irregularly oriented polygons with a per-sample variation in
  side count and edge shape.
- **Foliage** adds two scales of object-space value noise to the angular
  footprint. Noise changes the edge coverage, not the reconstructed position
  or depth, so the layered depth test remains authoritative.

The noise coordinates are derived from canopy-local positions and the source
texel footprint, rather than screen coordinates. This keeps the breakup attached
to the tree as the camera moves and avoids the characteristic screen-space
crawling of animated noise. The base colour and depth atlas data are unchanged.

Live controls (available while the depth impostor is active):

- **C** cycles square, round, angular and foliage styles.
- **[ / ]** decrease or increase edge-noise strength.
- **, / .** decrease or increase the noise scale.
- **- / =** decrease or increase angularity.

The default is foliage style with moderate angularity and noise. The controls
are clamped to safe ranges; switch back to square at any time for a direct
comparison. Noise evaluation is only performed in foliage style.

Quality controls:

- `--impostor-views`: capture directions (default 64)
- `--impostor-tile-size`: pixels per capture direction (default 512)
- `--impostor-layers`: depth layers per capture pixel (default 4)

The cache key includes recipe, seed, view count, tile size and layer count.
Assets from the earlier single-depth v2 format are not reused; the viewer
automatically creates a new layered v3 asset.

## Diagnostics panel

The panel reports:

- **Model triangles:** the full source polygon model's wood and leaf triangle
  counts, independent of the active canopy representation.
- **Active triangles:** wood triangles plus the polygon canopy triangles when
  polygon mode is active. The impostor itself is reported as depth-tested point
  samples, not falsely counted as two triangles.
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
default 64 views, 512px tiles and four layers, the three texture arrays require
approximately 640 MiB of GPU storage before small GL overheads. Reduce tile
size, view count or layer count when targeting a smaller memory budget.

## Validation

The test suite checks depth encoding, capture selection and runtime shader
compilation. The Xvfb-backed smoke test also inspects the baked atlases to
verify that secondary depth layers contain geometry, then renders the
forward-projected splats from the capture direction and a displaced camera.

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
- The runtime uses the nearest capture direction rather than blending views.
  Switching captures can cause a small change in leaf coverage as the camera
  crosses capture directions.
- Source texel density and projected point-splat size determine coverage.
  Very close views or severe grazing angles can still reveal holes or soften
  fine leaf detail.
- The capture is static. Wind animation requires multiple baked states or a
  separate animated near-field representation.
- Fine twigs remain optional because their appearance is highly dependent on
  target screen size and atlas resolution.
- Validate the baked result at the actual game camera distances before using
  the asset as a production vegetation LOD.
