# Canopy depth impostors

OakForge bakes the **canopy only** into a multi-view depth-impostor asset. This
is a hybrid representation: leaves are captured in the impostor, while the
trunk, roots and structural branches remain polygon geometry and can use the
game's normal mesh LOD system.

Bake the leaves (default):

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_canopy \
  --views 64 --tile-size 128 --columns 8
```

Optionally include terminal twig and spray tubes:

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_canopy_twigs \
  --views 64 --tile-size 256 --columns 8 --include-twigs
```

The default is deliberately **leaves only**. Use the optional twig capture to
compare whether those fine woody details survive at the intended on-screen
size and atlas resolution. It includes only branch roles `twig` and `spray`;
trunk, roots, primary, leader, secondary and tertiary branches are excluded.

## Output

The bake produces:

- `colour.png`: leaf (and optionally twig) colour and alpha coverage.
- `normal.png`: captured surface normals.
- `depth.png`: orthographic depth packed into RGB24 (big-endian).
- `manifest.json`: canopy bounds, capture directions, atlas rectangles,
  source component counts and bake settings.

The default asset is 8x8 tiles at 128x128 pixels (1056x1056 per atlas,
including 2px edge gutters around each tile), with 64 views. The manifest
bounds describe the canopy source geometry, **not the whole tree**. The runtime
can place the canopy impostor relative to the separately rendered wood mesh.

## Hybrid runtime arrangement

1. Render the trunk, roots and structural branches using conventional polygon
   LOD. OakForge's existing `build_tree_mesh` remains the complete polygon-mesh
   path.
2. Render the leaf impostor as a separate canopy representation, aligned using
   the canopy bounds in `manifest.json`.
3. Compute the direction from the canopy centre to the camera in tree-local
   space.
4. Find the three capture directions with the largest dot products.
5. Sample their colour, depth and normal tiles and blend by normalised
   positive dot-product weights. Use a denser bake if angular transitions are
   too visible.
6. Use depth for depth-aware UV correction (parallax/relief intersection) and
   write reconstructed fragment depth where the engine supports it. A plain
   blended billboard is a useful first integration, but does not provide the
   full parallax benefit of a depth impostor.
7. Use captured normals for approximate lighting. For moving sun directions,
   keep albedo unlit and apply the game's lighting model to the captured normal.
8. Keep shadows on the polygon wood geometry or use a separate low-cost canopy
   shadow proxy; the colour/depth atlas is not automatically a correct dynamic
   shadow caster.

## Current limitations

- This is a capture/asset pipeline, not yet an engine-specific runtime
  material. The consumer shader must implement directional view selection,
  depth correction and alpha clipping.
- Captures are static. Wind animation needs multiple baked states or a separate
  animated near-field representation.
- Twig inclusion is experimental. At low atlas resolutions, twigs may collapse
  to sub-pixel lines; compare the leaves-only and twig-enabled atlases at the
  intended game camera distance before adopting it.
- The default 64-view/128px settings are starting points. Validate silhouette
  stability at the intended on-screen canopy size before increasing texture
  cost.
- The canopy-only bake still renders high-detail source geometry offline. This
  does not increase the runtime polygon count for the trunk or branches.
