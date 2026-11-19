# Game-ready depth impostors

OakForge can bake a generated tree into a multi-view texture asset:

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_impostor \
  --views 64 --tile-size 128 --columns 8
```

The bake uses the complete generated tree as source geometry. It does **not**
simplify the source tree or change the growth recipe. It renders a Fibonacci
distribution of full-sphere view directions into three atlases:

- `colour.png`: foliage/wood colour and alpha coverage.
- `normal.png`: encoded view-captured surface normals.
- `depth.png`: orthographic depth packed into RGB24 (big-endian).
- `manifest.json`: bounds, capture directions, atlas rectangles and settings.

The default asset is 8x8 tiles at 128x128 pixels (1024x1024 per atlas), with
64 views. The intended runtime representation is one camera-facing quad: two
triangles per tree instance, with the view direction selecting nearby atlas
captures. The manifest contains the exact capture directions and UV rectangles
so the game renderer does not have to duplicate OakForge's sampling layout.

## Runtime integration notes

1. Compute the direction from the tree centre to the camera in tree-local space.
2. Find the three capture directions with the largest dot products.
3. Sample their colour/depth/normal tiles and blend by normalised positive
   dot-product weights. For better angular transitions, use a denser bake.
4. Use the depth texture for depth-aware UV correction (parallax/relief
   intersection) and write reconstructed fragment depth where the engine
   supports it. A plain blended billboard is a useful first integration, but
   does not provide the full parallax benefit of a depth impostor.
5. Use the captured normal for approximate lighting. For moving sun directions,
   keep albedo unlit and apply the game's lighting model to the captured normal.
6. Use a separate low-cost shadow proxy or a dedicated shadow impostor; the
   colour/depth atlas is not automatically a correct dynamic shadow caster.

The baker is an offline desktop-OpenGL operation. The resulting PNGs and JSON
manifest are ordinary game assets and have no Python/OpenGL runtime dependency.

## Current limitations

- The first implementation is a capture/asset pipeline, not an engine-specific
  runtime material. The consumer shader must implement directional view
  selection, depth correction and alpha clipping.
- Captures are static. Wind animation needs multiple baked states or a separate
  animated near-field representation.
- The default 64-view/128px settings are starting points. Validate silhouette
  stability at the intended on-screen tree size before increasing texture cost.
- The bake uses the full source mesh and therefore spends time/memory offline.
  This is intentional: runtime cost, rather than bake cost, is the optimisation
  target.
