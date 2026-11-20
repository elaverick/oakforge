# Canopy depth impostors and interactive LOD comparison

OakForge bakes the **canopy only** into a multi-view depth-impostor asset. This
is a hybrid representation: leaves are captured in the impostor, while the
trunk, roots and structural branches remain polygon geometry and can use the
game's normal mesh LOD system.

## Bake the canopy

Bake leaves only:

```bash
python -m oakforge.render.impostor \
  --config configs/english_oak.yaml \
  --output build/english_oak_canopy \
  --views 64 --tile-size 256 --columns 8
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

## Interactive representation comparison

Run the viewer normally:

```bash
python main.py --config configs/english_oak.yaml
```

On first launch for a recipe/seed/quality combination, OakForge automatically
bakes and caches the canopy atlas under `build/canopy_impostors/`. It reuses
the cache when the recipe hash, seed, leaf count, view count and tile size
match. The viewer starts in genuine polygon-leaf mode.

Press **I** to switch between:

- **Polygon leaves:** the original individual oak-leaf triangles rendered as
  genuine mesh geometry.
- **Depth impostor:** a camera-selected atlas view rendered on a two-triangle
  quad, with per-pixel depth reconstructed from the packed RGB24 depth texture
  and written back to the scene depth buffer.

The trunk, roots and branch mesh stays as polygon geometry in both modes. This
keeps the comparison focused on canopy representation. **L** hides/shows the
canopy; **1-5** switch structural branch levels; **W** toggles polygon
wireframe; **V** toggles VSync; **R** regenerates the tree and bakes the new
seed's impostor.

The runtime currently selects the single capture direction with the greatest
dot product against the camera direction. The atlas uses 64 full-sphere
Fibonacci directions by default. View blending and wind animation remain
possible follow-on improvements; nearest-view selection can show transitions
as the camera moves between capture directions.

## Diagnostics panel and metric definitions

The live panel reports:

- **Model triangles:** the full source polygon model's wood and leaf triangle
  counts. This does not change when switching representations.
- **Active triangles:** the wood triangles plus the active canopy representation
  (all leaf triangles in polygon mode, two quad triangles in impostor mode).
- **OakForge GPU allocations:** bytes requested for the viewer's tracked vertex
  buffers and textures. This is an exact sum of those resource sizes, not a
  driver-reported process total; it excludes driver overhead and small GL
  objects such as shader programs.
- **Device VRAM:** device-wide used/total memory from NVIDIA NVML when
  available. The figure includes the desktop and other processes.
- **Process VRAM:** graphics-process memory attributed to OakForge by NVML,
  when the driver exposes it.
- **GPU load and memory-controller load:** NVIDIA NVML utilisation samples.
- **FPS and frame time:** the viewer's measured frame interval. VSync is off by
  default to avoid capping both modes at the display refresh rate.
- **GPU scene time:** asynchronous OpenGL timer-query time for drawing the wood
  and canopy, excluding the grid and diagnostics overlay.

The NVIDIA telemetry is supplied by `nvidia-ml-py` (installed from
`requirements.txt`). If NVML is unavailable, the viewer attempts to read
device-wide total/free VRAM from `GL_NVX_gpu_memory_info`. Unsupported
utilisation and process-memory values are shown as `n/a`, not guessed.

The live panel also retains a per-representation benchmark after switching away
from that mode. Each reading is the average over the time spent in that mode
since the previous switch: FPS and CPU frame interval use wall-clock frame
timings, while GPU scene time uses asynchronous OpenGL timer queries. Press
`B` to clear the performance averages and start a fresh comparison. Tracked
canopy allocation is retained for both variants (polygon vertex buffer versus
impostor atlas textures and quad buffer), even though the inactive variant is
released. Device/process VRAM remains a driver-wide/process-wide sample and
can move due to unrelated desktop or GPU activity; use the OakForge tracked
allocation to isolate the actual representation resource difference.

## Runtime asset contents

- `colour.png`: leaf (and optionally twig) colour and alpha coverage.
- `normal.png`: captured surface normals.
- `depth.png`: orthographic depth packed into RGB24 (big-endian).
- `manifest.json`: canopy bounds, capture directions, atlas rectangles,
  depth near/far planes, source component counts and bake settings.

The manifest bounds describe the canopy source geometry, **not the whole tree**.
The runtime places the canopy representation relative to the separately
rendered wood mesh.

## Hybrid runtime arrangement

1. Build the polygon wood asset with `build_wood_mesh(tree)`. This excludes
   leaf polygons but retains all branch tiers for conventional LOD.
2. Render the leaf impostor as a separate canopy representation, aligned using
   the canopy bounds in `manifest.json`.
3. When using `--include-twigs`, call
   `build_wood_mesh(tree, include_terminal_twigs=False)` so terminal twigs and
   sprays are not drawn twice. The `build_tree_mesh` function remains the
   complete legacy path for the existing viewer and OBJ export.

## Limitations

- The current runtime selects one nearest atlas view; view blending has not yet
  been implemented.
- Captures are static. Wind animation needs multiple baked states or a separate
  animated near-field representation.
- Twig inclusion is experimental. At low atlas resolutions, twigs may collapse
  to sub-pixel lines; compare the leaves-only and twig-enabled atlases at the
  intended game camera distance before adopting them.
- The current atlas and depth shader are intended as a comparison/validation
  renderer. Validate angular transitions, depth edges and silhouette stability
  at game camera distances before integrating into a production renderer.
