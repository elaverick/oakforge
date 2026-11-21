## 2026-11-21 — Progressive runtime tree generation and LOD

- Added a default progressive viewer path that constructs the mature scaffold through tertiary branches first, with 17,760 deterministic branch-anchored proxy leaves, without waiting for terminal twigs, sprays, fused wood or individual-leaf placement.
- Added a CPU-only background worker which completes structural growth once, then runs foliage scatter passes incrementally: the first pass publishes LOD2 (one-quarter of actual leaves enlarged 1.2×), the second publishes LOD1 (one-half of actual leaves), and the remaining passes complete LOD0 (all 311,136 actual leaves). Far, mid, near and hero levels use progressively richer leaf polygon profiles.
- Added distance-based LOD selection with a two-metre hysteresis band and 0.35-second complementary screen-space dither transitions between ready representations. Added manual LOD selection on keys 1–4 and automatic distance selection on 5/A.
- Kept OpenGL operations on the viewer thread. CPU foliage mesh packages are published through a bounded queue and uploaded by the context-owning thread. LOD2/LOD1/LOD0 share a single immutable structural wood mesh.
- Added an explicit `--legacy-impostor-viewer` option to retain the previous baked depth-impostor versus polygon comparison workflow.
- Added architecture documentation, deterministic cluster tests, LOD-selection tests and a headless OpenGL progressive upload/draw smoke test. GitHub Actions now runs on this feature branch.

## 2026-11-20 — Interactive canopy representation benchmark

- Added an `I` hotkey to switch the live viewer between genuine leaf polygons and the depth-impostor atlas, keeping the trunk and branch mesh identical between modes.
- Added runtime atlas loading, nearest-view selection and per-fragment depth reconstruction into the scene depth buffer.
- Added a live diagnostics overlay for source/active triangle counts, canopy-specific and total tracked GPU allocations, device/process VRAM, GPU utilisation, FPS, frame interval and asynchronous GPU scene time.
- Added optional NVIDIA NVML telemetry through `nvidia-ml-py`, with an OpenGL VRAM query fallback and explicit `n/a` reporting for unavailable metrics.
- Added automatic recipe/seed/quality-keyed atlas caching for the viewer and re-baking when regenerating a tree.
- Added an Xvfb-backed runtime OpenGL shader/draw smoke test alongside the existing bake and unit-test workflow.
- The runtime currently uses nearest-view selection without cross-view blending; validate angular transitions and depth edges before production engine integration.

## 2026-11-20 — Canopy-only depth impostors

- Split foliage mesh construction from the complete tree mesh. Leaves are now the default and only source geometry for depth-impostor baking.
- Added `build_wood_mesh` so engine integrations can render polygon trunk/branch geometry independently; terminal twigs/sprays can be assigned to either wood geometry or the optional canopy capture without duplication. The complete `build_tree_mesh` API remains intact.
- Added optional terminal twig and spray tube capture with `--include-twigs`; trunk, roots and structural branch tiers are never included in the canopy atlas.
- Changed impostor bounds and manifest metadata to describe the canopy source geometry.
- Added regression tests proving leaves-only capture excludes wood, optional twigs add only terminal detail, and the complete polygon mesh path remains available.
- Reworked the depth-impostor documentation around the hybrid canopy-impostor plus polygon-wood-LOD arrangement.

## 2026-11-19 — Game-ready depth-impostor bake

- Added an offline OpenGL 3.3 baker that captures the complete generated tree from a Fibonacci distribution of view directions.
- Exported colour/coverage, encoded-normal and RGB24 orthographic-depth atlases with edge gutters and a JSON manifest containing bounds, view vectors and per-view UV rectangles.
- Default output is a 64-view, 128-pixel-per-view atlas intended for a two-triangle runtime quad; source geometry remains unchanged.
- Added renderer-independent tests for view sampling, camera basis construction and atlas layout.
- Documented the bake command, runtime sampling requirements and current limitations.
- Runtime engine material, depth reprojection, wind animation and shadow representation remain follow-on work.
 
## 2026-11-17 — Branch attachment integrity and close-detail twigs

- Added crown-boundary insets to branch growth. Terminal twigs and foliage sprays now finish inside the leaf-bearing envelope so bare wood ends are less likely to protrude beyond the crown silhouette.
- Added a configurable basal attachment embed and short collar swell to child branches. Their bases start slightly inside the parent axis and broaden locally before tapering, reducing disconnected-looking forks.
- Added trunk-surface collar cues for both primary limbs and all six upper leaders, rather than only the lower primary scaffold.
- Increased the implicit wood union blend width to soften intersections between the bole, roots and major limbs.
- Closed both ends of every separately meshed branch tube. Fine twigs and sprays use twelve-sided cross-sections instead of the previous eight-sided tubes.
- Added tier-specific minimum branch radii: tertiary supports, terminal twigs and canopy sprays no longer collapse to the shared 3 mm floor. Close-detail twig bases are at least 14 mm and tips at least 6 mm in the default recipe.
- Moved the terminal foliage placement range towards twig tips so small leaf clusters remain supported by visible wood.
- Changed leaf mesh assembly to bounded batches and direct packed-buffer copies. This avoids Python-level scalar iteration across the multi-million-triangle leaf mesh.
- Added tests for capped tubes, buried child attachments, collar swell, minimum twig radii and terminal branch endpoints inside the canopy envelope.
- Complete regression suite passes (28 tests). Full unleafed render mesh builds to 1,102,200 triangles; the full leaf-bearing mesh builds to 7,635,404 triangles with aligned position, normal and colour buffers.

# Changelog

## 2026-11-17 — Canopy growth and persistent source snapshot

- Added YAML-configurable crown envelope shaping to the branch-growth tool.
- Adjusted descendant direction inheritance so horizontal crown spread persists without amplifying upward pitch at each growth tier.
- Increased the primary scaffold to ten limbs and tuned branching settings for a wider, more irregular mature-oak crown.
- Added 1,120 short inner-canopy spray shoots to the default recipe: 160 attached to secondary limbs and 960 attached to tertiary limbs.
- Added sprays to the model's own `groups.sprays` collection and to foliage placement, separate from terminal twigs.
- Updated leaf placement to support clustered whorls and radial depth on both terminal twigs and sprays.
- Vectorised branch tube and foliage mesh construction to make the denser tree practical to render and export.
- Updated tests and documentation. The complete suite passes (24 tests).
- Preserved the original recovered project as a separate baseline archive.


## 2026-11-17 — Upper-crown convergence and scaffold correction

- Shortened the main bole from 11.8 m to 8.8 m so the central stem ends inside the lower crown rather than projecting as a bare spike.
- Split the crown framework into eight broad primary scaffold limbs and six separate upper leaders.
- Added height-weighted crown convergence to child-branch growth; upper boughs now gently turn back towards a spread-out target around the crown apex instead of extending every tier towards the perimeter.
- Increased leader branching to eight secondary boughs per leader and preserved the standard secondary branching on lower scaffold limbs.
- Added leaf-bearing sprays on primary limbs and upper leaders as well as secondary and tertiary branches.
- Expanded the canopy envelope to a broad, rounded profile and added deterministic target spread to avoid branches converging on one exact point.
- Added regression checks for the shortened bole, leader descendants, upper-crown foliage count and upper-centre canopy fill.
- Updated the root-study tests to use structural fixtures for placement and a separate coarser voxel fixture for watertightness, avoiding a large duplicate full-oak edge-deduplication pass.
- The default seed now generates 7,256 branches and 311,136 leaves. All 25 tests pass.
