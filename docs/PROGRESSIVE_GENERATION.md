# Progressive tree generation and runtime LOD

## Goal

A newly requested tree must become renderable quickly, beginning with a cheap but convincing distant representation. More expensive structural and foliage detail is then generated in the background and promoted into nearer LODs as the camera needs it. Generation must never block the render loop.

The procedural growth description is authoritative. Meshes, foliage clusters, depth impostors and future animated representations are derived products; none is allowed to become the source of botanical truth.

## Runtime representation contract

The initial implementation uses four ordered quality levels:

| Level | Intended use | Content |
|---|---|---|
| LOD3 | Far | Primary scaffold and a compact foliage-cluster canopy |
| LOD2 | Mid/far | Secondary and tertiary structure with denser cluster foliage |
| LOD1 | Mid/near | Fine shoots and sprays with a deterministic subset of actual leaves |
| LOD0 | Near | Full structural tree and detailed foliage |

The renderer selects the best ready representation appropriate to projected size/distance. If that representation is not ready, it keeps the best available coarser representation. A representation is never removed before its replacement is ready.

## Pipeline

1. Parse and validate the recipe on the caller thread.
2. Start a background generation worker. Execute the existing YAML recipe in order and publish immutable snapshots at structural milestones.
3. Build CPU meshes and foliage cluster representations on the worker. The worker must not call OpenGL.
4. Place completed CPU representations on a thread-safe queue.
5. The OpenGL-owning thread drains that queue and performs resource creation, uploads and retirement.
6. Select the desired LOD using projected crown size (distance is the first implementation's proxy); apply hysteresis to prevent thrashing.
7. Keep the last ready LOD visible while a more detailed level is being generated.

## Botanical and geometry rules

- Preserve deterministic recipe seeds and the existing growth hierarchy.
- Treat parent-child branch relationships and terminal foliage attachment as canonical.
- Use branch/twig assemblies and deterministic clustered foliage for economical mid-distance geometry.
- Preserve the existing layered depth impostor as an optional baked asset, not a prerequisite for first visibility.
- Never create detail by averaging unrelated depth samples or moving reconstructed depth positions.
- Do not build the full multi-million-triangle leaf mesh merely to make the first frame visible.
- Share immutable source data and cached geometry where possible; bound CPU work, memory and GPU uploads.

## Threading and lifecycle

- The background worker owns recipe execution and CPU mesh generation.
- The render thread exclusively owns OpenGL calls and all GL resource lifetimes.
- Worker failures are propagated to the viewer as a visible error while the last ready representation remains drawable.
- Regeneration cancels or invalidates older work by generation token; late results from an old seed are discarded.
- Shutdown signals the worker, joins it with a bounded timeout, and releases GL resources on the render thread.

## Verification

- Unit tests cover progressive milestones, deterministic output, stable parent links, queue ordering, failure propagation and cancellation.
- Mesh tests verify every representation has aligned packed buffers, finite values and non-empty geometry.
- OpenGL smoke tests create and draw each available LOD under a headless context.
- Performance measurements separate recipe time, mesh-build time, upload time, frame time and resident memory.
- CI must run on this feature branch as well as the existing branches.

## Delivery boundary

A first useful release is a runnable OakForge viewer that starts with a lower-cost crown/wood representation, refines in the background and automatically changes between multiple realistic LODs. Wind animation, terrain-aware root placement, Nanite-compatible cluster streaming and a production game-engine integration remain separate follow-on work.
