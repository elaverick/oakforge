# Progressive tree generation and runtime LOD

## Goal

OakForge's default viewer should display a useful oak quickly and refine it while the camera moves. The first representation is built from a reduced structural pass plus a deterministic, branch-anchored foliage proxy. A background worker then runs the complete botanical recipe and prepares successively finer representations. The renderer keeps drawing the best available LOD while more detailed work is in progress.

The procedural growth recipe and its generated branch/leaf model remain the botanical source of truth. Cluster leaves and meshes are derived representations.

## Runtime representation contract

LOD indices follow the conventional convention: LOD0 is the closest and most detailed representation; LOD3 is the farthest and cheapest.

| Level | Camera distance | Content |
|---|---:|---|
| LOD3 | 46 m and beyond | Trunk, roots and mature scaffold through tertiary branches, with 17,760 deterministic proxy leaves |
| LOD2 | 26–46 m | Complete branch hierarchy including twigs and sprays, with one-quarter of original leaves enlarged by 1.2× |
| LOD1 | 13–26 m | Complete hierarchy with one-half of original leaf placements and a more detailed leaf profile |
| LOD0 | Under 13 m | Complete hierarchy and every individual leaf with the full oak-lobed profile |

The three distance thresholds have a two-metre hysteresis band to reduce LOD oscillation. Representation changes use a 0.35-second complementary screen-space dither: outgoing pixels are discarded exactly where incoming pixels are revealed. This avoids alpha blending and transparency sorting. Distance is the first implementation's proxy for projected size; camera-space projected bounds should replace fixed distances when this becomes a multi-tree scene system.

If the desired LOD is not ready, the viewer continues to use the best available representation. It does not wait for a worker or remove the current geometry before a replacement is uploaded.

## Current generation pipeline

1. The main thread reads the recipe and executes the first two branching iterations through tertiary structure. It omits terminal twigs, sprays, the fused wood-surface bake and individual-leaf placement.
2. It places 17,760 deterministic proxy leaves in compact three-dimensional clusters along the tertiary scaffold, then builds the first wood and foliage meshes. This initial pass is synchronous, but avoids the thousands of terminal branches and 311,136 source leaves.
3. The viewer opens with LOD3.
4. A background CPU worker executes the complete recipe for the same seed. It then builds LOD2 (full branch hierarchy and one-quarter of source leaves, enlarged 1.2×), LOD1 (one-half of source leaves) and LOD0 (all source leaves), publishing each completed mesh package through a bounded queue.
5. The render thread drains completed packages and performs all OpenGL resource uploads. The CPU worker never calls OpenGL.
6. Camera distance selects the best ready representation. The renderer retains already-ready LODs so that the user can move in either direction without rebuilding them.

The current worker does not stream individual branch-generation milestones from inside the recipe interpreter. It runs the full recipe as one background generation task, then prepares and publishes the three higher-detail LODs in order. That is a deliberate first increment; finer-grained subtree scheduling is future work.

## Botanical and geometry rules

- Preserve deterministic recipe seeds and the existing branch-parent hierarchy.
- Treat the generated branch and leaf model as canonical; never derive botanical data back from a rendered mesh.
- Use deterministic leaf clusters anchored to real branch positions for distant LODs.
- Use deterministic branch-anchored proxy leaves for LOD3. LOD2 and LOD1 use actual foliage placements sampled at one-quarter and one-half respectively; LOD2 enlarges the sampled leaves 1.2× to preserve canopy coverage at its larger viewing distance. LOD0 uses all actual leaves.
- Reduce per-leaf polygon complexity with distinct far, mid, near and hero leaf profiles. The default profile retains the full alternating oak lobes.
- Preserve the existing layered depth-impostor baker as an optional offline representation. It is not required for first visibility or progressive runtime operation.
- The initial LOD omits the fused wood-surface bake, keeping its synchronous construction lighter. Finer LODs use the full recipe's fused structural wood surface.
- Full LOD0 is intentionally detailed: the default tree creates 311,136 individual leaves. Its CPU mesh and GPU upload are substantially larger than other LODs.

## Threading and lifecycle

- The caller thread owns the initial reduced tree generation and first mesh build.
- The worker thread owns complete recipe execution and CPU mesh construction for LOD2/1/0.
- The OpenGL-owning thread exclusively owns GL resource creation, upload, drawing and release.
- Worker errors are surfaced in the viewer while the last uploaded representation remains drawable.
- The queue is bounded so the worker cannot run arbitrarily far ahead of the viewer.
- Closing the viewer signals the worker to stop publishing and joins it with a bounded timeout. Python recipe execution itself is not pre-emptible in this version; a worker already inside the full recipe may finish in the background after the bounded join.
- Regeneration is available after the current generation worker has finished. It starts a new seed with the reduced LOD3 path and launches a new worker. Cancelling an active recipe and per-subtree generation tokens are future refinements.

## Verification

- Unit tests verify initial scaffold structure, same-seed agreement with the full recipe, deterministic cluster placement, packed mesh integrity and LOD2 foliage population.
- A headless OpenGL smoke test creates the viewer under Xvfb, uploads every LOD level through the production upload path and draws both wood and foliage buffers.
- The existing complete regression suite continues to cover tree generation, branch attachment, crown shape, root topology, foliage placement, recipe interpretation and the original impostor renderer.
- GitHub Actions runs on `feature/progressive-tree-generation` and executes both the normal suite and the headless progressive viewer smoke test.

## Deliberate follow-on work

The first deliverable is an interactive single-tree viewer with a progressively available four-level LOD hierarchy. It is not yet a complete forest streaming engine. The next architectural steps are per-subtree jobs, visibility/projected-size scheduling across multiple trees, shared instanced twig/leaf-cluster assemblies, memory-budgeted CPU/GPU caches, temporal wind animation and a true volumetric far-field canopy representation.
