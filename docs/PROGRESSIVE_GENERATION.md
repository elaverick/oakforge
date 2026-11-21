# Progressive tree generation and runtime LOD

## Goal

OakForge's default viewer should display a useful oak quickly and refine it while the camera moves. The first representation is built from a reduced structural pass plus deterministic, branch-attached foliage sprays. Each spray contains six compact, overlapping, individually oriented oak leaves. The canonical recipe still creates all terminal shoots and source leaves, but the runtime mesh is built from fixed per-LOD card budgets rather than triangulating every botanical leaf. A background worker completes the remaining branch/twig/spray and fused-wood stages, then advances the recipe's foliage scatter loop pass by pass. The renderer keeps drawing the best available LOD while more detailed work is in progress.

The procedural growth recipe and its generated branch/leaf model remain the botanical source of truth. Cluster leaves and meshes are derived representations.

## Runtime representation contract

LOD indices follow the conventional convention: LOD0 is the closest and most detailed representation; LOD3 is the farthest and cheapest.

| Level | Camera distance | Content |
|---|---:|---|
| LOD3 | 46 m and beyond | Fused trunk, roots and major scaffold limbs (primary, leaders and secondary); 3,072 alpha-cutout two-triangle oak cards arranged as 512 six-leaf sprays; tertiary tubes deferred |
| LOD2 | 26–46 m | Fused major scaffold, simplified tertiary tubes and 8,192 textured two-triangle oak-leaf cards sampled from the first foliage pass |
| LOD1 | 13–26 m | Fused major scaffold, simplified tertiary tubes and 16,384 textured two-triangle oak-leaf cards sampled from the first two foliage passes |
| LOD0 | Under 13 m | Fused major scaffold, simplified tertiary tubes and 40,960 textured two-triangle oak-leaf cards sampled across the complete source foliage |

The three distance thresholds have a two-metre hysteresis band to reduce LOD oscillation. Representation changes use a 0.35-second complementary screen-space dither: outgoing pixels are discarded exactly where incoming pixels are revealed. This avoids alpha blending and transparency sorting. Distance is the first implementation's proxy for projected size; camera-space projected bounds should replace fixed distances when this becomes a multi-tree scene system.

If the desired LOD is not ready, the viewer continues to use the best available representation. It does not wait for a worker or remove the current geometry before a replacement is uploaded.

## Current generation pipeline

1. The main thread reads the recipe and executes the first two branching iterations through tertiary structure. It omits terminal twigs, sprays and individual-leaf placement, but builds the fused trunk/root/primary/leader/secondary surface used by every refined LOD.
2. It places 3,072 cards as 512 compact six-leaf sprays, weighted by branch length and calibre and distributed over the primary-to-tertiary scaffold, then builds the first wood and foliage meshes. This initial pass is synchronous, but avoids the thousands of terminal branches and 311,136 source leaves.
3. The viewer opens with LOD3.
4. A background CPU worker executes the remaining recipe structure for the same seed: terminal twigs, sprays and the fused wood surface are generated once. It then executes one foliage scatter pass at a time. With the default four-pass oak recipe, pass one publishes LOD2 (8,192 cards sampled across the first source-leaf pass), pass two publishes LOD1 (16,384 cards sampled across the first two passes), and passes three and four complete the source foliage for LOD0 (40,960 cards sampled across the full 311,136-leaf population). Each ready mesh package is published through a bounded queue.
5. The branch hierarchy and fused wood surface are unchanged by foliage passes. The worker builds the structural wood mesh once and shares that immutable CPU mesh with LOD2, LOD1 and LOD0; only the foliage mesh is rebuilt at each milestone. This reduces repeated CPU mesh work and avoids retaining duplicate wood buffers.
6. The render thread drains completed packages and performs all OpenGL resource uploads. The CPU worker never calls OpenGL.
7. Camera distance selects the best ready representation. The renderer retains already-ready LODs so that the user can move in either direction without rebuilding them.

The worker streams foliage-pass milestones but does not yet stream individual branch-growth milestones or refine independent subtrees. Structural generation (including the fused wood-surface build) remains one background stage; within foliage generation, each recipe pass is a separate publish point. Finer-grained subtree scheduling and scene-wide multi-tree prioritisation remain future work.

## Botanical and geometry rules

- Preserve deterministic recipe seeds and the existing branch-parent hierarchy.
- Treat the generated branch and leaf model as canonical; never derive botanical data back from a rendered mesh.
- Use deterministic leaf clusters anchored to real branch positions for distant LODs.
- The canonical tree retains all 311,136 leaf placements. Runtime LOD meshes select deterministic subsets: 3,072 cards at LOD3, 8,192 at LOD2, 16,384 at LOD1 and 40,960 at LOD0. Each card is a UV-mapped two-triangle quad with an alpha-cutout oak silhouette, midrib, side veins and per-leaf atlas variation. All four variants come from the shared `oakforge/assets/oak_leaf_atlas.png` RGBA asset, used unchanged by the viewer and OBJ export. LOD3 cards are organised into compact overlapping six-leaf sprays around branch anchors.
- Preserve the full lobed source-leaf model for baking and close-up reference work. Runtime rendering uses two-triangle UV quads with an alpha-cutout oak-leaf texture atlas and fixed per-LOD budgets.
- Preserve the existing layered depth-impostor baker as an optional offline representation. It is not required for first visibility or progressive runtime operation.
- LOD3 and the refined LODs use the same fused trunk/root/primary/leader/secondary wood surface. Progressive runtime wood uses an 8-side build setting, resamples tertiary tubes to at most 4 segments for LOD2, LOD1 and LOD0, and omits terminal twig/spray tubes; the leaf-card canopy represents this fine growth at game distances. The English-oak fused wood surface uses a 0.20m voxel size for a lower polygon count.
- The canonical botanical model still creates 311,136 individual leaves, but LOD0 retains 40,960 cards at runtime. The `python -m oakforge.geometry_budget --instances 16` command builds each LOD and reports actual triangles and UV-aware buffer bytes, failing if any tree exceeds the 200,000-triangle target. OBJ export includes matching MTL files and `oak_leaf_atlas.png`.

## Threading and lifecycle

- The caller thread owns the initial reduced tree generation, cluster-spray placement and first mesh build.
- The worker thread owns the remaining structural recipe stages and incremental foliage-pass execution, plus CPU foliage-mesh construction for LOD2/1/0. The unchanged simplified structural wood mesh is built once and shared.
- The OpenGL-owning thread exclusively owns GL resource creation, upload, drawing and release.
- Worker errors are surfaced in the viewer while the last uploaded representation remains drawable.
- The queue is bounded so the worker cannot run arbitrarily far ahead of the viewer.
- Closing the viewer signals the worker to stop publishing and joins it with a bounded timeout. Structural generation and an individual foliage pass are not pre-emptible in this version; a worker already inside one of those stages may finish it in the background after the bounded join.
- Regeneration is available after the current generation worker has finished. It starts a new seed with the reduced LOD3 path and launches a new worker. Cancelling an active recipe and per-subtree generation tokens are future refinements.

## Verification

- Unit tests verify initial scaffold structure, same-seed agreement with the full recipe, deterministic cluster placement, packed mesh integrity, worker publication at each foliage-pass milestone and structural wood-mesh sharing across foliage LODs.
- A headless OpenGL smoke test creates the viewer under Xvfb, uploads every LOD level through the production UV-buffer and texture upload path, draws both wood and foliage buffers, and saves per-LOD preview images.
- The existing complete regression suite continues to cover tree generation, branch attachment, crown shape, root topology, foliage placement, recipe interpretation and the original impostor renderer.
- GitHub Actions runs on `feature/progressive-tree-generation` and executes both the normal suite and the headless progressive viewer smoke test.

## Deliberate follow-on work

The first deliverable is an interactive single-tree viewer with a progressively available four-level LOD hierarchy. It is not yet a complete forest streaming engine. The next architectural steps are per-subtree jobs, visibility/projected-size scheduling across multiple trees, shared instanced twig/leaf-cluster assemblies, memory-budgeted CPU/GPU caches, temporal wind animation and a true volumetric far-field canopy representation.
