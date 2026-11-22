# Progressive tree generation and runtime LOD

## Goal

OakForge's default viewer should display a useful oak quickly and refine it while the camera moves. The four levels deliberately use different canopy representations: LOD3 is a single irregular crown-volume shell; LOD2 and LOD1 are crossed multi-leaf clump cards attached to sprays and fine branches; only LOD0 resolves individual oak leaves. The canonical recipe still creates all terminal shoots and source leaves. A background worker completes the remaining branch/twig/spray and fused-wood stages, then advances the recipe's foliage scatter loop pass by pass. The renderer keeps drawing the best available LOD while more detailed work is in progress.

The procedural growth recipe and its generated branch/leaf model remain the botanical source of truth. Cluster leaves and meshes are derived representations.

## Runtime representation contract

LOD indices follow the conventional convention: LOD0 is the closest and most detailed representation; LOD3 is the farthest and cheapest.

| Level | Camera distance | Content |
|---|---:|---|
| LOD3 | 46 m and beyond | Fused trunk, roots and major scaffold limbs (primary, leaders and secondary); one irregular low-poly crown volume fitted to the leaf-bearing envelope; tertiary tubes deferred |
| LOD2 | 26–46 m | Fused major scaffold, simplified tertiary tubes and 1,300 crossed multi-leaf clumps attached to sprays and fine branches |
| LOD1 | 13–26 m | Fused major scaffold, simplified tertiary tubes and 2,600 denser, smaller crossed multi-leaf clumps |
| LOD0 | Under 13 m | Fused major scaffold, simplified tertiary tubes and 70,000 individual textured two-triangle oak-leaf cards sampled across the complete source foliage |

The three distance thresholds have a two-metre hysteresis band to reduce LOD oscillation. Representation changes use a 0.35-second complementary screen-space dither: outgoing pixels are discarded exactly where incoming pixels are revealed. This avoids alpha blending and transparency sorting. Distance is the first implementation's proxy for projected size; camera-space projected bounds should replace fixed distances when this becomes a multi-tree scene system.

If the desired LOD is not ready, the viewer continues to use the best available representation. It does not wait for a worker or remove the current geometry before a replacement is uploaded.

## Current generation pipeline

1. The main thread reads the recipe and executes the first two branching iterations through tertiary structure. It omits terminal twigs, sprays and individual-leaf placement, but builds the fused trunk/root/primary/leader/secondary surface used by every refined LOD.
2. It builds a low-poly, irregular crown-volume shell fitted around the early leaf-bearing branch envelope, then builds the first wood mesh. The shell is deliberately opaque and contains no individual leaf cards; it preserves the broad crown silhouette while keeping first visibility inexpensive.
3. The viewer opens with LOD3.
4. A background CPU worker executes the remaining recipe structure for the same seed: terminal twigs, sprays and the fused wood surface are generated once. It then executes one foliage scatter pass at a time. Pass one publishes LOD2 as 1,300 branch-attached clusters, each represented by three crossed cards sampling a multi-leaf clump atlas. Pass two publishes LOD1 as 2,600 smaller crossed clumps. Passes three and four complete the canonical source foliage; LOD0 then selects 70,000 individual leaves from the full 311,136-leaf population. Each ready mesh package is published through a bounded queue.
5. The branch hierarchy and fused wood surface are unchanged by foliage passes. The worker builds the structural wood mesh once and shares that immutable CPU mesh with LOD2, LOD1 and LOD0; only the foliage mesh is rebuilt at each milestone. This reduces repeated CPU mesh work and avoids retaining duplicate wood buffers.
6. The render thread drains completed packages and performs all OpenGL resource uploads. The CPU worker never calls OpenGL.
7. Camera distance selects the best ready representation. The renderer retains already-ready LODs so that the user can move in either direction without rebuilding them.

The worker streams foliage-pass milestones but does not yet stream individual branch-growth milestones or refine independent subtrees. Structural generation (including the fused wood-surface build) remains one background stage; within foliage generation, each recipe pass is a separate publish point. Finer-grained subtree scheduling and scene-wide multi-tree prioritisation remain future work.

## Botanical and geometry rules

- Preserve deterministic recipe seeds and the existing branch-parent hierarchy.
- Treat the generated branch and leaf model as canonical; never derive botanical data back from a rendered mesh.
- LOD3 must approximate the crown as a whole. It uses an opaque, low-poly, irregular shell fitted to the leaf-bearing envelope, with no individually textured leaves.
- LOD2 and LOD1 use branch-attached foliage-clump assemblies. Each clump is three crossed two-triangle cards using a composite atlas tile containing multiple oak leaves; the clumps are sampled from spray, twig and fine-branch positions.
- The canonical tree retains all 311,136 leaf placements. LOD0 selects 70,000 individual leaves, each represented by a two-triangle UV quad with an alpha-cutout oak silhouette, central vein, lateral veins and per-leaf atlas variation from `oakforge/assets/oak_leaf_atlas.png`.
- The clump atlas is generated deterministically from the source leaf atlas at runtime and exported as `oak_foliage_cluster_atlas.png`. The viewer and OBJ exporter use the same two texture sources.
- Preserve the existing layered depth-impostor baker as an optional offline representation. It is not required for first visibility or progressive runtime operation.
- LOD3 and the refined LODs use the same fused trunk/root/primary/leader/secondary wood surface. Progressive runtime wood uses an 8-side build setting, resamples tertiary tubes to at most 4 segments for LOD2, LOD1 and LOD0, and omits terminal twig/spray tubes; the crown proxy or crossed clumps represent this fine growth at game distances. The English-oak fused wood surface uses a 0.20m voxel size for a lower polygon count.
- The canonical botanical model still creates 311,136 individual leaves, but LOD0 retains 70,000 cards at runtime. LOD2/LOD1 use 1,300/2,600 clumps; LOD3 is a crown shell. The `python -m oakforge.geometry_budget --instances 16` command reports the representation kind, element counts, actual triangles and UV-aware buffer bytes, failing if any tree exceeds the 200,000-triangle target. OBJ export includes matching MTL files plus `oak_leaf_atlas.png` and `oak_foliage_cluster_atlas.png`.

## Threading and lifecycle

- The caller thread owns the initial reduced tree generation, cluster-spray placement and first mesh build.
- The worker thread owns the remaining structural recipe stages and incremental foliage-pass execution, plus CPU construction of the crown proxy, crossed clump meshes and detailed leaf mesh. The unchanged simplified structural wood mesh is built once and shared.
- The OpenGL-owning thread exclusively owns GL resource creation, upload, drawing and release.
- Worker errors are surfaced in the viewer while the last uploaded representation remains drawable.
- The queue is bounded so the worker cannot run arbitrarily far ahead of the viewer.
- Closing the viewer signals the worker to stop publishing and joins it with a bounded timeout. Structural generation and an individual foliage pass are not pre-emptible in this version; a worker already inside one of those stages may finish it in the background after the bounded join.
- Regeneration is available after the current generation worker has finished. It starts a new seed with the reduced LOD3 path and launches a new worker. Cancelling an active recipe and per-subtree generation tokens are future refinements.

## Verification

- Unit tests verify initial scaffold structure, same-seed agreement with the full recipe, crown-shell construction, deterministic branch-attached clumps, crossed-card winding, leaf and clump atlas generation, packed mesh integrity, worker publication at each foliage-pass milestone and structural wood-mesh sharing across foliage LODs.
- A headless OpenGL smoke test creates the viewer under Xvfb, uploads every LOD level through the production UV-buffer and per-LOD texture path, draws the opaque crown shell, cluster cards and individual leaves, and saves per-LOD preview images.
- The existing complete regression suite continues to cover tree generation, branch attachment, crown shape, root topology, foliage placement, recipe interpretation and the original impostor renderer.
- GitHub Actions runs on `feature/progressive-tree-generation` and executes both the normal suite and the headless progressive viewer smoke test.

## Deliberate follow-on work

The first deliverable is an interactive single-tree viewer with a progressively available four-level LOD hierarchy. It is not yet a complete forest streaming engine. The next architectural steps are per-subtree jobs, visibility/projected-size scheduling across multiple trees, shared instanced twig/leaf-cluster assemblies, memory-budgeted CPU/GPU caches, temporal wind animation and a true volumetric far-field canopy representation.
