# OakForge project state — 21 November 2026

## Recovery and source of truth

The authoritative source is the complete `oakforge/` project in the latest archive. The persistent Library folder is `/Projects/OakForge/`.

- `OakForge_source_baseline_2026-11-17.zip` — unchanged recovered baseline before canopy work.
- `OakForge_canopy_2026-11-17.zip` — earlier canopy revision.
- `OakForge_crown_convergence_2026-11-17.zip` — previous crown-architecture revision.
- `OakForge_branch_integrity_2026-11-17.zip` — **latest source**, including sealed branch tubes, attachment collars, terminal twig thickness and canopy-boundary insets.
- `OakForge_project_state_branch_integrity_2026-11-17.md` — current recovery record.

To recover after a conversation interruption, download and extract `OakForge_branch_integrity_2026-11-17.zip`, then continue from its `oakforge/` directory. Keep the earlier archives as rollback points. Do not depend on generated previews or the active conversation for source code.

## Progressive runtime generation — feature/progressive-tree-generation

The progressive runtime implementation is under active development on the dedicated branch `feature/progressive-tree-generation`, based on the then-current depth-impostor LOD branch. The default viewer path is now designed to show a reduced tree first and refine it in the background:

- LOD3 builds the trunk/root scaffold through tertiary branches and 17,760 deterministic branch-anchored proxy leaves.
- LOD2 uses the complete branch hierarchy, including twigs and sprays, with one-quarter of the 311,136 actual leaf placements enlarged 1.2×.
- LOD1 uses the complete hierarchy with one-half of actual leaf placements and a more detailed leaf outline.
- LOD0 uses the complete hierarchy and all 311,136 actual leaves with the full oak-lobed profile.
- Far, mid, near and hero foliage use progressively richer polygon profiles (9, 13, 17 and 21 triangles per leaf respectively).
- Automatic selection uses camera-distance bands of 46 m, 26 m and 13 m with two metres of hysteresis. Representation changes use a 0.35-second complementary screen-space dither.
- The initial reduced generation is synchronous. The background worker completes twigs, sprays and the fused wood surface once, then runs leaf scatter passes incrementally: default pass 1 publishes LOD2, pass 2 publishes LOD1, and passes 3–4 finish LOD0. It does not yet stream individual branch-growth milestones or independently refine subtrees.
- Structural generation and foliage-pass execution run on a CPU worker away from the render loop. The immutable structural wood mesh is built once and shared by LOD2/LOD1/LOD0; only foliage meshes change per pass. OpenGL uploads remain on the context-owning viewer thread.
- The previous baked depth-impostor comparison viewer remains available through `python main.py --legacy-impostor-viewer`.

Architecture and implementation notes: `docs/PROGRESSIVE_GENERATION.md`.

## Current implementation

- Modular Python generator with YAML-driven recipe execution and deterministic seed control.
- English oak bole and three-tier root architecture: primary buttresses, lateral surface roots and feeder roots.
- Manual and seeded procedural root placement.
- Eight broad primary scaffold limbs distributed along the lower bole.
- Six separate upper leaders attached in the upper bole. Each leader grows eight secondary boughs which fork into tertiary branches and terminal shoots.
- The main bole terminates at 8.8 m inside the lower crown rather than continuing as a bare central spear.
- Branches in the upper crown use height-weighted convergence steering towards a spread-out apex target.
- The crown envelope is broad and rounded, with a domed upper boundary and a controlled lower canopy boundary.
- Leaf-bearing `spray` shoots are generated from primary limbs, upper leaders, secondary boughs and tertiary branches. They are stored separately in `groups.sprays`.
- Individually oriented oak leaves are distributed on terminal twigs and canopy sprays.
- The coarse fused-wood surface contains the bole, all root tiers, primary limbs, upper leaders and secondary limbs.
- The trunk surface now has collar cues for both primary scaffold limbs and the six upper leaders.
- Separately meshed tertiary branches, twigs and sprays have sealed basal and terminal caps. Child bases start slightly inside their parent axes and widen into a short attachment collar.
- Fine woody tiers have configurable minimum base/tip radii. In the default recipe, twig bases are at least 14 mm and twig tips at least 6 mm.
- Terminal twigs and sprays are grown inside an inset crown envelope so their wood ends sit within the leaf-bearing mass.
- Leaf geometry is tessellated in bounded batches and packed directly into mesh buffers.

## Current default recipe statistics

Command:

```bash
python main.py --stats
```

Verified default-seed output:

- Total branches: 7,256
- Leaves: 311,136
- Groups: trunk 1; roots 31; primary 8; leaders 6; secondary 104; tertiary 624; twigs 4,992; sprays 1,490.
- Root tiers: 7 primary buttresses, 13 lateral roots, 11 feeder roots.
- Main bole height: 8.8 m.
- Crown envelope: horizontal radii 9.2 m and 8.6 m; top 14.5 m; lower boundary 3.45 m.
- Upper canopy (leaf base height at least 12 m): 28,498 leaves.
- Upper interior (height at least 12 m and normalized crown radius below 0.45): 21,011 leaves.

Counts are deterministic for seed 27.

## Verification

Command:

```bash
python -m unittest discover -s tests -v
```

Latest complete run: **28 tests passed in 31.1 seconds**.

Coverage includes branch hierarchy and parent references, crown bounds, upper leaders and upper-centre foliage, spray attachment, trunk/leader collars, minimum twig radii, terminal branch insets, sealed branch-tube ends, root geometry and placement, closed/outward-facing fused-surface topology, mesh buffers and deterministic recipe execution.

A full branch-only tree mesh was built successfully: 1,102,200 triangles with aligned position, normal and colour buffers. A full leaf-bearing mesh was also constructed and returned 7,635,404 triangles with aligned buffers; the headless Linux process then exceeded its timeout during interpreter teardown of the large in-memory leaf model. The OpenGL viewer has not yet been revalidated on the user's Windows desktop after these changes.

## Design boundaries and remaining work

- Soil and terrain integration are intentionally out of scope; roots remain tree geometry only.
- The crown is not an opaque sphere. Branch tiers remain distinct, with sprays filling gaps around the visible scaffold.
- No bark material/UV authoring, seasonal foliage, wind animation, collision-aware pruning, multi-tree forest scheduler or per-subtree generation task system yet.
- Close-detail twigs are now deliberately thicker; distant LOD should later replace or simplify fine twig geometry rather than thinning these close-detail supports until they disappear.
- The immediate next validation is to run the default viewer on Windows, toggle leaves with `L`, orbit around branch forks, and check whether the capped/insetted twig system removes the visible holes and unsupported leaf clusters.

## Useful commands

```bash
python -m pip install -r requirements.txt
python main.py --stats
python main.py
python main.py --config configs/oak_trunk_only.yaml --stats
python main.py --config configs/oak_roots_manual.yaml --stats
python -m unittest discover -s tests -v
```
