# OakForge — YAML-driven procedural English oak

OakForge is a small SpeedTree-inspired procedural tree generator. It builds an English oak (`Quercus robur`) from a hierarchy of growth tools, then displays the resulting shaded polygons in an interactive OpenGL 3.3 viewer.

The generator and renderer are deliberately separate. The growth tools produce a renderer-independent tree model; the viewer consumes that model. You can run the recipe interpreter and export geometry without installing the graphics dependencies.

The current English-oak recipe generates one trunk, seven main buttress roots, thirteen lateral surface roots and eleven finer feeder roots for its default seed. Its crown contains eight broad primary scaffold limbs, six upper leaders, 104 secondary boughs, 624 tertiary branches, 4,992 terminal twigs and 1,490 leaf-bearing sprays, carrying 311,136 individual oak leaves. The upper leaders fork into the crown and their descendants are gently steered towards a spread-out upper-crown target, avoiding a single exposed central pole. The coarse trunk, all root tiers, primary limbs, upper leaders and secondary limbs are fused into a continuous polygon surface. Main roots and offshoots can be generated procedurally from a seed or positioned individually in YAML. Use the supplied recipes to inspect the complete tree, an isolated trunk with procedural root networks, or an isolated trunk with manually positioned roots.

## What is included

- **Trunk generator** — creates an irregular, curved and tapered main stem with root flare and branch-collar cues.
- **Surface-root tools** — grow broad primary buttresses from the lower bole, then attach lateral and feeder roots with procedural or manually defined placements. Procedural placement is seed-driven and supports radial, jittered, random and staggered distributions.
- **Branching tool** — creates broad primary limbs, a distinct set of upper crown leaders, then secondary and tertiary boughs with inherited outward direction, height adaptation, upper-crown convergence and a configurable crown envelope.
- **Canopy spray tool path** — grows short leaf-bearing shoots from primary limbs, upper leaders, secondary and tertiary boughs to fill the crown interior and upper centre without replacing the visible scaffold with a solid sphere.
- **Twig tool** — creates the terminal shoots that carry foliage.
- **Oak foliage tool** — scatters individually oriented, lobed oak leaves in grouped whorls along terminal twigs and inner-canopy sprays, with configurable radial spread and colour variation.
- **Wood-surface tool** — fuses the trunk, roots and coarse limbs into one continuous implicit surface.
- **Packed mesh builders** — expose separate polygon wood (`build_wood_mesh`), foliage (`build_foliage_mesh`) and complete legacy (`build_tree_mesh`) meshes. Terminal twigs/sprays can be routed to either the polygon mesh or canopy impostor.
- **Layered canopy depth-impostor baker** — captures multiple depth-peeling layers of leaves into multi-view colour, normal and 16-bit depth atlases. Structural wood remains a separate polygon mesh for conventional LOD.
- **YAML recipe engine** — calls named tools and controls their execution order.
- **Progressive OpenGL viewer** — starts with a branch-anchored clustered canopy and coarse scaffold, then streams finer tree representations from a CPU worker. Distance-based LOD selection never waits for a finer representation. OpenGL uploads remain on the context-owning thread.
- **Legacy impostor viewer** — retains the baked layered depth-impostor versus polygon comparison for explicit A/B inspection.
- **OBJ exporter** — writes the generated geometry for inspection in Blender or another 3D package.

All source files are separated by responsibility under `oakforge/`. The default oak's structure and all tool inputs are declared in `configs/english_oak.yaml`.

## Requirements

- Python 3.10 or later
- NumPy
- PyYAML
- scikit-image for fused coarse-wood surfaces
- GLFW and PyOpenGL for the interactive viewer
- A desktop-capable OpenGL 3.3 driver for the viewer

Install the Python dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

On Windows, install a current graphics driver and run from a normal desktop session. The viewer uses GLFW to create an OpenGL 3.3 core-profile window.

## Run the progressive tree viewer

From the project root:

```bash
python main.py
```

OakForge first constructs a coarse tree with a real branch scaffold and a compact, deterministic cluster canopy. It then generates the full tree and prepares successively finer LODs on a background CPU worker. The viewer stays responsive and continues drawing the best available representation while refinement runs.

Use another recipe with:

```bash
python main.py --config configs/english_oak.yaml
```

Automatic LOD is selected by camera distance, with a two-metre hysteresis band around each fixed threshold in this first implementation:

| LOD | Camera distance | Representation |
|---|---:|---|
| LOD3 | 46 m and beyond | Primary/leader/secondary scaffold with branch-anchored foliage clusters |
| LOD2 | 26–46 m | Adds tertiary structure and denser clustered foliage |
| LOD1 | 13–26 m | Full branch hierarchy and one-eighth sampling of actual leaves |
| LOD0 | Under 13 m | Full branch hierarchy and all generated detailed leaves |

If the requested LOD is not ready yet, the viewer keeps the closest higher-quality representation already prepared. It never blocks the render loop waiting for the worker.

Viewer controls:

| Input | Action |
|---|---|
| Left mouse drag | Orbit around the tree |
| Mouse wheel | Zoom |
| `A` | Enable / pause automatic distance-based LOD |
| `1`–`5` | Manual structural inspection; pauses automatic LOD |
| `L` | Toggle canopy visibility |
| `W` | Toggle polygon wireframe |
| `V` | Toggle VSync (off by default for benchmarking) |
| `R` | Regenerate with the next seed after the current worker has finished |
| `F` | Reset the camera |
| `Escape` | Close the viewer |

The previous baked-impostor A/B viewer remains available with `python main.py --legacy-impostor-viewer`. The standalone `python -m oakforge.render.impostor` command and its 64-view layered depth captures are unchanged.

## Bake a canopy depth impostor

Bake the leaves separately from the polygon-based trunk and branches:

```bash
python -m oakforge.render.impostor --config configs/english_oak.yaml --output build/english_oak_canopy
```

The default captures leaves only. The legacy impostor viewer automatically bakes and caches a 64-view, 512px-per-view, four-layer canopy depth impostor the first time a recipe/seed/quality combination is opened. The default progressive viewer does not require this bake. Use `--impostor-views`, `--impostor-tile-size` and `--impostor-layers` to change that quality. The runtime forward-projects the actual reconstructed depth samples as depth-tested point splats. It selects one coherent capture view at a time rather than averaging unrelated leaf surfaces from neighbouring views. Add `--include-twigs` to the standalone bake command to capture terminal twig and spray tubes too.

In the legacy viewer, press `I` to switch between leaf polygons and the depth impostor. The default progressive viewer uses cluster and polygon LODs and does not require a pre-baked atlas. The diagnostics panel reports source-model and active triangle counts, per-representation geometry and tracked canopy allocations, device/process VRAM where supported, GPU utilisation, current FPS and frame time, and retained per-mode average FPS, CPU frame time and GPU scene time. Press `B` to reset the A/B averages. VSync is off by default so the two modes can be benchmarked without a 60Hz cap. Install `nvidia-ml-py` (included in requirements) for NVIDIA device/process telemetry; OpenGL VRAM querying is used as a fallback when available. Device/process VRAM is driver telemetry, while the viewer's tracked GPU allocations isolate its known buffers and textures. The default texture arrays require approximately 640 MiB of GPU storage. See [the canopy impostor notes](docs/DEPTH_IMPOSTORS.md) for details, quality trade-offs and metric definitions.

## Generate without a graphics window

Print the generated component counts:

```bash
python main.py --stats
```

Export the tree as an OBJ file:

```bash
python main.py --export-obj output/english_oak.obj
```

The OBJ includes vertex colours and normals. It can be opened in Blender and used as a geometry inspection artifact. The output is intentionally simple; material authoring and texture maps are not part of this first version.

## How the recipe works

The YAML file contains two top-level sections:

- `tree` contains species information, the random seed and named parameter groups.
- `pipeline` is an ordered list of tool calls and control-flow actions.

The oak pipeline creates the trunk and roots, grows the lower primary scaffold and a separate set of upper leaders, then generates secondary, tertiary and terminal branching tiers. Dedicated `for` loops add leaf-bearing sprays to the scaffold, upper leaders and later branch tiers to fill the upper and inner canopy. A final `if/then/else` controls foliage, and a `do_while` loop scatters the configured leaf passes over both terminal twigs and sprays.

A shortened example:

```yaml
pipeline:
  - call:
      tool: trunk.generate
      args:
        height: "$tree.parameters.trunk.height"
      save_as: trunk
      append_to: groups.trunk

  - for:
      each: "$groups.trunk"
      as: parent_branch
      do:
        - call:
            tool: branching.spawn
            args:
              parent: "$parent_branch"
              role: primary
              count: 8
            append_to: groups.primary

  - if:
      condition: "$tree.parameters.foliage.enabled"
      then:
        - for:
            each: "$groups.twigs"
            as: terminal_twig
            do:
              - call:
                  tool: foliage.scatter_oak
                  args:
                    twig: "$terminal_twig"
                    count: 8
                  append_to: groups.leaves
      else:
        - set:
            foliage_status: disabled
```

### Supported actions

| Action | Purpose |
|---|---|
| `call` | Invoke any registered tool by its name; supports `args`, `save_as` and `append_to` |
| `set` | Assign one or more recipe variables |
| `if` | Conditional execution with `condition`, `then` and `else` |
| `for` | Iterate over a list or a `range` expression |
| `while` | Test the condition before each iteration |
| `do_while` | Run the body once, then test the condition |
| `repeat` | Run a fixed number of iterations |

Expressions use references such as `$tree.parameters.primary.count` and safe operators such as `eq`, `lt`, `gte`, `and`, `or`, `not`, `add`, `multiply`, `divide`, `len` and `range`. The interpreter does not use Python `eval` or execute arbitrary YAML tags. `while`, `do_while` and `repeat` have iteration guards to prevent an accidental infinite recipe.

### Adding a tool

Create a separate Python module under `oakforge/tools/` and implement a function with this signature:

```python
def create_something(context, arguments):
    ...
```

Register it in `oakforge/tools/registry.py` under a descriptive name. The recipe can then call it by name, pass structured inputs and place its result in a named group. The recipe engine does not know what a trunk, branch, twig or leaf is; those behaviours live in their own tools.

## Source layout

```text
oakforge/
  core/
    context.py          Shared recipe variables and seeded randomness
    geometry.py         Curve, direction and coordinate-frame helpers
    mesh.py             Branch tube and lobed leaf polygon generation
    models.py           Branch, Leaf and TreeModel data structures
    recipe_engine.py    YAML action interpreter and control flow
  tools/
    registry.py         Named tool registration
    trunk.py            Trunk generator
    branching.py        Scaffold and upper-leader growth, crown convergence and envelope shaping
    twig.py             Terminal twigs and inner-canopy spray generator
    roots.py            Main buttress, lateral and feeder-root generation
    wood_surface.py     Smooth union of trunk, roots and coarse limbs
    foliage.py          Oak foliage placement
  render/
    opengl_viewer.py    OpenGL 3.3 interactive viewer
configs/
  english_oak.yaml      Complete English oak growth recipe
  oak_trunk_only.yaml   Isolated trunk and procedural roots
  oak_roots_manual.yaml Isolated trunk and manual root placements
main.py                 Command-line entry point
```

## Tests

Run the test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover the full hierarchical oak structure, parent relationships, upper-leader convergence and upper-centre foliage coverage, crown-envelope bounds, inner-canopy spray attachment, root and trunk geometry, closed fused-wood topology, three-dimensional foliage distribution, mesh output, deterministic seeds, recipe-controlled tool order, all loop forms, conditionals and loop safety guards.

## Current scope and limitations

This is a structural vegetation generator, not a clone of the complete SpeedTree product. It currently has deterministic seeded variation, a five-level structural hierarchy, a fused coarse-wood surface, buttress roots, lobed oak-leaf polygons and an inspection viewer. It does not yet implement collision-aware growth, spatial pruning volumes, artist editing handles, bark UVs, detailed bark materials, seasonal leaf states, wind animation or automatic multi-tier polygon LOD generation. The canopy impostor path is currently an experimental viewer comparison rather than a production-ready game renderer. The modular tool graph is intended to make those systems additive rather than requiring a rewrite of the recipe engine.

## Trunk development

The trunk generator is implemented independently in `oakforge/tools/trunk.py`.
Its YAML inputs define the stem height, basal radius, upper-stem radius, crown radius,
root-flare height, basal buttress count and strength, stem sweep, taper and broad
radius variation. The mesh builder turns the trunk's cross-section profile into
shaded polygons and applies the buttress shape near ground level.

To inspect the stem without branches or foliage:

```bash
python main.py --config configs/oak_trunk_only.yaml --export-obj output/oak_trunk.obj
```

The trunk study is intentionally separate from the complete oak recipe so the bole,
buttress roots and root branching network can be tuned before the crown is considered.


## Trunk shaping study

Use `python main.py --config configs/oak_trunk_only.yaml --stats` to generate the English oak's isolated bole and multi-tier surface-root system. The trunk tool exposes root flare, radial root spread, buttress count, non-linear taper, centreline curvature, cross-section irregularity and YAML-defined limb-collar swellings. Limb collars are visual junction cues; actual primary branches remain separate recipe-driven tools.

The procedural and manual root studies can be generated or exported with:

```bash
python main.py --config configs/oak_trunk_only.yaml --stats
python main.py --config configs/oak_trunk_only.yaml --export-obj output/oak_trunk.obj
python main.py --config configs/oak_roots_manual.yaml --export-obj output/oak_roots_manual.obj
```


### Trunk development recipe

`configs/oak_trunk_only.yaml` isolates the English oak bole and invokes `roots.spread` after `trunk.generate`. It then calls `roots.branch` twice to produce lateral and feeder roots. All root tiers remain separate child structures in the model and are fused with the bole by `wood_surface.build`.

Root placement has two YAML-driven modes:

- **Procedural:** explicitly set `distribution` and provide `count`. `radial` creates evenly spaced roots, `radial_jittered` creates an even base pattern with seeded angular variation, and `random` samples azimuths across the full circle. Length, collar height, thickness and direction can vary independently while remaining reproducible from the tree seed.
- **Manual:** provide a `placements` list. Each item requires `angle_degrees` and can set its own `attachment_height` (or `attachment_height_fraction`), `length`, `base_radius`, `tip_radius`, `tip_height`, `sweep_degrees` and `segments`. Omitted per-root values inherit the shared defaults. Manual placements are not randomly repositioned.

A recipe must explicitly choose one mode: if neither `distribution` nor `placements` is supplied, generation stops with a configuration error instead of guessing how the roots should be positioned.

`configs/english_oak.yaml` demonstrates procedural distribution. `configs/oak_roots_manual.yaml` demonstrates six individually positioned primary roots. The root tool samples the trunk centreline and radius at each requested attachment height, embeds the root collar slightly into the bole, then creates a descending swept profile for the fused wood surface.

Secondary and feeder roots are added with `roots.branch`. The tool accepts a named parent-root tier, seeded distribution (`staggered` or `random`), attachment-position range, branch-angle range, length and thickness variation. Manual placements can instead reference a parent's `placement_id` and specify `parent_position`, `angle_degrees`, `side`, length and radii. A generation pass must explicitly choose one of these modes. The default oak recipes generate two branching tiers beyond the main buttresses, with the thinnest feeder roots tapering beneath the soil plane so their tips disappear naturally instead of ending as exposed spikes.

The unified wood mesher uses a near-rounded cross-section for surface roots. Main buttresses receive a broad proximal collar and a modest mid-span swell; lateral and feeder roots use the shared YAML cross-section settings unless an individual placement overrides them. Roots can finish below the ground plane: the fused mesher clips the underground geometry, allowing the taper to disappear into the soil rather than exposing a pointed cap. Avoid setting `root_vertical_scale` and `root_thickness_scale` too far apart, as extreme ratios make roots read like flat fins.


## English oak structural target

The default recipe is tuned around an old, broad-crowned English oak rather than a young straight-stemmed plantation tree. The lower bole is heavy and fluted, the surface roots merge into the buttress flare, major scaffold limbs rise and spread from the trunk, and successive branching levels become progressively shorter and finer. The recipe favours a deliberately irregular crown rather than a perfect sphere.

The coarse-wood surface uses a smooth implicit union so the trunk, roots, scaffold limbs and secondary branches meet without visible tube intersections. Short segments along one branch are joined with a continuous minimum-distance field, avoiding a chain of bead-like joints. Fine tertiary and terminal branches remain separate tapered tubes so their hierarchy stays inspectable.

All vertex attributes are held in packed single-precision buffers. This keeps the high leaf count practical without changing the renderer-independent tree data model.


### Root placement and shape

`roots.spread` creates the primary buttress roots. Choose `distribution: radial`,
`radial_jittered` or `random` for seeded placement, or provide a `placements` list
with an explicit `angle_degrees`, attachment height, length and radius for each root.
`roots.branch` adds lateral or feeder roots to named parent roots. It also requires
either a procedural `distribution` or explicit `placements`; a missing choice is
reported as a recipe error rather than silently selecting a placement mode.

The fused wood surface uses near-rounded root cross-sections to avoid the thin,
blade-like buttresses produced by strongly flattened ellipses. Root collars swell
near the bole, root bodies carry a subtle mid-span belly, and tips taper below the
terrain plane so they disappear into the soil. Procedural roots use seeded lateral
meander and variable angular spacing; manual roots remain fixed to their YAML
coordinates unless curve variation is explicitly supplied.
