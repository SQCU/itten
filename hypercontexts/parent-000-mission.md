# Parent Hypercontext: Visualization Suite for Local Spectral Graph Methods

## Mission

Implement three visualizers that prove the local spectral methods actually work
by making their operation visible and interactive.

## Architectural Constraint (CRITICAL)

All three tools MUST follow this pattern:
```
┌─────────────────────────────────────────────────────────────┐
│  CORE: Pure data transform (stdin JSON → stdout JSON)      │
│  ├── No GUI dependencies in core logic                     │
│  ├── Serializable state (can save/load sessions)           │
│  └── Deterministic given same input                        │
├─────────────────────────────────────────────────────────────┤
│  CLI: Thin wrapper (argparse/typer)                         │
│  ├── --input file.json / stdin                             │
│  ├── --output file.json / stdout                           │
│  └── --render output.png (optional)                        │
├─────────────────────────────────────────────────────────────┤
│  GUI: Optional view/controller layer                        │
│  ├── Reads same JSON format                                │
│  ├── Writes same JSON format                               │
│  └── Just mediates human ↔ core                            │
└─────────────────────────────────────────────────────────────┘
```

Works identically: `cat input.json | tool` vs `tool --gui < input.json`

## The Three Deliverables

### 1. SimCity Lattice Mosaic (`lattice_mosaic/`)
- Mosaic tiles on convex surface (torus, sphere, egg)
- Shows neighborhoods as colored regions
- Bottlenecks visible as tile boundaries
- Demo: periodic pinch → visible "districts"

### 2. Infinite Bisection Visualizer (`bisect_viz/`)
- Graph grows from seed, animatable
- Bisection shown as red/blue coloring
- Fiedler values as node size/intensity
- Demo: watch partition emerge as graph reveals

### 3. Texture Synthesis Editor (`texture_synth/`)
- Input: 16x16-32x32 bitmap aperture (draw amongus)
- Output: Multiscale texture on 3D surface
- Compilable to PNG/normal-map (standalone asset)
- Demo: doodle → textured egg render

## Communication Protocol

### On Entry
1. Read this file (`hypercontexts/parent-000-mission.md`)
2. Understand your assigned deliverable
3. If spawning sub-subagents, write `hypercontexts/your-id-spawn.md` first

### On Exit
1. Write `hypercontexts/your-id-exit.md` with:
   - BLUF: Did it work? What exists now?
   - Files created/modified
   - Blockers or decisions made
   - Heat-ranked findings

### If Spawning Children
Tell them: "Read hypercontexts/ for parent chain. Write your exit context.
Your local parent is [X], doing [Y]. Report back in structured format."

## Dependencies Available
- numpy (already installed)
- Need: pillow (images), moderngl or similar (3D), possibly pygame/pyglet

## Codebase Location
`/home/bigboi/itten/`
- Existing: graph_view.py, spectral_ops.py, lattice_extrusion.py, texture_synthesis.py
- Extend these, don't rewrite core spectral logic

## Success Criteria
Each tool must:
1. Run headless: `echo '{}' | python tool.py --demo > output.json`
2. Render: `python tool.py --demo --render demo.png`
3. (Optional) GUI: `python tool.py --gui`
