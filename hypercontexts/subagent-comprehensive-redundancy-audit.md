# Hypercontext: Comprehensive Redundancy Audit

## Mission
Find ALL features that are redundantly reimplemented across the project. This is a comprehensive audit of every module, every file, every function that duplicates logic elsewhere.

## Audit Scope

### Modules to Examine
```
/home/bigboi/itten/
├── spectral_ops_fast.py          # Canonical spectral kernels
├── texture_synth/                 # Unified TUI system (new)
│   ├── geometry/
│   ├── inputs/
│   ├── render/
│   ├── synthesis/
│   └── tui/
├── texture_synth_v2/              # V2 rewrite (older)
│   ├── synthesize.py
│   ├── spectral_etch.py
│   ├── render_egg.py
│   ├── normals.py
│   ├── patterns.py
│   └── cli.py
├── texture_editor/                # Standalone editor
├── lattice_extrusion.py          # Root level
├── lattice_extrusion_v2/         # V2 of lattice
├── lattice_mosaic/               # Mosaic system
├── bisect_viz/                   # Graph bisection viz
├── local_pathfinding.py          # Root level
├── spectral_pathfind.py          # Root level (new)
├── graph_view.py                 # Root level
├── infinite_bisect.py            # Root level
└── texture_synthesis.py          # Root level (old?)
```

## Categories to Audit

### 1. 3D Rendering
Find all implementations of:
- Ray-mesh intersection
- UV mapping / texture sampling
- Lighting calculations (diffuse, specular)
- Normal perturbation
- 3D primitive generation (sphere, egg, icosahedron)

### 2. 2D Image Processing
Find all implementations of:
- Height field to normal map conversion
- Gradient computation
- Image blurring / smoothing
- Edge detection
- Contour extraction

### 3. Spectral Graph Operations
Find all implementations of:
- Laplacian matrix construction
- Eigenvector computation (Lanczos, dense, scipy)
- Heat diffusion
- Fiedler vector computation
- Graph partitioning / bisection

### 4. Pattern Generation
Find all implementations of:
- Amongus silhouette
- Checkerboard
- Noise (Perlin, simplex, random)
- Gradients
- SVG parsing
- Dragon curve / L-systems

### 5. Mesh Operations
Find all implementations of:
- Mesh data structures
- Vertex/face manipulation
- UV unwrapping
- Mesh transformations (translate, rotate, scale)
- Boolean operations (fuse, chop)

### 6. CLI / TUI
Find all implementations of:
- Argument parsing
- Interactive REPL
- Command parsing
- Configuration loading

### 7. File I/O
Find all implementations of:
- PNG saving
- Image loading
- Mesh export
- JSON/YAML config

### 8. Texture Synthesis Pipeline
Find all implementations of:
- Carrier + operand combination
- Theta-weighted spectral blending
- Nodal line extraction
- Height field synthesis

## Audit Method

For each category, run:
```bash
# Find function definitions
grep -rn "def function_name" --include="*.py" .

# Find class definitions
grep -rn "class ClassName" --include="*.py" .

# Find specific patterns
grep -rn "pattern" --include="*.py" .
```

## Output Format

For each redundancy found, document:

```
## [Category]: [Feature Name]

### Implementations Found

| Location | Function/Class | Lines | Notes |
|----------|----------------|-------|-------|
| file1.py:10 | func_a() | 50 | Original |
| file2.py:20 | func_b() | 45 | Duplicate |
| file3.py:30 | ClassC.method() | 60 | Variant |

### Differences
- Implementation A does X
- Implementation B does Y instead
- Implementation C adds Z

### Recommendation
- Keep: [which one]
- Delete: [which ones]
- Merge: [if needed]

### Priority: [1-5]
### Peril: [1-5]
```

## Deliverables

1. Complete redundancy table for all 8 categories
2. Dependency graph showing which modules import from which
3. Recommendations for consolidation
4. Risk assessment for each redundancy
