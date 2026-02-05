# Spec V2: Spectral-Dependent Visualization Tools

## Core Principle

Every operation MUST use local spectral queries that:
1. Cannot be replicated by fixed-kernel convolution
2. Cannot be computed via full matrix materialization (intractable)
3. Depend on graph structure beyond the explicit hop radius

---

## Tool 1: Lattice Extrusion with Fiedler-Aligned Geometry

### Concept

A "SimCity landscape" of islands, coasts, and bridges. Extrusion operations
add new lattice layers, but the geometry (where edges go, what period/shape)
is determined by LOCAL SPECTRAL PROPERTIES of the underlying territory.

### The Territory Graph

```python
class TerritoryGraph:
    """
    Islands (high λ₂) connected by bridges (low λ₂).
    Coasts are transition zones.
    """
    def __init__(self, islands: List[Region], bridges: List[Bridge]):
        # Islands: dense lattice patches
        # Bridges: narrow connections (1-3 edges wide)
        # This creates natural spectral variation
```

### Operations

**1. Expansion-Gated Extrusion**
```python
def expansion_gated_extrude(graph, threshold=1.5):
    """Only extrude where local expansion exceeds threshold."""
    for node in frontier:
        λ₂ = local_expansion_estimate(graph, node, radius=3, lanczos=15)
        if λ₂ > threshold:
            add_extruded_geometry(node)
        # Bridges have low λ₂ → extrusion skips them initially
        # Islands fill first, bridges fill later as neighbors provide support
```

**2. Fiedler-Aligned Triangles/Parallelograms**
```python
def fiedler_aligned_geometry(graph, node, geometry_type):
    """Orient new geometry along local Fiedler gradient."""
    fiedler, _ = local_fiedler_vector(graph, [node], radius=3, iterations=20)

    # Compute gradient direction
    gradient = np.zeros(2)
    for neighbor in graph.neighbors(node):
        if neighbor in fiedler:
            delta_fiedler = fiedler[neighbor] - fiedler[node]
            delta_pos = coord(neighbor) - coord(node)
            gradient += delta_fiedler * delta_pos

    # Triangles: point along gradient (toward partition boundary)
    # Parallelograms: align long axis with gradient
    return orient_geometry(geometry_type, gradient)
```

**3. Spectral-Gap Adaptive Period**
```python
def adaptive_period(graph, node, base_period=4):
    """Finer detail near bottlenecks, coarser in open regions."""
    λ₂ = local_expansion_estimate(graph, node)
    # Low λ₂ (bottleneck) → small period (fine detail)
    # High λ₂ (open) → large period (coarse detail)
    return max(2, int(base_period * (λ₂ / 2.0)))
```

### Output

- 3D mesh vertices from lattice coordinates
- Project onto convex surface (torus, egg, sphere)
- Color by: island membership, expansion value, Fiedler sign
- Render with depth shading

### Demo Scenario

1. Create territory: 3 islands connected by 2 bridges
2. Run expansion-gated extrusion for N steps
3. Watch: islands fill with aligned geometry, bridges stay sparse
4. As islands grow, their spectral influence "reaches" bridges
5. Eventually bridges fill with geometry oriented toward island centers
6. Final: 3D render on egg surface showing island/bridge structure

---

## Tool 2: Structure-Aware Texture Synthesis

### Concept

Carrier texture + Operand bitmap → Modulated output where the
spectral structure of BOTH inputs affects the result.

### Key Insight

The carrier texture defines a GRAPH (not just pixels):
- Nodes = pixels
- Edges = adjacency, BUT edge weights depend on carrier values
- Similar carrier values → strong edge (smooth region)
- Different carrier values → weak edge (edge/boundary)

This graph has spectral properties that vary spatially.

### Operations

**1. Carrier-Graph Construction**
```python
def carrier_to_graph(carrier_image, edge_threshold=0.1):
    """Convert carrier texture to weighted graph."""
    graph = WeightedImageGraph(carrier_image.shape)

    for (x, y) in pixels:
        for (nx, ny) in neighbors(x, y):
            # Edge weight = similarity of carrier values
            diff = abs(carrier[x,y] - carrier[nx,ny])
            weight = exp(-diff / edge_threshold)
            graph.set_edge_weight((x,y), (nx,ny), weight)

    return graph
    # Now spectral ops on this graph respect carrier structure
```

**2. Structure-Aware Spread**
```python
def structure_aware_kernel(carrier_graph, center, rotation_angle):
    """
    Spectral kernel that follows carrier structure.
    Near edges: anisotropic spread along edge
    Smooth regions: isotropic spread
    """
    # This is spectral_smoothing_kernel but on the WEIGHTED carrier graph
    # The weights cause the heat diffusion to follow carrier structure
    return spectral_smoothing_kernel(
        carrier_graph,  # weighted by carrier similarity
        center,
        rotation_angle
    )
```

**3. Expansion-Weighted Scale**
```python
def adaptive_scale_map(carrier_graph):
    """Compute rotation angle for each pixel based on local expansion."""
    scale_map = {}
    for node in carrier_graph.nodes():
        λ₂ = local_expansion_estimate(carrier_graph, node, radius=2)
        # High expansion (smooth) → large angle (global spread)
        # Low expansion (edge) → small angle (stay local)
        scale_map[node] = λ₂ / λ₂_max * (π/2)
    return scale_map
```

**4. Modulation**
```python
def modulate(carrier_graph, operand_bitmap, scale_map):
    """Apply operand to carrier using structure-aware kernels."""
    output = np.zeros_like(carrier_image)

    for (x, y) in pixels:
        if operand_bitmap[x % op_h, y % op_w] > 0:
            # Get structure-aware kernel at this location
            θ = scale_map[(x, y)]
            kernel = structure_aware_kernel(carrier_graph, (x,y), θ)

            # Apply kernel to accumulate influence
            for (kx, ky), weight in kernel.items():
                output[kx, ky] += weight * operand_bitmap[x % op_h, y % op_w]

    return output
```

**5. Normal Map Generation**
```python
def height_to_normals(height_field):
    """Standard height → normal map conversion."""
    dx = np.roll(height_field, -1, axis=1) - height_field
    dy = np.roll(height_field, -1, axis=0) - height_field

    normals = np.zeros((*height_field.shape, 3))
    normals[..., 0] = -dx
    normals[..., 1] = -dy
    normals[..., 2] = 1.0

    # Normalize
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norms + 1e-8)

    return normals
```

### Output

- Height map (grayscale)
- Normal map (RGB encoded)
- 3D egg render with normal-mapped lighting
- Show carrier structure visible through modulation

### Demo Scenarios

**Demo A: Amongus carrier, checkerboard operand**
- Carrier: tiled amongus silhouettes (creates edges at amongus boundaries)
- Operand: checkerboard pattern
- Result: checkerboard bumps that FOLLOW amongus outlines
- The spectral kernel spreads along amongus edges, not across them

**Demo B: Checkerboard carrier, amongus operand**
- Carrier: checkerboard (edges at square boundaries)
- Operand: single amongus bitmap
- Result: amongus bump that respects checkerboard grid
- Influence spreads within squares, weakens at boundaries

---

## Architecture (Same as V1)

```
┌─────────────────────────────────────────────────────────────┐
│  CORE: Pure data transform (stdin JSON → stdout JSON)      │
│  ├── No GUI dependencies in core logic                     │
│  ├── Serializable state (can save/load sessions)           │
│  └── Deterministic given same input                        │
├─────────────────────────────────────────────────────────────┤
│  CLI: Thin wrapper (argparse)                              │
│  ├── --input file.json / stdin                             │
│  ├── --output file.json / stdout                           │
│  └── --render output.png                                   │
├─────────────────────────────────────────────────────────────┤
│  Render: PIL + numpy for 2D, simple raycast for 3D egg     │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Notes

- Use vectorized numpy where possible
- Cache spectral kernels for repeated use at same location
- The carrier graph weights can be precomputed once
- Scale map can be computed once per carrier
- Consider numba for inner loops if needed

---

## Deliverables

### lattice_extrusion_v2/
- `territory.py` - Island/bridge graph construction
- `spectral_extrude.py` - Expansion-gated, Fiedler-aligned extrusion
- `mesh.py` - Convert lattice to 3D mesh on convex surface
- `render.py` - 3D render with depth/color
- `cli.py` - Demo modes, render output

### texture_synth_v2/
- `carrier_graph.py` - Weighted graph from carrier image
- `spectral_modulate.py` - Structure-aware spread, adaptive scale
- `normals.py` - Height field to normal map
- `render_egg.py` - 3D egg with normal-mapped texture
- `cli.py` - Demo modes with amongus/checkerboard

### Expected Outputs

```
lattice_extrusion_v2/
  demo_islands.png      - 3D egg with island/bridge lattice
  demo_bridges.png      - Same, colored by expansion value

texture_synth_v2/
  demo_amongus_checker.png  - Amongus carrier + checkerboard operand on egg
  demo_checker_amongus.png  - Checkerboard carrier + amongus operand on egg
  normal_map_a.png          - Normal map for demo A
  normal_map_b.png          - Normal map for demo B
```

---

## Communication Protocol

Read: `/home/bigboi/itten/hypercontexts/spec-v2-spectral-ops.md` (this file)
Write: `/home/bigboi/itten/hypercontexts/{tool-name}-v2-exit.md` on completion

Include in exit hypercontext:
- BLUF: success/failure
- Files created
- Demo commands that work
- Spectral ops actually used (prove it's not just fixed kernel)
