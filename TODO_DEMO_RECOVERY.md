# TODO: Demo Recovery via _even_cuter Composition

## What this is

Reimplementation of the first 3-4 project demos using the _even_cuter nn.Module
infrastructure as the spectral foundation. The goal is torch-compilable,
GPU-deployable code that demonstrates "there are fast 2026 ways to do these things"
rather than "relevant algorithms exist."

**torch.compile is the floor.** Code that doesn't compile isn't worth maintaining,
testing, or importing. This is the proxy for deployment-worthiness across
4-8GB gamer GPUs, phone NPUs, and tensor-core hardware.

## Source demos to recover

| Demo | Original spec | Key operation |
|------|--------------|---------------|
| 1. Spectral normal mapping | Amongus shape → spectral transform → normal map → 3D egg render | Height field from eigenvectors → bump-mapped surface |
| 2. Lattice extrusion | Base graph → extrude 3+ non-isomorphic lattices → 3D tiles | Expansion-gated geometry with spectral lattice selection |
| 3. Spectral pathfinding | Local spectra → approximate Dijkstra without O(n²) | Graph-local Fiedler for direction, low Wasserstein from optimal |
| 4. Texture atlas shader | Texture-texture cross-attention → UV-mapped deformation | SpectralShader on atlas → UV map to surface → intelligible deformations |

## Dependency graph

```
Phase A: GraphEmbedding (foundation)
   ├── Phase B: Rendering modules (independent — pure geometry)
   ├── Phase C: Lattice modules (needs A for expansion estimates)
   ├── Phase D: Pathfinding modules (needs A for local spectral probe)
   └── Phase E: Demo composition (needs A + B + C + D)
```

---

## Phase A: `spectral_graph_embedding.py` — Foundation

### The problem

`SpectralEmbedding` in `spectral_embedding_layer.py` only works on images.
It internally constructs a multi-radius image Laplacian. But 3 of 4 demos
operate on arbitrary graphs (TerritoryGraph, HeightfieldGraph, weighted grids).

Also: eigenvalues (lambda_2) are computed but discarded. The lattice demo
needs lambda_2 for expansion gating.

### What to create

**File: `spectral_graph_embedding.py`**

#### `GraphEmbedding(nn.Module)`

The core implicit layer that operates on ANY sparse Laplacian.

```python
class GraphEmbedding(nn.Module):
    """Compute eigenvectors/eigenvalues of a sparse Laplacian via Lanczos.

    This is the general case of SpectralEmbedding. SpectralEmbedding constructs
    an image Laplacian internally; GraphEmbedding accepts any prebuilt Laplacian.

    Architecture analog: Deep Equilibrium Model (Bai et al. 2019).
    The output is defined implicitly as the fixed point of L @ v = lambda * v.

    Returns both eigenvectors AND eigenvalues. Eigenvalues are load-bearing:
    lambda_2 (algebraic connectivity) gates lattice extrusion and measures
    local expansion. Discarding eigenvalues discards information.
    """
    def __init__(self, num_eigenvectors: int = 4, lanczos_iterations: int = 30):
        ...

    def forward(self, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            L: Sparse Laplacian (n, n) as torch.sparse_coo_tensor

        Returns:
            eigenvectors: (n, k) — columns are eigenvectors, Fiedler is [:, 0]
            eigenvalues: (k,) — lambda_2 is [0]
        """
```

Source: `spectral_ops_fast.py:lanczos_k_eigenvectors` (lines 643+)
and `spectral_ops_fast_cuter.py:lanczos_fiedler_gpu` (lines 44-101)

Properties to preserve:
- Mean subtraction (deflates trivial eigenvector)
- Full reorthogonalization (numerical stability)
- GPU-first with sparse matvec

#### `LocalSpectralProbe(nn.Module)`

Graph-local Fiedler computation for pathfinding and per-node expansion estimates.

```python
class LocalSpectralProbe(nn.Module):
    """Compute local spectral properties around a query point.

    Extracts a subgraph within `hop_radius` hops of the query node,
    builds a local Laplacian, computes Fiedler vector and lambda_2.

    This is the graph-local analog of SpectralEmbedding: it sees only
    a neighborhood, not the full graph. The pathfinder uses this to
    make locally-optimal decisions that approximate global Dijkstra
    with empirically low transport distance.

    Architecture analog: local attention window (Beltagy et al. 2020,
    Longformer) — restricts computation to a local context, accepts
    approximation error in exchange for O(local) cost.
    """
    def __init__(self, hop_radius: int = 2, lanczos_iterations: int = 15):
        ...

    def forward(self, adjacency: torch.Tensor, query_nodes: torch.Tensor,
                coords: Optional[torch.Tensor] = None
               ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Args:
            adjacency: Sparse (n, n) graph adjacency
            query_nodes: (q,) node indices to probe around
            coords: Optional (n, d) node coordinates

        Returns:
            fiedler_maps: Per-query local Fiedler vectors
            expansion_estimates: (q,) lambda_2 per query node
        """
```

Source: `spectral_ops_fns.py:local_fiedler_vector` (lines 169-207),
`local_expansion_estimate` (lines 210-227),
`expansion_map_batched` (lines 230+)

#### `ImageLaplacianBuilder(nn.Module)`

Factored out of SpectralEmbedding — builds the multi-radius image Laplacian.

```python
class ImageLaplacianBuilder(nn.Module):
    """Build weighted image Laplacian for spectral embedding.

    Constructs a graph where pixels are nodes and edges connect nearby
    pixels with weights based on color similarity. Multi-radius connectivity
    captures dither patterns that don't directly touch.

    Factored from SpectralEmbedding to allow reuse: the same Laplacian
    builder works for texture synthesis, normal mapping, and atlas operations.
    """
    def __init__(self, radii=[1,2,3,4,5,6], radius_weights=[1.0,0.6,0.4,0.3,0.2,0.1],
                 edge_threshold=0.15):
        ...

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Returns sparse Laplacian for the image graph."""
```

Source: `spectral_ops_fast_cuter.py:_build_multiscale_laplacian` (lines 104-151)

#### Refactor `SpectralEmbedding`

After Phase A, SpectralEmbedding becomes:
```python
class SpectralEmbedding(nn.Module):
    def __init__(self, ...):
        self.laplacian_builder = ImageLaplacianBuilder(...)
        self.graph_embedding = GraphEmbedding(...)
        # tiling logic stays here

    def forward(self, image) -> torch.Tensor:
        # tile, build Laplacian per tile via self.laplacian_builder,
        # embed via self.graph_embedding, blend
        ...

    def forward_with_eigenvalues(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        # same but returns eigenvalues too
        ...
```

### Numerical equivalence requirement

GraphEmbedding must produce bit-identical results to `lanczos_fiedler_gpu`
when given the same Laplacian. LocalSpectralProbe must match
`local_fiedler_vector` on the same subgraph.

---

## Phase B: `spectral_renderer.py` — Pure geometry, no spectral dependency

### The problem

The rendering pipeline (`texture/render/bump_render.py`, `lighting.py`,
`ray_cast.py`, `normals.py`) is ~1,500 lines of numpy. It works but:
- Not torch-compilable
- Not GPU-accelerated
- Not composable with _even_cuter shader modules

### What to create

**File: `spectral_renderer.py`** (single file, all rendering as Modules)

#### `HeightToNormals(nn.Module)`

```python
class HeightToNormals(nn.Module):
    """Convert height field to RGB-encoded normal map.

    Uses central differences for gradient computation.
    The normal map can be used as:
    1. Bump map for 3D rendering (perturb surface normals)
    2. Input to SpectralShaderBlock (the normal map IS an image)
    3. Visual representation of spectral structure
    """
    def __init__(self, strength: float = 1.0):
        ...

    def forward(self, height_field: torch.Tensor) -> torch.Tensor:
        """(H, W) → (H, W, 3) normal map in [0, 1]"""
```

Source: `texture/normals.py:height_to_normals`

#### `EggSurfaceRenderer(nn.Module)`

```python
class EggSurfaceRenderer(nn.Module):
    """Render texture on egg-shaped 3D surface with bump mapping.

    Pipeline:
    1. Ray-sphere intersection → egg deformation
    2. Spherical UV from intersection point
    3. Sample texture + normal map at UV
    4. Perturb surface normal by normal map (TBN frame)
    5. Blinn-Phong lighting with Fresnel

    This is the "visual echo" renderer: spectral structure in the
    height field produces visible 3D surface features via bump mapping.
    """
    def __init__(self, resolution: int = 512, egg_factor: float = 0.25,
                 light_dir: Tuple[float, float, float] = (0.5, 0.7, 1.0)):
        ...

    def forward(self, texture: torch.Tensor, normal_map: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) texture + (H, W, 3) normal map → (res, res, 3) rendered image"""
```

Source: `texture/render/bump_render.py:render_bumped_egg` (532 lines)

#### `BilinearSampler(nn.Module)`

```python
class BilinearSampler(nn.Module):
    """Bilinear texture sampling with wrapping. torch.compile compatible.

    Replaces numpy-based sample_texture_bilinear with pure torch.
    Supports (H, W) and (H, W, C) textures.
    """
    def forward(self, texture: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ...
```

Source: `texture/render/bump_render.py:sample_texture_bilinear` (lines 20-69)

### Properties to preserve
- Correct TBN frame computation
- Schlick Fresnel approximation
- Egg deformation from sphere
- Wrapping UV coordinates (tileable textures)

### torch.compile requirement
All rendering modules must compile with fullgraph=True (no sparse ops, no Python dicts).
This is pure geometry — there's no reason for graph breaks.

---

## Phase C: `spectral_lattice.py` — Lattice extrusion

### The problem

`lattice/extrude.py` (604 lines) uses numpy dataclasses, Python dicts for node state,
and imports graph-local spectral functions from `spectral_ops_fast.py`.
Needs reimplementation as torch Modules using `GraphEmbedding` from Phase A.

### What to create

**File: `spectral_lattice.py`**

#### `LatticeTypeSelector(nn.Module)`

```python
class LatticeTypeSelector(nn.Module):
    """Spectral-determined lattice type selection.

    Maps local spectral properties to geometry type:
    - High expansion (lambda_2) + low gradient → hex (open region)
    - Low expansion + high gradient → triangle (bottleneck)
    - Medium → square (neutral)
    - Theta rotates between geometry preferences

    Architecture analog: discrete routing / mixture of experts
    (Shazeer et al. 2017). Spectral properties determine which
    "expert" (lattice geometry) handles each spatial region.
    """
    def __init__(self, high_threshold=0.6, low_threshold=0.4):
        ...

    def forward(self, expansion: torch.Tensor, fiedler_gradient_mag: torch.Tensor,
                theta: float) -> torch.Tensor:
        """
        Args:
            expansion: (n,) local lambda_2 per node
            fiedler_gradient_mag: (n,) |∇Fiedler| per node
            theta: blend parameter

        Returns:
            lattice_types: (n,) integer tensor (0=square, 1=triangle, 2=hex)
        """
```

Source: `lattice/extrude.py:select_lattice_type` (lines 58-80+)

#### `ExpansionGatedExtruder(nn.Module)`

```python
class ExpansionGatedExtruder(nn.Module):
    """Spectral-gated graph extrusion.

    Extrudes geometry from a base graph, with extrusion depth gated by
    local spectral expansion (lambda_2). High-expansion regions extrude
    further; bottleneck regions (low lambda_2) get shallow extrusion.

    Uses GraphEmbedding for spectral properties and LatticeTypeSelector
    for geometry assignment.

    The extrusion is a graph operation: each layer adds nodes connected
    to the layer below, with connectivity determined by lattice type.
    """
    def __init__(self, graph_embedding: 'GraphEmbedding',
                 lattice_selector: 'LatticeTypeSelector',
                 expansion_threshold: float = 1.5,
                 max_layers: int = 5):
        ...

    def forward(self, adjacency: torch.Tensor, coords: torch.Tensor,
                theta: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            adjacency: Sparse (n, n) base graph
            coords: (n, 2) or (n, 3) node positions
            theta: spectral blend parameter

        Returns:
            extruded_coords: (m, 3) positions of all extruded nodes
            extruded_adjacency: Sparse (m, m) connectivity
            node_properties: (m, k) per-node metadata (layer, lattice_type, expansion, etc.)
        """
```

Source: `lattice/extrude.py:ExpansionGatedExtruder` class (604 lines)

### Key challenge

The existing code uses Python dicts and dataclasses for per-node state.
The reimplementation needs tensor-first representation:
- Node state as (n, k) tensor, not Dict[Tuple, ExtrudedNode]
- Frontier as boolean mask, not Set[Tuple]
- Extrusion as batched tensor operation, not sequential node processing

### Depends on: Phase A (GraphEmbedding, LocalSpectralProbe)

---

## Phase D: `spectral_pathfinder.py` — Graph-local spectral navigation

### The problem

`pathfinding/spectral.py` (399 lines) is numpy + dict-based. Uses
`local_fiedler_vector` from `spectral_ops_fns.py` for directional guidance.
Needs reimplementation as torch Module using `LocalSpectralProbe` from Phase A.

### What to create

**File: `spectral_pathfinder.py`**

#### `SpectralPathfinder(nn.Module)`

```python
class SpectralPathfinder(nn.Module):
    """Graph-local spectral navigation.

    At each step: compute local Fiedler vector around current position,
    choose neighbor most aligned with goal direction in spectral space.

    This is "Dijkstra but kinda wrong" — it uses only local spectral
    information (O(local_radius²) per step) instead of full graph access
    (O(n²) for Dijkstra). The approximation error is bounded: empirically
    low transport distance (Wasserstein or other) from optimal path.

    Architecture analog: greedy decoding with local attention.
    The model sees a local context window (spectral_radius hops),
    makes a locally-optimal decision, and advances. The spectral
    information provides a "soft gradient" toward the goal that
    regular greedy search lacks.

    The key insight: local spectral properties encode approximate global
    structure. The Fiedler gradient points "away from bottlenecks" —
    which is exactly the information Dijkstra uses implicitly via
    shortest-path relaxation.
    """
    def __init__(self, local_probe: 'LocalSpectralProbe',
                 spectral_weight: float = 0.3,
                 heuristic_weight: float = 0.7,
                 exploration_probability: float = 0.02):
        ...

    def forward(self, adjacency: torch.Tensor, coords: torch.Tensor,
                start: int, goal: int,
                max_steps: int = 1000) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            adjacency: Sparse (n, n) graph
            coords: (n, d) node positions
            start: start node index
            goal: goal node index
            max_steps: safety limit

        Returns:
            path: (path_len,) node indices
            diagnostics: dict with spectral_decisions, expansion_history, etc.
        """
```

Source: `pathfinding/spectral.py:SpectralPathfinder` (399 lines)

### Key challenge

Pathfinding is inherently sequential (each step depends on current position).
This means torch.compile with fullgraph=True is unlikely. But:
- The spectral probe (Lanczos on local subgraph) IS compilable
- The step logic can use `torch.compile(fullgraph=False)` with graph breaks at the loop

The value is: spectral computation on GPU, decision logic as thin Python glue.

### torch.compile strategy

```python
# The hot path (spectral probe) compiles:
compiled_probe = torch.compile(self.local_probe, fullgraph=False)

# The step loop has graph breaks but that's OK:
# each step does O(1) Python + O(local_radius²) compiled tensor ops
```

### Depends on: Phase A (LocalSpectralProbe)

---

## Phase E: Demo composition

### Demo 1: Spectral normal mapping

```python
# The composition that should work after A + B:
from spectral_graph_embedding import GraphEmbedding, ImageLaplacianBuilder
from spectral_shader_layers import SpectralShaderBlock
from spectral_renderer import HeightToNormals, EggSurfaceRenderer

image = load_image("amongus.png")
laplacian = ImageLaplacianBuilder()(image)
eigvecs, eigvals = GraphEmbedding(num_eigenvectors=8)(laplacian)

# Fiedler as height field → normal map → 3D render
height = eigvecs[:, 0].reshape(H, W)  # Fiedler
normals = HeightToNormals()(height)
rendered = EggSurfaceRenderer()(image, normals)

# Apply spectral shader → height field → render again ("visual echo")
shaded = SpectralShaderBlock()(image, height)
normals2 = HeightToNormals()(to_grayscale(shaded))
rendered2 = EggSurfaceRenderer()(shaded, normals2)
```

### Demo 2: Lattice extrusion

```python
# After A + B + C:
from spectral_graph_embedding import GraphEmbedding
from spectral_lattice import ExpansionGatedExtruder, LatticeTypeSelector
from spectral_renderer import EggSurfaceRenderer

graph = build_territory_graph(image)
extruder = ExpansionGatedExtruder(
    GraphEmbedding(), LatticeTypeSelector()
)
coords, adj, props = extruder(graph.adjacency, graph.coords, theta=0.5)
# props includes lattice_type per node → proves 3+ non-isomorphic types
```

### Demo 3: Spectral pathfinding

```python
# After A + D:
from spectral_graph_embedding import LocalSpectralProbe
from spectral_pathfinder import SpectralPathfinder

probe = LocalSpectralProbe(hop_radius=2, lanczos_iterations=15)
pathfinder = SpectralPathfinder(probe, spectral_weight=0.3)

path, diag = pathfinder(graph.adjacency, graph.coords, start=0, goal=999)
# Compare path cost vs Dijkstra optimal → measure transport distance
```

### Demo 4: Texture atlas shader

```python
# After A + B (already almost possible with current _even_cuter):
from spectral_shader_model import SpectralShader
from spectral_renderer import BilinearSampler, EggSurfaceRenderer, HeightToNormals

# Apply spectral shader to texture atlas
shader = SpectralShader.from_config(DEFAULT_CONFIG)
shaded_atlas, _ = shader(atlas_texture)

# UV-map to surface
sampler = BilinearSampler()
surface_color = sampler(shaded_atlas, u_coords, v_coords)
surface_normals = sampler(HeightToNormals()(to_grayscale(shaded_atlas)), u_coords, v_coords)

# Render → spectrally-intelligible deformations visible on 3D surface
rendered = EggSurfaceRenderer()(surface_color, surface_normals)
```

---

## Subagent workflow

### Phase A: 1 agent (reader+writer+tester)
- Read `spectral_ops_fast.py`, `spectral_ops_fns.py`, `spectral_ops_fast_cuter.py`,
  `spectral_embedding_layer.py`
- Write `spectral_graph_embedding.py` with GraphEmbedding, LocalSpectralProbe,
  ImageLaplacianBuilder
- Modify `spectral_embedding_layer.py` to add `forward_with_eigenvalues`
- Test numerical equivalence against source functions

### Phase B: 1 agent (reader+writer+tester)
- Read `texture/normals.py`, `texture/render/bump_render.py`,
  `texture/render/lighting.py`, `texture/render/ray_cast.py`
- Write `spectral_renderer.py` with HeightToNormals, EggSurfaceRenderer,
  BilinearSampler
- ALL pure torch, ALL torch.compile fullgraph=True compatible
- Test against numpy originals (within float32 tolerance — numpy→torch
  may introduce minor differences due to operation ordering)

### Phase C: 1 agent (after Phase A)
- Read `lattice/extrude.py`, `lattice/patterns.py`, `lattice/graph.py`
- Write `spectral_lattice.py` using GraphEmbedding from Phase A
- Tensor-first node state (no Python dicts in hot paths)

### Phase D: 1 agent (after Phase A)
- Read `pathfinding/spectral.py`, `pathfinding/graph.py`
- Write `spectral_pathfinder.py` using LocalSpectralProbe from Phase A
- torch.compile the spectral probe, accept graph breaks in step loop

### Phase E: 1 agent (after all above)
- Write 4 demo scripts using only _even_cuter + Phase A-D modules
- Verify each produces visually correct output
- Measure: torch.compile success, GPU utilization, wall-clock time

---

## Citations for docstrings

From TODO_EVEN_CUTER.md plus:
- Beltagy et al. 2020: "Longformer" (local attention windows)
- Shazeer et al. 2017: "Outrageously Large Neural Networks" (mixture of experts)
- Peyre & Cuturi 2019: "Computational Optimal Transport" (Wasserstein distance reference)

## Success criteria

1. All 4 demos produce visually correct output from Module-only code
2. All Modules torch.compile (fullgraph=True where possible, fullgraph=False for sparse ops)
3. Zero imports from `spectral_ops_fast.py` in demo scripts
4. Eigenvalues accessible where needed (lattice, pathfinding)
5. Same numerical results as numpy originals (within float32 tolerance)
