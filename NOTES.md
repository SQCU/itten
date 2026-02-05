# Graph Spectral Probes: Notes & TODOs

## Current Status (2026-01-30)

All three probes now run. Key fixes applied:
- Fixed import paths from previous session
- `PinchedLattice`: Creates actual topological bottlenecks (edge removal)
- `bump_at_point`: Computes proper tilted normals from height gradient

## Core Concept

**Problem**: Testing whether local spectral methods actually work requires seeing
them produce "obviously correct local solutions" rather than "wrong answers
because we didn't look at the whole graph."

**Solution**: Three probe applications where:
1. Locality is visibly correct (you can see it)
2. Wrongness would be obvious (not subtle numerical drift)
3. The same core methods are reused (not redefined per-probe)

---

## The Three Probes

### Probe 1: Local Pathfinding ✓ WORKING

**Status**: Working. Demonstrates ~96% path efficiency on lattice without global Dijkstra.
**Last run**: Path length 26 for Manhattan distance 25 (96.15% efficiency).

**What it shows**:
- Local expansion estimate identifies bottlenecks
- Fiedler vector provides "direction" without coordinates
- Paths are usually correct, occasionally suboptimal (phase error, not sign error)

**Key insight**: On infinite graphs, BFS cannot complete. Local spectral pathfinding
always produces *some* answer because it never needs the full graph.

**TODOs**:
- [ ] Add visualization of path on 2D lattice (matplotlib)
- [ ] Stress test: lattice with explicit obstacles (removed edges)
- [ ] Measure: how does path quality degrade with obstacle density?
- [ ] Compare: A* with heuristic vs local spectral without heuristic
- [ ] Add: random graph benchmark with planted path (known optimal)

---

### Probe 2: Lattice Extrusion / SimCity Neighborhoods ✓ WORKING

**Status**: Working. Now uses `PinchedLattice` with actual edge removal at bottlenecks.
**Last run**: Found 2 neighborhoods, expansion range 1.438-2.271.

**What it shows**:
- Periodic "pinch" deformation creates detectable bottlenecks
- Shear gradient creates directional variation in Fiedler alignment
- Recursive bisection finds clusters without global view

**Key insight**: The Fiedler gradient rotates as shear increases, showing that
local spectral methods capture the deformation direction.

**TODOs**:
- [ ] Actual SimCity-style visualization (colored neighborhoods on grid)
- [ ] Add more deformation types:
  - [ ] Spiral deformation (neighborhoods rotate around center)
  - [ ] Fractal pinch (bottlenecks at multiple scales)
  - [ ] Random local deletions (simulate "parks" or "rivers")
- [ ] Quantify: neighborhood count vs pinch frequency
- [ ] Add: 3D lattice extrusion (voxel neighborhoods)
- [ ] Compare: k-means on coordinates vs spectral clustering

---

### Probe 3: Non-Hierarchic Texture Synthesis ✓ WORKING

**Status**: Core kernel computation works. Bump now computes proper tilted normals.
**Last run**: Center normal tilted to nz=0.60, kernel radius scales with rotation angle.

**What it shows**:
- Spectral smoothing kernel spreads with rotation angle (replaces Gaussian)
- No explicit pyramid - scale emerges from spectral parameter
- Same kernel structure works on any graph topology

**Key insight**: Traditional texture pyramids bake in the hierarchy. Spectral
methods let "scale" be a continuous parameter that adapts to local structure.

**TODOs**:
- [ ] **Critical**: Fix bump_at_point coordinate handling
- [ ] Add PNG export (pillow/matplotlib)
- [ ] Interactive editing loop:
  - Click to place bump
  - Slider for rotation angle
  - Live preview of kernel spread
- [ ] Normal map export (RGB encoding of normals)
- [ ] Tile seamlessness verification
- [ ] Comparison:
  - [ ] Gaussian blur at matched radius
  - [ ] Laplacian pyramid reconstruction
  - [ ] Show where they differ (edges, corners)
- [ ] Advanced: Non-uniform connectivity graph (simulate fabric weave, brick patterns)
- [ ] Advanced: Anisotropic spectral kernel (oriented bumps for wood grain, etc.)

---

## Core Module Notes

### graph_view.py

Three GraphView implementations:

1. **InfiniteGraph**: Random generative graph. Calling `neighbors()` may spawn new nodes.
2. **LatticeGraph**: Infinite lattice with shear deformation. Coordinates wrap or extend.
3. **FunctionGraph**: Wrap any `Callable[[int], List[int]]` as a graph.

All satisfy the protocol: you can query neighbors but cannot enumerate all nodes.

### spectral_ops.py

Key functions:

- `local_laplacian_matvec`: Lx without materializing L. Only touches support(x) neighbors.
- `local_fiedler_vector`: Lanczos iteration, returns (fiedler_dict, lambda2).
- `local_expansion_estimate`: λ₂ of local neighborhood as expansion proxy.
- `local_bisect`: Sign of Fiedler vector partitions nodes.
- `spectral_node_embedding`: First k eigenvectors as node features.

**All functions work on infinite graphs** because they only query local neighborhoods.

---

## Theoretical Notes

### Rotation Angle ↔ Lanczos Iterations

The "rotation angle" in fractional Fourier transforms corresponds to how far
you've moved from vertex domain (θ=0) to spectral domain (θ=π/2).

For Lanczos:
- k iterations captures interactions up to k-hop distance
- More iterations ≈ higher angle ≈ more global information
- The mapping isn't linear; depends on eigenvalue distribution

### Convergence Behavior

Lanczos converges fastest for extremal eigenvalues (λ₁, λₙ).
λ₂ is near-extremal, so typically converges well.

Pathological cases:
- Clustered eigenvalues near λ₂ → slow convergence
- Highly regular graphs (lattices) → predictable spectrum, fast convergence
- Expander graphs → good gaps, fast convergence
- "Almost disconnected" graphs → tiny λ₂, may need many iterations

### Connection to RoPE

RoPE applies position-dependent rotations to attention queries/keys:
```
q_rotated = R(θ * position) @ q
```

This is fractional rotation where angle depends on sequence position.

Graph analogue: **Magnetic Laplacian** with position-dependent edge phases.
Eigenvectors become complex, encoding directional structure.

Potential application: Graph attention with position-aware spectral features.

### Connection to FlashAttention

FlashAttention computes attention without materializing full QKᵀ matrix.

Our local spectral methods compute Laplacian operations without materializing L.

Both exploit: "Global operation can be computed locally with clever bookkeeping."

---

## Potential Extensions

### AST Comparison Application

The original motivation: compare program ASTs across languages using spectral methods.

Workflow:
1. Parse programs to ASTs (tree-sitter, etc.)
2. Convert AST to graph (parent-child edges, type-similarity edges, etc.)
3. Compute local spectral embedding for each node
4. Compare programs via graph similarity metrics:
   - Spectral distance (compare eigenvalue distributions)
   - Embedding distance (compare node embedding distributions)
   - Bisection alignment (do they partition similarly?)

Key question: Does "similar AST spectral signature" correspond to 
"solves similar problems" or "written in similar style"?

### Streaming AST Analysis

```python
class ASTStreamView:
    def __init__(self, ast_generator: Iterator[AST]):
        self._gen = ast_generator
        self._cache = LRUCache()
    
    def neighbors(self, node):
        # Lazily expand AST as needed
        ...
```

This lets you compare infinite streams of programs (GitHub firehose, generative
model output) without ever having all programs in memory.

### GPU Acceleration

Current implementation: Python dicts, maximally GPU-hostile.

For torch/CUDA efficiency:
1. Materialize local subgraph as sparse CSR tensor
2. Batch multiple queries (different seeds) together  
3. Use torch.lobpcg for Lanczos
4. Window sliding: snapshot region, compute, discard, advance

The "infinite" part stays in Python generators. The "fast" part gets finite GPU windows.

---

## Running the Probes

```bash
cd /home/claude/graph_probes

# All probes
python probes/local_pathfinding.py
python probes/lattice_extrusion.py
python probes/texture_synthesis.py

# Or import in your code
from core import local_fiedler_vector, InfiniteGraph
from probes.local_pathfinding import LocalPathfinder
```

---

## Open Questions

1. **Optimal rotation angle selection**: Given a graph and task, can we predict
   the right θ without trial and error?

2. **Theoretical guarantees**: What's the approximation bound for k-iteration
   Lanczos on graphs with specific spectral properties?

3. **Composition of rotations**: Can we factor a global spectral operation into
   a sequence of local operations (like FFT butterfly)?

4. **Learning the kernel**: Instead of fixed spectral kernel, learn weights
   that optimize for task-specific similarity?

5. **Certifiable locality**: Can we prove that a computation *only* touched
   certain nodes, for privacy/verification purposes?

---

## The Generalization Problem (Core Research Direction)

The previous session was exploring: **Can we generalize local matrix operations
to local graph operations on "origin" or "reference" graphs of unbounded size?**

### What This Means

Traditional spectral methods assume finite graphs:
```
L = D - A          # Laplacian: O(n²) storage
v = eig(L)[:, 1]   # Fiedler: O(n³) or O(n² k) for k-Lanczos
```

Our approach works on infinite graphs by never materializing L:
```python
def local_laplacian_matvec(graph, x):
    # Only touches support(x) and 1-hop neighbors
    # Works identically whether graph has 100 or ∞ nodes
    for node, val in x.items():
        for neighbor in graph.neighbors(node):
            ...  # O(|support(x)| * avg_degree)
```

### The FFT Analogy

FFT factors a global O(n²) DFT into O(n log n) local butterflies:
```
DFT_n = (I ⊗ DFT_{n/2}) · T · (DFT_{n/2} ⊗ I)
```

Each butterfly is a local 2×2 rotation. Can we do similar for graphs?

**Hypothesis**: For graphs with hierarchical structure (trees, recursive
decompositions), there may exist factorizations of spectral transforms
into local operations that compose.

### Connection to Reference Graphs

The "reference graph" idea: Instead of computing on the actual graph G,
compute on a simpler reference graph G_ref that captures key properties:

1. **Lattice reference**: Any locally lattice-like graph can use lattice
   spectral methods as approximation
2. **Tree reference**: Hierarchical graphs → tree spectral decomposition
3. **Expander reference**: Well-connected regions → expander bounds

The local spectral methods give approximations whose quality depends on
how well the local structure matches the reference.

### Current Capability Demonstrated

The three probes show that local spectral methods:
1. **Pathfinding**: 96% efficient without global shortest-path
2. **Clustering**: Detects neighborhoods via local bisection
3. **Multiscale**: Kernel radius emerges from spectral parameter, not hierarchy

### Next Research Steps

1. **Formalize approximation bounds**: When does k-iteration Lanczos give
   ε-approximation to true Fiedler? Depends on spectral gap, graph regularity.

2. **Implement tree factorization**: For tree-structured graphs, the spectral
   decomposition should factor cleanly. Implement and test.

3. **Hybrid global/local**: Use local methods as preconditioners for global
   iteration. Local gives O(1) approximation, global refines.

4. **Streaming interface**: Process graph as stream of edges, maintaining
   spectral summary that allows local queries.
