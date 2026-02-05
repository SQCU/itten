# Hypercontext: Sparse Matrix/Graph Algorithm Implementations

## Mission
Demonstrate sparse matrix/graph superiority by implementing large-scale versions of three algorithms that avoid full matrix materialization.

## Three Algorithms to Implement

### 1. Large-Scale Texture Bump Map Etching
**Current**: `texture_synth_v2/spectral_etch.py` uses Lanczos on full image Laplacian
**Goal**: Tile-based or streaming approach for arbitrarily large images

Key insight: Eigenvector nodal lines are LOCAL features. We don't need global eigenvectors—we need local spectral structure.

Approach:
```python
def sparse_texture_etch(carrier, tile_size=64, overlap=16):
    # Process image in overlapping tiles
    # Each tile gets local Lanczos eigenvectors
    # Blend overlapping regions
    # Result: O(n * k * iters) where k << n
```

Files to modify:
- `spectral_ops_fast.py`: Add `local_eigenvectors_tiled()`
- `texture_synth_v2/spectral_etch.py`: Add streaming mode

### 2. Multi-Lattice Local Extrusion (Graph Spectra Guided)
**Concept**: Extrude 3D geometry from spectral structure
**Sparse requirement**: Work on graphs with millions of nodes

Current `spectral_ops_fast.py` has:
- `local_fiedler_vector()` - local spectral for seed nodes
- `expand_neighborhood()` - hop-limited expansion

New algorithm:
```python
def multi_lattice_extrusion(graph, seed_nodes, lattice_params):
    """
    1. Compute local spectral embedding at each seed
    2. Cluster seeds by spectral similarity
    3. Assign lattice type to each cluster
    4. Generate extrusion geometry from spectral field

    Key: Never materialize full adjacency matrix.
    Only query graph.neighbors(node) as needed.
    """
```

Lattice types:
- Triangular (high expansion, λ₂ large)
- Square (moderate expansion)
- Hexagonal (low expansion, λ₂ small)

### 3. Local Spectral Pathfinding (Low-Regret Dijkstra-like)
**Problem**: Find paths using local spectral direction instead of full Dijkstra

Insight: The Fiedler vector gradient points toward graph bisection boundary. Following this gradient is like following the "main axis" of the graph.

```python
def spectral_pathfind(graph, start, goal, max_hops=1000):
    """
    At each step:
    1. Compute local Fiedler vector (small neighborhood)
    2. Choose neighbor that best aligns with goal direction in spectral space
    3. Step to that neighbor

    No global shortest path computation needed.
    Complexity: O(hops * local_neighborhood_size * lanczos_iters)
    """
```

Comparison to Dijkstra:
- Dijkstra: O((V + E) log V), needs full graph access
- Spectral: O(path_length * local_cost), streaming compatible

The "low regret" comes from spectral direction being statistically aligned with true shortest path direction on structured graphs.

## GraphView Interface

All algorithms use this protocol (from `spectral_ops_fast.py`):
```python
class GraphView(Protocol):
    def neighbors(self, node: int) -> List[int]: ...
    def degree(self, node: int) -> int: ...
    def seed_nodes(self) -> List[int]: ...
```

This is the ONLY interface to the graph. No matrix access.

## Existing Sparse Primitives

From `spectral_ops_fast.py`:
```python
# Neighborhood expansion (BFS-like)
expand_neighborhood(graph, seed_nodes, hops) -> Set[int]

# Local Laplacian construction
build_sparse_laplacian(graph, active_nodes) -> (L, node_list, node_to_idx)

# GPU Lanczos for Fiedler
lanczos_fiedler_gpu(L, num_iterations) -> (fiedler, lambda2)

# Local Fiedler via protocol
local_fiedler_vector(graph, seed_nodes, num_iterations, hop_expansion)
    -> (fiedler_dict, lambda2)

# Batched expansion estimates
expansion_map_batched(graph, nodes, radius, k) -> Dict[int, float]

# Spectral embedding (multiple eigenvectors)
spectral_node_embedding(graph, seed_nodes, embedding_dim, hop_expansion)
    -> Dict[int, np.ndarray]
```

## Implementation Plan

### Phase 1: Tiled Texture Etching
```python
# In spectral_ops_fast.py
def compute_local_eigenvectors_tiled(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4
) -> np.ndarray:
    """
    Compute eigenvectors in overlapping tiles, blend results.
    Returns (H, W, k) eigenvector images.
    """
```

### Phase 2: Multi-Lattice Extrusion
```python
# New file: lattice_extrusion.py
def classify_local_expansion(graph, nodes, radius=3) -> Dict[int, str]:
    """Classify each node as 'high', 'medium', 'low' expansion."""

def assign_lattice_type(expansion_class: str) -> LatticeParams:
    """Map expansion class to lattice geometry."""

def generate_extrusion_mesh(graph, lattice_assignments) -> Mesh:
    """Generate 3D mesh from lattice assignments."""
```

### Phase 3: Spectral Pathfinding
```python
# New file: spectral_pathfind.py
def compute_spectral_direction(graph, node, goal, radius=2) -> int:
    """
    Return best neighbor to move toward goal using local spectral structure.
    """

def spectral_path(graph, start, goal, max_steps=1000) -> List[int]:
    """
    Find path using local spectral guidance.
    Returns path as list of node IDs.
    """

def compare_to_dijkstra(graph, start, goal) -> dict:
    """
    Compare spectral path to true shortest path.
    Returns: path_length_ratio, angle_error_mean, computation_time_ratio
    """
```

## Test Graphs

### Small (verification)
- 32x32 grid graph (1024 nodes)
- Random geometric graph (1000 nodes)

### Large (demonstration)
- 1024x1024 grid (1M nodes) - should work without full matrix
- Road network subset (100K nodes)
- Social network sample (sparse, irregular)

## Success Criteria

1. **Texture etching**: 4096x4096 image processed with < 4GB RAM
2. **Multi-lattice**: 100K node graph processed in < 10 seconds
3. **Pathfinding**: Path found with < 2x optimal length, 10x faster than Dijkstra

## Files to Create/Modify

```
spectral_ops_fast.py:
  - compute_local_eigenvectors_tiled()
  - spectral_direction_to_goal()

lattice_extrusion.py (new):
  - classify_local_expansion()
  - assign_lattice_type()
  - generate_extrusion_mesh()

spectral_pathfind.py (new):
  - spectral_path()
  - compare_to_dijkstra()

texture_synth_v2/spectral_etch.py:
  - compute_spectral_eigenvectors_tiled() (wrapper)
```

## Deliverables
1. All three algorithms implemented and tested
2. Benchmark results showing sparse superiority
3. Demo visualizations for each algorithm
4. Documentation of complexity and memory usage
