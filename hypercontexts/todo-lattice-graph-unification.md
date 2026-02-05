# TODO: Lattice Code & Graph Type Unification

## Problem Statement

THREE incompatible graph abstractions exist:
1. `texture_synthesis.py:65-140` → `ImageGraph` (deprecated but still imported)
2. `spectral_pathfind.py:444-486` → `LargeGridGraph`
3. `graph_view.py:14-37` → `GraphView` protocol (unused by texture editors)

Lattice code is similarly fragmented:
- `lattice_extrusion.py` (deprecated, 30KB)
- `lattice_extrusion_v2/` (4 files, current)

None of these connect to texture synthesis despite the spec requiring graph-aware texture operations.

## Goals

1. **ONE graph type**: Canonical `Graph` class in `spectral_ops_fast.py`
2. **ONE lattice module**: `lattice/` (collapse v1 and v2)
3. **CONNECTED**: Graph type usable by texture synthesis AND pathfinding AND lattice ops

## Phase 1: Graph Type Commitment

### Requirements for Canonical Graph Type

Must support:
- [ ] Regular 2D image grids (for texture synthesis)
- [ ] Arbitrary connectivity (for general graphs)
- [ ] Weighted edges (for spectral ops)
- [ ] Coordinate lookup (for pathfinding visualization)
- [ ] Sparse representation (for GPU/large graphs)
- [ ] Protocol-based (duck typing OK, but documented interface)

### Candidate Implementations

**Option A: Extend `GraphView` protocol**
```python
class GraphView(Protocol):
    def neighbors(self, node: int) -> Sequence[int]: ...
    def edge_weight(self, u: int, v: int) -> float: ...
    def coord_of(self, node: int) -> Tuple[float, ...]: ...
    def num_nodes(self) -> int: ...
```

**Option B: Concrete `Graph` class**
```python
@dataclass
class Graph:
    adjacency: scipy.sparse.csr_matrix  # or torch sparse
    coords: Optional[np.ndarray]  # (N, D) coordinates

    def neighbors(self, node: int) -> np.ndarray: ...
    def laplacian(self) -> scipy.sparse.csr_matrix: ...
```

**Option C: Keep multiple types, provide converters**
```python
def image_to_graph(image: np.ndarray) -> Graph: ...
def graph_to_laplacian(graph: Graph) -> sparse_matrix: ...
```

### Decision Required

Subagent must commit to ONE approach with justification based on:
- Performance (sparse GPU ops)
- Compatibility (existing spectral_ops_fast functions)
- Usability (texture synthesis API shouldn't need to think about graphs)

## Phase 2: Lattice Consolidation

### Current State

`lattice_extrusion.py` (v1):
- Monolithic 30KB file
- Contains mesh generation, extrusion, rendering
- Deprecated but not deleted

`lattice_extrusion_v2/`:
- `mesh.py` - Mesh dataclass
- `extrude.py` - Extrusion ops
- `render.py` - Rendering
- `demo.py` - Demos

### Target Structure
```
lattice/
├── __init__.py      # Public API
├── mesh.py          # Mesh dataclass (from v2)
├── extrude.py       # Extrusion operations
├── patterns.py      # Lattice pattern generation
├── render.py        # Rendering utilities
└── graph.py         # Lattice → Graph conversion (NEW - connects to unified Graph type)
```

### Critical New File: `lattice/graph.py`

This is the MISSING LINK. Must provide:
```python
def lattice_to_graph(lattice: Lattice) -> Graph:
    """Convert lattice structure to Graph for spectral ops."""
    ...

def graph_to_lattice(graph: Graph, embedding: np.ndarray) -> Lattice:
    """Convert Graph with 2D/3D embedding back to Lattice."""
    ...
```

## Phase 3: Integration with spectral_ops_fast.py

### Functions to Add

```python
# In spectral_ops_fast.py

@dataclass
class Graph:
    """Canonical graph representation for all spectral operations."""
    adjacency: torch.sparse.Tensor
    coords: Optional[torch.Tensor] = None

    @classmethod
    def from_image(cls, image: np.ndarray, connectivity: int = 4) -> 'Graph':
        """Create graph from image grid."""
        ...

    @classmethod
    def from_lattice(cls, lattice) -> 'Graph':
        """Create graph from lattice structure."""
        ...

    def laplacian(self, normalized: bool = False) -> torch.sparse.Tensor:
        """Compute graph Laplacian."""
        ...

# Existing functions should accept Graph:
def lanczos_k_eigenvectors(graph_or_laplacian: Union[Graph, torch.sparse.Tensor], k: int, ...) -> ...:
    ...
```

## Phase 4: Update All Consumers

Files that need updating to use canonical Graph:
- [ ] `texture_synthesis.py` - Replace ImageGraph
- [ ] `spectral_pathfind.py` - Replace LargeGridGraph
- [ ] `local_pathfinding.py` - Use canonical Graph
- [ ] `texture_synth_v2/synthesize.py` - Use Graph.from_image()
- [ ] `lattice_extrusion_v2/*` - Use lattice/graph.py converters

## Success Criteria

- [ ] `from spectral_ops_fast import Graph` works
- [ ] `Graph.from_image(texture)` creates graph for spectral ops
- [ ] `Graph.from_lattice(lattice)` creates graph for lattice ops
- [ ] Pathfinding accepts `Graph` directly
- [ ] Zero duplicate graph implementations
- [ ] `lattice_extrusion.py` (v1) deleted
- [ ] `graph_view.py` either deleted or imports from spectral_ops_fast

## Assigned Subagent

Subagent must:
1. Evaluate Options A/B/C and COMMIT to one
2. Design the `Graph` class interface in detail
3. Write `lattice/graph.py` conversion functions spec
4. List all files requiring updates with specific changes

---

*Hypercontext created: 2026-01-30*
*Status: PLANNING COMPLETE*

---

## Subagent Analysis (Completed)

### 1. Graph Type Audit

| Type | Location | Methods | Storage | GPU |
|------|----------|---------|---------|-----|
| `ImageGraph` | texture_synthesis.py:65-140 | neighbors, degree, seed_nodes, get/set_value | Dict-based | NO |
| `LargeGridGraph` | spectral_pathfind.py:444-486 | neighbors, degree, seed_nodes, coord_of | Generative | NO |
| `GraphView` | graph_view.py:14-37 | neighbors, degree, seed_nodes | Protocol only | N/A |
| `TerritoryGraph` | lattice_extrusion_v2/territory.py | GraphView + coord methods | Adjacency dict | NO |

### 2. DECISION: Option B - Concrete Graph Dataclass

**COMMITTED TO: Option B**

Justification:
1. **spectral_ops_fast compatibility**: Functions already take `torch.sparse.Tensor` Laplacians
2. **GPU support**: Concrete class can enforce sparse tensor storage
3. **Texture ease**: `Graph.from_image(carrier)` hides all graph details
4. **Pathfinding ease**: `Graph.neighbors(node)` for traversal, `Graph.coord_of(node)` for viz
5. **Lattice conversion**: `Graph.from_lattice(territory)` single canonical type

### 3. Canonical Graph Interface

```python
# Add to spectral_ops_fast.py

@dataclass
class Graph:
    """Canonical graph for all spectral operations."""
    adjacency: torch.Tensor  # Sparse (n x n)
    coords: Optional[torch.Tensor] = None  # (n, d)

    @property
    def num_nodes(self) -> int: ...

    def neighbors(self, node: int) -> List[int]: ...
    def degree(self, node: int) -> int: ...
    def seed_nodes(self) -> List[int]: ...
    def coord_of(self, node: int) -> Optional[Tuple[float, ...]]: ...
    def edge_weight(self, u: int, v: int) -> float: ...
    def laplacian(self, normalized: bool = False) -> torch.Tensor: ...

    @classmethod
    def from_image(cls, image, connectivity=4, edge_threshold=0.1) -> 'Graph': ...

    @classmethod
    def from_lattice(cls, lattice) -> 'Graph': ...

    @classmethod
    def from_graphview(cls, graph, seed_nodes=None, max_hops=10) -> 'Graph': ...
```

### 4. Lattice Migration

**DELETE**: `lattice_extrusion.py` (v1 deprecated, all in v2)

**MOVE** from `lattice_extrusion_v2/`:
| From | To |
|------|-----|
| `territory.py` | `lattice/patterns.py` |
| `spectral_extrude.py` | `lattice/extrude.py` |
| `mesh.py` | `lattice/mesh.py` |
| `render.py` | `lattice/render.py` |
| `cli.py` | `lattice/cli.py` |

**CREATE**: `lattice/graph.py` (THE MISSING LINK)
```python
def lattice_to_graph(lattice: TerritoryGraph) -> Graph:
    """Convert lattice to Graph for spectral ops."""
    return Graph.from_lattice(lattice)

def graph_to_lattice_coords(graph: Graph, node_data: Dict) -> Dict[Tuple, dict]:
    """Map spectral results back to lattice coordinates."""

def apply_spectral_to_lattice(lattice, spectral_fn, *args, **kwargs):
    """Convenience: lattice → Graph → spectral_fn → coords."""
```

### 5. Files to Update

| File | Change |
|------|--------|
| `spectral_ops_fast.py` | ADD `Graph` dataclass |
| `texture_synthesis.py` | REPLACE `ImageGraph` with `Graph.from_image()` |
| `spectral_pathfind.py` | REPLACE `LargeGridGraph` with `Graph` |
| `local_pathfinding.py` | Import from `spectral_ops_fast` |
| `graph_view.py` | ADD deprecation warning |
| `texture_editor/core.py` | Update imports |
| `bisect_viz/core.py` | Update imports |

### Target Structure

```
lattice/
├── __init__.py      # Public API
├── mesh.py          # Mesh, Vertex, Face
├── extrude.py       # ExpansionGatedExtruder
├── patterns.py      # Island, Bridge, TerritoryGraph
├── render.py        # Visualization
└── graph.py         # Lattice ↔ Graph conversion (NEW)
```
