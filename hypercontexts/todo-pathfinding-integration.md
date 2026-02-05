# TODO: Pathfinding Integration

## Problem Statement

Pathfinding is COMPLETELY DISCONNECTED from texture synthesis:
- `spectral_pathfind.py` (736 lines) - imported by ZERO texture modules
- `local_pathfinding.py` (422 lines) - same problem
- Original spec required: "pathfinding applied to surfaces with clipping/blocking geometry"

The `nodal_lines_demo.png` shows spectral graph ops work, but pathfinding on those surfaces was never implemented.

## Goals

1. **INTEGRATE** pathfinding with texture synthesis
2. **DEMONSTRATE** pathfinding on texture surfaces with blocking geometry
3. **UNIFY** pathfinding to use canonical `Graph` type from spectral_ops_fast.py

## The Original Vision (from spec)

> A pathfinder which could be applied to a surface with clipping/blocking geometry on it.

This means:
1. Take a texture/heightfield
2. Define "blocking" regions (high elevation? specific color? mask?)
3. Find paths that avoid blocking regions
4. Visualize paths on the texture surface

## Phase 1: Pathfinding Module Consolidation

### Current State

`spectral_pathfind.py`:
- `SpectralPathfinder` class
- Uses `local_fiedler_vector` for direction
- Expects `GraphView` protocol
- Has comparison to Dijkstra

`local_pathfinding.py`:
- Different approach (A*? Dijkstra?)
- Different graph expectations
- Possibly redundant with spectral_pathfind

### Target Structure
```
pathfinding/
├── __init__.py           # Public API: find_path, PathResult
├── spectral.py           # Spectral pathfinding (from spectral_pathfind.py)
├── classical.py          # A*, Dijkstra (from local_pathfinding.py)
├── surface.py            # NEW: Pathfinding on texture surfaces
└── visualization.py      # Path rendering on images/meshes
```

### Critical New File: `pathfinding/surface.py`

```python
def texture_to_pathfinding_graph(
    heightfield: np.ndarray,
    blocking_mask: Optional[np.ndarray] = None,
    elevation_cost: float = 1.0,
    blocking_cost: float = float('inf')
) -> Graph:
    """
    Convert texture heightfield to pathfinding graph.

    - Pixels become nodes
    - Edge weights based on elevation difference + blocking
    - Blocking regions have infinite cost (impassable)
    """
    ...

def find_surface_path(
    heightfield: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocking_mask: Optional[np.ndarray] = None,
    method: str = 'spectral'  # or 'astar', 'dijkstra'
) -> PathResult:
    """
    Find path on texture surface avoiding blocking geometry.
    """
    graph = texture_to_pathfinding_graph(heightfield, blocking_mask)
    if method == 'spectral':
        return spectral_find_path(graph, start, goal)
    else:
        return classical_find_path(graph, start, goal, method)

@dataclass
class PathResult:
    path: List[Tuple[int, int]]  # Pixel coordinates
    cost: float
    method: str
    graph: Graph  # For visualization
```

## Phase 2: Integration with Texture Module

### New Texture Functions

In `texture/core.py` (after unification):
```python
def synthesize_with_pathfinding(
    carrier: np.ndarray,
    operand: np.ndarray,
    path_start: Tuple[int, int],
    path_goal: Tuple[int, int],
    blocking_threshold: float = 0.8,  # Heights above this are blocking
    **synth_kwargs
) -> SynthesisResult:
    """
    Synthesize texture AND find path on result.

    Returns height field, normal map, AND optimal path.
    """
    result = synthesize(carrier, operand, **synth_kwargs)
    blocking_mask = result.height_field > blocking_threshold
    path = find_surface_path(
        result.height_field,
        path_start,
        path_goal,
        blocking_mask
    )
    result.path = path
    return result
```

### TUI Commands

```
> synthesize amongus with noise theta 0.5
[RENDER] step_0001_synthesize.png

> find path from (10,10) to (200,200) avoiding heights above 0.7
[RENDER] step_0002_path.png (shows path overlaid on texture)

> show blocking regions
[RENDER] step_0003_blocking.png (highlights impassable areas)
```

## Phase 3: Demonstration

### Demo Script: `demos/pathfinding_on_texture.py`

```python
"""
Demonstrates pathfinding on synthesized texture surfaces.

Creates:
1. Texture with nodal line "walls"
2. Blocking mask from high-frequency nodal regions
3. Path visualization avoiding walls
4. Comparison: spectral vs A* paths
"""

from texture import synthesize
from pathfinding import find_surface_path, visualize_path

# Create texture with prominent nodal lines (walls)
carrier = generate_amongus(256)
operand = generate_noise(256)
result = synthesize(carrier, operand, theta=0.3)  # Low theta = prominent Fiedler walls

# Define blocking: nodal lines become walls
blocking = detect_nodal_lines(result.height_field) > 0.5

# Find paths
start, goal = (20, 20), (230, 230)
spectral_path = find_surface_path(result.height_field, start, goal, blocking, method='spectral')
astar_path = find_surface_path(result.height_field, start, goal, blocking, method='astar')

# Visualize
visualize_path(result.height_field, spectral_path, 'outputs/spectral_path.png')
visualize_path(result.height_field, astar_path, 'outputs/astar_path.png')

# Compare
print(f"Spectral path length: {len(spectral_path.path)}, cost: {spectral_path.cost}")
print(f"A* path length: {len(astar_path.path)}, cost: {astar_path.cost}")
```

## Phase 4: Lattice Probe Connection

The spec also mentioned "lattice probe capabilities" applied to surfaces. This connects pathfinding to lattice:

```python
def probe_lattice_path(
    lattice: Lattice,
    surface: np.ndarray,
    start_node: int,
    goal_node: int
) -> PathResult:
    """
    Find path through lattice structure projected onto surface.

    - Lattice provides connectivity
    - Surface provides elevation costs
    - Blocking from surface geometry
    """
    graph = Graph.from_lattice(lattice)
    # Weight edges by surface elevation at lattice node positions
    weighted_graph = weight_graph_by_surface(graph, surface)
    return spectral_find_path(weighted_graph, start_node, goal_node)
```

## Success Criteria

- [ ] `from pathfinding import find_surface_path` works
- [ ] `find_surface_path(heightfield, start, goal, blocking)` returns valid path
- [ ] Path avoids blocking regions correctly
- [ ] Visualization shows path on texture surface
- [ ] Demo produces comparison of spectral vs classical paths
- [ ] TUI commands for pathfinding work
- [ ] Pathfinding uses canonical `Graph` type
- [ ] `spectral_pathfind.py` and `local_pathfinding.py` collapsed into `pathfinding/`

## Assigned Subagent

Subagent must:
1. Audit `spectral_pathfind.py` and `local_pathfinding.py` for reusable code
2. Design `texture_to_pathfinding_graph()` conversion
3. Specify `PathResult` dataclass fully
4. Write demo script skeleton
5. List TUI commands to add

---

*Hypercontext created: 2026-01-30*
*Status: PLANNING COMPLETE*

---

## Subagent Analysis (Completed)

### 1. Pathfinding Code Audit

**spectral_pathfind.py (736 lines)**:
- `SpectralPathfinder` class - local Fiedler guidance + geometric heuristic
- `dijkstra_path()` - standard shortest path
- `compare_to_dijkstra()` - comparison framework
- `LargeGridGraph`, `LargeRandomGraph` - test utilities

**local_pathfinding.py (422 lines)**:
- `expansion_gradient()` - prefer structured paths (bottlenecks)
- `fiedler_direction()` - spectral direction to target
- `LocalPathfinder` - combines expansion + Fiedler + heuristic

**RECOMMENDATION**: Use `SpectralPathfinder` as primary, incorporate `expansion_gradient` as enhancement.

### 2. HeightfieldGraph Design

```python
class HeightfieldGraph:
    """Image grid as pathfinding graph with elevation-weighted edges."""

    def __init__(self, heightfield, blocking_mask, elevation_cost_scale, connectivity): ...

    # GraphView protocol
    def neighbors(self, node) -> List[int]: ...
    def degree(self, node) -> int: ...
    def seed_nodes(self) -> List[int]: ...
    def coord_of(self, node) -> Tuple[int, int]: ...

    # Weighted pathfinding
    def edge_weight(self, from_node, to_node) -> float:
        """w = base_dist + elevation_scale * |h(to) - h(from)|"""

    def weighted_neighbors(self, node) -> List[Tuple[int, float]]: ...
    def is_blocked(self, node) -> bool: ...
    def to_sparse_laplacian(self) -> torch.Tensor: ...
```

### 3. PathResult Dataclass

```python
@dataclass
class PathResult:
    # Core
    path: List[Tuple[int, int]]   # (x, y) coordinates
    path_node_ids: List[int]
    total_cost: float
    path_length: int

    # Method info
    method: str                   # 'spectral', 'astar', 'dijkstra'
    success: bool

    # Performance
    nodes_visited: int
    computation_time: float

    # Spectral-specific
    spectral_stats: Dict[str, Any] = field(default_factory=dict)

    # Visualization
    visited_mask: Optional[np.ndarray] = None
    cost_field: Optional[np.ndarray] = None

    # Comparison
    optimal_cost: Optional[float] = None
    cost_ratio: Optional[float] = None

    def to_mask(self, height, width) -> np.ndarray: ...
    def to_colored_overlay(self, height, width, ...) -> np.ndarray: ...
    def summary(self) -> str: ...
```

### 4. TUI Commands

```
# Basic pathfinding
> find path from (10,10) to (200,200)
> find path from (10,10) to (200,200) using astar

# With blocking
> find path from (10,10) to (200,200) avoiding heights above 0.7

# Compare methods
> compare paths from (10,10) to (200,200)

# Visualize blocking
> show blocking above 0.7
```

Parser additions:
- `PathfindingCommand` dataclass with: action, start, goal, method, blocking_threshold
- Coordinate extraction via regex: `\((\d+)\s*,\s*(\d+)\)`
- Method keywords: 'astar', 'dijkstra', 'spectral', 'all'

### 5. Demo Script

`demos/pathfinding_on_texture.py`:
1. Generate texture with nodal "walls" (theta=0.3 for prominent Fiedler)
2. Create blocking from height threshold (>=0.7)
3. Find start/goal in open regions
4. Run spectral, A*, Dijkstra
5. Save visualizations: blocking, individual paths, comparison grid
6. Output metrics report

### 6. Module Structure

```
pathfinding/
├── __init__.py        # texture_to_pathfinding_graph, find_surface_path, PathResult
├── graph.py           # HeightfieldGraph
├── spectral.py        # SpectralPathfinder (from spectral_pathfind.py)
├── classical.py       # dijkstra_path, astar_path
├── surface.py         # High-level API: find_surface_path, compare_surface_paths
├── result.py          # PathResult dataclass
├── visualization.py   # visualize_path, visualize_blocking, visualize_comparison
└── compare.py         # compare_methods (from compare_to_dijkstra)
```

### Integration Checklist

- [ ] Create `pathfinding/` directory
- [ ] Move `SpectralPathfinder` to `pathfinding/spectral.py`
- [ ] Implement `astar_path` (not currently in codebase)
- [ ] Implement `HeightfieldGraph`
- [ ] Implement `texture_to_pathfinding_graph`
- [ ] Add `PathfindingCommand` to TUI parser
- [ ] Create demo script
- [ ] Test on sample textures
