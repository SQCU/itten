"""
Territory: Island/Bridge graph with clear spectral variation.

Islands are dense lattice patches with high local expansion (lambda_2).
Bridges are narrow connections (1-3 edges wide) with low local expansion.
Coasts are transition zones.

This creates natural spectral variation that the extrusion operations detect.

NOTE: Graph construction uses while loops (not hot path, OK per mandate).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


@dataclass
class Island:
    """A dense region of the territory (high lambda_2)."""
    center: Tuple[int, int]
    radius: int
    name: str = ""
    nodes: Set[Tuple[int, int]] = field(default_factory=set)

    def __post_init__(self):
        if not self.nodes:
            self._generate_nodes()

    def _generate_nodes(self):
        """Generate a roughly circular island region. Graph construction, not hot path."""
        cx, cy = self.center
        dx = -self.radius
        while dx <= self.radius:
            dy = -self.radius
            while dy <= self.radius:
                # Use L-infinity or L2 norm for shape
                if dx*dx + dy*dy <= self.radius * self.radius + self.radius:
                    self.nodes.add((cx + dx, cy + dy))
                dy += 1
            dx += 1


@dataclass
class Bridge:
    """A narrow connection between islands (low lambda_2)."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    width: int = 1  # Width in nodes
    name: str = ""
    nodes: Set[Tuple[int, int]] = field(default_factory=set)

    def __post_init__(self):
        if not self.nodes:
            self._generate_nodes()

    def _generate_nodes(self):
        """Generate nodes along the bridge path. Graph construction, not hot path."""
        x1, y1 = self.start
        x2, y2 = self.end

        # Use Bresenham-like approach
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)

        if steps == 0:
            self.nodes.add(self.start)
            return

        i = 0
        while i <= steps:
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # Add width perpendicular to direction
            if dx >= dy:
                # Mostly horizontal - add vertical width
                w = -(self.width // 2)
                while w < (self.width + 1) // 2:
                    self.nodes.add((x, y + w))
                    w += 1
            else:
                # Mostly vertical - add horizontal width
                w = -(self.width // 2)
                while w < (self.width + 1) // 2:
                    self.nodes.add((x + w, y))
                    w += 1
            i += 1


class TerritoryGraph:
    """
    Graph representing islands connected by bridges.

    Implements the GraphView protocol required by spectral_ops.
    Islands have high local lambda_2 (good expansion).
    Bridges have low local lambda_2 (bottlenecks).
    """

    def __init__(self, islands: List[Island], bridges: List[Bridge]):
        self.islands = islands
        self.bridges = bridges

        # Combine all nodes
        self._nodes: Set[Tuple[int, int]] = set()
        self._island_membership: Dict[Tuple[int, int], int] = {}
        self._bridge_membership: Dict[Tuple[int, int], int] = {}

        # Graph construction - not hot path
        i = 0
        while i < len(islands):
            island = islands[i]
            node_iter = iter(island.nodes)
            done = False
            while not done:
                try:
                    node = next(node_iter)
                    self._nodes.add(node)
                    self._island_membership[node] = i
                except StopIteration:
                    done = True
            i += 1

        i = 0
        while i < len(bridges):
            bridge = bridges[i]
            node_iter = iter(bridge.nodes)
            done = False
            while not done:
                try:
                    node = next(node_iter)
                    self._nodes.add(node)
                    self._bridge_membership[node] = i
                except StopIteration:
                    done = True
            i += 1

        # Build adjacency structure
        self._adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._build_adjacency()

        # Node to int mapping for spectral_ops compatibility
        self._node_to_id: Dict[Tuple[int, int], int] = {}
        self._id_to_node: Dict[int, Tuple[int, int]] = {}
        sorted_nodes = sorted(self._nodes)
        i = 0
        while i < len(sorted_nodes):
            node = sorted_nodes[i]
            self._node_to_id[node] = i
            self._id_to_node[i] = node
            i += 1

    def _build_adjacency(self):
        """Build 4-connected adjacency for all nodes. Graph construction, not hot path."""
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        node_iter = iter(self._nodes)
        done = False
        while not done:
            try:
                node = next(node_iter)
                x, y = node
                neighbors = []
                d_idx = 0
                while d_idx < len(directions):
                    dx, dy = directions[d_idx]
                    neighbor = (x + dx, y + dy)
                    if neighbor in self._nodes:
                        neighbors.append(neighbor)
                    d_idx += 1
                self._adjacency[node] = neighbors
            except StopIteration:
                done = True

    # GraphView protocol methods (using int IDs)
    def neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node (by ID)."""
        node = self._id_to_node.get(node_id)
        if node is None:
            return []
        adj = self._adjacency.get(node, [])
        result = []
        i = 0
        while i < len(adj):
            result.append(self._node_to_id[adj[i]])
            i += 1
        return result

    def degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return len(self.neighbors(node_id))

    def seed_nodes(self) -> List[int]:
        """Return some seed nodes for exploration."""
        # Return center of first island
        if self.islands:
            center = self.islands[0].center
            if center in self._node_to_id:
                return [self._node_to_id[center]]
        keys = list(self._id_to_node.keys())
        return keys[:1] if keys else []

    # Coordinate-based methods for convenience
    def neighbors_by_coord(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get neighbors of a node (by coordinate)."""
        return self._adjacency.get(coord, [])

    def node_id(self, coord: Tuple[int, int]) -> Optional[int]:
        """Get ID for a coordinate."""
        return self._node_to_id.get(coord)

    def node_coord(self, node_id: int) -> Optional[Tuple[int, int]]:
        """Get coordinate for an ID."""
        return self._id_to_node.get(node_id)

    def all_nodes(self) -> Set[Tuple[int, int]]:
        """Get all node coordinates."""
        return self._nodes.copy()

    def all_node_ids(self) -> List[int]:
        """Get all node IDs."""
        return list(self._id_to_node.keys())

    def is_on_island(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinate is on an island."""
        return coord in self._island_membership

    def is_on_bridge(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinate is on a bridge."""
        return coord in self._bridge_membership and coord not in self._island_membership

    def get_island_index(self, coord: Tuple[int, int]) -> Optional[int]:
        """Get island index for a coordinate."""
        return self._island_membership.get(coord)

    def get_bridge_index(self, coord: Tuple[int, int]) -> Optional[int]:
        """Get bridge index for a coordinate."""
        return self._bridge_membership.get(coord)

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x_min, y_min, x_max, y_max)."""
        if not self._nodes:
            return (0, 0, 0, 0)
        # Vectorized bounds computation
        nodes_arr = np.array(list(self._nodes))
        xs = nodes_arr[:, 0]
        ys = nodes_arr[:, 1]
        return (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))


def create_islands_and_bridges(
    num_islands: int = 3,
    island_radius: int = 5,
    bridge_width: int = 1,
    spacing: int = 15
) -> TerritoryGraph:
    """
    Create a territory with islands arranged horizontally, connected by bridges.

    This creates clear spectral variation:
    - Islands have high lambda_2 (many paths, good expansion)
    - Bridges have low lambda_2 (bottleneck, few paths)
    """
    islands = []
    bridges = []

    # Create islands along x-axis (graph construction, not hot path)
    i = 0
    while i < num_islands:
        center = (i * spacing, 0)
        island = Island(
            center=center,
            radius=island_radius,
            name=f"Island_{i}"
        )
        islands.append(island)
        i += 1

    # Create bridges between consecutive islands
    i = 0
    while i < num_islands - 1:
        # Bridge starts at right edge of island i, ends at left edge of island i+1
        start = (i * spacing + island_radius, 0)
        end = ((i + 1) * spacing - island_radius, 0)

        bridge = Bridge(
            start=start,
            end=end,
            width=bridge_width,
            name=f"Bridge_{i}_{i+1}"
        )
        bridges.append(bridge)
        i += 1

    return TerritoryGraph(islands, bridges)


def create_demo_territory() -> TerritoryGraph:
    """
    Create the demo territory: 3 islands connected by 2 narrow bridges.

    This is the exact scenario from the spec:
    - 3 dense island regions
    - 2 narrow bridges connecting them
    - Clear spectral distinction between islands and bridges
    """
    return create_islands_and_bridges(
        num_islands=3,
        island_radius=6,
        bridge_width=1,
        spacing=18
    )


# ============================================================================
# Deformed and Pinched Lattice Classes
# (Moved from deprecated lattice_extrusion.py)
# ============================================================================

from typing import Callable


class DeformedLattice:
    """
    A lattice with position-dependent deformation.

    The deformation is defined by a function that modifies neighbor offsets
    based on position. This creates local variation in the graph structure.
    """

    def __init__(
        self,
        dims: int = 2,
        deformation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        self.dims = dims
        self.deformation_fn = deformation_fn or (lambda x: np.eye(dims))
        self._known: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}

        # Seed
        origin = tuple([0] * dims)
        self._known[origin] = set()

    def _coord_to_id(self, coord: Tuple[int, ...]) -> int:
        result = 0
        for i, c in enumerate(coord):
            result += (c + 5000) * (10001 ** i)
        return result

    def _id_to_coord(self, node: int) -> Tuple[int, ...]:
        coord = []
        for _ in range(self.dims):
            coord.append((node % 10001) - 5000)
            node //= 10001
        return tuple(coord)

    def _base_offsets(self) -> List[np.ndarray]:
        """Von Neumann neighborhood offsets."""
        offsets = []
        for d in range(self.dims):
            for delta in [-1, 1]:
                off = np.zeros(self.dims)
                off[d] = delta
                offsets.append(off)
        return offsets

    def neighbors(self, node: int) -> List[int]:
        coord = np.array(self._id_to_coord(node))
        coord_tuple = tuple(coord.astype(int).tolist())

        # Get local deformation matrix
        local_deform = self.deformation_fn(coord)

        neighbor_ids = []
        for offset in self._base_offsets():
            # Apply local deformation to offset
            deformed_offset = local_deform @ offset
            neighbor_coord = coord + np.round(deformed_offset).astype(int)
            neighbor_tuple = tuple(neighbor_coord.tolist())

            # Register this edge
            if coord_tuple not in self._known:
                self._known[coord_tuple] = set()
            self._known[coord_tuple].add(neighbor_tuple)

            if neighbor_tuple not in self._known:
                self._known[neighbor_tuple] = set()
            self._known[neighbor_tuple].add(coord_tuple)

            neighbor_ids.append(self._coord_to_id(neighbor_tuple))

        return neighbor_ids

    def degree(self, node: int) -> int:
        return len(self._base_offsets())

    def seed_nodes(self) -> List[int]:
        return [self._coord_to_id(tuple([0] * self.dims))]

    def coord_of(self, node: int) -> Tuple[int, ...]:
        return self._id_to_coord(node)


class PinchedLattice:
    """
    Lattice with periodic bottlenecks via edge removal.

    At x = k*period, vertical edges are removed with high probability,
    creating actual topological bottlenecks that spectral methods can detect.
    """

    def __init__(self, period: int = 8, bottleneck_prob: float = 0.7, rng_seed: int = 42):
        self.period = period
        self.bottleneck_prob = bottleneck_prob
        self._rng = np.random.default_rng(rng_seed)
        self._edge_cache: Dict[Tuple[Tuple[int,int], Tuple[int,int]], bool] = {}

    def _coord_to_id(self, coord: Tuple[int, int]) -> int:
        x, y = coord
        return (y + 5000) * 10001 + (x + 5000)

    def _id_to_coord(self, node: int) -> Tuple[int, int]:
        x = (node % 10001) - 5000
        y = (node // 10001) - 5000
        return (x, y)

    def _edge_exists(self, c1: Tuple[int,int], c2: Tuple[int,int]) -> bool:
        """Check if edge exists (may be removed at bottleneck)."""
        key = (min(c1, c2), max(c1, c2))
        if key not in self._edge_cache:
            x1, y1 = c1
            x2, y2 = c2

            # Vertical edges at bottleneck x-positions may be removed
            is_vertical = (x1 == x2)
            at_bottleneck = (x1 % self.period == 0)

            if is_vertical and at_bottleneck:
                # Use deterministic random based on edge position
                edge_seed = hash(key) & 0xFFFFFFFF
                self._edge_cache[key] = (edge_seed % 100) >= (self.bottleneck_prob * 100)
            else:
                self._edge_cache[key] = True

        return self._edge_cache[key]

    def neighbors(self, node: int) -> List[int]:
        x, y = self._id_to_coord(node)
        result = []

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_coord = (x + dx, y + dy)
            if self._edge_exists((x, y), neighbor_coord):
                result.append(self._coord_to_id(neighbor_coord))

        return result

    def degree(self, node: int) -> int:
        return len(self.neighbors(node))

    def seed_nodes(self) -> List[int]:
        return [self._coord_to_id((0, 0))]

    def coord_of(self, node: int) -> Tuple[int, int]:
        return self._id_to_coord(node)


def periodic_pinch_deformation(period: int = 10, pinch_strength: float = 0.5):
    """
    Create periodic "bottlenecks" by contracting one dimension periodically.

    This is like having main roads every N blocks.
    Note: This only affects geometry, not topology. See PinchedLattice for
    actual connectivity changes.
    """
    def deform(coord: np.ndarray) -> np.ndarray:
        x = coord[0]
        # Near multiples of period, squeeze the y-direction
        distance_to_pinch = min(abs(x % period), period - abs(x % period))
        squeeze = 1.0 - pinch_strength * np.exp(-distance_to_pinch**2 / 4)

        return np.array([
            [1.0, 0.0],
            [0.0, squeeze]
        ])
    return deform


def shear_gradient_deformation(shear_rate: float = 0.02):
    """
    Shear that increases with position.

    Creates neighborhoods that are progressively more skewed.
    """
    def deform(coord: np.ndarray) -> np.ndarray:
        x = coord[0]
        shear = shear_rate * x
        return np.array([
            [1.0, shear],
            [0.0, 1.0]
        ])
    return deform
