"""
HeightfieldGraph - Image grid as pathfinding graph with elevation-weighted edges.

Wraps a 2D heightfield as a graph where:
- Nodes are pixels (encoded as row * width + col)
- Edges connect neighboring pixels (4-connectivity or 8-connectivity)
- Edge weights depend on elevation difference and blocking
"""

from typing import List, Tuple, Dict, Set, Optional, Protocol
import numpy as np


class GraphView(Protocol):
    """Graph you can query but not enumerate."""
    def neighbors(self, node: int) -> List[int]: ...
    def degree(self, node: int) -> int: ...
    def seed_nodes(self) -> List[int]: ...


class HeightfieldGraph:
    """
    Image grid as pathfinding graph with elevation-weighted edges.

    Implements GraphView protocol for compatibility with spectral operations.
    """

    def __init__(
        self,
        heightfield: np.ndarray,
        blocking_mask: Optional[np.ndarray] = None,
        elevation_cost_scale: float = 1.0,
        connectivity: str = "4"  # "4" or "8"
    ):
        """
        Initialize heightfield graph.

        Args:
            heightfield: 2D array of elevation values (0-1 normalized)
            blocking_mask: Optional boolean mask where True = blocked/impassable
            elevation_cost_scale: Weight for elevation difference in edge cost
            connectivity: "4" for von Neumann, "8" for Moore neighborhood
        """
        if heightfield.ndim != 2:
            raise ValueError(f"Heightfield must be 2D, got {heightfield.ndim}D")

        self.heightfield = heightfield.astype(np.float32)
        self.height, self.width = heightfield.shape
        self.n_nodes = self.height * self.width

        # Blocking mask: True means blocked
        if blocking_mask is not None:
            self.blocking_mask = blocking_mask.astype(bool)
        else:
            self.blocking_mask = np.zeros_like(heightfield, dtype=bool)

        self.elevation_cost_scale = elevation_cost_scale
        self.connectivity = connectivity

        # Precompute neighbor offsets
        if connectivity == "8":
            # 8-connectivity (Moore): includes diagonals
            self._offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            self._base_distances = [
                1.414, 1.0, 1.414,
                1.0,        1.0,
                1.414, 1.0, 1.414
            ]
        else:
            # 4-connectivity (von Neumann): axis-aligned only
            self._offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            self._base_distances = [1.0, 1.0, 1.0, 1.0]

    def _coord_to_id(self, row: int, col: int) -> int:
        """Convert (row, col) to node ID."""
        return row * self.width + col

    def _id_to_coord(self, node: int) -> Tuple[int, int]:
        """Convert node ID to (row, col)."""
        return node // self.width, node % self.width

    def coord_of(self, node: int) -> Tuple[int, int]:
        """Get (row, col) coordinates for a node."""
        return self._id_to_coord(node)

    def is_blocked(self, node: int) -> bool:
        """Check if a node is blocked."""
        r, c = self._id_to_coord(node)
        return self.blocking_mask[r, c]

    def is_valid(self, row: int, col: int) -> bool:
        """Check if (row, col) is within bounds and not blocked."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return not self.blocking_mask[row, col]

    def neighbors(self, node: int) -> List[int]:
        """
        Get neighbors of a node (excluding blocked nodes).

        Implements GraphView protocol.
        """
        r, c = self._id_to_coord(node)
        result = []

        idx = 0
        while idx < len(self._offsets):
            dr, dc = self._offsets[idx]
            nr, nc = r + dr, c + dc

            if self.is_valid(nr, nc):
                result.append(self._coord_to_id(nr, nc))

            idx += 1

        return result

    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return len(self.neighbors(node))

    def seed_nodes(self) -> List[int]:
        """Return seed nodes (corners of image that aren't blocked)."""
        seeds = []
        corners = [
            (0, 0),
            (0, self.width - 1),
            (self.height - 1, 0),
            (self.height - 1, self.width - 1)
        ]
        idx = 0
        while idx < len(corners):
            r, c = corners[idx]
            if not self.blocking_mask[r, c]:
                seeds.append(self._coord_to_id(r, c))
            idx += 1

        if not seeds:
            # Find first non-blocked node
            r = 0
            while r < self.height:
                c = 0
                while c < self.width:
                    if not self.blocking_mask[r, c]:
                        return [self._coord_to_id(r, c)]
                    c += 1
                r += 1

        return seeds

    def edge_weight(self, from_node: int, to_node: int) -> float:
        """
        Compute edge weight between adjacent nodes.

        Weight = base_distance + elevation_scale * |h(to) - h(from)|

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            Edge weight (infinite if target is blocked)
        """
        to_r, to_c = self._id_to_coord(to_node)

        # Blocked nodes have infinite cost
        if self.blocking_mask[to_r, to_c]:
            return float('inf')

        from_r, from_c = self._id_to_coord(from_node)

        # Base distance (Euclidean for diagonal moves)
        dr = abs(to_r - from_r)
        dc = abs(to_c - from_c)
        if dr + dc == 2:  # Diagonal
            base_dist = 1.414
        else:
            base_dist = 1.0

        # Elevation cost (going up or down)
        h_from = self.heightfield[from_r, from_c]
        h_to = self.heightfield[to_r, to_c]
        elevation_cost = abs(h_to - h_from)

        return base_dist + self.elevation_cost_scale * elevation_cost

    def weighted_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """
        Get neighbors with edge weights.

        Returns list of (neighbor_id, edge_weight) tuples.
        """
        neighbors = self.neighbors(node)
        result = []

        idx = 0
        while idx < len(neighbors):
            neighbor = neighbors[idx]
            weight = self.edge_weight(node, neighbor)
            result.append((neighbor, weight))
            idx += 1

        return result

    def heuristic(self, node: int, goal: int) -> float:
        """
        A* heuristic: Euclidean distance to goal.

        Admissible because actual cost >= Euclidean distance.
        """
        r1, c1 = self._id_to_coord(node)
        r2, c2 = self._id_to_coord(goal)
        return np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    def manhattan_distance(self, node: int, goal: int) -> float:
        """Manhattan distance heuristic (for 4-connectivity)."""
        r1, c1 = self._id_to_coord(node)
        r2, c2 = self._id_to_coord(goal)
        return abs(r1 - r2) + abs(c1 - c2)

    def to_sparse_laplacian(self):
        """
        Build sparse Laplacian matrix for spectral operations.

        Only includes non-blocked nodes.
        Returns (L_sparse, node_list, node_to_idx).
        """
        import torch

        # Collect non-blocked nodes
        node_list = []
        r = 0
        while r < self.height:
            c = 0
            while c < self.width:
                if not self.blocking_mask[r, c]:
                    node_list.append(self._coord_to_id(r, c))
                c += 1
            r += 1

        n = len(node_list)
        node_to_idx: Dict[int, int] = {}
        idx = 0
        while idx < n:
            node_to_idx[node_list[idx]] = idx
            idx += 1

        # Collect edges
        rows = []
        cols = []
        vals = []
        degrees = []

        i = 0
        while i < n:
            node = node_list[i]
            neighbors = self.neighbors(node)
            deg = 0
            j = 0
            while j < len(neighbors):
                neighbor = neighbors[j]
                if neighbor in node_to_idx:
                    k = node_to_idx[neighbor]
                    rows.append(i)
                    cols.append(k)
                    # Use negative edge weight for off-diagonal
                    weight = self.edge_weight(node, neighbor)
                    if weight < float('inf'):
                        vals.append(-1.0 / (1.0 + weight))  # Normalize
                        deg += 1
                j += 1
            degrees.append(deg)
            i += 1

        # Add diagonal
        i = 0
        while i < n:
            rows.append(i)
            cols.append(i)
            vals.append(float(degrees[i]))
            i += 1

        # Build sparse tensor
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        L = torch.sparse_coo_tensor(indices, values, size=(n, n))

        return L, node_list, node_to_idx


def heightfield_from_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to heightfield.

    Args:
        image: RGB or grayscale image (H, W) or (H, W, C)

    Returns:
        Heightfield as float32 array normalized to [0, 1]
    """
    if image.ndim == 3:
        # Convert to grayscale using luminance formula
        heightfield = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )
    else:
        heightfield = image.astype(np.float32)

    # Normalize to [0, 1]
    h_min = heightfield.min()
    h_max = heightfield.max()
    if h_max > h_min:
        heightfield = (heightfield - h_min) / (h_max - h_min)
    else:
        heightfield = np.zeros_like(heightfield)

    return heightfield.astype(np.float32)
