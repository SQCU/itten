"""
PathResult dataclass for pathfinding results.

Captures path, cost, method, and visualization support.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import time


@dataclass
class PathResult:
    """
    Result of a pathfinding operation.

    Contains the path, cost metrics, and visualization helpers.
    """
    # Core path data
    path: List[Tuple[int, int]]  # (row, col) coordinates
    path_node_ids: List[int] = field(default_factory=list)
    total_cost: float = 0.0
    path_length: int = 0

    # Method info
    method: str = 'unknown'  # 'spectral', 'astar', 'dijkstra'
    success: bool = False

    # Performance metrics
    nodes_visited: int = 0
    computation_time: float = 0.0

    # Spectral-specific stats
    spectral_stats: Dict[str, Any] = field(default_factory=dict)

    # Visualization data (optional)
    visited_mask: Optional[np.ndarray] = None
    cost_field: Optional[np.ndarray] = None

    # Comparison metrics (set when comparing to optimal)
    optimal_cost: Optional[float] = None
    cost_ratio: Optional[float] = None

    def __post_init__(self):
        """Compute derived fields."""
        if self.path_length == 0:
            self.path_length = len(self.path)

    def to_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create binary mask of path pixels.

        Args:
            height: Image height
            width: Image width

        Returns:
            Boolean mask where path pixels are True
        """
        mask = np.zeros((height, width), dtype=bool)
        idx = 0
        while idx < len(self.path):
            r, c = self.path[idx]
            if 0 <= r < height and 0 <= c < width:
                mask[r, c] = True
            idx += 1
        return mask

    def to_colored_overlay(
        self,
        height: int,
        width: int,
        path_color: Tuple[int, int, int] = (255, 0, 0),
        start_color: Tuple[int, int, int] = (0, 255, 0),
        goal_color: Tuple[int, int, int] = (0, 0, 255),
        line_width: int = 2
    ) -> np.ndarray:
        """
        Create RGBA overlay with colored path.

        Args:
            height: Image height
            width: Image width
            path_color: RGB color for path (default red)
            start_color: RGB color for start point (default green)
            goal_color: RGB color for goal point (default blue)
            line_width: Width of path line in pixels

        Returns:
            RGBA image (height, width, 4) as uint8
        """
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        if not self.path:
            return overlay

        # Draw path with line width
        half_width = line_width // 2
        idx = 0
        while idx < len(self.path):
            r, c = self.path[idx]
            # Draw square around path point for line width
            dr = -half_width
            while dr <= half_width:
                dc = -half_width
                while dc <= half_width:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        overlay[rr, cc, 0] = path_color[0]
                        overlay[rr, cc, 1] = path_color[1]
                        overlay[rr, cc, 2] = path_color[2]
                        overlay[rr, cc, 3] = 255
                    dc += 1
                dr += 1
            idx += 1

        # Mark start point
        if len(self.path) >= 1:
            r, c = self.path[0]
            dr = -half_width - 1
            while dr <= half_width + 1:
                dc = -half_width - 1
                while dc <= half_width + 1:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        overlay[rr, cc, 0] = start_color[0]
                        overlay[rr, cc, 1] = start_color[1]
                        overlay[rr, cc, 2] = start_color[2]
                        overlay[rr, cc, 3] = 255
                    dc += 1
                dr += 1

        # Mark goal point
        if len(self.path) >= 2:
            r, c = self.path[-1]
            dr = -half_width - 1
            while dr <= half_width + 1:
                dc = -half_width - 1
                while dc <= half_width + 1:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        overlay[rr, cc, 0] = goal_color[0]
                        overlay[rr, cc, 1] = goal_color[1]
                        overlay[rr, cc, 2] = goal_color[2]
                        overlay[rr, cc, 3] = 255
                    dc += 1
                dr += 1

        return overlay

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Path Result ({self.method})",
            f"  Success: {self.success}",
            f"  Path length: {self.path_length} steps",
            f"  Total cost: {self.total_cost:.2f}",
            f"  Nodes visited: {self.nodes_visited}",
            f"  Computation time: {self.computation_time:.3f}s",
        ]

        if self.optimal_cost is not None:
            lines.append(f"  Optimal cost: {self.optimal_cost:.2f}")
        if self.cost_ratio is not None:
            lines.append(f"  Cost ratio: {self.cost_ratio:.2f}x optimal")
        if self.spectral_stats:
            lines.append(f"  Spectral stats: {self.spectral_stats}")

        return '\n'.join(lines)


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
