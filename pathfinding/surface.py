"""
Surface Pathfinding - High-level API for pathfinding on texture surfaces.

This is the main integration point between pathfinding and texture synthesis.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np

from .graph import HeightfieldGraph, heightfield_from_image
from .result import PathResult
from .spectral import spectral_path
from .classical import dijkstra_path, astar_path, compare_methods


def texture_to_pathfinding_graph(
    heightfield: np.ndarray,
    blocking_mask: Optional[np.ndarray] = None,
    elevation_cost_scale: float = 1.0,
    connectivity: str = "8"
) -> HeightfieldGraph:
    """
    Convert texture heightfield to pathfinding graph.

    Args:
        heightfield: 2D array of heights (normalized to [0,1] recommended)
        blocking_mask: Optional boolean mask where True = blocked
        elevation_cost_scale: Multiplier for elevation difference in edge cost
        connectivity: "4" for von Neumann, "8" for Moore neighborhood

    Returns:
        HeightfieldGraph ready for pathfinding

    Example:
        >>> heightfield = np.random.rand(100, 100)
        >>> blocking = heightfield > 0.8  # Block high areas
        >>> graph = texture_to_pathfinding_graph(heightfield, blocking)
        >>> result = find_surface_path(heightfield, (10, 10), (90, 90), blocking)
    """
    # Normalize heightfield if needed
    h_min = heightfield.min()
    h_max = heightfield.max()
    if h_max > h_min:
        normalized = (heightfield - h_min) / (h_max - h_min)
    else:
        normalized = np.zeros_like(heightfield, dtype=np.float32)

    return HeightfieldGraph(
        heightfield=normalized,
        blocking_mask=blocking_mask,
        elevation_cost_scale=elevation_cost_scale,
        connectivity=connectivity
    )


def find_surface_path(
    heightfield: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocking_mask: Optional[np.ndarray] = None,
    method: str = 'spectral',
    elevation_cost_scale: float = 1.0,
    connectivity: str = "8",
    max_steps: int = 2000,
    max_nodes: int = 100000
) -> PathResult:
    """
    Find path on texture surface avoiding blocking geometry.

    This is the main API for pathfinding on textures.

    Args:
        heightfield: 2D array of heights
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        blocking_mask: Optional boolean mask where True = blocked
        method: 'spectral', 'astar', or 'dijkstra'
        elevation_cost_scale: Multiplier for elevation difference in edge cost
        connectivity: "4" for von Neumann, "8" for Moore neighborhood
        max_steps: Maximum steps for spectral method
        max_nodes: Maximum nodes for classical methods

    Returns:
        PathResult with path coordinates and statistics

    Example:
        >>> heightfield = texture_synthesis_result.height_field
        >>> blocking = heightfield > 0.7
        >>> result = find_surface_path(
        ...     heightfield, (10, 10), (200, 200),
        ...     blocking_mask=blocking,
        ...     method='spectral'
        ... )
        >>> print(result.summary())
    """
    graph = texture_to_pathfinding_graph(
        heightfield, blocking_mask, elevation_cost_scale, connectivity
    )

    if method == 'spectral':
        return spectral_path(graph, start, goal, max_steps)
    elif method == 'astar':
        return astar_path(graph, start, goal, max_nodes)
    elif method == 'dijkstra':
        return dijkstra_path(graph, start, goal, max_nodes)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spectral', 'astar', or 'dijkstra'")


def compare_surface_paths(
    heightfield: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocking_mask: Optional[np.ndarray] = None,
    elevation_cost_scale: float = 1.0,
    connectivity: str = "8",
    max_steps: int = 2000,
    max_nodes: int = 100000
) -> Dict[str, PathResult]:
    """
    Compare all pathfinding methods on the same texture surface.

    Runs Dijkstra (optimal), A*, and spectral pathfinding, computing
    cost ratios relative to optimal.

    Args:
        heightfield: 2D array of heights
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        blocking_mask: Optional boolean mask where True = blocked
        elevation_cost_scale: Multiplier for elevation difference
        connectivity: "4" or "8"
        max_steps: Maximum steps for spectral method
        max_nodes: Maximum nodes for classical methods

    Returns:
        Dict mapping method name to PathResult

    Example:
        >>> results = compare_surface_paths(heightfield, start, goal, blocking)
        >>> for method, result in results.items():
        ...     print(f"{method}: cost={result.total_cost:.2f}, ratio={result.cost_ratio:.2f}x")
    """
    graph = texture_to_pathfinding_graph(
        heightfield, blocking_mask, elevation_cost_scale, connectivity
    )

    return compare_methods(graph, start, goal, max_steps, max_nodes)


def find_valid_endpoints(
    blocking_mask: np.ndarray,
    margin: int = 10
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Find valid start and goal points that aren't blocked.

    Tries to find points near opposite corners.

    Args:
        blocking_mask: Boolean mask where True = blocked
        margin: Margin from image edges to search

    Returns:
        (start, goal) as (row, col) tuples

    Raises:
        ValueError: If no valid endpoints found
    """
    height, width = blocking_mask.shape

    # Find start near top-left
    start = None
    r = margin
    while r < height // 4 and start is None:
        c = margin
        while c < width // 4 and start is None:
            if not blocking_mask[r, c]:
                start = (r, c)
            c += 1
        r += 1

    if start is None:
        # Fallback: find any non-blocked point in top-left quadrant
        r = 0
        while r < height // 2 and start is None:
            c = 0
            while c < width // 2 and start is None:
                if not blocking_mask[r, c]:
                    start = (r, c)
                c += 1
            r += 1

    # Find goal near bottom-right
    goal = None
    r = height - margin - 1
    while r >= 3 * height // 4 and goal is None:
        c = width - margin - 1
        while c >= 3 * width // 4 and goal is None:
            if not blocking_mask[r, c]:
                goal = (r, c)
            c -= 1
        r -= 1

    if goal is None:
        # Fallback: find any non-blocked point in bottom-right quadrant
        r = height - 1
        while r >= height // 2 and goal is None:
            c = width - 1
            while c >= width // 2 and goal is None:
                if not blocking_mask[r, c]:
                    goal = (r, c)
                c -= 1
            r -= 1

    if start is None or goal is None:
        raise ValueError("Could not find valid start/goal points - too much blocking")

    return start, goal


def create_blocking_mask(
    heightfield: np.ndarray,
    threshold: float = 0.7,
    above: bool = True
) -> np.ndarray:
    """
    Create blocking mask from heightfield threshold.

    Args:
        heightfield: 2D array of heights
        threshold: Height threshold
        above: If True, block heights >= threshold; if False, block <= threshold

    Returns:
        Boolean mask where True = blocked
    """
    if above:
        return heightfield >= threshold
    else:
        return heightfield <= threshold
