"""
Spectral Pathfinding - Local spectral guidance for path finding.

Uses local Fiedler vector to guide pathfinding without full graph materialization.
Moved from spectral_pathfind.py for integration with texture surfaces.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Set, Dict, Tuple, Optional, Protocol
import numpy as np
import time

from spectral_ops_fast import (
    local_fiedler_vector,
    local_expansion_estimate,
    expand_neighborhood,
    DEVICE
)

from .result import PathResult, Timer
from .graph import HeightfieldGraph


class GraphView(Protocol):
    """Graph you can query but not enumerate."""
    def neighbors(self, node: int) -> List[int]: ...
    def degree(self, node: int) -> int: ...
    def seed_nodes(self) -> List[int]: ...


class SpectralPathfinder:
    """
    Local spectral pathfinding: at each step, compute local Fiedler vector
    and choose neighbor most aligned with goal direction in spectral space.

    This uses only local graph queries - never materializes full graph.

    On structured graphs (grids, lattices), combines spectral guidance with
    geometric heuristic for efficient navigation.
    """

    def __init__(
        self,
        graph: GraphView,
        spectral_radius: int = 2,
        lanczos_iters: int = 15,
        use_expansion_bias: bool = False,
        exploration_probability: float = 0.02,
        spectral_weight: float = 0.3,
        heuristic_weight: float = 0.7,
        rng_seed: Optional[int] = None
    ):
        """
        Initialize spectral pathfinder.

        Args:
            graph: GraphView supporting neighbors() queries
            spectral_radius: neighborhood size for local Fiedler computation
            lanczos_iters: iterations for Lanczos algorithm
            use_expansion_bias: bias toward lower expansion (corridors/structure)
            exploration_probability: random exploration to avoid local minima
            spectral_weight: weight for spectral component (0-1)
            heuristic_weight: weight for geometric heuristic (0-1)
            rng_seed: random seed for reproducibility
        """
        self.graph = graph
        self.spectral_radius = spectral_radius
        self.lanczos_iters = lanczos_iters
        self.use_expansion_bias = use_expansion_bias
        self.exploration_probability = exploration_probability
        self.spectral_weight = spectral_weight
        self.heuristic_weight = heuristic_weight
        self._rng = np.random.default_rng(rng_seed)

        # Check if graph has coordinates
        self._has_coords = hasattr(graph, 'coord_of') or hasattr(graph, '_id_to_coord')

    def _get_coord(self, node: int) -> Optional[Tuple[float, float]]:
        """Get coordinates for a node if available."""
        if hasattr(self.graph, 'coord_of'):
            coord = self.graph.coord_of(node)
            return (float(coord[0]), float(coord[1]))
        elif hasattr(self.graph, '_id_to_coord'):
            coord = self.graph._id_to_coord(node)
            return (float(coord[0]), float(coord[1]))
        return None

    def _manhattan_distance(self, node1: int, node2: int) -> float:
        """Compute Manhattan distance if coordinates available."""
        c1 = self._get_coord(node1)
        c2 = self._get_coord(node2)
        if c1 is None or c2 is None:
            return 0.0
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

    def compute_spectral_direction(
        self,
        current: int,
        goal: int,
        visited: Set[int]
    ) -> Tuple[int, float]:
        """
        Compute best neighbor to move toward goal using local spectral structure.

        Combines:
        1. Spectral guidance (Fiedler direction)
        2. Geometric heuristic (if coordinates available)
        3. Expansion bias (optional)

        Returns (best_neighbor, confidence_score).
        """
        neighbors = self.graph.neighbors(current)

        # Filter out visited nodes
        unvisited = [n for n in neighbors if n not in visited]

        if not unvisited:
            # All neighbors visited - allow backtracking to least recently visited
            unvisited = neighbors

        if not unvisited:
            return current, 0.0

        # Direct connection to goal?
        if goal in unvisited:
            return goal, 1.0

        scores: Dict[int, float] = {}

        # Component 1: Geometric heuristic (if coordinates available)
        if self._has_coords and self.heuristic_weight > 0:
            current_dist = self._manhattan_distance(current, goal)
            heuristic_scores = []
            idx = 0
            while idx < len(unvisited):
                neighbor = unvisited[idx]
                neighbor_dist = self._manhattan_distance(neighbor, goal)
                # Score: how much closer does this move get us?
                # Positive = getting closer
                heuristic_scores.append(current_dist - neighbor_dist)
                idx += 1

            # Normalize heuristic scores
            h_min = min(heuristic_scores)
            h_max = max(heuristic_scores)
            h_range = h_max - h_min if h_max > h_min else 1.0

            idx = 0
            while idx < len(unvisited):
                neighbor = unvisited[idx]
                scores[neighbor] = self.heuristic_weight * (heuristic_scores[idx] - h_min) / h_range
                idx += 1

        # Component 2: Spectral guidance (Fiedler direction)
        if self.spectral_weight > 0:
            # Compute local Fiedler centered on current
            active_nodes = expand_neighborhood(
                self.graph, {current}, self.spectral_radius
            )

            # Include goal in seeds if in neighborhood
            seeds = [current]
            if goal in active_nodes:
                seeds.append(goal)

            fiedler, lambda2 = local_fiedler_vector(
                self.graph,
                seed_nodes=seeds,
                num_iterations=self.lanczos_iters,
                hop_expansion=self.spectral_radius
            )

            if fiedler and current in fiedler:
                current_val = fiedler.get(current, 0.0)
                goal_val = fiedler.get(goal, 0.0)

                # Determine target direction in spectral space
                target_direction = np.sign(goal_val - current_val)
                if abs(goal_val - current_val) < 1e-8:
                    target_direction = 1.0

                spectral_scores = []
                idx = 0
                while idx < len(unvisited):
                    neighbor = unvisited[idx]
                    neighbor_val = fiedler.get(neighbor, current_val)
                    delta = neighbor_val - current_val
                    spectral_scores.append(delta * target_direction)
                    idx += 1

                # Normalize spectral scores
                s_min = min(spectral_scores)
                s_max = max(spectral_scores)
                s_range = s_max - s_min if s_max > s_min else 1.0

                idx = 0
                while idx < len(unvisited):
                    neighbor = unvisited[idx]
                    if neighbor not in scores:
                        scores[neighbor] = 0.0
                    scores[neighbor] += self.spectral_weight * (spectral_scores[idx] - s_min) / s_range
                    idx += 1

        # Component 3: Expansion bias (optional)
        if self.use_expansion_bias:
            idx = 0
            while idx < len(unvisited):
                neighbor = unvisited[idx]
                expansion = local_expansion_estimate(
                    self.graph, neighbor, radius=1, num_lanczos=5
                )
                if neighbor not in scores:
                    scores[neighbor] = 0.0
                scores[neighbor] -= expansion * 0.1
                idx += 1

        # Ensure all neighbors have scores
        idx = 0
        while idx < len(unvisited):
            neighbor = unvisited[idx]
            if neighbor not in scores:
                scores[neighbor] = 0.0
            idx += 1

        # Select best neighbor
        best_neighbor = max(scores.keys(), key=lambda n: scores[n])
        best_score = scores[best_neighbor]

        # Normalize confidence to [0, 1]
        score_range = max(scores.values()) - min(scores.values()) + 1e-10
        confidence = (best_score - min(scores.values())) / score_range

        return best_neighbor, confidence

    def find_path(
        self,
        start: int,
        goal: int,
        max_steps: int = 1000
    ) -> Tuple[List[int], bool, Dict]:
        """
        Find path from start to goal using local spectral guidance.

        Returns:
            path: list of node IDs from start to goal
            success: whether goal was reached
            stats: dict with path statistics
        """
        current = start
        path = [current]
        visited = {current}

        total_confidence = 0.0
        stuck_count = 0

        step = 0
        while step < max_steps:
            if current == goal:
                return path, True, {
                    'steps': step,
                    'path_length': len(path),
                    'avg_confidence': total_confidence / max(step, 1),
                    'nodes_visited': len(visited)
                }

            # Check if goal is a direct neighbor
            if goal in self.graph.neighbors(current):
                path.append(goal)
                return path, True, {
                    'steps': step + 1,
                    'path_length': len(path),
                    'avg_confidence': total_confidence / max(step + 1, 1),
                    'nodes_visited': len(visited)
                }

            # Random exploration?
            if self._rng.random() < self.exploration_probability:
                neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
                if neighbors:
                    next_node = self._rng.choice(neighbors)
                    confidence = 0.5
                else:
                    next_node, confidence = self.compute_spectral_direction(
                        current, goal, visited
                    )
            else:
                next_node, confidence = self.compute_spectral_direction(
                    current, goal, visited
                )

            total_confidence += confidence

            if next_node == current:
                stuck_count += 1
                if stuck_count > 10:
                    # Really stuck - try random walk
                    all_neighbors = self.graph.neighbors(current)
                    if all_neighbors:
                        next_node = self._rng.choice(all_neighbors)
                        stuck_count = 0
            else:
                stuck_count = 0

            current = next_node
            path.append(current)
            visited.add(current)
            step += 1

        return path, False, {
            'steps': max_steps,
            'path_length': len(path),
            'avg_confidence': total_confidence / max_steps,
            'nodes_visited': len(visited)
        }


def spectral_path(
    graph: HeightfieldGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 2000,
    spectral_radius: int = 2,
    lanczos_iters: int = 15,
    rng_seed: int = 42
) -> PathResult:
    """
    Find path using spectral pathfinding.

    Args:
        graph: HeightfieldGraph to navigate
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        max_steps: Maximum path steps
        spectral_radius: Neighborhood size for Fiedler computation
        lanczos_iters: Lanczos iterations
        rng_seed: Random seed

    Returns:
        PathResult with path and statistics
    """
    start_node = graph._coord_to_id(start[0], start[1])
    goal_node = graph._coord_to_id(goal[0], goal[1])

    # Check start/goal validity
    if graph.is_blocked(start_node):
        return PathResult(
            path=[],
            method='spectral',
            success=False,
            total_cost=float('inf')
        )

    if graph.is_blocked(goal_node):
        return PathResult(
            path=[],
            method='spectral',
            success=False,
            total_cost=float('inf')
        )

    pathfinder = SpectralPathfinder(
        graph,
        spectral_radius=spectral_radius,
        lanczos_iters=lanczos_iters,
        rng_seed=rng_seed
    )

    with Timer() as timer:
        node_path, success, stats = pathfinder.find_path(start_node, goal_node, max_steps)

    # Convert node IDs to coordinates
    coord_path = []
    total_cost = 0.0
    idx = 0
    while idx < len(node_path):
        node = node_path[idx]
        coord_path.append(graph.coord_of(node))
        if idx > 0:
            total_cost += graph.edge_weight(node_path[idx - 1], node)
        idx += 1

    return PathResult(
        path=coord_path,
        path_node_ids=node_path,
        total_cost=total_cost,
        path_length=len(coord_path),
        method='spectral',
        success=success,
        nodes_visited=stats['nodes_visited'],
        computation_time=timer.elapsed,
        spectral_stats={
            'avg_confidence': stats.get('avg_confidence', 0),
            'steps': stats.get('steps', 0)
        }
    )
