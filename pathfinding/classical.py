"""
Classical Pathfinding Algorithms - Dijkstra and A*.

These are standard graph search algorithms for comparison with spectral methods.
"""

from typing import List, Dict, Set, Tuple, Optional
import heapq
import time

from .result import PathResult, Timer
from .graph import HeightfieldGraph


def dijkstra_path(
    graph: HeightfieldGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_nodes: int = 100000
) -> PathResult:
    """
    Dijkstra's algorithm for shortest path.

    Args:
        graph: HeightfieldGraph to navigate
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        max_nodes: Maximum nodes to visit before giving up

    Returns:
        PathResult with path and statistics
    """
    start_node = graph._coord_to_id(start[0], start[1])
    goal_node = graph._coord_to_id(goal[0], goal[1])

    # Check start/goal validity
    if graph.is_blocked(start_node):
        return PathResult(
            path=[],
            method='dijkstra',
            success=False,
            total_cost=float('inf')
        )

    if graph.is_blocked(goal_node):
        return PathResult(
            path=[],
            method='dijkstra',
            success=False,
            total_cost=float('inf')
        )

    with Timer() as timer:
        distances: Dict[int, float] = {start_node: 0}
        predecessors: Dict[int, int] = {}
        visited: Set[int] = set()

        # Priority queue: (distance, node)
        heap = [(0.0, start_node)]

        while heap and len(visited) < max_nodes:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            if current == goal_node:
                # Reconstruct path
                path_nodes = [goal_node]
                node = goal_node
                while node in predecessors:
                    node = predecessors[node]
                    path_nodes.append(node)
                path_nodes.reverse()

                # Convert to coordinates
                coord_path = []
                idx = 0
                while idx < len(path_nodes):
                    coord_path.append(graph.coord_of(path_nodes[idx]))
                    idx += 1

                return PathResult(
                    path=coord_path,
                    path_node_ids=path_nodes,
                    total_cost=dist,
                    path_length=len(coord_path),
                    method='dijkstra',
                    success=True,
                    nodes_visited=len(visited),
                    computation_time=timer.elapsed
                )

            # Explore neighbors
            weighted_neighbors = graph.weighted_neighbors(current)
            idx = 0
            while idx < len(weighted_neighbors):
                neighbor, weight = weighted_neighbors[idx]
                if neighbor not in visited:
                    new_dist = dist + weight
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current
                        heapq.heappush(heap, (new_dist, neighbor))
                idx += 1

    # Path not found
    return PathResult(
        path=[],
        method='dijkstra',
        success=False,
        total_cost=float('inf'),
        nodes_visited=len(visited),
        computation_time=timer.elapsed
    )


def astar_path(
    graph: HeightfieldGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_nodes: int = 100000,
    use_euclidean: bool = True
) -> PathResult:
    """
    A* algorithm for shortest path with heuristic.

    Args:
        graph: HeightfieldGraph to navigate
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        max_nodes: Maximum nodes to visit before giving up
        use_euclidean: Use Euclidean heuristic (True) or Manhattan (False)

    Returns:
        PathResult with path and statistics
    """
    start_node = graph._coord_to_id(start[0], start[1])
    goal_node = graph._coord_to_id(goal[0], goal[1])

    # Check start/goal validity
    if graph.is_blocked(start_node):
        return PathResult(
            path=[],
            method='astar',
            success=False,
            total_cost=float('inf')
        )

    if graph.is_blocked(goal_node):
        return PathResult(
            path=[],
            method='astar',
            success=False,
            total_cost=float('inf')
        )

    # Select heuristic
    if use_euclidean:
        heuristic = graph.heuristic
    else:
        heuristic = graph.manhattan_distance

    with Timer() as timer:
        # g_score: cost from start to node
        g_score: Dict[int, float] = {start_node: 0}
        # f_score: g_score + heuristic
        f_score: Dict[int, float] = {start_node: heuristic(start_node, goal_node)}

        predecessors: Dict[int, int] = {}
        visited: Set[int] = set()

        # Priority queue: (f_score, counter, node)
        # Counter for tie-breaking (FIFO among equal f-scores)
        counter = 0
        heap = [(f_score[start_node], counter, start_node)]

        while heap and len(visited) < max_nodes:
            _, _, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            if current == goal_node:
                # Reconstruct path
                path_nodes = [goal_node]
                node = goal_node
                while node in predecessors:
                    node = predecessors[node]
                    path_nodes.append(node)
                path_nodes.reverse()

                # Convert to coordinates
                coord_path = []
                idx = 0
                while idx < len(path_nodes):
                    coord_path.append(graph.coord_of(path_nodes[idx]))
                    idx += 1

                return PathResult(
                    path=coord_path,
                    path_node_ids=path_nodes,
                    total_cost=g_score[goal_node],
                    path_length=len(coord_path),
                    method='astar',
                    success=True,
                    nodes_visited=len(visited),
                    computation_time=timer.elapsed
                )

            # Explore neighbors
            weighted_neighbors = graph.weighted_neighbors(current)
            idx = 0
            while idx < len(weighted_neighbors):
                neighbor, weight = weighted_neighbors[idx]
                if neighbor not in visited:
                    tentative_g = g_score[current] + weight
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        predecessors[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal_node)
                        f_score[neighbor] = f
                        counter += 1
                        heapq.heappush(heap, (f, counter, neighbor))
                idx += 1

    # Path not found
    return PathResult(
        path=[],
        method='astar',
        success=False,
        total_cost=float('inf'),
        nodes_visited=len(visited),
        computation_time=timer.elapsed
    )


def compare_methods(
    graph: HeightfieldGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 2000,
    dijkstra_max_nodes: int = 50000
) -> Dict[str, PathResult]:
    """
    Compare all pathfinding methods on the same start/goal.

    Args:
        graph: HeightfieldGraph to navigate
        start: (row, col) start coordinates
        goal: (row, col) goal coordinates
        max_steps: Max steps for spectral method
        dijkstra_max_nodes: Max nodes for Dijkstra/A*

    Returns:
        Dict mapping method name to PathResult
    """
    from .spectral import spectral_path

    results = {}

    # Run Dijkstra first (optimal)
    results['dijkstra'] = dijkstra_path(graph, start, goal, dijkstra_max_nodes)

    # Run A*
    results['astar'] = astar_path(graph, start, goal, dijkstra_max_nodes)

    # Run spectral
    results['spectral'] = spectral_path(graph, start, goal, max_steps)

    # Add cost ratios if Dijkstra found optimal
    if results['dijkstra'].success:
        optimal_cost = results['dijkstra'].total_cost
        method_name = 'astar'
        while method_name in ['astar', 'spectral']:
            result = results[method_name]
            if result.success:
                result.optimal_cost = optimal_cost
                if optimal_cost > 0:
                    result.cost_ratio = result.total_cost / optimal_cost
            if method_name == 'astar':
                method_name = 'spectral'
            else:
                break

    return results
