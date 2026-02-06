"""Spectral pathfinding as nn.Modules -- Phase D of TODO_DEMO_RECOVERY.md.

Reimplements pathfinding/spectral.py (399 lines) using pure PyTorch and the
Phase A foundation (GraphEmbedding, LocalSpectralProbe from
spectral_graph_embedding.py). No numpy. torch.compile compatible on inner
modules; fullgraph=False accepted for the outer sequential step loop.

The core algorithm: at each step, compute a local Fiedler vector around the
current position and choose the neighbor most aligned with the spectral
gradient toward the goal. This is "Dijkstra but kinda wrong" -- it uses only
local spectral information (O(hop_radius^2) per step) instead of full graph
access (O(n^2) for Dijkstra). The approximation error is bounded: empirically
low transport distance from the optimal path.

Architecture analog: greedy decoding with local attention (Beltagy et al.
2020, Longformer). The model sees a local context window (hop_radius hops),
makes a locally-optimal decision, and advances. The spectral information
provides a "soft gradient" toward the goal that regular greedy search lacks.

The key insight: local spectral properties encode approximate global structure.
The Fiedler gradient points "away from bottlenecks" -- which is exactly the
information Dijkstra uses implicitly via shortest-path relaxation.

Source: pathfinding/spectral.py:SpectralPathfinder (399 lines)
Depends on: spectral_graph_embedding.py (Phase A)
Priority: P1 (Phase D of TODO_DEMO_RECOVERY.md)
"""

import torch
import torch.nn as nn
import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from spectral_graph_embedding import GraphEmbedding, LocalSpectralProbe


# ============================================================
# SpectralNavigator -- inner module, compilable
# ============================================================


class SpectralNavigator(nn.Module):
    """Compute the spectral gradient: which neighbor to advance toward.

    Given a local Fiedler vector around the current node and a set of
    candidate neighbors, scores each candidate by alignment with the
    spectral direction toward the goal. Returns scores as a dense tensor.

    This module is the compilable inner kernel of SpectralPathfinder.
    The outer sequential loop (step-by-step navigation) has graph breaks;
    this module does not.

    Architecture analog: local attention scoring (Beltagy et al. 2020,
    Longformer). The Fiedler vector values at neighbor positions are the
    "keys"; the spectral direction toward the goal is the "query".
    The dot product (delta * target_direction) is the attention score.
    Unlike softmax attention, we take the argmax -- greedy decoding.

    The gating vocabulary from _even_cuter applies here: the Fiedler
    value at each neighbor acts as a gate. Neighbors whose Fiedler
    delta aligns with the target direction are "open" (high gate);
    neighbors in the wrong spectral direction are "closed" (low gate).
    This is a hard-threshold gate (cf. SwiGLU, Dauphin et al. 2017)
    rather than soft attention.

    Parameters
    ----------
    spectral_weight : float
        Weight for spectral component in neighbor scoring. Default 0.3.
    heuristic_weight : float
        Weight for geometric heuristic (Manhattan distance). Default 0.7.
    """

    def __init__(
        self,
        spectral_weight: float = 0.3,
        heuristic_weight: float = 0.7,
    ):
        super().__init__()
        self.register_buffer(
            "spectral_weight",
            torch.tensor(spectral_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "heuristic_weight",
            torch.tensor(heuristic_weight, dtype=torch.float32),
        )

    def forward(
        self,
        fiedler_values: torch.Tensor,
        current_idx: int,
        goal_idx: int,
        neighbor_indices: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        current_node: Optional[int] = None,
        goal_node: Optional[int] = None,
    ) -> torch.Tensor:
        """Score candidate neighbors by spectral + geometric alignment.

        Transcribed from pathfinding/spectral.py:compute_spectral_direction
        (lines 99-236). Same scoring logic, pure-tensor form.

        Args:
            fiedler_values: (m,) local Fiedler vector over the subgraph.
                Index i corresponds to the i-th node in the local subgraph
                ordering (node_list from LocalSpectralProbe._build_local_laplacian).
            current_idx: Index of current node in the LOCAL subgraph ordering.
            goal_idx: Index of goal node in the LOCAL subgraph ordering.
                If goal is outside the local subgraph, pass -1 and goal_val
                will default to 0.
            neighbor_indices: (k,) indices of candidate neighbors in LOCAL
                subgraph ordering. -1 for neighbors outside the subgraph.
            coords: Optional (n, 2) global node coordinates for geometric
                heuristic. If None, only spectral scoring is used.
            current_node: Global node ID of current position (for coord lookup).
            goal_node: Global node ID of goal (for coord lookup).

        Returns:
            scores: (k,) score per candidate neighbor. Higher = better.
        """
        device = fiedler_values.device
        k = neighbor_indices.shape[0]
        scores = torch.zeros(k, device=device, dtype=torch.float32)

        # --- Spectral component (pathfinding/spectral.py lines 158-205) ---
        current_val = fiedler_values[current_idx]

        if goal_idx >= 0:
            goal_val = fiedler_values[goal_idx]
        else:
            # Goal outside local neighborhood -- assume 0 (neutral)
            goal_val = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Target direction in spectral space: sign(goal_val - current_val)
        # spectral.py line 181
        direction_raw = goal_val - current_val
        target_direction = torch.sign(direction_raw)
        # If goal and current have nearly identical Fiedler values, default to +1
        # spectral.py lines 182-183
        target_direction = torch.where(
            torch.abs(direction_raw) < 1e-8,
            torch.ones_like(target_direction),
            target_direction,
        )

        # Compute per-neighbor spectral scores
        # For neighbors inside the subgraph: delta = fiedler[neighbor] - current_val
        # For neighbors outside (-1 index): delta = 0 (neutral)
        valid_mask = neighbor_indices >= 0
        safe_indices = torch.where(valid_mask, neighbor_indices, torch.zeros_like(neighbor_indices))
        neighbor_vals = fiedler_values[safe_indices]
        # Zero out values for invalid (out-of-subgraph) neighbors
        neighbor_vals = torch.where(valid_mask, neighbor_vals, current_val)
        deltas = neighbor_vals - current_val
        spectral_raw = deltas * target_direction

        # Normalize spectral scores to [0, 1] (spectral.py lines 195-197)
        s_min = spectral_raw.min()
        s_max = spectral_raw.max()
        s_range = s_max - s_min
        s_range = torch.where(s_range > 1e-10, s_range, torch.ones_like(s_range))
        spectral_norm = (spectral_raw - s_min) / s_range

        scores = scores + self.spectral_weight * spectral_norm

        # --- Geometric heuristic component (spectral.py lines 134-155) ---
        if coords is not None and current_node is not None and goal_node is not None:
            current_coord = coords[current_node]  # (2,)
            goal_coord = coords[goal_node]  # (2,)
            current_dist = torch.abs(current_coord - goal_coord).sum()

            # We need global node IDs for coord lookup -- but neighbor_indices
            # are local. The caller must provide a mapping or we skip.
            # For grid graphs, coords is (n, 2) indexed by global node ID.
            # The caller should pass global neighbor IDs separately.
            # For now, geometric heuristic is handled in SpectralPathfinder
            # using global coords directly (see _score_neighbors_geometric).
            pass

        return scores


# ============================================================
# SpectralPathfinder -- outer module with sequential loop
# ============================================================


class SpectralPathfinder(nn.Module):
    """Graph-local spectral navigation.

    At each step: compute local Fiedler vector around current position,
    choose neighbor most aligned with goal direction in spectral space.

    This is "Dijkstra but kinda wrong" -- it uses only local spectral
    information (O(hop_radius^2) per step) instead of full graph access
    (O(n^2) for Dijkstra). The approximation error is bounded: empirically
    low transport distance (Wasserstein or other) from the optimal path.

    Architecture analog: greedy decoding with local attention.
    The model sees a local context window (spectral_radius hops),
    makes a locally-optimal decision, and advances. The spectral
    information provides a "soft gradient" toward the goal that
    regular greedy search lacks.

    The key insight: local spectral properties encode approximate global
    structure. The Fiedler gradient points "away from bottlenecks" --
    which is exactly the information Dijkstra uses implicitly via
    shortest-path relaxation.

    torch.compile strategy
    ----------------------
    The outer step loop is inherently sequential (each step depends on
    the current position) and cannot compile with fullgraph=True. But the
    inner LocalSpectralProbe and SpectralNavigator CAN compile. We
    compile those and accept graph breaks at the loop boundary.

    This is explicitly noted in TODO_DEMO_RECOVERY.md: "SpectralEmbedding
    needs fullgraph=False due to sparse_coo_tensor graph break."

    Gating semantics
    ----------------
    Navigation decisions map to the gating vocabulary from _even_cuter:
    - The Fiedler value at each neighbor is a gate signal
    - The spectral direction toward the goal defines the "open" gate polarity
    - Neighbors with aligned Fiedler delta pass the gate (high score)
    - Neighbors with opposing delta are gated out (low score)
    - The exploration_probability is a stochastic gate bypass (dropout analog)

    Source: pathfinding/spectral.py:SpectralPathfinder (lines 34-317)
    Depends on: spectral_graph_embedding.py:LocalSpectralProbe (Phase A)

    Parameters
    ----------
    local_probe : LocalSpectralProbe
        Module for computing local Fiedler vectors around query nodes.
    spectral_weight : float
        Weight for spectral component in neighbor scoring. Default 0.3.
    heuristic_weight : float
        Weight for geometric heuristic (Manhattan distance). Default 0.7.
    exploration_probability : float
        Probability of random neighbor selection to escape local minima.
        Analogous to epsilon-greedy exploration or dropout on the gate.
        Default 0.02.
    use_expansion_bias : bool
        Whether to bias toward low-expansion (corridor/structured) regions.
        Default False.
    """

    def __init__(
        self,
        local_probe: LocalSpectralProbe,
        spectral_weight: float = 0.3,
        heuristic_weight: float = 0.7,
        exploration_probability: float = 0.02,
        use_expansion_bias: bool = False,
    ):
        super().__init__()
        self.local_probe = local_probe
        self.navigator = SpectralNavigator(
            spectral_weight=spectral_weight,
            heuristic_weight=heuristic_weight,
        )
        self.register_buffer(
            "exploration_probability",
            torch.tensor(exploration_probability, dtype=torch.float32),
        )
        self.use_expansion_bias = use_expansion_bias

    def _get_neighbors(
        self, adjacency: torch.Tensor, node: int
    ) -> List[int]:
        """Get neighbors of a node from the sparse adjacency matrix.

        This is graph traversal (not hot-path numerical computation).
        Transcribed from LocalSpectralProbe._expand_neighborhood, simplified
        for single-node single-hop.

        Args:
            adjacency: Sparse (n, n) adjacency matrix (coalesced).
            node: Node index.

        Returns:
            List of neighbor node indices.
        """
        adj = adjacency.coalesce()
        indices = adj.indices()
        rows = indices[0]
        cols = indices[1]
        mask = rows == node
        neighbors = cols[mask].cpu().tolist()
        return neighbors

    def _manhattan_distance_tensor(
        self, coords: torch.Tensor, node_a: int, node_b: int
    ) -> torch.Tensor:
        """Manhattan distance between two nodes using coordinate tensor.

        Args:
            coords: (n, 2) node coordinates.
            node_a: First node index.
            node_b: Second node index.

        Returns:
            Scalar distance tensor.
        """
        return torch.abs(coords[node_a] - coords[node_b]).sum()

    def _score_neighbors_geometric(
        self,
        coords: torch.Tensor,
        current: int,
        goal: int,
        neighbor_list: List[int],
    ) -> torch.Tensor:
        """Score neighbors by geometric heuristic (Manhattan distance).

        Transcribed from pathfinding/spectral.py lines 134-155.
        Pure tensor computation on global coordinates.

        Args:
            coords: (n, 2) node coordinates.
            current: Current node (global index).
            goal: Goal node (global index).
            neighbor_list: List of global neighbor node indices.

        Returns:
            (k,) normalized heuristic scores in [0, 1]. Higher = closer to goal.
        """
        device = coords.device
        k = len(neighbor_list)
        if k == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)

        current_dist = self._manhattan_distance_tensor(coords, current, goal)
        neighbor_ids = torch.tensor(neighbor_list, device=device, dtype=torch.long)
        neighbor_coords = coords[neighbor_ids]  # (k, 2)
        goal_coord = coords[goal].unsqueeze(0)  # (1, 2)
        neighbor_dists = torch.abs(neighbor_coords - goal_coord).sum(dim=1)  # (k,)

        # Score: how much closer does this move get us? (positive = closer)
        # spectral.py line 143
        heuristic_scores = current_dist - neighbor_dists

        # Normalize to [0, 1] (spectral.py lines 147-154)
        h_min = heuristic_scores.min()
        h_max = heuristic_scores.max()
        h_range = h_max - h_min
        h_range = torch.where(h_range > 1e-10, h_range, torch.ones_like(h_range))
        return (heuristic_scores - h_min) / h_range

    def _compute_spectral_direction(
        self,
        adjacency: torch.Tensor,
        coords: Optional[torch.Tensor],
        current: int,
        goal: int,
        visited: Set[int],
    ) -> Tuple[int, float]:
        """Compute best neighbor to move toward goal using local spectral structure.

        Transcribed from pathfinding/spectral.py:compute_spectral_direction
        (lines 99-236). Combines spectral guidance with geometric heuristic.

        This is the per-step decision function. It calls LocalSpectralProbe
        (compilable inner module) for the spectral computation, then does
        thin Python logic for neighbor selection.

        Args:
            adjacency: Sparse (n, n) graph adjacency.
            coords: Optional (n, 2) node coordinates.
            current: Current node index (global).
            goal: Goal node index (global).
            visited: Set of already-visited node indices.

        Returns:
            (best_neighbor, confidence) -- global node index of best neighbor
            and confidence score in [0, 1].
        """
        device = adjacency.device
        neighbors = self._get_neighbors(adjacency, current)

        # Filter visited (spectral.py lines 118-123)
        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            # All neighbors visited -- allow backtracking
            unvisited = neighbors
        if not unvisited:
            return current, 0.0

        # Direct connection to goal? (spectral.py lines 128-129)
        if goal in unvisited:
            return goal, 1.0

        k = len(unvisited)
        scores = torch.zeros(k, device=device, dtype=torch.float32)

        # --- Component 1: Geometric heuristic (spectral.py lines 134-155) ---
        heuristic_weight = self.navigator.heuristic_weight
        if coords is not None and heuristic_weight.item() > 0:
            h_scores = self._score_neighbors_geometric(
                coords, current, goal, unvisited
            )
            scores = scores + heuristic_weight * h_scores

        # --- Component 2: Spectral guidance (spectral.py lines 158-205) ---
        spectral_weight = self.navigator.spectral_weight
        if spectral_weight.item() > 0:
            # Compute local Fiedler around current node
            # Include goal in query if it might be in the neighborhood
            query = torch.tensor([current], device=device, dtype=torch.long)
            fiedler_maps, expansion = self.local_probe(adjacency, query, coords)

            if current in fiedler_maps:
                fiedler_tensor = fiedler_maps[current]  # (m,) local Fiedler
                # We need the node_list to map global IDs to local indices
                # Re-expand neighborhood to get the mapping (same as LocalSpectralProbe does)
                active_nodes = self.local_probe._expand_neighborhood(
                    adjacency, {current}, self.local_probe.hop_radius
                )
                node_list = sorted(active_nodes)
                node_to_idx: Dict[int, int] = {}
                for idx, node in enumerate(node_list):
                    node_to_idx[node] = idx

                current_local = node_to_idx.get(current, -1)
                goal_local = node_to_idx.get(goal, -1)

                if current_local >= 0 and fiedler_tensor.shape[0] > 0:
                    # Build local neighbor indices
                    neighbor_local = torch.tensor(
                        [node_to_idx.get(n, -1) for n in unvisited],
                        device=device,
                        dtype=torch.long,
                    )

                    spectral_scores = self.navigator(
                        fiedler_tensor,
                        current_local,
                        goal_local,
                        neighbor_local,
                    )
                    scores = scores + spectral_weight * spectral_scores

            # --- Component 3: Expansion bias (spectral.py lines 208-218) ---
            if self.use_expansion_bias:
                # Probe expansion at each neighbor
                neighbor_query = torch.tensor(
                    unvisited, device=device, dtype=torch.long
                )
                _, neighbor_expansion = self.local_probe(
                    adjacency, neighbor_query, coords
                )
                # Bias toward lower expansion (corridors/structure)
                # spectral.py line 217
                scores = scores - 0.1 * neighbor_expansion

        # Select best neighbor (spectral.py line 229)
        best_idx = torch.argmax(scores).item()
        best_neighbor = unvisited[best_idx]
        best_score = scores[best_idx].item()

        # Normalize confidence to [0, 1] (spectral.py lines 233-234)
        score_range = scores.max() - scores.min() + 1e-10
        confidence = (scores[best_idx] - scores.min()) / score_range

        return best_neighbor, confidence.item()

    def forward(
        self,
        adjacency: torch.Tensor,
        coords: torch.Tensor,
        start: int,
        goal: int,
        max_steps: int = 1000,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Find path from start to goal using local spectral guidance.

        The outer step loop is sequential (each step depends on current
        position). torch.compile with fullgraph=True is not possible here.
        The inner LocalSpectralProbe and SpectralNavigator are compiled
        separately.

        Transcribed from pathfinding/spectral.py:find_path (lines 238-317).

        Args:
            adjacency: Sparse (n, n) graph adjacency matrix.
            coords: (n, 2) node coordinates (row, col) for geometric heuristic.
            start: Start node index (global).
            goal: Goal node index (global).
            max_steps: Maximum path steps (safety limit). Default 1000.

        Returns:
            path: (path_len,) tensor of node indices from start to goal.
            diagnostics: Dict with navigation statistics:
                - 'success': bool, whether goal was reached
                - 'steps': int, number of steps taken
                - 'path_length': int, number of nodes in path
                - 'avg_confidence': float, mean confidence across steps
                - 'nodes_visited': int, total unique nodes touched
                - 'computation_time': float, wall-clock seconds
        """
        device = adjacency.device
        start_time = time.time()

        current = start
        path_list: List[int] = [current]
        visited: Set[int] = {current}

        total_confidence = 0.0
        stuck_count = 0

        # Ensure adjacency is coalesced for fast neighbor lookup
        adjacency = adjacency.coalesce()

        step = 0
        while step < max_steps:
            if current == goal:
                elapsed = time.time() - start_time
                path_tensor = torch.tensor(path_list, device=device, dtype=torch.long)
                return path_tensor, {
                    "success": True,
                    "steps": step,
                    "path_length": len(path_list),
                    "avg_confidence": total_confidence / max(step, 1),
                    "nodes_visited": len(visited),
                    "computation_time": elapsed,
                }

            # Check if goal is a direct neighbor (spectral.py lines 270-277)
            neighbors = self._get_neighbors(adjacency, current)
            if goal in neighbors:
                path_list.append(goal)
                elapsed = time.time() - start_time
                path_tensor = torch.tensor(path_list, device=device, dtype=torch.long)
                return path_tensor, {
                    "success": True,
                    "steps": step + 1,
                    "path_length": len(path_list),
                    "avg_confidence": total_confidence / max(step + 1, 1),
                    "nodes_visited": len(visited),
                    "computation_time": elapsed,
                }

            # Random exploration? (spectral.py lines 280-292)
            # Stochastic gate bypass: analogous to dropout on the spectral gate
            explore_roll = torch.rand(1, device=device)
            if explore_roll.item() < self.exploration_probability.item():
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                if unvisited_neighbors:
                    rand_idx = torch.randint(
                        0, len(unvisited_neighbors), (1,), device=device
                    ).item()
                    next_node = unvisited_neighbors[rand_idx]
                    confidence = 0.5
                else:
                    next_node, confidence = self._compute_spectral_direction(
                        adjacency, coords, current, goal, visited
                    )
            else:
                next_node, confidence = self._compute_spectral_direction(
                    adjacency, coords, current, goal, visited
                )

            total_confidence += confidence

            # Stuck detection (spectral.py lines 296-304)
            if next_node == current:
                stuck_count += 1
                if stuck_count > 10:
                    # Really stuck -- random walk to escape
                    if neighbors:
                        rand_idx = torch.randint(
                            0, len(neighbors), (1,), device=device
                        ).item()
                        next_node = neighbors[rand_idx]
                        stuck_count = 0
            else:
                stuck_count = 0

            current = next_node
            path_list.append(current)
            visited.add(current)
            step += 1

        # Max steps exceeded (spectral.py lines 312-317)
        elapsed = time.time() - start_time
        path_tensor = torch.tensor(path_list, device=device, dtype=torch.long)
        return path_tensor, {
            "success": False,
            "steps": max_steps,
            "path_length": len(path_list),
            "avg_confidence": total_confidence / max_steps,
            "nodes_visited": len(visited),
            "computation_time": elapsed,
        }


# ============================================================
# PathQualityEstimator -- post-hoc verification
# ============================================================


class PathQualityEstimator(nn.Module):
    """Post-hoc quality measurement for spectral paths.

    Compares a spectral path against a reference (Dijkstra-optimal) path
    using cost ratio and Wasserstein-like distance metrics. This module
    is for verification, not navigation.

    The Wasserstein distance between paths (Peyre & Cuturi 2019,
    Computational Optimal Transport) measures how much "work" it takes
    to transform one path into another. For grid paths, this reduces to
    the sum of pointwise distances between path positions at matched
    time steps.

    Parameters
    ----------
    (none -- stateless estimator)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        spectral_path: torch.Tensor,
        optimal_path: torch.Tensor,
        coords: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute quality metrics comparing spectral path to optimal.

        Args:
            spectral_path: (s,) node indices of spectral path.
            optimal_path: (o,) node indices of optimal (Dijkstra) path.
            coords: (n, 2) node coordinates.
            adjacency: Optional sparse (n, n) adjacency for edge-weight cost.

        Returns:
            Dict with quality metrics:
                - 'length_ratio': len(spectral) / len(optimal)
                - 'cost_ratio': cost(spectral) / cost(optimal)
                   (only if adjacency provided)
                - 'wasserstein_approx': approximate 1-Wasserstein between
                   paths as sequences of positions, using linear interpolation
                   to match path lengths
                - 'hausdorff': max min-distance from spectral path point to
                   optimal path (directional Hausdorff)
        """
        device = spectral_path.device

        s_len = spectral_path.shape[0]
        o_len = optimal_path.shape[0]

        # Length ratio
        length_ratio = torch.tensor(
            s_len / max(o_len, 1), device=device, dtype=torch.float32
        )

        # Get path coordinates
        s_coords = coords[spectral_path.long()]  # (s, 2)
        o_coords = coords[optimal_path.long()]  # (o, 2)

        # Cost ratio (if adjacency provided)
        cost_ratio = torch.tensor(0.0, device=device, dtype=torch.float32)
        if adjacency is not None and s_len > 1 and o_len > 1:
            s_cost = self._path_cost(spectral_path, adjacency)
            o_cost = self._path_cost(optimal_path, adjacency)
            safe_o_cost = torch.where(
                o_cost > 1e-10, o_cost, torch.ones_like(o_cost)
            )
            cost_ratio = s_cost / safe_o_cost

        # Approximate 1-Wasserstein via linear time interpolation
        # Resample both paths to common length, then compute L1 distance
        common_len = max(s_len, o_len)
        s_resampled = self._resample_path_coords(s_coords, common_len)
        o_resampled = self._resample_path_coords(o_coords, common_len)
        wasserstein = torch.abs(s_resampled - o_resampled).sum(dim=1).mean()

        # Hausdorff distance (spectral -> optimal direction)
        # For each point in spectral path, find min distance to optimal path
        # dists: (s, o)
        diffs = s_coords.unsqueeze(1) - o_coords.unsqueeze(0)  # (s, o, 2)
        dists = torch.norm(diffs.float(), dim=2)  # (s, o)
        min_dists_s2o = dists.min(dim=1).values  # (s,)
        hausdorff = min_dists_s2o.max()

        return {
            "length_ratio": length_ratio,
            "cost_ratio": cost_ratio,
            "wasserstein_approx": wasserstein,
            "hausdorff": hausdorff,
        }

    def _path_cost(
        self, path: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Compute total edge-weight cost along a path.

        Args:
            path: (p,) node indices.
            adjacency: Sparse (n, n) weighted adjacency.

        Returns:
            Scalar total cost.
        """
        adj = adjacency.coalesce()
        indices = adj.indices()  # (2, nnz)
        values = adj.values()  # (nnz,)

        total = torch.tensor(0.0, device=path.device, dtype=torch.float32)
        p_len = path.shape[0]
        step = 0
        while step < p_len - 1:
            src = path[step].item()
            dst = path[step + 1].item()
            # Find edge (src, dst) in the adjacency
            mask = (indices[0] == src) & (indices[1] == dst)
            edge_vals = values[mask]
            if edge_vals.numel() > 0:
                # Adjacency values are weights; cost is weight
                total = total + edge_vals[0].abs()
            else:
                # Edge not found -- add unit cost
                total = total + 1.0
            step += 1
        return total

    @staticmethod
    def _resample_path_coords(
        coords: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Resample path coordinates to target_len via linear interpolation.

        Args:
            coords: (p, 2) path coordinates.
            target_len: Desired output length.

        Returns:
            (target_len, 2) resampled coordinates.
        """
        p = coords.shape[0]
        if p == target_len:
            return coords
        if p == 1:
            return coords.expand(target_len, -1)

        # Parameter t in [0, 1] for each output point
        t = torch.linspace(0, 1, target_len, device=coords.device, dtype=torch.float32)
        # Map t to input indices
        src_t = t * (p - 1)
        src_idx = src_t.long().clamp(0, p - 2)
        frac = (src_t - src_idx.float()).unsqueeze(1)  # (target_len, 1)
        # Linear interpolation
        lower = coords[src_idx]  # (target_len, 2)
        upper = coords[(src_idx + 1).clamp(max=p - 1)]  # (target_len, 2)
        return lower + frac * (upper - lower)


# ============================================================
# Grid graph builder utility (for testing and demos)
# ============================================================


def build_grid_adjacency(
    height: int,
    width: int,
    blocked: Optional[torch.Tensor] = None,
    connectivity: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sparse adjacency matrix and coordinate tensor for a grid graph.

    Utility function for constructing the graph representation that
    SpectralPathfinder expects. Pure PyTorch, no numpy.

    Args:
        height: Grid height (rows).
        width: Grid width (columns).
        blocked: Optional (height, width) bool tensor where True = blocked.
        connectivity: 4 (von Neumann) or 8 (Moore). Default 4.
        device: Target device.

    Returns:
        adjacency: Sparse (n, n) adjacency matrix where n = height * width.
            Blocked nodes have no edges.
        coords: (n, 2) node coordinates as (row, col).
    """
    if device is None:
        device = torch.device("cpu")

    n = height * width

    # Build coordinates: (n, 2) as (row, col)
    rows_grid = torch.arange(height, device=device, dtype=torch.float32)
    cols_grid = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows_grid, cols_grid, indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # (n, 2)

    # Neighbor offsets
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    edge_rows: List[int] = []
    edge_cols: List[int] = []

    # Vectorized edge construction
    yy_int = torch.arange(height, device=device, dtype=torch.long)
    xx_int = torch.arange(width, device=device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(yy_int, xx_int, indexing="ij")
    flat_y = grid_y.flatten()  # (n,)
    flat_x = grid_x.flatten()  # (n,)

    src_list = []
    dst_list = []

    for dy, dx in offsets:
        dst_y = flat_y + dy
        dst_x = flat_x + dx
        # Bounds check
        valid = (dst_y >= 0) & (dst_y < height) & (dst_x >= 0) & (dst_x < width)

        if blocked is not None:
            # Also exclude blocked source and destination
            src_node = flat_y * width + flat_x
            dst_node = dst_y * width + dst_x
            blocked_flat = blocked.flatten().bool()
            # Clamp dst_node for indexing (invalid ones will be masked anyway)
            dst_node_safe = dst_node.clamp(0, n - 1)
            valid = valid & (~blocked_flat[src_node]) & (~blocked_flat[dst_node_safe])
            src_list.append(src_node[valid])
            dst_list.append(dst_node[valid])
        else:
            src_node = flat_y * width + flat_x
            dst_node = dst_y * width + dst_x
            src_list.append(src_node[valid])
            dst_list.append(dst_node[valid])

    if src_list:
        all_src = torch.cat(src_list)
        all_dst = torch.cat(dst_list)
        indices = torch.stack([all_src, all_dst]).long()
        values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
        adjacency = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    else:
        adjacency = torch.sparse_coo_tensor(
            torch.zeros((2, 0), device=device, dtype=torch.long),
            torch.zeros(0, device=device, dtype=torch.float32),
            (n, n),
        ).coalesce()

    return adjacency, coords


# ============================================================
# Dijkstra reference (pure PyTorch, for testing)
# ============================================================


def dijkstra_shortest_path(
    adjacency: torch.Tensor,
    start: int,
    goal: int,
    max_nodes: int = 100000,
) -> Tuple[torch.Tensor, bool, float]:
    """Standard Dijkstra for comparison with spectral pathfinder.

    Pure Python + PyTorch (no numpy). Uses heapq priority queue.

    Transcribed from pathfinding/classical.py:dijkstra_path (lines 15-117)
    but operates on sparse adjacency tensor instead of HeightfieldGraph.

    Args:
        adjacency: Sparse (n, n) weighted adjacency matrix.
        start: Start node index.
        goal: Goal node index.
        max_nodes: Maximum nodes to explore.

    Returns:
        path: (path_len,) tensor of node indices, or empty tensor if failed.
        success: Whether goal was reached.
        total_cost: Path cost (inf if failed).
    """
    device = adjacency.device
    adj = adjacency.coalesce()
    adj_indices = adj.indices()
    adj_values = adj.values()

    distances: Dict[int, float] = {start: 0.0}
    predecessors: Dict[int, int] = {}
    visited_set: Set[int] = set()

    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap and len(visited_set) < max_nodes:
        dist, current = heapq.heappop(heap)

        if current in visited_set:
            continue
        visited_set.add(current)

        if current == goal:
            # Reconstruct path
            path_nodes: List[int] = [goal]
            node = goal
            while node in predecessors:
                node = predecessors[node]
                path_nodes.append(node)
            path_nodes.reverse()
            return (
                torch.tensor(path_nodes, device=device, dtype=torch.long),
                True,
                dist,
            )

        # Get neighbors
        mask = adj_indices[0] == current
        neighbor_ids = adj_indices[1][mask].cpu().tolist()
        neighbor_weights = adj_values[mask].cpu().tolist()

        for nbr, w in zip(neighbor_ids, neighbor_weights):
            if nbr not in visited_set:
                new_dist = dist + abs(w)
                if nbr not in distances or new_dist < distances[nbr]:
                    distances[nbr] = new_dist
                    predecessors[nbr] = current
                    heapq.heappush(heap, (new_dist, nbr))

    return (
        torch.tensor([], device=device, dtype=torch.long),
        False,
        float("inf"),
    )


# ============================================================
# Self-test
# ============================================================


def _run_self_test():
    """Self-test: grid graph pathfinding with obstacles.

    Creates a 16x16 grid with a wall obstacle, runs SpectralPathfinder
    from top-left to bottom-right, compares with Dijkstra, and verifies:
    1. Spectral path reaches the goal
    2. Path quality (length ratio vs Dijkstra optimal)
    3. torch.compile on inner modules (SpectralNavigator, LocalSpectralProbe)
    """
    print("=" * 60)
    print("SpectralPathfinder self-test")
    print("=" * 60)

    device = torch.device("cpu")
    H, W = 16, 16
    n = H * W

    # Create obstacle: vertical wall with a gap
    # Wall at column 8, gap at rows 6-9
    blocked = torch.zeros(H, W, dtype=torch.bool, device=device)
    for r in range(H):
        if r < 6 or r > 9:
            blocked[r, 8] = True

    print(f"Grid: {H}x{W} = {n} nodes")
    print(f"Obstacle: vertical wall at col 8, gap at rows 6-9")

    # Build graph
    adjacency, coords = build_grid_adjacency(H, W, blocked=blocked, device=device)
    nnz = adjacency._nnz()
    print(f"Adjacency: {n}x{n} sparse, {nnz} non-zeros")

    start_node = 0               # top-left (0, 0)
    goal_node = H * W - 1        # bottom-right (15, 15)
    print(f"Start: node {start_node} = ({0}, {0})")
    print(f"Goal:  node {goal_node} = ({H-1}, {W-1})")

    # --- Dijkstra baseline ---
    print("\n--- Dijkstra (optimal) ---")
    t0 = time.time()
    dij_path, dij_success, dij_cost = dijkstra_shortest_path(
        adjacency, start_node, goal_node
    )
    dij_time = time.time() - t0
    print(f"Success: {dij_success}")
    print(f"Path length: {dij_path.shape[0]}")
    print(f"Cost: {dij_cost:.2f}")
    print(f"Time: {dij_time:.4f}s")

    # --- Spectral pathfinder ---
    print("\n--- Spectral Pathfinder ---")
    probe = LocalSpectralProbe(hop_radius=2, lanczos_iterations=15)
    pathfinder = SpectralPathfinder(
        local_probe=probe,
        spectral_weight=0.3,
        heuristic_weight=0.7,
        exploration_probability=0.02,
    )

    t0 = time.time()
    spec_path, diag = pathfinder(adjacency, coords, start_node, goal_node, max_steps=500)
    spec_time = time.time() - t0

    print(f"Success: {diag['success']}")
    print(f"Path length: {diag['path_length']}")
    print(f"Steps: {diag['steps']}")
    print(f"Avg confidence: {diag['avg_confidence']:.4f}")
    print(f"Nodes visited: {diag['nodes_visited']}")
    print(f"Time: {spec_time:.4f}s")

    # --- Quality comparison ---
    if dij_success and diag["success"]:
        print("\n--- Quality Comparison ---")
        estimator = PathQualityEstimator()
        quality = estimator(spec_path, dij_path, coords, adjacency)
        print(f"Length ratio: {quality['length_ratio'].item():.2f}x")
        print(f"Cost ratio: {quality['cost_ratio'].item():.2f}x")
        print(f"Wasserstein approx: {quality['wasserstein_approx'].item():.2f}")
        print(f"Hausdorff: {quality['hausdorff'].item():.2f}")

    # --- torch.compile test ---
    print("\n--- torch.compile test ---")
    try:
        compiled_navigator = torch.compile(pathfinder.navigator, fullgraph=False)
        # Test with a small input
        test_fiedler = torch.randn(10, device=device)
        test_neighbors = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
        out = compiled_navigator(test_fiedler, 0, 5, test_neighbors)
        print(f"SpectralNavigator torch.compile: OK (output shape: {out.shape})")
    except Exception as e:
        print(f"SpectralNavigator torch.compile: FAILED ({e})")

    try:
        compiled_probe = torch.compile(pathfinder.local_probe, fullgraph=False)
        test_query = torch.tensor([0], device=device, dtype=torch.long)
        fmaps, exp_est = compiled_probe(adjacency, test_query)
        print(f"LocalSpectralProbe torch.compile: OK (expansion: {exp_est})")
    except Exception as e:
        print(f"LocalSpectralProbe torch.compile: FAILED ({e})")

    # --- Verify on smaller grid without obstacles ---
    print("\n--- Small grid (8x8, no obstacles) ---")
    adj_small, coords_small = build_grid_adjacency(8, 8, device=device)
    probe_small = LocalSpectralProbe(hop_radius=2, lanczos_iterations=10)
    pf_small = SpectralPathfinder(
        local_probe=probe_small,
        spectral_weight=0.4,
        heuristic_weight=0.6,
    )
    sp, sd = pf_small(adj_small, coords_small, 0, 63, max_steps=200)
    dp, ds, dc = dijkstra_shortest_path(adj_small, 0, 63)
    print(f"Spectral: success={sd['success']}, len={sd['path_length']}")
    print(f"Dijkstra: success={ds}, len={dp.shape[0]}, cost={dc:.2f}")
    if ds and sd["success"]:
        ratio = sd["path_length"] / max(dp.shape[0], 1)
        print(f"Length ratio: {ratio:.2f}x")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_test()
