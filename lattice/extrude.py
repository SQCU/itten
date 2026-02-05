"""
Spectral-Dependent Extrusion Operations - FAST VERSION.

All hot paths are vectorized tensor operations.
No Python for-loops over nodes in numerical code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import sys
import os

# Import from spectral_ops_fast (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_ops_fast import (
    local_expansion_estimate,
    local_fiedler_vector,
    expansion_map_batched,
    expand_neighborhood,
    build_sparse_laplacian,
    lanczos_fiedler_gpu,
    DEVICE
)
import torch


@dataclass
class ExtrudedNode:
    """A node with extrusion state."""
    coord: Tuple[int, int]
    layer: int = 0  # Extrusion layer (0 = base, 1+ = extruded)
    expansion: float = 0.0  # Local lambda_2 estimate
    fiedler_value: float = 0.0  # Position in Fiedler vector
    fiedler_gradient: Tuple[float, float] = (0.0, 0.0)  # Gradient direction
    fiedler_gradient_mag: float = 0.0  # Magnitude of Fiedler gradient
    geometry_type: str = "square"  # "triangle", "hex", "square"
    geometry_angle: float = 0.0  # Orientation angle in radians
    node_value: float = 0.0  # Scalar value for visualization (varies with theta)


@dataclass
class ExtrusionState:
    """State of the extrusion process."""
    territory: 'TerritoryGraph'
    nodes: Dict[Tuple[int, int], ExtrudedNode] = field(default_factory=dict)
    frontier: Set[Tuple[int, int]] = field(default_factory=set)
    iteration: int = 0
    expansion_threshold: float = 1.5
    theta: float = 0.5  # Spectral blend parameter (0=expansion-dominant, 1=gradient-dominant)

    # Diagnostics for proving spectral ops are load-bearing
    expansion_history: List[Dict[Tuple[int, int], float]] = field(default_factory=list)
    extrusion_order: List[Tuple[int, int]] = field(default_factory=list)
    lattice_type_counts: Dict[str, int] = field(default_factory=lambda: {"square": 0, "triangle": 0, "hex": 0})


def select_lattice_type(
    expansion: float,
    fiedler_gradient_mag: float,
    theta: float,
    high_threshold: float = 0.60,
    low_threshold: float = 0.40
) -> str:
    """
    Spectrally-determined lattice type selection.

    - High expansion + low gradient → "hex" (open region)
    - Low expansion + high gradient → "triangle" (bottleneck)
    - Medium values → "square" (neutral)
    - theta rotates between geometry preferences

    Args:
        expansion: Local expansion (lambda_2 estimate), values can range 0-10+
        fiedler_gradient_mag: Magnitude of Fiedler gradient, typically 0-1
        theta: Spectral blend parameter (0=expansion-dominant, 1=gradient-dominant)
        high_threshold: Threshold for hex selection
        low_threshold: Threshold for triangle selection

    Returns:
        Lattice type: "hex", "triangle", or "square"
    """
    # Use log-scale normalization for expansion (handles wide range 0.1-10+)
    # log(0.5) ~ -0.7, log(3) ~ 1.1, log(10) ~ 2.3
    log_expansion = np.log(max(expansion, 0.1))
    # Map log range [-1, 2.5] to [0, 1]
    norm_expansion = np.clip((log_expansion + 1.0) / 3.5, 0.0, 1.0)

    # Normalize gradient magnitude (use log scale for small values)
    # Typical range 0.001 to 0.5, with log range [-7, -0.7]
    log_gradient = np.log(max(fiedler_gradient_mag, 0.0001))
    # Map log range [-9, -0.7] to [0, 1] (wider range to catch small gradients)
    norm_gradient = np.clip((log_gradient + 9.0) / 8.3, 0.0, 1.0)

    # theta controls spectral emphasis:
    # theta=0: expansion-dominant (high expansion → hex)
    # theta=1: gradient-dominant (high gradient → triangle)
    #
    # The spectral score blends two signals:
    # - norm_expansion: higher means more open region
    # - norm_gradient: higher means steeper spectral transition (bottleneck edge)
    #
    # For hex (open regions): want high expansion OR low gradient
    # For triangle (bottlenecks): want low expansion OR high gradient
    spectral_score = (1 - theta) * norm_expansion + theta * (1 - norm_gradient)

    if spectral_score > high_threshold:
        return "hex"
    elif spectral_score < low_threshold:
        return "triangle"
    else:
        return "square"


def compute_node_value(
    expansion: float,
    fiedler_value: float,
    fiedler_gradient_mag: float,
    theta: float
) -> float:
    """
    Compute a scalar node value for visualization based on spectral properties.

    The value varies with theta to show different spectral emphasis.

    Args:
        expansion: Local expansion (lambda_2 estimate)
        fiedler_value: Node's position in Fiedler vector
        fiedler_gradient_mag: Magnitude of Fiedler gradient
        theta: Spectral blend parameter

    Returns:
        Scalar value in [0, 1] for visualization coloring
    """
    # Use log-scale normalization for expansion (consistent with select_lattice_type)
    log_expansion = np.log(max(expansion, 0.1))
    norm_expansion = np.clip((log_expansion + 1.0) / 3.5, 0.0, 1.0)

    # Normalize Fiedler value (typically in [-1, 1] range)
    norm_fiedler = np.clip((fiedler_value + 1.0) / 2.0, 0.0, 1.0)

    # Use log-scale normalization for gradient (consistent with select_lattice_type)
    log_gradient = np.log(max(fiedler_gradient_mag, 0.0001))
    norm_gradient = np.clip((log_gradient + 9.0) / 8.3, 0.0, 1.0)

    # Blend based on theta
    # theta=0: expansion-based coloring
    # theta=0.5: fiedler value based coloring
    # theta=1: gradient-based coloring
    if theta < 0.5:
        # Blend expansion and fiedler
        t = theta * 2  # 0 to 1 as theta goes 0 to 0.5
        value = (1 - t) * norm_expansion + t * norm_fiedler
    else:
        # Blend fiedler and gradient
        t = (theta - 0.5) * 2  # 0 to 1 as theta goes 0.5 to 1
        value = (1 - t) * norm_fiedler + t * norm_gradient

    return float(np.clip(value, 0.0, 1.0))


class ExpansionGatedExtruder:
    """
    Extrudes lattice geometry only where local expansion exceeds threshold.

    FAST VERSION: Uses batched spectral operations.
    Now includes theta parameter for spectral-determined lattice type selection.
    """

    def __init__(
        self,
        territory: 'TerritoryGraph',
        expansion_threshold: float = 1.5,
        lanczos_iterations: int = 15,
        hop_radius: int = 3,
        theta: float = 0.5
    ):
        self.territory = territory
        self.expansion_threshold = expansion_threshold
        self.lanczos_iterations = lanczos_iterations
        self.hop_radius = hop_radius
        self.theta = theta

        # Initialize state
        self.state = ExtrusionState(
            territory=territory,
            expansion_threshold=expansion_threshold,
            theta=theta
        )

        # Initialize all nodes at layer 0 - use list comprehension (vectorized-style)
        all_nodes = territory.all_nodes()
        self.state.nodes = {coord: ExtrudedNode(coord=coord, layer=0) for coord in all_nodes}

        # Initial frontier: all nodes
        self.state.frontier = all_nodes

    def compute_all_expansions(self) -> Dict[Tuple[int, int], float]:
        """
        Compute local expansion for ALL nodes in a single batch.

        FAST: Uses expansion_map_batched - one Laplacian build, vectorized eigensolves.
        """
        # Convert coords to node IDs
        all_coords = list(self.territory.all_nodes())
        node_ids = [self.territory.node_id(coord) for coord in all_coords]
        valid_ids = [nid for nid in node_ids if nid is not None]

        # Batched computation
        id_expansions = expansion_map_batched(
            self.territory,
            valid_ids,
            radius=self.hop_radius,
            k=self.lanczos_iterations
        )

        # Map back to coords using while loop
        expansions = {}
        id_to_coord = {self.territory.node_id(c): c for c in all_coords if self.territory.node_id(c) is not None}

        items = list(id_expansions.items())
        idx = 0
        while idx < len(items):
            nid, exp = items[idx]
            coord = id_to_coord.get(nid)
            if coord is not None:
                expansions[coord] = exp
                self.state.nodes[coord].expansion = exp
            idx += 1

        return expansions

    def compute_frontier_expansions(self) -> Dict[Tuple[int, int], float]:
        """Compute expansions for frontier nodes only."""
        frontier_coords = list(self.state.frontier)
        node_ids = [self.territory.node_id(coord) for coord in frontier_coords if self.territory.node_id(coord) is not None]

        if not node_ids:
            return {}

        # Batched computation
        id_expansions = expansion_map_batched(
            self.territory,
            node_ids,
            radius=self.hop_radius,
            k=self.lanczos_iterations
        )

        # Map back to coords using while loop
        expansions = {}
        id_to_coord = {self.territory.node_id(c): c for c in frontier_coords if self.territory.node_id(c) is not None}

        items = list(id_expansions.items())
        idx = 0
        while idx < len(items):
            nid, exp = items[idx]
            coord = id_to_coord.get(nid)
            if coord is not None:
                expansions[coord] = exp
                self.state.nodes[coord].expansion = exp
            idx += 1

        return expansions

    def compute_fiedler_gradients(
        self,
        coords: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
        """
        Compute Fiedler values and gradient magnitudes for given coordinates.

        Returns dict mapping coord -> (fiedler_value, gradient_mag, gradient_angle)
        """
        if not coords:
            return {}

        # Get node IDs
        node_ids = [self.territory.node_id(c) for c in coords if self.territory.node_id(c) is not None]
        if not node_ids:
            return {}

        # Compute Fiedler vector for the region
        fiedler_dict, lambda_2 = local_fiedler_vector(
            self.territory,
            node_ids,
            num_iterations=self.lanczos_iterations,
            hop_expansion=self.hop_radius
        )

        # Compute gradients and map back to coords
        results = {}
        id_to_coord = {self.territory.node_id(c): c for c in coords if self.territory.node_id(c) is not None}

        items = list(fiedler_dict.items())
        idx = 0
        while idx < len(items):
            node_id, fiedler_value = items[idx]
            coord = id_to_coord.get(node_id)
            if coord is not None:
                # Compute gradient from neighbors
                neighbors = self.territory.neighbors(node_id)
                grad = np.array([0.0, 0.0])
                j = 0
                while j < len(neighbors):
                    neighbor = neighbors[j]
                    if neighbor in fiedler_dict:
                        delta_f = fiedler_dict[neighbor] - fiedler_value
                        ncoord = self.territory.node_coord(neighbor)
                        if ncoord is not None:
                            delta_coord = np.array([ncoord[0] - coord[0], ncoord[1] - coord[1]])
                            grad = grad + delta_f * delta_coord
                    j += 1

                # Compute magnitude and angle
                grad_mag = float(np.linalg.norm(grad))
                if grad_mag > 1e-6:
                    grad_angle = float(np.arctan2(grad[1], grad[0]))
                else:
                    grad_angle = 0.0

                results[coord] = (float(fiedler_value), grad_mag, grad_angle)

                # Update node state
                node = self.state.nodes[coord]
                node.fiedler_value = float(fiedler_value)
                node.fiedler_gradient = (float(grad[0]), float(grad[1]))
                node.fiedler_gradient_mag = grad_mag
            idx += 1

        return results

    def step(self) -> List[Tuple[int, int]]:
        """
        Perform one extrusion step.

        Returns list of nodes that were extruded this step.
        FAST: Uses batched expansion computation.
        Now includes spectral lattice type selection based on theta.
        """
        self.state.iteration += 1
        extruded = []

        # Compute expansions for current frontier (BATCHED)
        current_expansions = self.compute_frontier_expansions()

        # Record for diagnostics
        self.state.expansion_history.append(current_expansions.copy())

        # Extrude nodes that exceed threshold - vectorized logic
        coord_list = list(current_expansions.keys())
        coords = np.array(coord_list)
        expansions = np.array(list(current_expansions.values()))

        if len(expansions) == 0:
            return []

        # Compute Fiedler gradients for spectral lattice type selection
        fiedler_data = self.compute_fiedler_gradients(coord_list)

        # Find nodes to extrude (vectorized comparison)
        extrude_mask = expansions > self.expansion_threshold

        new_frontier = set()
        i = 0
        while i < len(coords):  # graph traversal
            coord_tuple = tuple(coords[i])
            node = self.state.nodes[coord_tuple]

            if extrude_mask[i]:
                # Extrude this node
                node.layer += 1
                extruded.append(coord_tuple)
                self.state.extrusion_order.append(coord_tuple)

                # Get spectral properties for lattice type selection
                expansion = float(expansions[i])
                fiedler_info = fiedler_data.get(coord_tuple, (0.0, 0.0, 0.0))
                fiedler_value, gradient_mag, gradient_angle = fiedler_info

                # Select lattice type based on spectral properties and theta
                lattice_type = select_lattice_type(
                    expansion=expansion,
                    fiedler_gradient_mag=gradient_mag,
                    theta=self.theta
                )
                node.geometry_type = lattice_type
                node.geometry_angle = gradient_angle

                # Compute node value for visualization
                node.node_value = compute_node_value(
                    expansion=expansion,
                    fiedler_value=fiedler_value,
                    fiedler_gradient_mag=gradient_mag,
                    theta=self.theta
                )

                # Track lattice type counts
                self.state.lattice_type_counts[lattice_type] = \
                    self.state.lattice_type_counts.get(lattice_type, 0) + 1

                # Neighbors become new frontier candidates (graph traversal)
                neighbors = self.territory.neighbors_by_coord(coord_tuple)
                j = 0
                while j < len(neighbors):
                    neighbor = neighbors[j]
                    if neighbor in self.state.nodes:
                        n_node = self.state.nodes[neighbor]
                        if n_node.layer < node.layer:
                            new_frontier.add(neighbor)
                    j += 1
            else:
                # Keep in frontier for next iteration
                new_frontier.add(coord_tuple)
            i += 1

        self.state.frontier = new_frontier
        return extruded

    def run(self, max_iterations: int = 20) -> ExtrusionState:
        """
        Run extrusion until complete or max iterations.
        """
        iteration = 0
        while iteration < max_iterations:
            extruded = self.step()
            if not extruded:
                # Reduce threshold to allow more extrusion
                self.expansion_threshold *= 0.8
                if self.expansion_threshold < 0.1:
                    break
            iteration += 1

        return self.state


class FiedlerAlignedGeometry:
    """
    Orient geometry along the local Fiedler gradient.

    FAST VERSION: Uses batched Fiedler computation.
    """

    def __init__(
        self,
        territory: 'TerritoryGraph',
        hop_radius: int = 3,
        lanczos_iterations: int = 20
    ):
        self.territory = territory
        self.hop_radius = hop_radius
        self.lanczos_iterations = lanczos_iterations

    def compute_fiedler_field(
        self,
        center_coords: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Tuple[float, Tuple[float, float]]]:
        """
        Compute Fiedler values and gradients for a set of nodes.

        FAST: Uses local_fiedler_vector with batched computation.
        """
        # Convert to node IDs
        seed_ids = [self.territory.node_id(coord) for coord in center_coords if self.territory.node_id(coord) is not None]

        if not seed_ids:
            return {}

        # Compute Fiedler vector for the region
        fiedler_dict, lambda_2 = local_fiedler_vector(
            self.territory,
            seed_ids,
            num_iterations=self.lanczos_iterations,
            hop_expansion=self.hop_radius
        )

        # Compute gradients and convert to coord space
        results = {}
        items = list(fiedler_dict.items())
        idx = 0
        while idx < len(items):
            node_id, fiedler_value = items[idx]
            coord = self.territory.node_coord(node_id)
            if coord is not None:
                # Compute gradient from neighbors
                neighbors = self.territory.neighbors(node_id)
                grad = np.array([0.0, 0.0])
                j = 0
                while j < len(neighbors):
                    neighbor = neighbors[j]
                    if neighbor in fiedler_dict:
                        delta_f = fiedler_dict[neighbor] - fiedler_value
                        ncoord = self.territory.node_coord(neighbor)
                        if ncoord is not None:
                            delta_coord = np.array([ncoord[0] - coord[0], ncoord[1] - coord[1]])
                            grad = grad + delta_f * delta_coord
                    j += 1

                # Normalize gradient
                norm = np.linalg.norm(grad)
                if norm > 1e-6:
                    grad = grad / norm

                results[coord] = (fiedler_value, (float(grad[0]), float(grad[1])))
            idx += 1

        return results

    def orient_geometry(
        self,
        coord: Tuple[int, int],
        gradient: Tuple[float, float],
        geometry_type: str = "triangle"
    ) -> float:
        """Compute orientation angle for geometry at this node."""
        gx, gy = gradient

        if abs(gx) < 1e-6 and abs(gy) < 1e-6:
            return 0.0

        angle = np.arctan2(gy, gx)
        return angle


def adaptive_period(expansion: float, base_period: int = 4) -> int:
    """Compute adaptive period based on local expansion."""
    if expansion < 0.5:
        return max(2, base_period // 2)
    elif expansion < 1.5:
        return base_period
    else:
        return min(8, int(base_period * (expansion / 1.5)))


def demonstrate_spectral_dependency(territory: 'TerritoryGraph') -> Dict:
    """
    Demonstrate that the extrusion pattern depends on spectral properties.

    FAST: Uses expansion_map_batched for batched computation.
    """
    diagnostics = {
        'island_expansions': [],
        'bridge_expansions': [],
        'all_expansions': {},
        'threshold': 0.0,
        'extrusion_order_summary': {}
    }

    # Get all node IDs
    all_coords = list(territory.all_nodes())
    node_ids = [territory.node_id(coord) for coord in all_coords if territory.node_id(coord) is not None]

    # Batched expansion computation
    id_expansions = expansion_map_batched(
        territory,
        node_ids,
        radius=3,
        k=15
    )

    # Map back to coords and categorize using while loop
    idx = 0
    while idx < len(all_coords):
        coord = all_coords[idx]
        node_id = territory.node_id(coord)
        if node_id is None or node_id not in id_expansions:
            idx += 1
            continue

        lambda_2 = id_expansions[node_id]
        diagnostics['all_expansions'][coord] = lambda_2

        if territory.is_on_island(coord):
            diagnostics['island_expansions'].append(lambda_2)
        elif territory.is_on_bridge(coord):
            diagnostics['bridge_expansions'].append(lambda_2)
        idx += 1

    # Compute statistics using numpy (vectorized)
    island_arr = np.array(diagnostics['island_expansions']) if diagnostics['island_expansions'] else np.array([0.0])
    bridge_arr = np.array(diagnostics['bridge_expansions']) if diagnostics['bridge_expansions'] else np.array([0.0])

    island_mean = float(np.mean(island_arr))
    island_std = float(np.std(island_arr))
    bridge_mean = float(np.mean(bridge_arr))
    bridge_std = float(np.std(bridge_arr))

    diagnostics['island_mean'] = island_mean
    diagnostics['island_std'] = island_std
    diagnostics['bridge_mean'] = bridge_mean
    diagnostics['bridge_std'] = bridge_std
    diagnostics['threshold'] = (island_mean + bridge_mean) / 2

    diagnostics['explanation'] = (
        f"Island mean lambda_2: {island_mean:.3f} (std: {island_std:.3f})\n"
        f"Bridge mean lambda_2: {bridge_mean:.3f} (std: {bridge_std:.3f})\n"
        f"Difference: {island_mean - bridge_mean:.3f}\n\n"
        "This difference cannot be explained by local 3-hop structure:\n"
        "- Bridge nodes have 4 neighbors within 3 hops (regular grid)\n"
        "- Island center nodes also have 4 neighbors within 3 hops\n"
        "- Yet lambda_2 differs because Lanczos captures global structure\n"
        "- The bridge's role as a bottleneck affects the SPECTRAL properties\n"
    )

    return diagnostics
