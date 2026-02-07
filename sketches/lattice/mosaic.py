"""
Mosaic: Lattice visualization as colored tiles.

Nodes become tiles, colored by expansion/Fiedler values.
Bottlenecks appear as tile boundaries.

This module contains no GUI dependencies. All functions operate on
serializable data structures and can be used in headless pipelines.

JSON in/out pattern:
    state = state_from_json(input_json)
    state = extract_neighborhoods(state)
    output_json = state_to_json(state)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import math
import numpy as np

from .patterns import PinchedLattice, DeformedLattice

# CANONICAL IMPORT - one source of truth
from spectral_ops_fast import (
    local_fiedler_vector,
    local_expansion_estimate,
    local_bisect,
    expand_neighborhood,
    DEVICE
)


@dataclass
class Tile:
    """A mosaic tile representing a neighborhood cluster."""
    id: int
    nodes: List[Tuple[int, int]]
    color: Tuple[int, int, int]
    expansion_mean: float
    is_bottleneck: bool = False
    boundary_nodes: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class LatticeState:
    """
    Complete state of a lattice mosaic visualization.
    Fully serializable to JSON.
    """
    # Lattice configuration
    lattice_type: str = "pinched"  # "pinched" or "deformed"
    period: int = 8
    bottleneck_prob: float = 0.7
    rng_seed: int = 42

    # Region of interest
    x_range: Tuple[int, int] = (-4, 20)
    y_range: Tuple[int, int] = (-8, 8)
    seed_coord: Tuple[int, int] = (4, 0)

    # Computed data
    nodes: List[Tuple[int, int]] = field(default_factory=list)
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    expansion_map: Dict[str, float] = field(default_factory=dict)
    neighborhoods: List[List[Tuple[int, int]]] = field(default_factory=list)
    tiles: List[Dict] = field(default_factory=list)

    # Surface projection
    surface_type: str = "torus"  # "torus", "cylinder", "flat"
    surface_params: Dict[str, float] = field(default_factory=lambda: {
        "major_radius": 3.0,
        "minor_radius": 1.0
    })
    projected_coords: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


def coord_key(coord: Tuple[int, int]) -> str:
    """Convert coordinate tuple to JSON-safe string key."""
    return f"{coord[0]},{coord[1]}"


def key_to_coord(key: str) -> Tuple[int, int]:
    """Convert string key back to coordinate tuple."""
    x, y = key.split(",")
    return (int(x), int(y))


def create_lattice(state: LatticeState):
    """Create a lattice instance from state configuration."""
    if state.lattice_type == "pinched":
        return PinchedLattice(
            period=state.period,
            bottleneck_prob=state.bottleneck_prob,
            rng_seed=state.rng_seed
        )
    else:
        # Deformed lattice with periodic pinch
        from lattice import periodic_pinch_deformation
        return DeformedLattice(
            dims=2,
            deformation_fn=periodic_pinch_deformation(
                period=state.period,
                pinch_strength=0.5
            )
        )


def extract_region(state: LatticeState) -> LatticeState:
    """
    Extract nodes and edges from the configured region.
    Pure function: returns new state with nodes/edges populated.
    """
    lattice = create_lattice(state)

    nodes = []
    edges = []
    edge_set = set()

    for x in range(state.x_range[0], state.x_range[1] + 1):
        for y in range(state.y_range[0], state.y_range[1] + 1):
            coord = (x, y)
            node_id = lattice._coord_to_id(coord)
            nodes.append(coord)

            # Get neighbors and add edges
            for neighbor_id in lattice.neighbors(node_id):
                neighbor_coord = lattice.coord_of(neighbor_id)
                # Only add edge if neighbor is in region
                if (state.x_range[0] <= neighbor_coord[0] <= state.x_range[1] and
                    state.y_range[0] <= neighbor_coord[1] <= state.y_range[1]):
                    edge = tuple(sorted([coord, neighbor_coord]))
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edges.append(edge)

    # Create new state with updated nodes/edges
    new_state = LatticeState(**{**asdict(state),
                                 'nodes': nodes,
                                 'edges': list(edges)})
    return new_state


def compute_expansion_map(state: LatticeState) -> LatticeState:
    """
    Compute local expansion estimate for each node.
    High values = well-connected, low values = bottleneck.
    """
    lattice = create_lattice(state)

    expansion = {}
    for coord in state.nodes:
        node_id = lattice._coord_to_id(coord)
        exp_val = local_expansion_estimate(lattice, node_id, radius=2, num_lanczos=8)
        expansion[coord_key(coord)] = float(exp_val)

    new_state = LatticeState(**{**asdict(state), 'expansion_map': expansion})
    return new_state


def extract_neighborhoods(
    state: LatticeState,
    min_size: int = 4,
    max_depth: int = 3
) -> LatticeState:
    """
    Detect neighborhoods via recursive spectral bisection.
    Returns state with neighborhoods populated.
    """
    lattice = create_lattice(state)

    # Build node set for region
    node_set = {lattice._coord_to_id(c) for c in state.nodes}

    neighborhoods = []

    def recursive_bisect(nodes: Set[int], depth: int = 0):
        if len(nodes) < min_size or depth >= max_depth:
            if len(nodes) > 0:
                coords = [lattice.coord_of(n) for n in nodes]
                neighborhoods.append(coords)
            return

        # Try spectral bisection
        seed_list = list(nodes)[:5]
        partition_a, partition_b, cut = local_bisect(
            lattice,
            seed_nodes=seed_list,
            num_iterations=20,
            hop_expansion=1
        )

        # Filter to only nodes in our set
        partition_a = partition_a & nodes
        partition_b = partition_b & nodes

        # Check if bisection is meaningful
        if len(partition_a) < 3 or len(partition_b) < 3:
            coords = [lattice.coord_of(n) for n in nodes]
            neighborhoods.append(coords)
            return

        # Compute internal edges
        internal_a = sum(
            1 for n in partition_a
            for m in lattice.neighbors(n)
            if m in partition_a
        ) / 2
        internal_b = sum(
            1 for n in partition_b
            for m in lattice.neighbors(n)
            if m in partition_b
        ) / 2

        # Accept bisection if cut is relatively small
        if cut < 0.4 * (internal_a + internal_b + 1):
            recursive_bisect(partition_a, depth + 1)
            recursive_bisect(partition_b, depth + 1)
        else:
            coords = [lattice.coord_of(n) for n in nodes]
            neighborhoods.append(coords)

    # Start recursion
    recursive_bisect(node_set)

    new_state = LatticeState(**{**asdict(state), 'neighborhoods': neighborhoods})
    return new_state


def generate_tile_colors(n: int, rng_seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors for tiles."""
    rng = np.random.default_rng(rng_seed)

    colors = []
    for i in range(n):
        # Use golden angle for hue distribution
        hue = (i * 0.618033988749895) % 1.0
        # Vary saturation and value slightly
        sat = 0.5 + 0.3 * rng.random()
        val = 0.7 + 0.2 * rng.random()

        # HSV to RGB
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = val * (1 - sat)
        q = val * (1 - f * sat)
        t = val * (1 - (1 - f) * sat)

        if h_i == 0:
            r, g, b = val, t, p
        elif h_i == 1:
            r, g, b = q, val, p
        elif h_i == 2:
            r, g, b = p, val, t
        elif h_i == 3:
            r, g, b = p, q, val
        elif h_i == 4:
            r, g, b = t, p, val
        else:
            r, g, b = val, p, q

        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return colors


def map_to_tiles(state: LatticeState) -> LatticeState:
    """
    Convert neighborhoods to tiles with colors and boundary detection.
    """
    tiles = []
    colors = generate_tile_colors(len(state.neighborhoods), state.rng_seed)

    # Build coord -> neighborhood index mapping
    coord_to_hood = {}
    for i, hood in enumerate(state.neighborhoods):
        for coord in hood:
            coord_to_hood[coord_key(coord)] = i

    # Build edge set for boundary detection
    edge_set = set()
    for e in state.edges:
        edge_set.add((coord_key(e[0]), coord_key(e[1])))
        edge_set.add((coord_key(e[1]), coord_key(e[0])))

    for i, hood in enumerate(state.neighborhoods):
        # Compute mean expansion
        exp_vals = [state.expansion_map.get(coord_key(c), 0.0) for c in hood]
        exp_mean = sum(exp_vals) / len(exp_vals) if exp_vals else 0.0

        # Find boundary nodes (adjacent to different neighborhood)
        boundary = []
        for coord in hood:
            ck = coord_key(coord)
            is_boundary = False
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (coord[0] + dx, coord[1] + dy)
                nk = coord_key(neighbor)
                if (ck, nk) in edge_set:
                    neighbor_hood = coord_to_hood.get(nk, -1)
                    if neighbor_hood != i:
                        is_boundary = True
                        break
            if is_boundary:
                boundary.append(coord)

        # Determine if this is a bottleneck region
        is_bottleneck = exp_mean < 0.15  # Low expansion = bottleneck

        tile = Tile(
            id=i,
            nodes=hood,
            color=colors[i],
            expansion_mean=exp_mean,
            is_bottleneck=is_bottleneck,
            boundary_nodes=boundary
        )
        tiles.append(asdict(tile))

    new_state = LatticeState(**{**asdict(state), 'tiles': tiles})
    return new_state


def project_to_surface(state: LatticeState) -> LatticeState:
    """
    Project 2D lattice coordinates onto 3D surface.
    Supports: torus, cylinder, flat (no projection).
    """
    projected = {}

    x_min, x_max = state.x_range
    y_min, y_max = state.y_range
    x_span = x_max - x_min
    y_span = y_max - y_min

    R = state.surface_params.get("major_radius", 3.0)
    r = state.surface_params.get("minor_radius", 1.0)

    for coord in state.nodes:
        x, y = coord

        # Normalize to [0, 1]
        u = (x - x_min) / max(x_span, 1)
        v = (y - y_min) / max(y_span, 1)

        if state.surface_type == "torus":
            # Parametric torus
            theta = 2 * math.pi * u  # around major circle
            phi = 2 * math.pi * v    # around minor circle

            px = (R + r * math.cos(phi)) * math.cos(theta)
            py = (R + r * math.cos(phi)) * math.sin(theta)
            pz = r * math.sin(phi)

        elif state.surface_type == "cylinder":
            # Parametric cylinder (wrapped in x, linear in y)
            theta = 2 * math.pi * u

            px = R * math.cos(theta)
            py = R * math.sin(theta)
            pz = v * (y_span / 2)

        else:  # flat
            px = x
            py = y
            pz = 0.0

        projected[coord_key(coord)] = (px, py, pz)

    new_state = LatticeState(**{**asdict(state), 'projected_coords': projected})
    return new_state


def state_to_json(state: LatticeState) -> str:
    """Serialize state to JSON string."""
    return json.dumps(asdict(state), indent=2)


def state_from_json(json_str: str) -> LatticeState:
    """Deserialize state from JSON string."""
    data = json.loads(json_str)

    # Convert tuple fields back from lists
    if 'x_range' in data and isinstance(data['x_range'], list):
        data['x_range'] = tuple(data['x_range'])
    if 'y_range' in data and isinstance(data['y_range'], list):
        data['y_range'] = tuple(data['y_range'])
    if 'seed_coord' in data and isinstance(data['seed_coord'], list):
        data['seed_coord'] = tuple(data['seed_coord'])

    # Convert node lists back to tuples
    if 'nodes' in data:
        data['nodes'] = [tuple(n) for n in data['nodes']]
    if 'edges' in data:
        data['edges'] = [(tuple(e[0]), tuple(e[1])) for e in data['edges']]
    if 'neighborhoods' in data:
        data['neighborhoods'] = [[tuple(c) for c in hood] for hood in data['neighborhoods']]

    return LatticeState(**data)


def run_full_pipeline(state: LatticeState) -> LatticeState:
    """
    Run the complete analysis pipeline:
    1. Extract region
    2. Compute expansion map
    3. Extract neighborhoods
    4. Map to tiles
    5. Project to surface
    """
    state = extract_region(state)
    state = compute_expansion_map(state)
    state = extract_neighborhoods(state)
    state = map_to_tiles(state)
    state = project_to_surface(state)
    return state


def create_demo_state() -> LatticeState:
    """Create a demo state with good defaults for visualization."""
    return LatticeState(
        lattice_type="pinched",
        period=8,
        bottleneck_prob=0.8,
        rng_seed=42,
        x_range=(-2, 22),
        y_range=(-6, 6),
        seed_coord=(4, 0),
        surface_type="torus",
        surface_params={"major_radius": 3.0, "minor_radius": 1.0}
    )
