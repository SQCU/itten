"""
Lattice/Graph Conversion: THE MISSING LINK.

This module bridges lattice structures (TerritoryGraph) and the canonical
Graph type in spectral_ops_fast.py.

Key functions:
- lattice_to_graph: Convert lattice to Graph for spectral operations
- graph_to_lattice_coords: Map spectral results back to lattice coordinates
- apply_spectral_to_lattice: Convenience wrapper for spectral ops on lattices

The Graph class in spectral_ops_fast.py is THE canonical graph representation.
All lattice operations should convert to/from Graph when spectral operations
are needed.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import sys
import os

# Import from spectral_ops_fast (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_ops_fast import Graph, DEVICE

# Import local types
from .patterns import TerritoryGraph


def lattice_to_graph(
    lattice: TerritoryGraph,
    device=None
) -> Tuple['Graph', Dict[int, Tuple[int, int]]]:
    """
    Convert a lattice structure to the canonical Graph type.

    Args:
        lattice: TerritoryGraph (or compatible) with:
            - all_node_ids() or all_nodes(): iterable of nodes
            - neighbors(node_id): list of neighbor IDs
            - node_coord(node_id): (x, y) tuple
        device: Target device (defaults to DEVICE)

    Returns:
        graph: Graph with lattice connectivity
        id_to_coord: Dict mapping Graph node indices to original lattice coords

    The id_to_coord dict is essential for mapping spectral results back to
    lattice coordinates.
    """
    if device is None:
        device = DEVICE

    # Get all node IDs from the lattice
    if hasattr(lattice, 'all_node_ids'):
        original_ids = list(lattice.all_node_ids())
    elif hasattr(lattice, 'all_nodes'):
        # TerritoryGraph uses coords, need to map through node_id
        all_coords = lattice.all_nodes()
        original_ids = [lattice.node_id(c) for c in all_coords if lattice.node_id(c) is not None]
    else:
        raise ValueError("Lattice must have all_node_ids() or all_nodes() method")

    # Build mapping from original IDs to Graph indices
    original_to_idx = {orig_id: idx for idx, orig_id in enumerate(original_ids)}

    # Build reverse mapping: Graph index -> lattice coordinate
    id_to_coord: Dict[int, Tuple[int, int]] = {}
    idx = 0
    while idx < len(original_ids):
        orig_id = original_ids[idx]
        if hasattr(lattice, 'node_coord'):
            coord = lattice.node_coord(orig_id)
            if coord is not None:
                id_to_coord[idx] = coord
        idx += 1

    # Create Graph using the from_lattice class method
    graph = Graph.from_lattice(lattice, device=device)

    return graph, id_to_coord


def graph_to_lattice_coords(
    graph: 'Graph',
    node_data: Dict[int, Any],
    id_to_coord: Dict[int, Tuple[int, int]]
) -> Dict[Tuple[int, int], Any]:
    """
    Map spectral results from Graph node indices back to lattice coordinates.

    Args:
        graph: The Graph that was created from a lattice
        node_data: Dict mapping Graph node indices to some data (e.g., Fiedler values)
        id_to_coord: The id_to_coord dict returned by lattice_to_graph()

    Returns:
        coord_data: Dict mapping lattice (x, y) coords to the same data

    Example:
        graph, id_to_coord = lattice_to_graph(territory)
        L = graph.laplacian()
        fiedler_dict, _ = local_fiedler_vector(...)  # Returns {node_id: value}
        coord_fiedler = graph_to_lattice_coords(graph, fiedler_dict, id_to_coord)
        # coord_fiedler is now {(x, y): fiedler_value}
    """
    coord_data: Dict[Tuple[int, int], Any] = {}

    items = list(node_data.items())
    idx = 0
    while idx < len(items):
        node_idx, value = items[idx]
        if node_idx in id_to_coord:
            coord_data[id_to_coord[node_idx]] = value
        idx += 1

    return coord_data


def apply_spectral_to_lattice(
    lattice: TerritoryGraph,
    spectral_fn: Callable,
    *args,
    device=None,
    **kwargs
) -> Tuple[Any, Dict[int, Tuple[int, int]]]:
    """
    Convenience: apply a spectral function to a lattice.

    Pipeline:
        1. Convert lattice to Graph
        2. Apply spectral function to Graph
        3. Return result and id_to_coord for mapping back

    Args:
        lattice: TerritoryGraph or compatible
        spectral_fn: Function that takes Graph (or its Laplacian) as first arg
        *args: Additional positional args for spectral_fn
        device: Target device
        **kwargs: Keyword args for spectral_fn

    Returns:
        result: Result of spectral_fn
        id_to_coord: For mapping results back to coordinates

    Example:
        # Compute Laplacian eigenvectors on a lattice
        from spectral_ops_fast import lanczos_k_eigenvectors

        def my_spectral_fn(graph, num_eigs):
            L = graph.laplacian()
            return lanczos_k_eigenvectors(L, num_eigs)

        (eigenvecs, eigenvals), id_to_coord = apply_spectral_to_lattice(
            territory,
            my_spectral_fn,
            num_eigs=4
        )
    """
    graph, id_to_coord = lattice_to_graph(lattice, device=device)
    result = spectral_fn(graph, *args, **kwargs)
    return result, id_to_coord


def compute_lattice_expansion_map(
    lattice: TerritoryGraph,
    radius: int = 3,
    num_lanczos: int = 15,
    device=None
) -> Dict[Tuple[int, int], float]:
    """
    Compute local spectral expansion for all nodes in a lattice.

    This is a convenience function that:
    1. Converts lattice to Graph
    2. Computes batched expansion estimates
    3. Maps results back to lattice coordinates

    Args:
        lattice: TerritoryGraph with island/bridge structure
        radius: Neighborhood radius for expansion estimate
        num_lanczos: Lanczos iterations
        device: Target device

    Returns:
        expansion_map: Dict mapping (x, y) coords to lambda_2 estimates
    """
    from spectral_ops_fast import expansion_map_batched

    # Get all node IDs
    if hasattr(lattice, 'all_node_ids'):
        node_ids = list(lattice.all_node_ids())
    else:
        all_coords = lattice.all_nodes()
        node_ids = [lattice.node_id(c) for c in all_coords if lattice.node_id(c) is not None]

    # Compute expansions using batched method
    id_expansions = expansion_map_batched(
        lattice,  # TerritoryGraph implements GraphView
        node_ids,
        radius=radius,
        k=num_lanczos
    )

    # Map back to coordinates
    result: Dict[Tuple[int, int], float] = {}
    idx = 0
    while idx < len(node_ids):
        node_id = node_ids[idx]
        if node_id in id_expansions:
            coord = lattice.node_coord(node_id)
            if coord is not None:
                result[coord] = id_expansions[node_id]
        idx += 1

    return result


def compute_lattice_fiedler(
    lattice: TerritoryGraph,
    seed_coords: Optional[List[Tuple[int, int]]] = None,
    num_iterations: int = 30,
    hop_expansion: int = 3,
    device=None
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    Compute Fiedler vector for a lattice region.

    Args:
        lattice: TerritoryGraph
        seed_coords: Starting coordinates (defaults to first island center)
        num_iterations: Lanczos iterations
        hop_expansion: Neighborhood expansion
        device: Target device

    Returns:
        fiedler_by_coord: Dict mapping (x, y) to Fiedler values
        lambda2: Second smallest eigenvalue
    """
    from spectral_ops_fast import local_fiedler_vector

    # Get seed node IDs
    if seed_coords is None:
        seed_ids = lattice.seed_nodes()
    else:
        seed_ids = [lattice.node_id(c) for c in seed_coords if lattice.node_id(c) is not None]

    if not seed_ids:
        return {}, 0.0

    # Compute Fiedler vector
    fiedler_dict, lambda2 = local_fiedler_vector(
        lattice,  # TerritoryGraph implements GraphView
        seed_ids,
        num_iterations=num_iterations,
        hop_expansion=hop_expansion
    )

    # Map back to coordinates
    result: Dict[Tuple[int, int], float] = {}
    items = list(fiedler_dict.items())
    idx = 0
    while idx < len(items):
        node_id, value = items[idx]
        coord = lattice.node_coord(node_id)
        if coord is not None:
            result[coord] = value
        idx += 1

    return result, lambda2


def lattice_laplacian(
    lattice: TerritoryGraph,
    normalized: bool = False,
    device=None
):
    """
    Get the Laplacian matrix for a lattice.

    Args:
        lattice: TerritoryGraph
        normalized: If True, return normalized Laplacian
        device: Target device

    Returns:
        L: Sparse Laplacian tensor
        id_to_coord: For mapping indices back to coordinates
    """
    graph, id_to_coord = lattice_to_graph(lattice, device=device)
    L = graph.laplacian(normalized=normalized)
    return L, id_to_coord
