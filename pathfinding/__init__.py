"""
Pathfinding Module - Spectral and classical pathfinding on surfaces.

This module provides pathfinding algorithms that work on 2D heightfields
and texture surfaces, with support for blocking geometry.

Main API:
    - find_surface_path: Find path on texture surface
    - texture_to_pathfinding_graph: Convert heightfield to graph
    - compare_surface_paths: Compare all methods on same input
    - PathResult: Dataclass for pathfinding results
    - HeightfieldGraph: Graph representation of heightfield

Example:
    >>> import numpy as np
    >>> from pathfinding import find_surface_path, create_blocking_mask
    >>>
    >>> # Create a heightfield (or use texture synthesis output)
    >>> heightfield = np.random.rand(100, 100)
    >>>
    >>> # Create blocking mask for high areas
    >>> blocking = create_blocking_mask(heightfield, threshold=0.8)
    >>>
    >>> # Find path using spectral method
    >>> result = find_surface_path(
    ...     heightfield,
    ...     start=(10, 10),
    ...     goal=(90, 90),
    ...     blocking_mask=blocking,
    ...     method='spectral'
    ... )
    >>>
    >>> print(result.summary())
"""

from .surface import (
    texture_to_pathfinding_graph,
    find_surface_path,
    compare_surface_paths,
    find_valid_endpoints,
    create_blocking_mask,
)

from .result import PathResult

from .graph import HeightfieldGraph, heightfield_from_image

from .visualization import (
    visualize_path,
    visualize_blocking,
    visualize_comparison,
    create_summary_image,
    save_image,
)

# Also expose individual methods for direct use
from .spectral import spectral_path, SpectralPathfinder
from .classical import dijkstra_path, astar_path

__all__ = [
    # Main API
    'find_surface_path',
    'texture_to_pathfinding_graph',
    'compare_surface_paths',
    'find_valid_endpoints',
    'create_blocking_mask',

    # Result type
    'PathResult',

    # Graph type
    'HeightfieldGraph',
    'heightfield_from_image',

    # Visualization
    'visualize_path',
    'visualize_blocking',
    'visualize_comparison',
    'create_summary_image',
    'save_image',

    # Individual methods
    'spectral_path',
    'SpectralPathfinder',
    'dijkstra_path',
    'astar_path',
]
