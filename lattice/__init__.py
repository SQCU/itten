"""
Lattice: Spectral-Dependent Lattice Construction and Graph Conversion.

This module provides:
1. Lattice patterns (Islands, Bridges, TerritoryGraph)
2. Spectral-dependent extrusion operations
3. Mesh generation for 3D visualization
4. Graph conversion (lattice <-> spectral_ops_fast.Graph)

Key insight: The local_expansion_estimate() function uses Lanczos iteration
to approximate lambda_2 of the local neighborhood. This gives information
about graph structure BEYOND the explicit hop radius via Krylov subspace
projection.

The graph.py submodule provides the CRITICAL LINK between lattice structures
and the canonical Graph type in spectral_ops_fast.py.
"""

from .patterns import (
    TerritoryGraph, Island, Bridge, create_islands_and_bridges,
    PinchedLattice, DeformedLattice, periodic_pinch_deformation, shear_gradient_deformation
)
from .extrude import ExpansionGatedExtruder, FiedlerAlignedGeometry, ExtrudedNode, ExtrusionState
from .mesh import LatticeToMesh, EggSurface, TorusSurface, SphereSurface, Mesh, Vertex, Face
from .render import render_3d_egg_mesh, render_expansion_heatmap, render_extrusion_layers, save_render
from .graph import lattice_to_graph, graph_to_lattice_coords, apply_spectral_to_lattice

# Mosaic visualization (nodes as colored tiles)
from .mosaic import (
    LatticeState, Tile, extract_neighborhoods, compute_expansion_map,
    map_to_tiles, project_to_surface, state_to_json, state_from_json
)
from .mosaic_render import render_flat_mosaic

__all__ = [
    # Patterns
    'TerritoryGraph', 'Island', 'Bridge', 'create_islands_and_bridges',
    'PinchedLattice', 'DeformedLattice', 'periodic_pinch_deformation', 'shear_gradient_deformation',
    # Extrusion
    'ExpansionGatedExtruder', 'FiedlerAlignedGeometry', 'ExtrudedNode', 'ExtrusionState',
    # Mesh
    'LatticeToMesh', 'EggSurface', 'TorusSurface', 'SphereSurface', 'Mesh', 'Vertex', 'Face',
    # Render
    'render_3d_egg_mesh', 'render_expansion_heatmap', 'render_extrusion_layers', 'save_render',
    # Graph conversion (THE MISSING LINK)
    'lattice_to_graph', 'graph_to_lattice_coords', 'apply_spectral_to_lattice',
    # Mosaic visualization
    'LatticeState', 'Tile', 'extract_neighborhoods', 'compute_expansion_map',
    'map_to_tiles', 'project_to_surface', 'state_to_json', 'state_from_json',
    'render_flat_mosaic',
]
