"""
Geometry module for texture synthesis.

Provides composable 3D primitives and operations.
"""

from .mesh import Mesh
from .primitives import Icosahedron, Sphere, Egg
from .operations import fuse, chop, squash

__all__ = [
    'Mesh',
    'Icosahedron', 'Sphere', 'Egg',
    'fuse', 'chop', 'squash'
]
