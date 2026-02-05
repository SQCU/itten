# Hypercontext: Geometry Module

## Mission
Create `texture_synth/geometry/` module with composable 3D primitives and operations.

## Files to Create

### `texture_synth/geometry/__init__.py`
```python
from .primitives import Icosahedron, Sphere, Egg, Mesh
from .operations import fuse, chop, squash, translate, rotate, scale
```

### `texture_synth/geometry/mesh.py`
Core mesh data structure:
```python
@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray     # (M, 3) triangle indices
    uvs: np.ndarray       # (N, 2) UV coordinates

    def translate(self, x, y, z) -> 'Mesh': ...
    def rotate(self, axis, angle) -> 'Mesh': ...
    def scale(self, sx, sy, sz) -> 'Mesh': ...
    def copy(self) -> 'Mesh': ...
```

### `texture_synth/geometry/primitives.py`
Generate primitive meshes with proper UVs:
```python
def Icosahedron(radius=1.0, subdivisions=0) -> Mesh:
    """Icosahedron with spherical UV mapping."""

def Sphere(radius=1.0, segments=32) -> Mesh:
    """UV sphere."""

def Egg(radius=1.0, pointiness=0.25, segments=32) -> Mesh:
    """Egg shape with proper UVs."""
```

### `texture_synth/geometry/operations.py`
Mesh operations:
```python
def fuse(*meshes: Mesh) -> Mesh:
    """Combine meshes (union of vertices/faces)."""

def chop(mesh: Mesh, plane_normal: tuple, plane_origin: tuple) -> Mesh:
    """Cut mesh by plane, keep half above plane."""

def squash(mesh: Mesh, axis: str, factor: float) -> Mesh:
    """Scale along axis. factor=0.7 means 30% shorter."""

def boolean_union(a: Mesh, b: Mesh) -> Mesh: ...
def boolean_intersect(a: Mesh, b: Mesh) -> Mesh: ...
```

## Icosahedron Implementation

```python
def icosahedron_base_vertices():
    """20 vertices of unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]) / np.sqrt(1 + phi**2)

    return vertices

def icosahedron_faces():
    """20 triangular faces."""
    return np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
```

## UV Mapping for Convex Surfaces

Spherical UV mapping:
```python
def spherical_uv(vertices):
    """Map 3D vertices to UV using spherical projection."""
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    unit = vertices / norms

    # θ = atan2(z, x), φ = asin(y)
    theta = np.arctan2(unit[:, 2], unit[:, 0])
    phi = np.arcsin(np.clip(unit[:, 1], -1, 1))

    u = (theta / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi + 0.5

    return np.column_stack([u, v])
```

## Chop Operation

```python
def chop(mesh: Mesh, plane_normal, plane_origin) -> Mesh:
    """Cut mesh by plane."""
    normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    origin = np.array(plane_origin)

    # Signed distance from plane
    distances = np.dot(mesh.vertices - origin, normal)

    # Keep vertices above plane
    keep_mask = distances >= 0

    # For faces, need to handle intersection...
    # (Simplified: keep faces where all 3 vertices are above)
    face_keep = keep_mask[mesh.faces].all(axis=1)

    # Remap indices
    new_indices = np.cumsum(keep_mask) - 1
    new_vertices = mesh.vertices[keep_mask]
    new_faces = new_indices[mesh.faces[face_keep]]
    new_uvs = mesh.uvs[keep_mask]

    return Mesh(new_vertices, new_faces, new_uvs)
```

## Test Cases

```python
# Two fused icosahedrons
mesh = fuse(
    Icosahedron(),
    Icosahedron().translate(1.5, 0, 0)
)

# Three icosahedrons
mesh = fuse(
    Icosahedron(),
    Icosahedron().translate(1.5, 0, 0),
    Icosahedron().translate(0.75, 1.3, 0)
)

# Chop one in half
mesh = chop(mesh, plane_normal=(0, 1, 0), plane_origin=(0, 0.5, 0))

# Squash 30%
mesh = squash(mesh, axis='y', factor=0.7)
```

## Deliverables
1. `texture_synth/geometry/mesh.py`
2. `texture_synth/geometry/primitives.py`
3. `texture_synth/geometry/operations.py`
4. `texture_synth/geometry/__init__.py`
5. Working test that creates fused/chopped/squashed icosahedrons
