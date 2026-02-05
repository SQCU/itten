# Hypercontext: Unified Render Module

## Mission
Create `texture_synth/render/` module that:
1. Wraps textures onto arbitrary convex meshes
2. Uses dichromatic PBR for unambiguous normal differentiation
3. Produces rendered output for each edit step

## Key Insight: Dichromatic Reflection

Two-component model:
1. **Body reflection** (diffuse): Light penetrates surface, scatters, exits. Color = material color.
2. **Interface reflection** (specular): Light bounces at surface. Color = light color (or tinted for metals).

For normal map differentiation:
- Body: **WARM** (copper/orange: [200, 120, 80])
- Interface: **COOL** (blue-white: [180, 200, 255])
- High anisotropy so specular follows bump direction
- Fresnel blend between warm diffuse and cool specular

## Files to Create

### `texture_synth/render/__init__.py`
```python
from .pbr import render_mesh, render_mesh_dichromatic
from .uv_mapping import apply_texture_to_mesh
from .trace import RenderTrace
```

### `texture_synth/render/pbr.py`
Unified PBR renderer:
```python
def render_mesh(
    mesh: Mesh,
    height_field: np.ndarray,
    normal_map: np.ndarray,
    output_size: int = 512,
    mode: str = 'dichromatic',  # 'dichromatic', 'anisotropic', 'raking', 'iridescent'
    **kwargs
) -> np.ndarray:
    """
    Render textured mesh.

    Performs:
    1. Ray-mesh intersection for each pixel
    2. UV lookup at intersection point
    3. Sample height/normal from texture
    4. Compute shading based on mode
    5. Return RGB image
    """
    ...

def render_mesh_dichromatic(
    mesh: Mesh,
    height_field: np.ndarray,
    normal_map: np.ndarray,
    output_size: int = 512,
    body_color: tuple = (200, 120, 80),    # Warm copper
    interface_color: tuple = (180, 200, 255),  # Cool blue-white
    anisotropy: float = 0.8,
    light_angle: float = 15.0,
    bump_strength: float = 0.7,
) -> np.ndarray:
    """
    Dichromatic PBR rendering.

    Body reflection: warm, Lambertian
    Interface reflection: cool, anisotropic specular following bump direction
    Fresnel: blends body↔interface based on viewing angle
    """
    # Setup
    body = np.array(body_color, dtype=np.float64)
    interface = np.array(interface_color, dtype=np.float64)

    # For each pixel...
    for py in range(output_size):
        for px in range(output_size):
            # Ray-mesh intersection
            hit, uv, normal = ray_mesh_intersect(mesh, px, py, output_size)
            if not hit:
                continue

            # Sample textures
            height = sample_bilinear(height_field, uv)
            tex_normal = sample_bilinear(normal_map, uv)

            # Perturb surface normal
            perturbed = perturb_normal(normal, tex_normal, bump_strength)

            # Tangent direction from height gradient (for anisotropy)
            tangent = compute_tangent_from_gradient(height_field, uv)

            # Lighting
            ndotl = max(0, dot(perturbed, light_dir))
            ndotv = max(0, dot(perturbed, view_dir))

            # Body (diffuse)
            diffuse = ndotl

            # Interface (anisotropic specular)
            specular = anisotropic_specular(
                perturbed, tangent, light_dir, view_dir, anisotropy
            )

            # Fresnel blend
            fresnel = schlick_fresnel(ndotv, F0=0.04)
            color = (1 - fresnel) * body * diffuse + fresnel * interface * specular

            img[py, px] = color

    return img
```

### Anisotropic Specular (Ward Model)
```python
def anisotropic_specular(normal, tangent, light, view, anisotropy):
    """
    Ward anisotropic specular.

    anisotropy controls elongation:
    - 0: isotropic (circular highlight)
    - 1: fully stretched along tangent
    """
    bitangent = np.cross(normal, tangent)

    half_vec = normalize(light + view)

    # Anisotropic roughness
    alpha_t = 0.1 * (1 - anisotropy) + 0.01  # Along tangent (rough if aniso low)
    alpha_b = 0.1  # Along bitangent (constant)

    # Ward model
    hdott = np.dot(half_vec, tangent)
    hdotb = np.dot(half_vec, bitangent)
    hdotn = np.dot(half_vec, normal)

    exponent = -((hdott/alpha_t)**2 + (hdotb/alpha_b)**2) / (hdotn**2)
    spec = np.exp(exponent) / (4 * np.pi * alpha_t * alpha_b)

    return spec * max(0, np.dot(normal, light))
```

### `texture_synth/render/uv_mapping.py`
```python
def apply_texture_to_mesh(mesh: Mesh, height_field: np.ndarray) -> Mesh:
    """
    Store height values as vertex attributes for rendering.
    """
    # Sample height at each vertex's UV
    heights = np.array([
        sample_bilinear(height_field, uv) for uv in mesh.uvs
    ])
    mesh.vertex_heights = heights
    return mesh

def compute_tangent_frame(mesh: Mesh, height_field: np.ndarray):
    """
    Compute per-vertex tangent frames from texture gradient.
    """
    tangents = []
    for i, uv in enumerate(mesh.uvs):
        # Sample height gradient
        du = 1.0 / height_field.shape[1]
        dv = 1.0 / height_field.shape[0]
        h_u = sample_bilinear(height_field, (uv[0] + du, uv[1]))
        h_l = sample_bilinear(height_field, (uv[0] - du, uv[1]))
        grad_u = (h_u - h_l) / (2 * du)

        # Tangent in U direction, projected to surface
        tangent = project_to_tangent_plane(mesh.vertex_normals[i], [1, 0, grad_u])
        tangents.append(normalize(tangent))

    mesh.tangents = np.array(tangents)
    return mesh
```

### `texture_synth/render/trace.py`
Track render history:
```python
class RenderTrace:
    """Track every rendered output."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.history = []

    def render_and_save(
        self,
        mesh: Mesh,
        height_field: np.ndarray,
        command: str,
        **render_kwargs
    ) -> Path:
        """Render current state and save to trace."""
        from .pbr import render_mesh_dichromatic

        normal_map = height_to_normals(height_field)
        img = render_mesh_dichromatic(mesh, height_field, normal_map, **render_kwargs)

        filename = f"step_{self.step:04d}_{slugify(command)}.png"
        path = self.output_dir / filename

        Image.fromarray(img).save(path)

        self.history.append({
            'step': self.step,
            'command': command,
            'path': str(path)
        })
        self.step += 1

        print(f"[RENDER] {path}")
        return path

    def save_manifest(self):
        """Save JSON manifest of all renders."""
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.history, f, indent=2)
```

## Integration with Mesh

Need simple ray-mesh intersection:
```python
def ray_mesh_intersect(mesh, px, py, output_size):
    """
    Cast ray from pixel (px, py) into scene.

    Returns: (hit: bool, uv: tuple, normal: ndarray)
    """
    # Camera setup (orthographic for simplicity)
    ray_origin = np.array([
        (px / output_size - 0.5) * 2,
        (0.5 - py / output_size) * 2,
        -10  # Far behind mesh
    ])
    ray_dir = np.array([0, 0, 1])  # Looking down +Z

    # Test each triangle
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        hit, t, u, v = ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2)
        if hit:
            # Interpolate UV
            uv0, uv1, uv2 = mesh.uvs[face]
            uv = (1-u-v) * uv0 + u * uv1 + v * uv2

            # Compute normal
            normal = normalize(np.cross(v1-v0, v2-v0))

            return True, uv, normal

    return False, None, None
```

## Deliverables
1. `texture_synth/render/pbr.py` - Dichromatic + anisotropic renderer
2. `texture_synth/render/uv_mapping.py` - UV and tangent utilities
3. `texture_synth/render/trace.py` - Render history tracking
4. `texture_synth/render/__init__.py`
5. Test showing dichromatic render of icosahedron with amongus texture
