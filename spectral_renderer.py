"""
Pure-torch rendering pipeline for spectral surface visualization.

All modules are torch.compile compatible with fullgraph=True.
This is pure geometry -- no sparse ops, no dicts, no Python control flow
in the hot path.

Reimplements the numpy rendering code from texture/render/ as nn.Modules:
- HeightToNormals: height field -> RGB-encoded normal map
- BilinearSampler: bilinear texture sampling with UV wrapping
- EggSurfaceRenderer: full 3D egg renderer with bump mapping + Blinn-Phong

Source references:
- texture/normals.py: height_to_normals
- texture/render/bump_render.py: sample_texture_bilinear, render_bumped_egg
- texture/render/lighting.py: lambertian_diffuse, blinn_phong_specular, schlick_fresnel
- texture/render/ray_cast.py: ray_sphere_test, egg_deformation, spherical_uv_from_ray
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class HeightToNormals(nn.Module):
    """Convert (H, W) height field to (H, W, 3) RGB-encoded normal map.

    Uses central differences for gradient, constructs normal as (-dh/dx, -dh/dy, 1),
    normalizes, maps to [0, 1] range for RGB encoding.

    Source: texture/normals.py:height_to_normals
    """

    def __init__(self, strength: float = 1.0):
        super().__init__()
        self.strength = strength

    def forward(self, height_field: torch.Tensor) -> torch.Tensor:
        """(H, W) -> (H, W, 3) normal map in [0, 1]."""
        h = height_field.float()

        # Normalize height to [0, 1]
        h_min = h.min()
        h_max = h.max()
        denom = h_max - h_min
        # Avoid division by zero for flat fields
        h = torch.where(denom > 0, (h - h_min) / denom, h - h_min)

        # Central differences with wrapping (roll-based, matches numpy original)
        dx = (torch.roll(h, -1, dims=1) - torch.roll(h, 1, dims=1)) / 2.0
        dy = (torch.roll(h, -1, dims=0) - torch.roll(h, 1, dims=0)) / 2.0

        # Scale by strength
        dx = dx * self.strength
        dy = dy * self.strength

        # Build normal vectors: n = (-dx, -dy, 1)
        ones = torch.ones_like(dx)
        normals = torch.stack([-dx, -dy, ones], dim=-1)  # (H, W, 3)

        # Normalize to unit length
        norm = torch.norm(normals, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-10)
        normals = normals / norm

        # Map from [-1, 1] to [0, 1] for RGB encoding
        normals = (normals + 1.0) / 2.0

        return normals


class BilinearSampler(nn.Module):
    """Bilinear texture sampling with UV wrapping.

    Pure torch implementation of bilinear interpolation at arbitrary UV coordinates.
    Handles both (H, W) and (H, W, C) textures.

    Source: texture/render/bump_render.py:sample_texture_bilinear (lines 20-69)
    """

    def forward(self, texture: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sample texture at (u, v) coordinates in [0, 1].

        Args:
            texture: (H, W) or (H, W, C) texture tensor
            u: UV u-coordinates, any shape
            v: UV v-coordinates, same shape as u

        Returns:
            Sampled values. Shape matches u/v (plus channel dim if texture has channels).
        """
        h, w = texture.shape[0], texture.shape[1]

        # Wrap UVs to [0, 1)
        u = u % 1.0
        v = v % 1.0

        # Convert to pixel coordinates (center-pixel convention)
        x = u * w - 0.5
        y = v * h - 0.5

        x0 = torch.floor(x).long() % w
        y0 = torch.floor(y).long() % h
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h

        # Fractional parts
        fx = x - torch.floor(x)
        fy = y - torch.floor(y)

        # Sample four corners
        if texture.ndim == 2:
            v00 = texture[y0, x0]
            v01 = texture[y0, x1]
            v10 = texture[y1, x0]
            v11 = texture[y1, x1]
        else:
            v00 = texture[y0, x0, :]
            v01 = texture[y0, x1, :]
            v10 = texture[y1, x0, :]
            v11 = texture[y1, x1, :]
            fx = fx.unsqueeze(-1)
            fy = fy.unsqueeze(-1)

        result = (v00 * (1 - fx) * (1 - fy) +
                  v01 * fx * (1 - fy) +
                  v10 * (1 - fx) * fy +
                  v11 * fx * fy)

        return result


class EggSurfaceRenderer(nn.Module):
    """Render bump-mapped egg surface from texture + normal map.

    The "visual echo" renderer: spectral structure in the height field
    becomes visible 3D surface features through bump mapping.

    Pipeline:
    1. Ray generation: grid of rays from camera through pixel grid
    2. Ray-sphere intersection: find where rays hit a unit sphere
    3. Egg deformation: deform sphere to egg shape (one axis compressed)
    4. Spherical UV: compute UV coordinates from intersection point
    5. Texture + normal sampling: BilinearSampler at UV coordinates
    6. TBN frame: compute tangent/bitangent/normal frame on egg surface
    7. Normal perturbation: apply normal map via TBN frame
    8. Lighting: Blinn-Phong with Schlick Fresnel

    Source: texture/render/bump_render.py:render_bumped_egg (~200 lines)
    """

    def __init__(
        self,
        resolution: int = 512,
        egg_factor: float = 0.25,
        bump_strength: float = 1.0,
        light_dir: Tuple[float, float, float] = (0.5, 0.7, 1.0),
        specular_power: float = 32.0,
        fresnel_ior: float = 1.5,
        ambient: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.egg_factor = egg_factor
        self.bump_strength = bump_strength
        self.specular_power = specular_power
        self.ambient = ambient

        # Normalize and register light direction
        ld = torch.tensor(light_dir, dtype=torch.float32)
        ld = ld / torch.norm(ld)
        self.register_buffer("light_dir", ld)

        # View direction (looking down -Z, so view is +Z)
        self.register_buffer("view_dir", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        # Precompute F0 from IOR: F0 = ((ior-1)/(ior+1))^2
        f0 = ((fresnel_ior - 1.0) / (fresnel_ior + 1.0)) ** 2
        self.register_buffer("f0", torch.tensor(f0, dtype=torch.float32))

        # Bilinear sampler
        self.sampler = BilinearSampler()

        # Precompute pixel grid (registered as buffer for device portability)
        y_coords = torch.arange(resolution, dtype=torch.float32)
        x_coords = torch.arange(resolution, dtype=torch.float32)
        # meshgrid: px[i,j] = x_coords[j], py[i,j] = y_coords[i]
        px, py = torch.meshgrid(x_coords, y_coords, indexing="xy")
        # Normalized device coordinates with 0.85 scale
        scale = 0.85
        nx = (px / resolution - 0.5) * 2.0 / scale
        ny = (0.5 - py / resolution) * 2.0 / scale
        self.register_buffer("nx", nx)
        self.register_buffer("ny", ny)

    # ------------------------------------------------------------------
    # Lighting helpers (pure static methods on tensors)
    # ------------------------------------------------------------------

    @staticmethod
    def _lambertian_diffuse(normal: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        """N dot L clamped to [0, 1].  normal: (..., 3), light_dir: (3,)."""
        return torch.clamp(torch.sum(normal * light_dir, dim=-1), min=0.0)

    @staticmethod
    def _blinn_phong_specular(
        normal: torch.Tensor,
        light_dir: torch.Tensor,
        view_dir: torch.Tensor,
        power: float,
    ) -> torch.Tensor:
        """Blinn-Phong specular: (N dot H)^power."""
        half_vec = light_dir + view_dir
        half_vec = half_vec / torch.norm(half_vec)
        ndoth = torch.clamp(torch.sum(normal * half_vec, dim=-1), min=0.0)
        return ndoth.pow(power)

    @staticmethod
    def _schlick_fresnel(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Schlick Fresnel: F0 + (1 - F0)(1 - cos_theta)^5."""
        return f0 + (1.0 - f0) * (1.0 - cos_theta).pow(5)

    @staticmethod
    def _compute_tbn_frame(
        base_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tangent and bitangent from a surface normal field.

        Uses Gram-Schmidt: project 'up' (or 'right' at poles) onto tangent plane.

        Args:
            base_normal: (..., 3)

        Returns:
            tangent: (..., 3), bitangent: (..., 3)
        """
        # Reference vectors
        up = torch.zeros_like(base_normal)
        up[..., 1] = 1.0

        right = torch.zeros_like(base_normal)
        right[..., 0] = 1.0

        # Detect poles where base_normal is nearly parallel to up
        dot_up = torch.sum(base_normal * up, dim=-1)  # (...)
        parallel_mask = (torch.abs(dot_up) > 0.99).unsqueeze(-1)  # (..., 1)

        ref = torch.where(parallel_mask, right, up)

        # Gram-Schmidt: tangent = normalize(ref - (ref . n) * n)
        ref_dot_n = torch.sum(ref * base_normal, dim=-1, keepdim=True)
        tangent = ref - ref_dot_n * base_normal
        tangent_len = torch.norm(tangent, dim=-1, keepdim=True)
        tangent = tangent / torch.clamp(tangent_len, min=1e-6)

        # Bitangent = normal x tangent
        bitangent = torch.linalg.cross(base_normal, tangent, dim=-1)

        return tangent, bitangent

    @staticmethod
    def _perturb_normal(
        base_normal: torch.Tensor,
        tangent: torch.Tensor,
        bitangent: torch.Tensor,
        tex_normal: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Perturb surface normal using tangent-space normal map sample.

        Args:
            base_normal: (..., 3) surface normal
            tangent: (..., 3)
            bitangent: (..., 3)
            tex_normal: (..., 3) tangent-space normal in [-1, 1]
            strength: blend factor [0, 1]

        Returns:
            Perturbed, normalized normal (..., 3)
        """
        # Transform tangent-space normal to world space via TBN matrix
        world_normal = (
            tex_normal[..., 0:1] * tangent
            + tex_normal[..., 1:2] * bitangent
            + tex_normal[..., 2:3] * base_normal
        )

        # Blend between base and world_normal
        result = base_normal * (1.0 - strength) + world_normal * strength

        # Normalize
        result_len = torch.norm(result, dim=-1, keepdim=True)
        result = result / torch.clamp(result_len, min=1e-6)
        return result

    # ------------------------------------------------------------------
    # Ray-surface helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ray_sphere_test(nx: torch.Tensor, ny: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ray-sphere intersection for unit sphere.

        Returns:
            inside_mask: (H, W) bool, z: (H, W) depth
        """
        r_sq = nx * nx + ny * ny
        inside_mask = r_sq <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r_sq, min=0.0))
        return inside_mask, z

    @staticmethod
    def _egg_deformation(
        nx: torch.Tensor,
        ny: torch.Tensor,
        z: torch.Tensor,
        egg_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deform sphere coordinates into egg shape.

        Returns:
            x_egg, z_egg, egg_mod
        """
        egg_mod = 1.0 - egg_factor * ny
        safe = torch.abs(egg_mod) > 0.01
        x_egg = torch.where(safe, nx / egg_mod, nx)
        z_egg = torch.where(safe, z / egg_mod, z)
        return x_egg, z_egg, egg_mod

    @staticmethod
    def _spherical_uv(
        ny: torch.Tensor,
        z_egg: torch.Tensor,
        x_egg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spherical UV mapping.

        Returns:
            u, v in [0, 1]
        """
        phi = torch.acos(torch.clamp(ny, -1.0, 1.0))
        theta = torch.atan2(z_egg, x_egg)
        u = (theta / (2.0 * math.pi) + 0.5) % 1.0
        v = phi / math.pi
        return u, v

    @staticmethod
    def _egg_surface_normal(
        nx: torch.Tensor,
        ny: torch.Tensor,
        z: torch.Tensor,
        egg_factor: float,
    ) -> torch.Tensor:
        """Analytical surface normal for egg geometry.

        Returns:
            (H, W, 3) unit normals
        """
        egg_mod = 1.0 - egg_factor * ny
        surf_normal = torch.stack([nx * egg_mod, ny, z * egg_mod], dim=-1)
        surf_len = torch.norm(surf_normal, dim=-1, keepdim=True)
        surf_len = torch.clamp(surf_len, min=1e-6)
        return surf_normal / surf_len

    # ------------------------------------------------------------------
    # Height-field displacement (same as compute_displaced_normal)
    # ------------------------------------------------------------------

    def _compute_displaced_normal(
        self,
        base_normal: torch.Tensor,
        height_field: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        displacement_scale: float,
    ) -> torch.Tensor:
        """Displace surface normal using height-field gradients.

        Matches texture/render/bump_render.py:compute_displaced_normal.
        """
        th, tw = height_field.shape[0], height_field.shape[1]
        du = 1.0 / tw
        dv = 1.0 / th

        # Central differences in UV space
        h_u_plus = self.sampler(height_field, u + du, v)
        h_u_minus = self.sampler(height_field, u - du, v)
        h_v_plus = self.sampler(height_field, u, v + dv)
        h_v_minus = self.sampler(height_field, u, v - dv)

        dh_du = (h_u_plus - h_u_minus) / (2.0 * du) * displacement_scale
        dh_dv = (h_v_plus - h_v_minus) / (2.0 * dv) * displacement_scale

        # TBN from base_normal
        tangent, bitangent = self._compute_tbn_frame(base_normal)

        # Perturb: n' = normalize(n - dh_du * T - dh_dv * B)
        perturbed = (
            base_normal
            - dh_du.unsqueeze(-1) * tangent
            - dh_dv.unsqueeze(-1) * bitangent
        )
        perturbed_len = torch.norm(perturbed, dim=-1, keepdim=True)
        perturbed = torch.where(
            perturbed_len > 1e-6,
            perturbed / perturbed_len,
            base_normal,
        )
        return perturbed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, texture: torch.Tensor, normal_map: torch.Tensor) -> torch.Tensor:
        """Render the egg surface.

        Args:
            texture: (H, W, 3) RGB texture in [0, 1]
            normal_map: (H, W, 3) normal map from HeightToNormals in [0, 1]

        Returns:
            rendered: (resolution, resolution, 3) float32 image in [0, 1].
                      Background pixels are 0.
        """
        nx = self.nx
        ny = self.ny

        # --- 1. Ray-sphere intersection ---
        inside_mask, z = self._ray_sphere_test(nx, ny)  # (R, R), (R, R)

        # --- 2. Egg deformation ---
        x_egg, z_egg, egg_mod = self._egg_deformation(nx, ny, z, self.egg_factor)

        # --- 3. Spherical UV ---
        u, v = self._spherical_uv(ny, z_egg, x_egg)

        # --- 4. Sample texture and normal map ---
        tex_color = self.sampler(texture, u, v)       # (R, R, 3)
        tex_normal = self.sampler(normal_map, u, v)    # (R, R, 3)

        # --- 5. Surface normal (analytical egg normal) ---
        base_normal = self._egg_surface_normal(nx, ny, z, self.egg_factor)

        # --- 6. Height-field displacement ---
        # Derive a scalar height from the texture for displacement
        # (average of RGB channels serves as height proxy, matching original pipeline)
        height_field_2d = texture.mean(dim=-1)  # (Ht, Wt)
        displaced_normal = self._compute_displaced_normal(
            base_normal, height_field_2d, u, v, displacement_scale=0.1,
        )

        # --- 7. Normal-map perturbation via TBN ---
        # Convert [0,1] encoded normal to [-1,1] tangent-space
        ts_normal = tex_normal * 2.0 - 1.0

        tangent, bitangent = self._compute_tbn_frame(displaced_normal)
        final_normal = self._perturb_normal(
            displaced_normal, tangent, bitangent, ts_normal, self.bump_strength,
        )

        # --- 8. Lighting ---
        # Lambertian diffuse
        ndotl = self._lambertian_diffuse(final_normal, self.light_dir)

        # Blinn-Phong specular
        specular = self._blinn_phong_specular(
            final_normal, self.light_dir, self.view_dir, self.specular_power,
        )

        # Fresnel
        ndotv = torch.clamp(
            torch.sum(final_normal * self.view_dir, dim=-1), min=0.01,
        )
        fresnel = self._schlick_fresnel(ndotv, self.f0)

        # Height-based ambient occlusion
        height_sample = self.sampler(height_field_2d, u, v)
        ao = 0.7 + 0.3 * height_sample

        # Combine: diffuse + specular with Fresnel blend
        diffuse_contrib = tex_color * (ndotl.unsqueeze(-1) + self.ambient)
        specular_contrib = (specular * fresnel).unsqueeze(-1)

        color = (1.0 - fresnel.unsqueeze(-1)) * diffuse_contrib + specular_contrib
        color = color * ao.unsqueeze(-1)

        # Clamp to [0, 1]
        color = torch.clamp(color, 0.0, 1.0)

        # Mask out background (outside sphere)
        color = color * inside_mask.unsqueeze(-1).float()

        return color
