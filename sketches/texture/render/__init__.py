"""
Render module for texture synthesis.

Provides PBR rendering, UV mapping, and render trace tracking.
"""

from .pbr import render_mesh, render_mesh_dichromatic, height_to_normals
from .trace import RenderTrace
from .lighting import (
    lambertian_diffuse,
    blinn_phong_specular,
    schlick_fresnel,
    ward_anisotropic_specular,
    compute_tbn_frame,
    perturb_normal,
    iridescence_color,
)
from .ray_cast import (
    ray_sphere_test,
    egg_deformation,
    spherical_uv_from_ray,
)
from .bump_render import (
    render_bumped_egg,
    render_color_only_egg,
    create_comparison_grid,
)

__all__ = [
    'render_mesh', 'render_mesh_dichromatic', 'height_to_normals',
    'RenderTrace',
    # Lighting functions
    'lambertian_diffuse',
    'blinn_phong_specular',
    'schlick_fresnel',
    'ward_anisotropic_specular',
    'compute_tbn_frame',
    'perturb_normal',
    'iridescence_color',
    # Ray casting functions
    'ray_sphere_test',
    'egg_deformation',
    'spherical_uv_from_ray',
    # Bump mapping functions
    'render_bumped_egg',
    'render_color_only_egg',
    'create_comparison_grid',
]
