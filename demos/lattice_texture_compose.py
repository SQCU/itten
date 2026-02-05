#!/usr/bin/env python3
"""
Lattice Texture Composition Demo

Demonstrates that lattice extrusion composes with texture transforms on 3D surfaces.

Key insight: Lattice extrusion creates geometry that "perches" on a textured surface.
The texture transform (controlled by theta) affects:
1. The base surface texture appearance
2. How the lattice geometry visually integrates with the surface

This demo:
1. Creates a lattice pattern (TerritoryGraph with islands and bridges)
2. Applies ExpansionGatedExtruder to create 3D geometry
3. Creates base surface textures using synthesize() with different theta values
4. Maps textures onto the surface
5. Places lattice geometry on top of textured surfaces
6. Renders composed results showing theta's effect on the composition

Output saved to: demo_output/lattice_compose/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import lattice module
from lattice import (
    TerritoryGraph, Island, Bridge, create_islands_and_bridges,
    PinchedLattice,
    ExpansionGatedExtruder, FiedlerAlignedGeometry, ExtrudedNode, ExtrusionState,
    EggSurface, LatticeToMesh, Mesh,
    render_3d_egg_mesh, render_expansion_heatmap, save_render,
    lattice_to_graph
)
from lattice.graph import compute_lattice_expansion_map, compute_lattice_fiedler

# Import texture synthesis
from texture import synthesize, TextureResult

# Import PIL for image operations
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_territory_pattern() -> TerritoryGraph:
    """
    Create a TerritoryGraph with clear spectral variation.

    Islands have high local expansion (lambda_2).
    Bridges have low local expansion (bottlenecks).
    """
    # Create 3 islands connected by narrow bridges
    territory = create_islands_and_bridges(
        num_islands=3,
        island_radius=6,
        bridge_width=2,
        spacing=20
    )

    return territory


def extrude_territory(territory: TerritoryGraph) -> ExtrusionState:
    """
    Apply ExpansionGatedExtruder to create 3D geometry.

    Nodes with high local expansion get extruded to higher layers.
    Nodes in bottleneck regions (bridges) stay at lower layers.
    """
    extruder = ExpansionGatedExtruder(
        territory=territory,
        expansion_threshold=1.2,  # Threshold for extrusion
        lanczos_iterations=15,
        hop_radius=3
    )

    # Compute all expansion values first
    print("  Computing local expansion values...")
    expansions = extruder.compute_all_expansions()

    # Run extrusion process
    print("  Running extrusion...")
    state = extruder.run(max_iterations=10)

    return state


def compute_fiedler_orientations(
    territory: TerritoryGraph,
    state: ExtrusionState
) -> ExtrusionState:
    """
    Compute Fiedler vector to orient geometry.

    The Fiedler gradient gives a natural orientation for each node's geometry.
    """
    print("  Computing Fiedler field...")

    fiedler_aligner = FiedlerAlignedGeometry(
        territory=territory,
        hop_radius=3,
        lanczos_iterations=20
    )

    # Get all node coordinates
    all_coords = list(territory.all_nodes())

    # Compute Fiedler field
    fiedler_field = fiedler_aligner.compute_fiedler_field(all_coords)

    # Update node orientations
    idx = 0
    while idx < len(all_coords):
        coord = all_coords[idx]
        if coord in fiedler_field and coord in state.nodes:
            fiedler_value, gradient = fiedler_field[coord]
            state.nodes[coord].fiedler_value = fiedler_value
            state.nodes[coord].fiedler_gradient = gradient
            state.nodes[coord].geometry_angle = fiedler_aligner.orient_geometry(
                coord, gradient, "triangle"
            )
        idx += 1

    return state


def create_surface_texture(
    carrier: np.ndarray,
    theta: float,
    size: int = 256
) -> TextureResult:
    """
    Create a surface texture using spectral synthesis.

    Args:
        carrier: Carrier pattern for texture structure
        theta: Spectral emphasis parameter [0, 1]
               0 = coarse structure (Fiedler dominates)
               1 = fine structure (higher eigenvectors)
        size: Output size

    Returns:
        TextureResult with height_field and normal_map
    """
    # Use noise as operand for natural variation
    result = synthesize(
        carrier=carrier,
        operand='noise',
        theta=theta,
        gamma=0.3,
        num_eigenvectors=8,
        edge_threshold=0.1,
        output_size=size,
        normal_strength=2.0,
        mode='spectral',
        return_diagnostics=True
    )

    return result


def render_textured_egg_with_lattice(
    nodes: Dict[Tuple[int, int], ExtrudedNode],
    territory: TerritoryGraph,
    texture_result: TextureResult,
    output_size: int = 800,
    egg_factor: float = 0.25
) -> np.ndarray:
    """
    Render lattice on egg surface with texture applied.

    Combines:
    1. Base egg surface with texture color/lighting
    2. Lattice node geometry on top

    The texture affects both the base surface appearance AND how the
    lattice geometry visually integrates.
    """
    # Get height field as base for coloring
    height_field = texture_result.height_field
    normal_map = texture_result.normal_map

    # Normalize light direction
    light = np.array([0.5, 0.7, 0.5], dtype=np.float32)
    light = light / np.linalg.norm(light)

    # Get bounds for UV mapping
    bounds = territory.get_bounds()
    x_min, y_min, x_max, y_max = bounds
    x_range = max(1, x_max - x_min)
    y_range = max(1, y_max - y_min)

    # Create coordinate grids
    y_coords = np.arange(output_size, dtype=np.float32)
    x_coords = np.arange(output_size, dtype=np.float32)
    px, py = np.meshgrid(x_coords, y_coords)

    # Normalized device coordinates
    scale = 0.85
    nx = (px / output_size - 0.5) * 2 / scale
    ny = (0.5 - py / output_size) * 2 / scale

    # Ray-sphere intersection
    r_sq = nx * nx + ny * ny
    inside_mask = r_sq <= 1.0

    # Z coordinate on sphere
    z = np.sqrt(np.maximum(0, 1.0 - r_sq))

    # Egg deformation
    egg_mod = 1.0 - egg_factor * ny

    # Surface normal
    surf_normal = np.stack([nx * egg_mod, ny, z * egg_mod], axis=-1)
    surf_len = np.linalg.norm(surf_normal, axis=-1, keepdims=True)
    surf_len = np.where(surf_len > 1e-6, surf_len, 1.0)
    surf_normal = surf_normal / surf_len

    # UV coordinates for texture sampling
    x_egg = np.where(np.abs(egg_mod) > 0.01, nx / egg_mod, nx)
    z_egg = np.where(np.abs(egg_mod) > 0.01, z / egg_mod, z)

    phi = np.arccos(np.clip(ny, -1, 1))
    u = (np.arctan2(z_egg, x_egg) / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi

    # Sample texture at UV coordinates
    tex_h, tex_w = height_field.shape
    tex_u_idx = (u * (tex_w - 1)).astype(np.int32)
    tex_v_idx = (v * (tex_h - 1)).astype(np.int32)
    tex_u_idx = np.clip(tex_u_idx, 0, tex_w - 1)
    tex_v_idx = np.clip(tex_v_idx, 0, tex_h - 1)

    sampled_height = height_field[tex_v_idx, tex_u_idx]

    # Sample normal map for per-pixel lighting
    sampled_normal_r = normal_map[tex_v_idx, tex_u_idx, 0]
    sampled_normal_g = normal_map[tex_v_idx, tex_u_idx, 1]
    sampled_normal_b = normal_map[tex_v_idx, tex_u_idx, 2]

    # Convert normal from [0,1] to [-1,1]
    tex_normal = np.stack([
        sampled_normal_r * 2 - 1,
        sampled_normal_g * 2 - 1,
        sampled_normal_b * 2 - 1
    ], axis=-1)

    # Combine surface normal with texture normal (tangent space approximation)
    combined_normal = surf_normal + 0.3 * tex_normal
    combined_normal = combined_normal / (np.linalg.norm(combined_normal, axis=-1, keepdims=True) + 1e-6)

    # Lighting with combined normals
    ndotl = np.maximum(0, np.sum(combined_normal * light, axis=-1))
    ambient = 0.25
    diffuse = 0.75 * ndotl
    lighting = ambient + diffuse

    # Base color from texture height (blue-green gradient)
    base_r = 60 + 40 * sampled_height
    base_g = 100 + 100 * sampled_height
    base_b = 180 - 60 * sampled_height

    # Apply lighting to base colors
    color_r = base_r * lighting
    color_g = base_g * lighting
    color_b = base_b * lighting

    # Now render lattice nodes on top
    node_radius = max(2, output_size // (2 * max(x_range, y_range) + 10))
    radius_sq = node_radius * node_radius

    # Build arrays of node data
    node_list = list(nodes.items())
    n_nodes = len(node_list)

    if n_nodes > 0:
        coords = np.array([item[0] for item in node_list], dtype=np.float32)
        expansions = np.array([item[1].expansion for item in node_list], dtype=np.float32)
        layers = np.array([item[1].layer for item in node_list], dtype=np.float32)

        # Compute UV for all nodes
        node_u = (coords[:, 0] - x_min) / x_range if x_range > 0 else np.full(n_nodes, 0.5)
        node_v = 1.0 - (coords[:, 1] - y_min) / y_range if y_range > 0 else np.full(n_nodes, 0.5)

        # Compute 3D positions on egg
        theta_node = node_u * 2 * np.pi - np.pi
        phi_node = node_v * np.pi

        sin_phi = np.sin(phi_node)
        cos_phi = np.cos(phi_node)

        px_3d = sin_phi * np.cos(theta_node)
        py_3d = cos_phi
        pz_3d = sin_phi * np.sin(theta_node)

        # Apply egg deformation and layer offset (extrusion height)
        egg_mod_node = 1.0 - egg_factor * py_3d
        layer_offset = 1.0 + layers * 0.03  # Extruded nodes sit higher
        px_3d = px_3d * egg_mod_node * layer_offset
        pz_3d = pz_3d * egg_mod_node * layer_offset

        # Screen coordinates
        screen_x = ((px_3d * scale / 2 + 0.5) * output_size).astype(np.int32)
        screen_y = ((0.5 - py_3d * scale / 2) * output_size).astype(np.int32)

        # Visibility mask
        visible = pz_3d > 0.01

        # Determine colors based on island/bridge membership
        is_island = np.array([territory.is_on_island(tuple(c)) for c in coords], dtype=bool)
        is_bridge = np.array([territory.is_on_bridge(tuple(c)) for c in coords], dtype=bool)

        # Compute base colors
        exp_factor = np.minimum(1.0, expansions / 2.5)
        layer_factor = np.minimum(1.0, layers / 3.0)

        # Island colors: green tones, brighter with expansion
        island_r = 40 + 30 * layer_factor
        island_g = 180 + 60 * exp_factor
        island_b = 100 - 30 * exp_factor

        # Bridge colors: orange tones
        bridge_r = 220 - 40 * exp_factor
        bridge_g = 140 + 40 * exp_factor
        bridge_b = 60 + 20 * layer_factor

        # Neutral colors
        neutral_r = np.full(n_nodes, 150.0, dtype=np.float32)
        neutral_g = np.full(n_nodes, 150.0, dtype=np.float32)
        neutral_b = np.full(n_nodes, 150.0, dtype=np.float32)

        # Select colors
        node_r = np.where(is_island, island_r, np.where(is_bridge, bridge_r, neutral_r))
        node_g = np.where(is_island, island_g, np.where(is_bridge, bridge_g, neutral_g))
        node_b = np.where(is_island, island_b, np.where(is_bridge, bridge_b, neutral_b))

        # Create disc kernel
        disc_y, disc_x = np.meshgrid(
            np.arange(-node_radius, node_radius + 1),
            np.arange(-node_radius, node_radius + 1),
            indexing='ij'
        )
        disc_mask = (disc_x * disc_x + disc_y * disc_y) <= radius_sq
        disc_offsets = np.stack([disc_y[disc_mask], disc_x[disc_mask]], axis=1)

        # Draw nodes (sorted by z for proper occlusion)
        z_order = np.argsort(-pz_3d)  # Back to front

        i = 0
        while i < n_nodes:
            node_idx = z_order[i]
            if visible[node_idx]:
                sx, sy = screen_x[node_idx], screen_y[node_idx]
                if 0 <= sx < output_size and 0 <= sy < output_size:
                    pixel_y = sy + disc_offsets[:, 0]
                    pixel_x = sx + disc_offsets[:, 1]

                    valid = (pixel_x >= 0) & (pixel_x < output_size) & (pixel_y >= 0) & (pixel_y < output_size)
                    pixel_y = pixel_y[valid]
                    pixel_x = pixel_x[valid]

                    in_egg = inside_mask[pixel_y, pixel_x]
                    pixel_y = pixel_y[in_egg]
                    pixel_x = pixel_x[in_egg]

                    # Apply node lighting based on layer (higher = brighter)
                    node_lighting = 0.8 + 0.2 * layer_factor[node_idx]

                    color_r[pixel_y, pixel_x] = node_r[node_idx] * node_lighting
                    color_g[pixel_y, pixel_x] = node_g[node_idx] * node_lighting
                    color_b[pixel_y, pixel_x] = node_b[node_idx] * node_lighting
            i += 1

    # Create output image
    img = np.full((output_size, output_size, 3), [30, 30, 45], dtype=np.uint8)
    img[inside_mask, 0] = np.clip(color_r[inside_mask], 0, 255).astype(np.uint8)
    img[inside_mask, 1] = np.clip(color_g[inside_mask], 0, 255).astype(np.uint8)
    img[inside_mask, 2] = np.clip(color_b[inside_mask], 0, 255).astype(np.uint8)

    return img


def render_texture_only(
    texture_result: TextureResult,
    output_size: int = 400
) -> np.ndarray:
    """
    Render texture as a flat 2D image for comparison.
    """
    height_field = texture_result.height_field

    # Resize to output size
    from scipy.ndimage import zoom
    zoom_factor = output_size / height_field.shape[0]
    resized = zoom(height_field, zoom_factor, order=1)

    # Convert to RGB with blue-green gradient
    r = (60 + 80 * resized).clip(0, 255).astype(np.uint8)
    g = (100 + 120 * resized).clip(0, 255).astype(np.uint8)
    b = (180 - 80 * resized).clip(0, 255).astype(np.uint8)

    img = np.stack([r, g, b], axis=-1)
    return img


def create_carrier_from_territory(
    territory: TerritoryGraph,
    size: int = 256
) -> np.ndarray:
    """
    Create a carrier pattern from territory structure.

    Islands = 1.0 (high intensity)
    Bridges = 0.5 (medium intensity)
    Background = 0.0 (low intensity)
    """
    bounds = territory.get_bounds()
    x_min, y_min, x_max, y_max = bounds

    # Add margin
    margin = 5
    width = x_max - x_min + 1 + 2 * margin
    height = y_max - y_min + 1 + 2 * margin

    # Create small carrier
    carrier_small = np.zeros((height, width), dtype=np.float32)

    all_nodes = territory.all_nodes()
    node_iter = iter(all_nodes)
    done = False
    while not done:
        try:
            coord = next(node_iter)
            px = coord[0] - x_min + margin
            py = coord[1] - y_min + margin

            if 0 <= px < width and 0 <= py < height:
                if territory.is_on_island(coord):
                    carrier_small[py, px] = 1.0
                elif territory.is_on_bridge(coord):
                    carrier_small[py, px] = 0.5
                else:
                    carrier_small[py, px] = 0.3
        except StopIteration:
            done = True

    # Resize to target size
    from scipy.ndimage import zoom
    zoom_factor = size / max(width, height)
    carrier = zoom(carrier_small, zoom_factor, order=1)

    # Crop or pad to exact size
    if carrier.shape[0] > size:
        carrier = carrier[:size, :]
    if carrier.shape[1] > size:
        carrier = carrier[:, :size]
    if carrier.shape[0] < size or carrier.shape[1] < size:
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:carrier.shape[0], :carrier.shape[1]] = carrier
        carrier = padded

    return carrier


def run_demo():
    """Run the lattice texture composition demo."""
    print("=" * 70)
    print("LATTICE TEXTURE COMPOSITION DEMO")
    print("=" * 70)
    print("\nDemonstrating that lattice extrusion composes with texture transforms")
    print("on 3D surfaces.\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "demo_output" / "lattice_compose"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Step 1: Create lattice pattern
    print("\n" + "-" * 50)
    print("STEP 1: Creating lattice pattern (TerritoryGraph)")
    print("-" * 50)

    territory = create_territory_pattern()
    n_nodes = len(territory.all_nodes())
    n_islands = len(territory.islands)
    n_bridges = len(territory.bridges)

    print(f"  Created territory with {n_nodes} nodes")
    print(f"  Islands: {n_islands}")
    print(f"  Bridges: {n_bridges}")

    # Step 2: Apply extrusion
    print("\n" + "-" * 50)
    print("STEP 2: Applying ExpansionGatedExtruder")
    print("-" * 50)

    state = extrude_territory(territory)

    # Count extrusion layers
    layer_counts = {}
    node_iter = iter(state.nodes.values())
    done = False
    while not done:
        try:
            node = next(node_iter)
            layer = node.layer
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
        except StopIteration:
            done = True

    print(f"  Extrusion complete. Layer distribution:")
    for layer in sorted(layer_counts.keys()):
        print(f"    Layer {layer}: {layer_counts[layer]} nodes")

    # Compute Fiedler orientations
    state = compute_fiedler_orientations(territory, state)
    print("  Fiedler orientations computed.")

    # Step 3: Create carrier pattern from territory
    print("\n" + "-" * 50)
    print("STEP 3: Creating carrier pattern for texture synthesis")
    print("-" * 50)

    carrier = create_carrier_from_territory(territory, size=256)
    print(f"  Carrier shape: {carrier.shape}")
    print(f"  Carrier range: [{carrier.min():.2f}, {carrier.max():.2f}]")

    # Save carrier visualization
    carrier_img = (carrier * 255).clip(0, 255).astype(np.uint8)
    carrier_rgb = np.stack([carrier_img, carrier_img, carrier_img], axis=-1)
    save_render(carrier_rgb, str(output_dir / "00_carrier_pattern.png"))
    print(f"  Saved: 00_carrier_pattern.png")

    # Step 4: Test with different theta values
    print("\n" + "-" * 50)
    print("STEP 4: Testing composition with different theta values")
    print("-" * 50)

    theta_values = [0.1, 0.5, 0.9]
    results = {}

    for theta in theta_values:
        print(f"\n  Processing theta = {theta}:")

        # Create texture with this theta
        print(f"    Synthesizing texture...")
        texture_result = create_surface_texture(carrier, theta=theta, size=256)

        print(f"    Height field stats:")
        print(f"      Mean: {texture_result.metadata['height_mean']:.3f}")
        print(f"      Std: {texture_result.metadata['height_std']:.3f}")

        # Save texture preview
        texture_img = render_texture_only(texture_result, output_size=400)
        save_render(texture_img, str(output_dir / f"01_texture_theta_{theta:.1f}.png"))
        print(f"    Saved: 01_texture_theta_{theta:.1f}.png")

        # Render composed result
        print(f"    Rendering composed egg surface...")
        composed_img = render_textured_egg_with_lattice(
            state.nodes,
            territory,
            texture_result,
            output_size=800,
            egg_factor=0.25
        )
        save_render(composed_img, str(output_dir / f"02_composed_theta_{theta:.1f}.png"))
        print(f"    Saved: 02_composed_theta_{theta:.1f}.png")

        results[theta] = {
            'texture': texture_result,
            'composed_img': composed_img
        }

    # Step 5: Save comparison images
    print("\n" + "-" * 50)
    print("STEP 5: Creating comparison images")
    print("-" * 50)

    # Create side-by-side texture comparison
    if HAS_PIL:
        texture_imgs = []
        for theta in theta_values:
            img = render_texture_only(results[theta]['texture'], output_size=300)
            texture_imgs.append(Image.fromarray(img))

        # Combine horizontally
        total_width = sum(img.width for img in texture_imgs) + 20 * (len(texture_imgs) - 1)
        comparison = Image.new('RGB', (total_width, 300 + 40), color=(30, 30, 45))

        x_offset = 0
        for i, (theta, img) in enumerate(zip(theta_values, texture_imgs)):
            comparison.paste(img, (x_offset, 0))
            x_offset += img.width + 20

        comparison.save(str(output_dir / "03_texture_comparison.png"))
        print(f"  Saved: 03_texture_comparison.png")

        # Create side-by-side composed comparison
        composed_imgs = []
        for theta in theta_values:
            img = Image.fromarray(results[theta]['composed_img'])
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            composed_imgs.append(img)

        total_width = sum(img.width for img in composed_imgs) + 20 * (len(composed_imgs) - 1)
        comparison = Image.new('RGB', (total_width, 400), color=(30, 30, 45))

        x_offset = 0
        for img in composed_imgs:
            comparison.paste(img, (x_offset, 0))
            x_offset += img.width + 20

        comparison.save(str(output_dir / "04_composed_comparison.png"))
        print(f"  Saved: 04_composed_comparison.png")

    # Save expansion heatmap
    print("\n  Saving expansion heatmap...")
    heatmap_img = render_expansion_heatmap(state.nodes, territory, output_size=600)
    save_render(heatmap_img, str(output_dir / "05_expansion_heatmap.png"))
    print(f"  Saved: 05_expansion_heatmap.png")

    # Save bare lattice on egg (no texture, for comparison)
    print("\n  Saving lattice without texture...")
    bare_img = render_3d_egg_mesh(state.nodes, territory, output_size=800)
    save_render(bare_img, str(output_dir / "06_bare_lattice.png"))
    print(f"  Saved: 06_bare_lattice.png")

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\nKey observations:")
    print("-" * 50)
    print("1. Lattice extrusion creates 3D geometry based on local spectral expansion")
    print("   - Islands (high expansion) extrude to higher layers")
    print("   - Bridges (low expansion) remain at lower layers")
    print("")
    print("2. Texture transform (theta) affects the composition:")
    print("   - theta=0.1: Coarse structure, Fiedler-dominated nodal lines")
    print("   - theta=0.5: Balanced structure, mixed eigenvector influence")
    print("   - theta=0.9: Fine structure, higher eigenvectors dominate")
    print("")
    print("3. The composition shows:")
    print("   - Base surface texture coloring changes with theta")
    print("   - Lattice geometry (colored nodes) sits on top of texture")
    print("   - Node brightness reflects extrusion layer (height)")
    print("   - Visual integration differs based on theta-controlled texture")

    print(f"\nOutput files saved to: {output_dir}")
    print("\nFiles:")
    print("  00_carrier_pattern.png      - Territory pattern as carrier")
    print("  01_texture_theta_*.png      - Synthesized textures at different theta")
    print("  02_composed_theta_*.png     - Composed lattice + texture renders")
    print("  03_texture_comparison.png   - Side-by-side texture comparison")
    print("  04_composed_comparison.png  - Side-by-side composed comparison")
    print("  05_expansion_heatmap.png    - Local expansion values")
    print("  06_bare_lattice.png         - Lattice without texture (baseline)")

    return results


if __name__ == "__main__":
    run_demo()
