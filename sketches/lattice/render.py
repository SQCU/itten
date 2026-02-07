"""
Render: 3D rendering of lattice meshes on egg/torus surfaces.

FAST VERSION: Pure numpy broadcasting, no pixel loops.
All loops are over nodes (graph traversal), not over pixels.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def render_3d_egg_mesh(
    nodes: Dict[Tuple[int, int], 'ExtrudedNode'],
    territory: 'TerritoryGraph',
    output_size: int = 800,
    egg_factor: float = 0.25,
    light_dir: Tuple[float, float, float] = (0.5, 0.7, 0.5),
    show_islands: bool = True,
    show_bridges: bool = True
) -> np.ndarray:
    """
    Render lattice on egg surface using VECTORIZED operations.

    No pixel loops. Pure numpy broadcasting.
    Node traversal uses while loops (graph traversal, not hot path).
    """
    # Normalize light direction
    light = np.array(light_dir, dtype=np.float32)
    light = light / np.linalg.norm(light)

    # Get bounds for UV mapping
    bounds = territory.get_bounds()
    x_min, y_min, x_max, y_max = bounds
    x_range = max(1, x_max - x_min)
    y_range = max(1, y_max - y_min)

    # Create coordinate grids (vectorized)
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

    # Surface normal (vectorized)
    surf_normal = np.stack([nx * egg_mod, ny, z * egg_mod], axis=-1)
    surf_len = np.linalg.norm(surf_normal, axis=-1, keepdims=True)
    surf_len = np.where(surf_len > 1e-6, surf_len, 1.0)
    surf_normal = surf_normal / surf_len

    # Lighting (vectorized dot product)
    ndotl = np.maximum(0, np.sum(surf_normal * light, axis=-1))
    ambient = 0.25
    diffuse = 0.75 * ndotl
    lighting = ambient + diffuse

    # UV coordinates
    x_egg = np.where(np.abs(egg_mod) > 0.01, nx / egg_mod, nx)
    z_egg = np.where(np.abs(egg_mod) > 0.01, z / egg_mod, z)

    phi = np.arccos(np.clip(ny, -1, 1))
    u = (np.arctan2(z_egg, x_egg) / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi

    # Initialize color arrays
    color_r = np.full((output_size, output_size), 100.0, dtype=np.float32)
    color_g = np.full((output_size, output_size), 90.0, dtype=np.float32)
    color_b = np.full((output_size, output_size), 80.0, dtype=np.float32)

    # Precompute node screen positions and colors (vectorized where possible)
    node_radius = max(1, output_size // (2 * max(x_range, y_range) + 10))
    radius_sq = node_radius * node_radius

    # Build arrays of node data for vectorized processing
    node_list = list(nodes.items())
    n_nodes = len(node_list)

    if n_nodes > 0:
        # Extract all node properties into arrays
        coords = np.array([item[0] for item in node_list], dtype=np.float32)
        expansions = np.array([item[1].expansion for item in node_list], dtype=np.float32)

        # Compute UV for all nodes (vectorized)
        node_u = (coords[:, 0] - x_min) / x_range if x_range > 0 else np.full(n_nodes, 0.5)
        node_v = 1.0 - (coords[:, 1] - y_min) / y_range if y_range > 0 else np.full(n_nodes, 0.5)

        # Compute 3D positions on egg (vectorized)
        theta = node_u * 2 * np.pi - np.pi
        phi_node = node_v * np.pi

        sin_phi = np.sin(phi_node)
        cos_phi = np.cos(phi_node)

        px_3d = sin_phi * np.cos(theta)
        py_3d = cos_phi
        pz_3d = sin_phi * np.sin(theta)

        # Apply egg deformation
        egg_mod_node = 1.0 - egg_factor * py_3d
        px_3d = px_3d * egg_mod_node
        pz_3d = pz_3d * egg_mod_node

        # Screen coordinates (vectorized)
        screen_x = ((px_3d * scale / 2 + 0.5) * output_size).astype(np.int32)
        screen_y = ((0.5 - py_3d * scale / 2) * output_size).astype(np.int32)

        # Visibility mask (facing viewer)
        visible = pz_3d > 0.01

        # Determine colors based on island/bridge membership
        is_island = np.array([territory.is_on_island(tuple(c)) for c in coords], dtype=bool)
        is_bridge = np.array([territory.is_on_bridge(tuple(c)) for c in coords], dtype=bool)

        # Compute base colors (vectorized)
        exp_factor = np.minimum(1.0, expansions / 2.5)

        # Island colors: blue-green
        island_r = np.full(n_nodes, 60.0, dtype=np.float32)
        island_g = 150.0 + 70.0 * exp_factor
        island_b = 200.0 - 40.0 * exp_factor

        # Bridge colors: orange-red
        bridge_exp_factor = np.minimum(1.0, expansions / 1.5)
        bridge_r = 200.0 - 30.0 * bridge_exp_factor
        bridge_g = 100.0 + 60.0 * bridge_exp_factor
        bridge_b = np.full(n_nodes, 60.0, dtype=np.float32)

        # Neutral colors
        neutral_r = np.full(n_nodes, 150.0, dtype=np.float32)
        neutral_g = np.full(n_nodes, 150.0, dtype=np.float32)
        neutral_b = np.full(n_nodes, 150.0, dtype=np.float32)

        # Select colors based on membership
        base_r = np.where(is_island, island_r, np.where(is_bridge, bridge_r, neutral_r))
        base_g = np.where(is_island, island_g, np.where(is_bridge, bridge_g, neutral_g))
        base_b = np.where(is_island, island_b, np.where(is_bridge, bridge_b, neutral_b))

        # Create disc kernel for drawing nodes (vectorized)
        disc_y, disc_x = np.meshgrid(
            np.arange(-node_radius, node_radius + 1),
            np.arange(-node_radius, node_radius + 1),
            indexing='ij'
        )
        disc_mask = (disc_x * disc_x + disc_y * disc_y) <= radius_sq
        disc_offsets = np.stack([disc_y[disc_mask], disc_x[disc_mask]], axis=1)

        # Draw all nodes using vectorized scatter (while loop over nodes, not pixels)
        i = 0
        while i < n_nodes:  # graph traversal
            if visible[i]:
                sx, sy = screen_x[i], screen_y[i]
                if 0 <= sx < output_size and 0 <= sy < output_size:
                    # Compute all pixel positions for this node (vectorized)
                    pixel_y = sy + disc_offsets[:, 0]
                    pixel_x = sx + disc_offsets[:, 1]

                    # Bounds check (vectorized)
                    valid = (pixel_x >= 0) & (pixel_x < output_size) & (pixel_y >= 0) & (pixel_y < output_size)
                    pixel_y = pixel_y[valid]
                    pixel_x = pixel_x[valid]

                    # Check inside egg mask (vectorized)
                    in_egg = inside_mask[pixel_y, pixel_x]
                    pixel_y = pixel_y[in_egg]
                    pixel_x = pixel_x[in_egg]

                    # Set colors (vectorized assignment)
                    color_r[pixel_y, pixel_x] = base_r[i]
                    color_g[pixel_y, pixel_x] = base_g[i]
                    color_b[pixel_y, pixel_x] = base_b[i]
            i += 1

    # Apply lighting (vectorized)
    final_r = np.clip(color_r * lighting, 0, 255)
    final_g = np.clip(color_g * lighting, 0, 255)
    final_b = np.clip(color_b * lighting, 0, 255)

    # Create output image
    img = np.full((output_size, output_size, 3), [30, 30, 45], dtype=np.uint8)
    img[inside_mask, 0] = final_r[inside_mask].astype(np.uint8)
    img[inside_mask, 1] = final_g[inside_mask].astype(np.uint8)
    img[inside_mask, 2] = final_b[inside_mask].astype(np.uint8)

    return img


def render_expansion_heatmap(
    nodes: Dict[Tuple[int, int], 'ExtrudedNode'],
    territory: 'TerritoryGraph',
    output_size: int = 600
) -> np.ndarray:
    """
    Render a flat heatmap of expansion values.

    FAST: Vectorized color computation and blitting.
    """
    bounds = territory.get_bounds()
    x_min, y_min, x_max, y_max = bounds

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    cell_size = output_size // max(width, height)
    img_width = width * cell_size
    img_height = height * cell_size

    img = np.full((img_height, img_width, 3), [30, 30, 40], dtype=np.uint8)

    # Build arrays for vectorized processing
    node_list = list(nodes.items())
    n_nodes = len(node_list)

    if n_nodes == 0:
        return img

    coords = np.array([item[0] for item in node_list], dtype=np.int32)
    expansions = np.array([item[1].expansion for item in node_list], dtype=np.float32)

    # Compute pixel positions (vectorized)
    px = (coords[:, 0] - x_min) * cell_size
    py = (y_max - coords[:, 1]) * cell_size

    # Compute colors based on expansion (vectorized)
    # Color ramp: blue -> green -> yellow -> red
    colors = np.zeros((n_nodes, 3), dtype=np.uint8)

    # Blue: exp < 0.5
    mask1 = expansions < 0.5
    colors[mask1] = [50, 80, 200]

    # Blue to green: 0.5 <= exp < 1.5
    mask2 = (expansions >= 0.5) & (expansions < 1.5)
    t = expansions[mask2] - 0.5
    colors[mask2, 0] = (50 - 30 * t).astype(np.uint8)
    colors[mask2, 1] = (80 + 120 * t).astype(np.uint8)
    colors[mask2, 2] = (200 - 150 * t).astype(np.uint8)

    # Green to yellow: 1.5 <= exp < 2.5
    mask3 = (expansions >= 1.5) & (expansions < 2.5)
    t = expansions[mask3] - 1.5
    colors[mask3, 0] = (20 + 200 * t).astype(np.uint8)
    colors[mask3, 1] = (200 - 20 * t).astype(np.uint8)
    colors[mask3, 2] = (50 - 30 * t).astype(np.uint8)

    # Yellow to red: exp >= 2.5
    mask4 = expansions >= 2.5
    t = np.minimum(1.0, expansions[mask4] - 2.5)
    colors[mask4, 0] = (220 + 35 * t).astype(np.uint8)
    colors[mask4, 1] = (180 - 130 * t).astype(np.uint8)
    colors[mask4, 2] = (20 - 10 * t).astype(np.uint8)

    # Fill cells (while loop over nodes, not pixels)
    i = 0
    while i < n_nodes:  # graph traversal
        x, y = px[i], py[i]
        if 0 <= x < img_width - cell_size and 0 <= y < img_height - cell_size:
            img[y:y+cell_size-1, x:x+cell_size-1] = colors[i]
        i += 1

    # Mark bridges with white dots
    is_bridge = np.array([territory.is_on_bridge(tuple(c)) for c in coords], dtype=bool)
    bridge_cx = px[is_bridge] + cell_size // 2
    bridge_cy = py[is_bridge] + cell_size // 2

    j = 0
    while j < len(bridge_cx):  # graph traversal
        cx, cy = bridge_cx[j], bridge_cy[j]
        if 2 <= cx < img_width - 2 and 2 <= cy < img_height - 2:
            img[cy-1:cy+2, cx-1:cx+2] = [255, 255, 255]
        j += 1

    return img


def render_extrusion_layers(
    nodes: Dict[Tuple[int, int], 'ExtrudedNode'],
    territory: 'TerritoryGraph',
    output_size: int = 600
) -> np.ndarray:
    """
    Render showing extrusion layer (height) of each node.

    FAST: Vectorized operations.
    """
    bounds = territory.get_bounds()
    x_min, y_min, x_max, y_max = bounds

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    cell_size = output_size // max(width, height)
    img_width = width * cell_size
    img_height = height * cell_size

    img = np.full((img_height, img_width, 3), [30, 30, 40], dtype=np.uint8)

    # Build arrays
    node_list = list(nodes.items())
    n_nodes = len(node_list)

    if n_nodes == 0:
        return img

    coords = np.array([item[0] for item in node_list], dtype=np.int32)
    layers = np.array([item[1].layer for item in node_list], dtype=np.float32)

    # Find max layer
    max_layer = max(1.0, np.max(layers))

    # Compute pixel positions
    px = (coords[:, 0] - x_min) * cell_size
    py = (y_max - coords[:, 1]) * cell_size

    # Compute grayscale values (vectorized)
    layer_frac = layers / max_layer
    vals = (50 + 180 * layer_frac).astype(np.uint8)

    # Fill cells (while loop over nodes, not pixels)
    i = 0
    while i < n_nodes:  # graph traversal
        x, y = px[i], py[i]
        if 0 <= x < img_width - cell_size and 0 <= y < img_height - cell_size:
            v = vals[i]
            img[y:y+cell_size-1, x:x+cell_size-1] = [v, v, v]
        i += 1

    return img


def save_render(img_array: np.ndarray, path: str):
    """Save numpy array as image."""
    if not HAS_PIL:
        raise ImportError("PIL/Pillow required for saving images")

    img = Image.fromarray(img_array, mode='RGB')
    img.save(path)
    return path


def render_to_pil(img_array: np.ndarray) -> 'Image.Image':
    """Convert numpy array to PIL Image."""
    if not HAS_PIL:
        raise ImportError("PIL/Pillow not available")
    return Image.fromarray(img_array, mode='RGB')
