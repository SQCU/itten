"""
Render: PIL-based 2D rendering for lattice mosaic visualization.

Generates PNG images showing:
- Colored tiles for neighborhoods
- Darker boundaries between tiles
- Expansion intensity (brightness)
- Bottleneck markers
"""

import math
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

from .mosaic import LatticeState, coord_key, key_to_coord


def darken(color: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
    """Darken a color by a factor."""
    return tuple(int(c * factor) for c in color)


def lighten(color: Tuple[int, int, int], factor: float = 0.3) -> Tuple[int, int, int]:
    """Lighten a color towards white."""
    return tuple(int(c + (255 - c) * factor) for c in color)


def blend_with_expansion(
    base_color: Tuple[int, int, int],
    expansion: float,
    exp_min: float,
    exp_max: float
) -> Tuple[int, int, int]:
    """
    Adjust color brightness based on expansion value.
    High expansion = brighter, low expansion = darker.
    """
    if exp_max <= exp_min:
        return base_color

    norm = (expansion - exp_min) / (exp_max - exp_min)
    norm = max(0.0, min(1.0, norm))

    # Map to brightness range [0.5, 1.0]
    brightness = 0.5 + 0.5 * norm

    return tuple(int(c * brightness) for c in base_color)


def render_flat_mosaic(
    state: LatticeState,
    cell_size: int = 20,
    border_width: int = 2,
    show_expansion: bool = True,
    show_boundaries: bool = True,
    show_grid: bool = False
) -> Image.Image:
    """
    Render a flat 2D mosaic image.

    Args:
        state: LatticeState with tiles computed
        cell_size: Pixel size of each lattice cell
        border_width: Width of tile boundaries
        show_expansion: Adjust brightness by expansion value
        show_boundaries: Draw dark borders between tiles
        show_grid: Draw light grid between all cells

    Returns:
        PIL Image
    """
    x_min, x_max = state.x_range
    y_min, y_max = state.y_range

    width = (x_max - x_min + 1) * cell_size
    height = (y_max - y_min + 1) * cell_size

    # Create image with dark background
    img = Image.new('RGB', (width, height), (30, 30, 40))
    draw = ImageDraw.Draw(img)

    # Build coordinate to tile mapping
    coord_to_tile = {}
    for tile in state.tiles:
        for node in tile['nodes']:
            coord_to_tile[coord_key(tuple(node))] = tile

    # Get expansion range for normalization
    if state.expansion_map:
        exp_values = list(state.expansion_map.values())
        exp_min = min(exp_values)
        exp_max = max(exp_values)
    else:
        exp_min = exp_max = 0.0

    # Draw each cell
    for coord in state.nodes:
        ck = coord_key(coord)
        x, y = coord

        # Compute pixel coordinates (flip y for image coordinates)
        px = (x - x_min) * cell_size
        py = (y_max - y) * cell_size  # Flip y

        tile = coord_to_tile.get(ck)
        if tile is None:
            continue

        base_color = tuple(tile['color'])

        # Adjust for expansion if enabled
        if show_expansion and ck in state.expansion_map:
            exp_val = state.expansion_map[ck]
            color = blend_with_expansion(base_color, exp_val, exp_min, exp_max)
        else:
            color = base_color

        # Check if this is a boundary node
        is_boundary = any(
            coord_key(tuple(bc)) == ck
            for bc in tile.get('boundary_nodes', [])
        )

        # Draw the cell
        if show_boundaries and is_boundary:
            # Draw with border
            draw.rectangle(
                [px, py, px + cell_size - 1, py + cell_size - 1],
                fill=darken(color, 0.4),
                outline=None
            )
            # Inner rectangle
            draw.rectangle(
                [px + border_width, py + border_width,
                 px + cell_size - 1 - border_width, py + cell_size - 1 - border_width],
                fill=color,
                outline=None
            )
        else:
            draw.rectangle(
                [px, py, px + cell_size - 1, py + cell_size - 1],
                fill=color,
                outline=None
            )

        # Mark bottleneck regions
        if tile.get('is_bottleneck', False):
            # Draw a small marker in the center
            cx = px + cell_size // 2
            cy = py + cell_size // 2
            marker_size = cell_size // 6
            draw.ellipse(
                [cx - marker_size, cy - marker_size,
                 cx + marker_size, cy + marker_size],
                fill=(255, 80, 80)
            )

    # Draw grid if requested
    if show_grid:
        grid_color = (60, 60, 70)
        for x in range(x_min, x_max + 2):
            px = (x - x_min) * cell_size
            draw.line([(px, 0), (px, height)], fill=grid_color, width=1)
        for y in range(y_min, y_max + 2):
            py = (y_max - y + 1) * cell_size
            draw.line([(0, py), (width, py)], fill=grid_color, width=1)

    # Draw period markers at bottleneck x-positions
    for x in range(x_min, x_max + 1):
        if x % state.period == 0:
            px = (x - x_min) * cell_size + cell_size // 2
            # Draw small triangle at top
            draw.polygon(
                [(px - 5, 5), (px + 5, 5), (px, 15)],
                fill=(200, 200, 200)
            )

    return img


def render_torus_projection(
    state: LatticeState,
    width: int = 800,
    height: int = 600,
    rotation: float = 0.3,
    tilt: float = 0.4
) -> Image.Image:
    """
    Render a 3D-projected view of the torus mosaic.

    Uses simple orthographic projection with depth sorting.
    """
    img = Image.new('RGB', (width, height), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    if not state.projected_coords:
        return img

    # Build coordinate to tile mapping
    coord_to_tile = {}
    for tile in state.tiles:
        for node in tile['nodes']:
            coord_to_tile[coord_key(tuple(node))] = tile

    # Rotation matrices
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    # Project and sort by depth
    projected_cells = []

    R = state.surface_params.get("major_radius", 3.0)
    scale = min(width, height) / (2 * (R + 2))

    for coord in state.nodes:
        ck = coord_key(coord)
        if ck not in state.projected_coords:
            continue

        px, py, pz = state.projected_coords[ck]

        # Apply rotation around Z axis
        x2 = px * cos_r - py * sin_r
        y2 = px * sin_r + py * cos_r

        # Apply tilt around X axis
        y3 = y2 * cos_t - pz * sin_t
        z3 = y2 * sin_t + pz * cos_t

        # Orthographic projection
        screen_x = width // 2 + int(x2 * scale)
        screen_y = height // 2 - int(y3 * scale)
        depth = z3

        tile = coord_to_tile.get(ck)
        if tile:
            projected_cells.append((depth, screen_x, screen_y, tile, ck))

    # Sort by depth (back to front)
    projected_cells.sort(key=lambda x: x[0])

    # Get expansion range
    if state.expansion_map:
        exp_values = list(state.expansion_map.values())
        exp_min = min(exp_values)
        exp_max = max(exp_values)
    else:
        exp_min = exp_max = 0.0

    # Draw cells
    cell_size = int(scale * 0.3)  # Approximate cell size

    for depth, sx, sy, tile, ck in projected_cells:
        base_color = tuple(tile['color'])

        # Depth-based shading
        depth_factor = 0.4 + 0.6 * (depth / (R + 2) * 0.5 + 0.5)
        depth_factor = max(0.3, min(1.0, depth_factor))

        # Expansion-based adjustment
        if ck in state.expansion_map:
            exp_val = state.expansion_map[ck]
            color = blend_with_expansion(base_color, exp_val, exp_min, exp_max)
        else:
            color = base_color

        # Apply depth shading
        color = tuple(int(c * depth_factor) for c in color)

        # Check if boundary
        is_boundary = any(
            coord_key(tuple(bc)) == ck
            for bc in tile.get('boundary_nodes', [])
        )

        # Draw as small rectangle/circle
        if is_boundary:
            draw.rectangle(
                [sx - cell_size, sy - cell_size,
                 sx + cell_size, sy + cell_size],
                fill=darken(color, 0.5),
                outline=None
            )
            draw.rectangle(
                [sx - cell_size + 2, sy - cell_size + 2,
                 sx + cell_size - 2, sy + cell_size - 2],
                fill=color,
                outline=None
            )
        else:
            draw.rectangle(
                [sx - cell_size, sy - cell_size,
                 sx + cell_size, sy + cell_size],
                fill=color,
                outline=None
            )

    return img


def render_cylinder_projection(
    state: LatticeState,
    width: int = 800,
    height: int = 400,
    rotation: float = 0.0
) -> Image.Image:
    """
    Render the cylinder projection view.
    """
    # Update state to use cylinder projection
    state.surface_type = "cylinder"

    img = Image.new('RGB', (width, height), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    if not state.projected_coords:
        return img

    coord_to_tile = {}
    for tile in state.tiles:
        for node in tile['nodes']:
            coord_to_tile[coord_key(tuple(node))] = tile

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    R = state.surface_params.get("major_radius", 3.0)
    scale = min(width, height) / (2 * (R + 1))

    projected_cells = []

    for coord in state.nodes:
        ck = coord_key(coord)
        if ck not in state.projected_coords:
            continue

        px, py, pz = state.projected_coords[ck]

        # Rotate around Z axis
        x2 = px * cos_r - py * sin_r
        y2 = px * sin_r + py * cos_r

        screen_x = width // 2 + int(x2 * scale)
        screen_y = height // 2 - int(pz * scale * 2)  # Stretch vertically
        depth = y2

        tile = coord_to_tile.get(ck)
        if tile:
            projected_cells.append((depth, screen_x, screen_y, tile, ck))

    projected_cells.sort(key=lambda x: x[0])

    cell_size = int(scale * 0.25)

    for depth, sx, sy, tile, ck in projected_cells:
        base_color = tuple(tile['color'])
        depth_factor = 0.5 + 0.5 * (depth / R * 0.5 + 0.5)
        depth_factor = max(0.4, min(1.0, depth_factor))

        color = tuple(int(c * depth_factor) for c in base_color)

        draw.rectangle(
            [sx - cell_size, sy - cell_size,
             sx + cell_size, sy + cell_size],
            fill=color,
            outline=None
        )

    return img


def render_combined(
    state: LatticeState,
    mode: str = "flat",
    **kwargs
) -> Image.Image:
    """
    Unified render interface.

    Args:
        state: LatticeState with tiles computed
        mode: "flat", "torus", or "cylinder"
        **kwargs: Passed to specific render function

    Returns:
        PIL Image
    """
    if mode == "flat":
        return render_flat_mosaic(state, **kwargs)
    elif mode == "torus":
        return render_torus_projection(state, **kwargs)
    elif mode == "cylinder":
        return render_cylinder_projection(state, **kwargs)
    else:
        raise ValueError(f"Unknown render mode: {mode}")


def save_render(state: LatticeState, path: str, mode: str = "flat", **kwargs):
    """Render and save to file."""
    img = render_combined(state, mode=mode, **kwargs)
    img.save(path)
    return path
