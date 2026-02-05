"""
Pattern generators for texture synthesis.

Provides vectorized pattern generation for carriers and operands.
All functions return (H, W) arrays normalized to [0, 1].

Includes CG-derived operand generators that create operands with
spatial structure from natural images - unlike uniform noise which
produces "mushy" blends with no features from the operand.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import sobel, grey_dilation, grey_erosion
from typing import Optional, Callable, List, Tuple


def generate_amongus(size: int = 64) -> np.ndarray:
    """
    Generate an Among Us character silhouette (vectorized).

    Args:
        size: Output bitmap size (square)

    Returns:
        2D array (size, size) with values 0.0 (background) or 1.0 (character)
    """
    bitmap = np.zeros((size, size))

    # Scale factor relative to reference size 16
    s = size / 16.0

    def sc(v):
        return int(v * s)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:size, 0:size]

    # Body (main oval)
    cy, cx = sc(8), sc(8)
    ry, rx = sc(5), sc(4)
    dy = (y_coords - cy) / max(ry, 1)
    dx = (x_coords - cx) / max(rx, 1)
    body_mask = (dy*dy + dx*dx < 1.0) & (y_coords >= sc(3)) & (y_coords < sc(13)) & (x_coords >= sc(4)) & (x_coords < sc(12))
    bitmap[body_mask] = 1.0

    # Visor (rectangular bump)
    visor_mask = (y_coords >= sc(4)) & (y_coords < sc(8)) & (x_coords >= sc(8)) & (x_coords < min(sc(14), size))
    bitmap[visor_mask] = 1.0

    # Backpack
    backpack_mask = (y_coords >= sc(5)) & (y_coords < sc(11)) & (x_coords >= sc(1)) & (x_coords < sc(5))
    bitmap[backpack_mask] = 1.0

    # Legs (two small rectangles)
    leg1_mask = (y_coords >= sc(12)) & (y_coords < min(sc(15), size)) & (x_coords >= sc(4)) & (x_coords < sc(7))
    leg2_mask = (y_coords >= sc(12)) & (y_coords < min(sc(15), size)) & (x_coords >= sc(9)) & (x_coords < sc(12))
    bitmap[leg1_mask] = 1.0
    bitmap[leg2_mask] = 1.0

    return bitmap.astype(np.float32)


def generate_checkerboard(size: int = 64, tile_size: int = 8) -> np.ndarray:
    """
    Generate checkerboard pattern (vectorized).

    Args:
        size: Output bitmap size (square)
        tile_size: Size of each checker square

    Returns:
        2D array with checkerboard (0.0 and 1.0 values)
    """
    y_coords, x_coords = np.mgrid[0:size, 0:size]
    tile_x = x_coords // tile_size
    tile_y = y_coords // tile_size
    bitmap = ((tile_x + tile_y) % 2 == 0).astype(np.float32)

    return bitmap


def generate_noise(
    size: int = 64,
    scale: float = 4.0,
    seed: int = 42,
    octaves: int = 4
) -> np.ndarray:
    """
    Generate multi-octave noise pattern (Perlin-like).

    Args:
        size: Output image size
        scale: Base frequency scale
        seed: Random seed
        octaves: Number of noise octaves

    Returns:
        2D array normalized to [0, 1]
    """
    np.random.seed(seed)

    result = np.zeros((size, size), dtype=np.float32)

    for octave in range(octaves):
        freq = scale * (2 ** octave)
        amplitude = 1.0 / (2 ** octave)

        # Simple value noise
        grid_size = max(2, int(size / freq))
        noise_grid = np.random.randn(grid_size, grid_size)

        # Interpolate to full size
        y_coords = np.linspace(0, grid_size - 1, size)
        x_coords = np.linspace(0, grid_size - 1, size)

        from scipy.ndimage import map_coordinates
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.array([yy, xx])

        interpolated = map_coordinates(noise_grid, coords, order=1, mode='wrap')
        result += amplitude * interpolated

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result


def generate_dragon_curve(
    size: int = 64,
    iterations: int = 10,
    line_width: float = 2.0
) -> np.ndarray:
    """
    Generate dragon curve pattern using L-system.

    L-system: F -> F+G, G -> F-G

    Args:
        size: Output image size
        iterations: L-system iterations
        line_width: Width of the curve in pixels

    Returns:
        2D array with dragon curve pattern
    """
    # Generate dragon curve points
    points = _dragon_curve_points(iterations)

    # Rasterize to image
    return _rasterize_curve(points, size, line_width)


def _dragon_curve_points(iterations: int) -> List[Tuple[float, float]]:
    """Generate dragon curve points using L-system."""
    # L-system rules
    # F -> F+G (turn right, draw F, turn left, draw G)
    # G -> F-G (turn right, draw F, turn right, draw G)

    # Simplified: just generate the turn sequence
    # Then convert to points

    # Generate turn sequence
    turns = []
    for _ in range(iterations):
        new_turns = []
        for t in turns:
            new_turns.append(t)
        new_turns.append(1)  # Right turn
        for t in reversed(turns):
            new_turns.append(-t if t != 0 else 1)
        turns = new_turns

    # Convert turns to points
    x, y = 0.0, 0.0
    dx, dy = 1.0, 0.0
    points = [(x, y)]

    for turn in turns:
        # Move forward
        x += dx
        y += dy
        points.append((x, y))

        # Turn
        if turn == 1:  # Right
            dx, dy = dy, -dx
        elif turn == -1:  # Left
            dx, dy = -dy, dx

    # One more forward move
    x += dx
    y += dy
    points.append((x, y))

    return points


def _rasterize_curve(
    points: List[Tuple[float, float]],
    size: int,
    line_width: float
) -> np.ndarray:
    """Rasterize curve points to image."""
    if not points:
        return np.zeros((size, size), dtype=np.float32)

    # Convert to numpy array
    pts = np.array(points)

    # Normalize to [0, size-1] with padding
    padding = 0.1 * size
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)
    scale = (size - 2 * padding) / max(x_range, y_range)

    pts_norm = (pts - [x_min, y_min]) * scale + padding

    # Create image
    image = np.zeros((size, size), dtype=np.float32)

    # Draw line segments
    y_coords, x_coords = np.mgrid[0:size, 0:size]

    for i in range(len(pts_norm) - 1):
        p1 = pts_norm[i]
        p2 = pts_norm[i + 1]

        # Distance from each pixel to line segment
        dist = _point_to_segment_distance(x_coords, y_coords, p1, p2)

        # Anti-aliased line
        line_mask = np.clip(1.0 - dist / line_width, 0, 1)
        image = np.maximum(image, line_mask)

    return image


def _point_to_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> np.ndarray:
    """Compute distance from points to line segment."""
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # Project point onto line
    if dx * dx + dy * dy < 1e-10:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    t = np.clip(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy), 0, 1)

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def generate_gradient(
    size: int = 64,
    direction: str = 'horizontal'
) -> np.ndarray:
    """
    Generate gradient pattern.

    Args:
        size: Output image size
        direction: 'horizontal', 'vertical', or 'radial'

    Returns:
        2D array normalized to [0, 1]
    """
    y, x = np.ogrid[:size, :size]
    xn = x / (size - 1)
    yn = y / (size - 1)

    if direction == 'vertical':
        return yn.astype(np.float32)
    elif direction == 'radial':
        cx, cy = 0.5, 0.5
        dist = np.sqrt((xn - cx) ** 2 + (yn - cy) ** 2)
        dist = dist / dist.max()
        return (1.0 - dist).astype(np.float32)
    else:  # horizontal
        return (xn * np.ones_like(yn)).astype(np.float32)


def generate_tiled_amongus(size: int = 64, amongus_size: int = 16) -> np.ndarray:
    """
    Generate tiled amongus silhouettes.

    Args:
        size: Output size
        amongus_size: Size of each amongus tile

    Returns:
        2D array with tiled amongus pattern
    """
    single = generate_amongus(amongus_size)

    # Tile using numpy fancy indexing
    y_idx = np.arange(size) % amongus_size
    x_idx = np.arange(size) % amongus_size
    bitmap = single[y_idx[:, None], x_idx[None, :]]

    return bitmap


def generate_circles(
    size: int = 64,
    num_circles: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Generate random circles pattern.

    Args:
        size: Output image size
        num_circles: Number of circles to generate
        seed: Random seed

    Returns:
        2D array with circles
    """
    np.random.seed(seed)

    image = np.zeros((size, size), dtype=np.float32)
    y_coords, x_coords = np.mgrid[0:size, 0:size]

    for _ in range(num_circles):
        cx = np.random.randint(0, size)
        cy = np.random.randint(0, size)
        radius = np.random.randint(size // 10, size // 3)

        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        circle = (dist < radius).astype(np.float32)
        image = np.maximum(image, circle)

    return image


def generate_varied_amongus(
    size: int = 128,
    num_instances: int = 4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate multiple amongus silhouettes with random transforms.

    Args:
        size: Output bitmap size
        num_instances: Number of amongus silhouettes
        seed: Random seed

    Returns:
        2D array (size, size) normalized to [0, 1]
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((size, size))

    base_amongus_size = size // 4

    for _ in range(num_instances):
        # Random transforms
        scale = rng.uniform(0.5, 1.5)
        rotation = rng.uniform(-30, 30)
        offset_y = rng.uniform(-size // 3, size // 3)
        offset_x = rng.uniform(-size // 3, size // 3)

        # Generate base amongus
        instance_size = int(base_amongus_size * scale)
        instance_size = max(8, min(instance_size, size))

        small_amongus = generate_amongus(instance_size)

        # Create canvas and place amongus
        canvas = np.zeros((size, size))
        start_y = (size - instance_size) // 2
        start_x = (size - instance_size) // 2
        end_y = start_y + instance_size
        end_x = start_x + instance_size

        # Handle boundaries
        src_start_y = max(0, -start_y)
        src_start_x = max(0, -start_x)
        src_end_y = instance_size - max(0, end_y - size)
        src_end_x = instance_size - max(0, end_x - size)

        dst_start_y = max(0, start_y)
        dst_start_x = max(0, start_x)
        dst_end_y = min(size, end_y)
        dst_end_x = min(size, end_x)

        canvas[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
            small_amongus[src_start_y:src_end_y, src_start_x:src_end_x]

        # Apply transforms
        transformed = ndimage.rotate(canvas, rotation, reshape=False, mode='constant')
        transformed = ndimage.shift(transformed, [offset_y, offset_x], mode='constant')

        result = np.maximum(result, transformed)

    # Normalize
    if result.max() > 0:
        result = result / result.max()

    return result.astype(np.float32)


def generate_amongus_stretched(
    size: int = 64,
    stretch_x: float = 1.5,
    stretch_y: float = 1.0
) -> np.ndarray:
    """
    Generate a stretched Among Us character silhouette.

    Args:
        size: Output bitmap size (square)
        stretch_x: Horizontal stretch factor (>1 stretches, <1 compresses)
        stretch_y: Vertical stretch factor (>1 stretches, <1 compresses)

    Returns:
        2D array (size, size) with values 0.0 (background) or 1.0 (character)
    """
    # Generate base amongus at the output size
    base = generate_amongus(size)

    # Create coordinate grids for output
    y_out, x_out = np.mgrid[0:size, 0:size]

    # Map output to centered coordinates
    center = size / 2.0
    y_centered = y_out - center
    x_centered = x_out - center

    # Apply inverse stretch (to get input coordinates)
    # Stretching output means compressing input sampling
    y_in = y_centered * stretch_y + center
    x_in = x_centered * stretch_x + center

    # Sample using map_coordinates
    coords = np.array([y_in, x_in])
    result = ndimage.map_coordinates(base, coords, order=1, mode='constant', cval=0.0)

    return result.astype(np.float32)


def generate_amongus_sheared(
    size: int = 64,
    shear_angle: float = 15.0
) -> np.ndarray:
    """
    Generate a sheared Among Us character silhouette.

    Args:
        size: Output bitmap size (square)
        shear_angle: Shear angle in degrees

    Returns:
        2D array (size, size) with values 0.0 (background) or 1.0 (character)
    """
    # Generate base amongus
    base = generate_amongus(size)

    # Shear transformation
    shear_rad = np.radians(shear_angle)
    shear_factor = np.tan(shear_rad)

    # Create coordinate grids
    y_out, x_out = np.mgrid[0:size, 0:size]

    # Center coordinates
    center = size / 2.0
    y_centered = y_out - center
    x_centered = x_out - center

    # Apply inverse shear to get input coordinates
    # Shear matrix: [[1, shear], [0, 1]]
    # Inverse: [[1, -shear], [0, 1]]
    y_in = y_centered + center
    x_in = x_centered - shear_factor * y_centered + center

    coords = np.array([y_in, x_in])
    result = ndimage.map_coordinates(base, coords, order=1, mode='constant', cval=0.0)

    return result.astype(np.float32)


def generate_amongus_rotated(
    size: int = 64,
    angle: float = 45.0
) -> np.ndarray:
    """
    Generate a rotated Among Us character silhouette.

    Args:
        size: Output bitmap size (square)
        angle: Rotation angle in degrees (counter-clockwise)

    Returns:
        2D array (size, size) with values 0.0 (background) or 1.0 (character)
    """
    # Generate base amongus
    base = generate_amongus(size)

    # Use scipy.ndimage.rotate
    result = ndimage.rotate(base, angle, reshape=False, mode='constant', cval=0.0, order=1)

    return result.astype(np.float32)


def generate_amongus_tessellated(
    size: int = 64,
    copies_x: int = 3,
    copies_y: int = 3
) -> np.ndarray:
    """
    Generate a tessellated grid of Among Us characters.

    Args:
        size: Output bitmap size (square)
        copies_x: Number of copies in horizontal direction
        copies_y: Number of copies in vertical direction

    Returns:
        2D array (size, size) with tiled amongus pattern
    """
    # Calculate size for each tile
    tile_size_y = size // copies_y
    tile_size_x = size // copies_x

    # Use minimum to keep aspect ratio
    tile_size = min(tile_size_x, tile_size_y)

    # Generate single amongus at tile size
    single = generate_amongus(tile_size)

    # Create output array
    result = np.zeros((size, size), dtype=np.float32)

    # Place copies
    for iy in range(copies_y):
        for ix in range(copies_x):
            # Calculate position (centered in each cell)
            start_y = iy * (size // copies_y) + (size // copies_y - tile_size) // 2
            start_x = ix * (size // copies_x) + (size // copies_x - tile_size) // 2

            end_y = start_y + tile_size
            end_x = start_x + tile_size

            # Handle boundary clipping
            if end_y > size:
                end_y = size
            if end_x > size:
                end_x = size

            src_end_y = end_y - start_y
            src_end_x = end_x - start_x

            result[start_y:end_y, start_x:end_x] = single[:src_end_y, :src_end_x]

    return result


def generate_amongus_warped(
    size: int = 64,
    warp_strength: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a smoothly warped Among Us character silhouette.

    Uses a random displacement field with smooth interpolation.

    Args:
        size: Output bitmap size (square)
        warp_strength: Strength of the warp (0.0 to 1.0, relative to size)
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) with warped amongus
    """
    rng = np.random.default_rng(seed)

    # Generate base amongus
    base = generate_amongus(size)

    # Create smooth displacement fields using low-frequency noise
    # Use a small grid and interpolate
    grid_size = 8
    displacement_scale = warp_strength * size

    # Random displacement grids
    dx_grid = rng.uniform(-1, 1, (grid_size, grid_size)) * displacement_scale
    dy_grid = rng.uniform(-1, 1, (grid_size, grid_size)) * displacement_scale

    # Interpolate to full size using zoom
    zoom_factor = size / grid_size
    dx_field = ndimage.zoom(dx_grid, zoom_factor, order=3)
    dy_field = ndimage.zoom(dy_grid, zoom_factor, order=3)

    # Ensure same size (zoom might produce slightly different size)
    dx_field = dx_field[:size, :size]
    dy_field = dy_field[:size, :size]

    # Create coordinate grids
    y_out, x_out = np.mgrid[0:size, 0:size]

    # Apply displacement to get input coordinates
    y_in = y_out - dy_field
    x_in = x_out - dx_field

    # Sample
    coords = np.array([y_in, x_in])
    result = ndimage.map_coordinates(base, coords, order=1, mode='constant', cval=0.0)

    return result.astype(np.float32)


def generate_amongus_random_transform(
    size: int = 64,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate an Among Us character with random combination of transforms.

    Applies random stretch, shear, rotation, and warp.

    Args:
        size: Output bitmap size (square)
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) with randomly transformed amongus
    """
    rng = np.random.default_rng(seed)

    # Generate base amongus at same size
    base = generate_amongus(size)

    center = size / 2.0

    # Random transform parameters (moderate ranges to keep figure visible)
    stretch_x = rng.uniform(0.9, 1.2)
    stretch_y = rng.uniform(0.9, 1.2)
    shear_angle = rng.uniform(-12, 12)
    rotation_angle = rng.uniform(-25, 25)
    warp_strength = rng.uniform(0.0, 0.08)

    # Build combined affine transformation matrix
    # Order: stretch -> shear -> rotate

    # Rotation matrix
    theta = np.radians(rotation_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Shear matrix
    shear_rad = np.radians(shear_angle)
    shear_matrix = np.array([[1, np.tan(shear_rad)], [0, 1]])

    # Stretch matrix (for coordinate mapping: stretch output = compress input sampling)
    stretch_matrix = np.array([[stretch_y, 0], [0, stretch_x]])

    # Combined matrix (applied in reverse order for coordinate mapping)
    combined = stretch_matrix @ shear_matrix @ rot_matrix

    # Create coordinate grids for output
    y_out, x_out = np.mgrid[0:size, 0:size]

    # Center output coordinates
    y_centered = y_out - center
    x_centered = x_out - center

    # Stack coordinates for matrix multiplication
    coords_out = np.stack([y_centered.ravel(), x_centered.ravel()], axis=0)

    # Apply combined inverse transform
    coords_in = combined @ coords_out

    # Reshape and add center offset
    y_in = coords_in[0].reshape(size, size) + center
    x_in = coords_in[1].reshape(size, size) + center

    # Apply warp if strength > 0
    if warp_strength > 0:
        grid_size = 6
        displacement_scale = warp_strength * size
        dx_grid = rng.uniform(-1, 1, (grid_size, grid_size)) * displacement_scale
        dy_grid = rng.uniform(-1, 1, (grid_size, grid_size)) * displacement_scale

        zoom_factor = size / grid_size
        dx_field = ndimage.zoom(dx_grid, zoom_factor, order=3)[:size, :size]
        dy_field = ndimage.zoom(dy_grid, zoom_factor, order=3)[:size, :size]

        y_in = y_in - dy_field
        x_in = x_in - dx_field

    # Sample
    coords = np.array([y_in, x_in])
    result = ndimage.map_coordinates(base, coords, order=1, mode='constant', cval=0.0)

    return result.astype(np.float32)


# =============================================================================
# CG-Derived Operand Generators
# =============================================================================
# These functions create operands with spatial structure derived from natural
# images. Unlike uniform noise, they preserve recognizable features that
# survive spectral blending, showing visible contribution from BOTH carrier
# AND operand.


def generate_edges_operand(
    image: np.ndarray,
    method: str = 'sobel'
) -> np.ndarray:
    """
    Extract edges from a natural image as operand.

    Edge detection creates an operand with spatial structure corresponding
    to the original image's features, rather than random noise.

    Args:
        image: (H, W) grayscale image [0, 1] or [0, 255]
        method: 'sobel' for Sobel edge detection

    Returns:
        2D array normalized to [0, 1] with edge magnitudes
    """
    # Normalize input to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    if method == 'sobel':
        # Compute Sobel gradients in x and y
        grad_x = sobel(img, axis=1, mode='reflect')
        grad_y = sobel(img, axis=0, mode='reflect')

        # Compute gradient magnitude
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        raise ValueError(f"Unknown edge method: {method}")

    # Normalize to [0, 1]
    if edge_mag.max() > edge_mag.min():
        edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min())

    return edge_mag.astype(np.float32)


def generate_posterized_operand(
    image: np.ndarray,
    levels: int = 4
) -> np.ndarray:
    """
    Quantize a natural image to N intensity levels.

    Posterization creates bands/regions with sharp boundaries,
    preserving the image's spatial structure while simplifying
    the intensity distribution.

    Args:
        image: (H, W) grayscale image [0, 1] or [0, 255]
        levels: Number of quantization levels (default 4)

    Returns:
        2D array with posterized values in [0, 1]
    """
    # Normalize input to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Quantize: map [0, 1] to [0, levels-1] then back to [0, 1]
    quantized = np.floor(img * levels) / (levels - 1)
    quantized = np.clip(quantized, 0, 1)

    return quantized.astype(np.float32)


def generate_threshold_operand(
    image: np.ndarray,
    percentile: float = 50
) -> np.ndarray:
    """
    Create binary threshold operand at given percentile.

    Binary thresholding creates a silhouette-like pattern that
    preserves the broad spatial structure of the original image.

    Args:
        image: (H, W) grayscale image [0, 1] or [0, 255]
        percentile: Threshold percentile (0-100, default 50 = median)

    Returns:
        2D binary array with values 0.0 or 1.0
    """
    # Normalize input to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Compute threshold value at percentile
    threshold = np.percentile(img, percentile)

    # Apply binary threshold
    binary = (img >= threshold).astype(np.float32)

    return binary


def generate_morpho_operand(
    image: np.ndarray,
    operation: str = 'gradient',
    structure_size: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to create structure-aware operand.

    The morphological gradient (dilation - erosion) creates an edge-like
    effect that respects the image's structure. This produces operands
    with clear spatial organization that survives spectral blending.

    Args:
        image: (H, W) grayscale image [0, 1] or [0, 255]
        operation: 'gradient' (dilation - erosion), 'dilation', 'erosion'
        structure_size: Size of the structuring element (default 3)

    Returns:
        2D array normalized to [0, 1]
    """
    # Normalize input to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Create structuring element
    structure = np.ones((structure_size, structure_size))

    if operation == 'gradient':
        # Morphological gradient = dilation - erosion
        dilated = grey_dilation(img, footprint=structure)
        eroded = grey_erosion(img, footprint=structure)
        result = dilated - eroded
    elif operation == 'dilation':
        result = grey_dilation(img, footprint=structure)
    elif operation == 'erosion':
        result = grey_erosion(img, footprint=structure)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")

    # Normalize to [0, 1]
    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min())

    return result.astype(np.float32)


def generate_tiled_operand(
    image: np.ndarray,
    tile_size: int = 16
) -> np.ndarray:
    """
    Tile a cropped region of the image to create periodic operand.

    Takes a region from the center of the image and tiles it,
    creating periodic structure while preserving natural image content.

    Args:
        image: (H, W) grayscale image [0, 1] or [0, 255]
        tile_size: Size of the tile to extract and repeat (default 16)

    Returns:
        2D array with tiled pattern, normalized to [0, 1]
    """
    # Normalize input to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    H, W = img.shape

    # Extract tile from center of image
    center_y = H // 2
    center_x = W // 2
    half_tile = tile_size // 2

    # Ensure tile fits in image
    tile_size = min(tile_size, min(H, W))
    half_tile = tile_size // 2

    y_start = max(0, center_y - half_tile)
    y_end = y_start + tile_size
    x_start = max(0, center_x - half_tile)
    x_end = x_start + tile_size

    tile = img[y_start:y_end, x_start:x_end]

    # Tile to fill the original image size
    # Use modular indexing
    y_idx = np.arange(H) % tile.shape[0]
    x_idx = np.arange(W) % tile.shape[1]

    tiled = tile[y_idx[:, None], x_idx[None, :]]

    return tiled.astype(np.float32)


def generate_amongus_scattered(
    size: int = 128,
    num_copies: int = 3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a few amongus silhouettes with random affine transforms (NOT dense tiling).

    Each copy gets random position, rotation, scale, and shear. This creates
    recognizable features without the dense periodicity of tessellation.

    Args:
        size: Output bitmap size (square)
        num_copies: Number of amongus copies to place (default 3)
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) with scattered amongus pattern, normalized to [0, 1]
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((size, size), dtype=np.float32)

    # Base amongus size (smaller so multiple fit)
    base_size = size // 3

    for _ in range(num_copies):
        # Random transform parameters
        angle_deg = rng.uniform(-30, 30)  # Rotation in degrees
        scale = rng.uniform(0.7, 1.3)     # Scale factor
        shear = rng.uniform(-0.2, 0.2)    # Shear factor
        tx = rng.uniform(0, size * 0.5)   # Translation X
        ty = rng.uniform(0, size * 0.5)   # Translation Y

        # Generate base amongus at scaled size
        actual_size = int(base_size * scale)
        actual_size = max(8, min(actual_size, size - 4))
        base = generate_amongus(actual_size)

        # Build affine transform matrix
        # We'll apply: rotate -> shear -> translate
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix
        rot = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Shear matrix
        shear_mat = np.array([
            [1, shear],
            [0, 1]
        ])

        # Combined transform (inverse for coordinate mapping)
        transform = rot @ shear_mat
        inv_transform = np.linalg.inv(transform)

        # Create canvas for this copy
        canvas = np.zeros((size, size), dtype=np.float32)

        # Create output coordinate grid
        y_out, x_out = np.mgrid[0:size, 0:size]

        # Center for rotation
        center_out = np.array([size / 2, size / 2])
        center_base = np.array([actual_size / 2, actual_size / 2])

        # Apply inverse transform to get input coordinates
        # First translate output coords to rotation center
        y_centered = y_out - (ty + center_out[0])
        x_centered = x_out - (tx + center_out[1])

        # Apply inverse transform
        y_in = inv_transform[0, 0] * y_centered + inv_transform[0, 1] * x_centered + center_base[0]
        x_in = inv_transform[1, 0] * y_centered + inv_transform[1, 1] * x_centered + center_base[1]

        # Sample from base using scipy
        coords = np.array([y_in, x_in])
        transformed = ndimage.map_coordinates(base, coords, order=1, mode='constant', cval=0.0)

        # Composite onto result
        result = np.maximum(result, transformed)

    # Normalize
    if result.max() > 0:
        result = result / result.max()

    return result.astype(np.float32)
