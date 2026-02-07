"""
Procedural surface texture generators.

Provides three distinct surface textures for testing spectral transform variance:
- Marble: smooth gradients with organic veins (perlin-like noise)
- Brick: structured rectangular brick pattern
- Noise: high-frequency random noise

All functions return (H, W) arrays normalized to [0, 1].
"""

import numpy as np
from scipy import ndimage
from typing import Optional


def generate_marble(
    size: int = 128,
    vein_scale: float = 8.0,
    turbulence_octaves: int = 6,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate marble texture with smooth gradients and organic veins.

    Uses turbulence (multi-octave noise) to distort a sine wave pattern,
    creating realistic vein-like structures.

    Args:
        size: Output image size (square)
        vein_scale: Base frequency of vein pattern
        turbulence_octaves: Number of noise octaves for turbulence
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) normalized to [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate turbulence (sum of multi-octave noise)
    turbulence = np.zeros((size, size), dtype=np.float32)

    for octave in range(turbulence_octaves):
        freq = 2 ** octave
        amplitude = 1.0 / freq

        # Generate noise at this octave
        grid_size = max(4, size // (2 ** (turbulence_octaves - octave - 1)))
        noise_grid = np.random.randn(grid_size, grid_size).astype(np.float32)

        # Smooth interpolation to full size
        from scipy.ndimage import zoom
        scale_factor = size / grid_size
        interpolated = zoom(noise_grid, scale_factor, order=3)

        # Ensure exact size match
        if interpolated.shape[0] != size or interpolated.shape[1] != size:
            interpolated = interpolated[:size, :size]
            if interpolated.shape[0] < size:
                pad_h = size - interpolated.shape[0]
                interpolated = np.pad(interpolated, ((0, pad_h), (0, 0)), mode='edge')
            if interpolated.shape[1] < size:
                pad_w = size - interpolated.shape[1]
                interpolated = np.pad(interpolated, ((0, 0), (0, pad_w)), mode='edge')

        turbulence += amplitude * interpolated

    # Create coordinate grid
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    # Marble pattern: sine wave distorted by turbulence
    # The key insight: veins follow sin(x + turbulence)
    vein_pattern = np.sin(
        (x / size * vein_scale + turbulence * 5.0) * np.pi
    )

    # Add secondary vein direction for more organic feel
    vein_pattern2 = np.sin(
        ((x + y) / size * vein_scale * 0.7 + turbulence * 3.0) * np.pi
    )

    # Combine veins
    combined = 0.6 * vein_pattern + 0.4 * vein_pattern2

    # Add smooth base gradient
    base_gradient = np.sin(y / size * np.pi * 0.5) * 0.2

    marble = combined + base_gradient + turbulence * 0.3

    # Normalize to [0, 1]
    marble = (marble - marble.min()) / (marble.max() - marble.min() + 1e-10)

    return marble.astype(np.float32)


def generate_brick(
    size: int = 128,
    brick_width: int = 32,
    brick_height: int = 16,
    mortar_width: int = 2,
    mortar_value: float = 0.2,
    variation: float = 0.1,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate structured rectangular brick pattern.

    Creates a classic brick wall pattern with offset rows,
    mortar lines, and subtle color variation per brick.

    Args:
        size: Output image size (square)
        brick_width: Width of each brick in pixels
        brick_height: Height of each brick in pixels
        mortar_width: Width of mortar lines in pixels
        mortar_value: Grayscale value for mortar [0, 1]
        variation: Random variation in brick color
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) normalized to [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    brick = np.zeros((size, size), dtype=np.float32)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:size, 0:size]

    # Determine row index
    row_height = brick_height + mortar_width
    row_idx = y_coords // row_height

    # Position within row
    y_in_row = y_coords % row_height

    # Offset every other row by half brick width
    x_offset = (row_idx % 2) * (brick_width // 2)
    x_adjusted = (x_coords + x_offset) % size

    # Determine column index (brick index within row)
    col_width = brick_width + mortar_width
    col_idx = x_adjusted // col_width

    # Position within brick
    x_in_brick = x_adjusted % col_width

    # Create mortar mask (horizontal and vertical mortar lines)
    is_h_mortar = y_in_row >= brick_height
    is_v_mortar = x_in_brick >= brick_width
    is_mortar = is_h_mortar | is_v_mortar

    # Generate random brick colors
    num_rows = (size // row_height) + 2
    num_cols = (size // col_width) + 2
    brick_colors = 0.5 + variation * np.random.randn(num_rows, num_cols)
    brick_colors = np.clip(brick_colors, 0.3, 0.8)

    # Map each pixel to its brick color
    row_idx_clipped = np.clip(row_idx, 0, num_rows - 1)
    col_idx_clipped = np.clip(col_idx, 0, num_cols - 1)
    brick_color_field = brick_colors[row_idx_clipped, col_idx_clipped]

    # Apply colors
    brick = np.where(is_mortar, mortar_value, brick_color_field)

    # Add subtle noise for texture
    noise = 0.03 * np.random.randn(size, size)
    brick = brick + noise

    # Normalize to [0, 1]
    brick = np.clip(brick, 0, 1)

    return brick.astype(np.float32)


def generate_noise(
    size: int = 128,
    high_freq: bool = True,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate high-frequency random noise texture.

    Creates pure random noise, optionally blended with some
    spatial coherence for visual interest.

    Args:
        size: Output image size (square)
        high_freq: If True, pure random noise; if False, adds some smoothing
        seed: Random seed for reproducibility

    Returns:
        2D array (size, size) normalized to [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base random noise
    noise = np.random.rand(size, size).astype(np.float32)

    if not high_freq:
        # Add some spatial coherence with slight blur
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=1.0)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

    return noise.astype(np.float32)


# Convenience function to get all surfaces
def generate_all_surfaces(size: int = 128, seed: int = 42) -> dict:
    """
    Generate all three surface textures.

    Args:
        size: Output image size for all surfaces
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'marble', 'brick', 'noise' mapping to (size, size) arrays
    """
    return {
        'marble': generate_marble(size, seed=seed),
        'brick': generate_brick(size, seed=seed),
        'noise': generate_noise(size, seed=seed),
    }
