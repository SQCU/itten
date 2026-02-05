"""
Dither-aware tiled spectral transform.

Use the tiled approach to make multi-radius connectivity tractable:
- Each tile is small (e.g., 64x64 = 4096 pixels)
- We CAN afford dense multi-radius edges per tile
- Dither patterns within tiles become connected
- No giant global matrix ever materializes

This is the "clever" approach: dither-okay weighted-graphs over
spatially-compact textures in tractable time/flops/memory.

NOTE: This module now delegates to spectral_ops_fast.py for the core
implementation. The functions here are thin wrappers for demo purposes.
"""
import numpy as np
from PIL import Image
from typing import List

# Use the canonical implementation from spectral_ops_fast
from spectral_ops_fast import (
    compute_local_eigenvectors_tiled_dither,
    compute_local_eigenvectors_tiled,
)


def dither_aware_tiled_spectral(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    radii: List[int] = [1, 2, 3, 4, 5, 6],
    radius_weights: List[float] = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    edge_threshold: float = 0.15
) -> np.ndarray:
    """
    Compute eigenvectors using dither-aware multi-scale connectivity.

    Delegates to compute_local_eigenvectors_tiled_dither from spectral_ops_fast.

    Args:
        image: (H, W) grayscale image in [0, 1] or [0, 255]
        tile_size: size of each tile (e.g., 64)
        overlap: overlap between adjacent tiles (e.g., 16)
        num_eigenvectors: how many eigenvectors per tile
        radii: list of connectivity radii
        radius_weights: weight for each radius level
        edge_threshold: intensity difference threshold for edge weights

    Returns:
        (H, W, num_eigenvectors) array of eigenvector fields
    """
    return compute_local_eigenvectors_tiled_dither(
        image,
        tile_size=tile_size,
        overlap=overlap,
        num_eigenvectors=num_eigenvectors,
        radii=radii,
        radius_weights=radius_weights,
        edge_threshold=edge_threshold
    )


def standard_tiled_spectral(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Standard tiled spectral (radius=1 only) for comparison.

    Delegates to compute_local_eigenvectors_tiled from spectral_ops_fast.
    """
    return compute_local_eigenvectors_tiled(
        image,
        tile_size=tile_size,
        overlap=overlap,
        num_eigenvectors=num_eigenvectors,
        edge_threshold=edge_threshold
    )


def demo_dither_tiled():
    """
    Demo: compare standard vs dither-aware tiled spectral on the dithered image.

    Uses the canonical implementation from spectral_ops_fast.py.
    """
    print("=" * 60)
    print("DITHER-AWARE TILED SPECTRAL DEMO")
    print("=" * 60)

    # Load full resolution dithered image
    img_path = "demo_output/inputs/1bit redraw.png"
    img_pil = Image.open(img_path).convert('L')
    img = np.array(img_pil).astype(np.float32) / 255.0
    H, W = img.shape

    print(f"\nImage: {W}x{H} = {H*W} pixels (FULL RESOLUTION)")

    # Method 1: Standard connectivity (radius 1 only)
    print("\n--- Method 1: Standard tiled (radius=1 only) ---")
    evecs_standard = standard_tiled_spectral(
        img,
        tile_size=64,
        overlap=16,
        num_eigenvectors=4,
        edge_threshold=0.1
    )

    # Method 2: Dither-aware connectivity (radii 1-6)
    print("\n--- Method 2: Dither-aware tiled (radii=1-6) ---")
    evecs_dither = dither_aware_tiled_spectral(
        img,
        tile_size=64,
        overlap=16,
        num_eigenvectors=4,
        radii=[1, 2, 3, 4, 5, 6],
        radius_weights=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
        edge_threshold=0.15
    )

    # Visualize Fiedler vectors (eigenvector index 1)
    visualize_dither_comparison(img, evecs_standard[:,:,1], evecs_dither[:,:,1])

    return img, evecs_standard, evecs_dither


def visualize_dither_comparison(img, fiedler_std, fiedler_dither):
    """Compare Fiedler vectors from standard vs dither-aware."""
    from PIL import ImageDraw

    H, W = img.shape

    def to_colormap(field):
        """Convert field to blue-white-red colormap."""
        f = (field - field.min()) / (field.max() - field.min() + 1e-8)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:,:,0] = (np.clip(f - 0.5, 0, 0.5) * 2 * 255).astype(np.uint8)
        rgb[:,:,2] = (np.clip(0.5 - f, 0, 0.5) * 2 * 255).astype(np.uint8)
        rgb[:,:,1] = (255 - np.abs(f * 255 - 128).astype(np.uint8))
        return rgb

    # Create comparison image
    margin = 10
    label_h = 25

    canvas_w = 3 * W + 4 * margin
    canvas_h = H + 2 * margin + label_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), (50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    # Original
    img_uint8 = (img * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(img_uint8).convert('RGB'), (margin, margin + label_h))
    draw.text((margin + 5, 5), "Original (full res)", fill=(255, 255, 255))

    # Standard Fiedler
    std_rgb = to_colormap(fiedler_std)
    canvas.paste(Image.fromarray(std_rgb), (2*margin + W, margin + label_h))
    draw.text((2*margin + W + 5, 5), "Standard (r=1)", fill=(255, 255, 255))

    # Dither-aware Fiedler
    dither_rgb = to_colormap(fiedler_dither)
    canvas.paste(Image.fromarray(dither_rgb), (3*margin + 2*W, margin + label_h))
    draw.text((3*margin + 2*W + 5, 5), "Dither-aware (r=1-6)", fill=(255, 255, 255))

    canvas.save('demo_output/dither_tiled_spectral.png')
    print(f"\nSaved: demo_output/dither_tiled_spectral.png")


if __name__ == "__main__":
    demo_dither_tiled()
