"""
Rasterization: turn graph (coords, colors) into pixel image.

Given:
    coords: (N, 2) continuous coordinates
    colors: (N, 3) RGB values

Produce:
    image: (H, W, 3) pixel array

This is the final step that converts graph-space operations back to pixels.
All the work happens in graph space; we rasterize once at the end.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional


def rasterize_nearest(
    coords: torch.Tensor,       # (N, 2) as (x, y)
    colors: torch.Tensor,       # (N, 3) RGB in [0, 1]
    H: int, W: int,
    background: Optional[torch.Tensor] = None  # (H, W, 3) or None for white
) -> torch.Tensor:              # (H, W, 3)
    """
    Rasterize points to nearest pixel via scatter.

    Multiple points at same pixel: last write wins (simple, fast).
    """
    device = coords.device
    dtype = colors.dtype

    if background is None:
        img = torch.ones(H, W, 3, device=device, dtype=dtype)
    else:
        img = background.clone()

    # Round to nearest pixel
    px = coords[:, 0].round().long().clamp(0, W - 1)
    py = coords[:, 1].round().long().clamp(0, H - 1)

    # Flatten for indexing
    flat_idx = py * W + px  # (N,)

    # Scatter colors (last write wins)
    img_flat = img.view(-1, 3)  # (H*W, 3)
    img_flat[flat_idx] = colors

    return img_flat.view(H, W, 3)


def rasterize_accumulate(
    coords: torch.Tensor,       # (N, 2) as (x, y)
    colors: torch.Tensor,       # (N, 3) RGB in [0, 1]
    H: int, W: int,
    background: Optional[torch.Tensor] = None
) -> torch.Tensor:              # (H, W, 3)
    """
    Rasterize with accumulation: average colors at each pixel.

    Multiple points at same pixel: averaged together.
    """
    device = coords.device
    dtype = colors.dtype

    # Round to nearest pixel
    px = coords[:, 0].round().long().clamp(0, W - 1)
    py = coords[:, 1].round().long().clamp(0, H - 1)

    flat_idx = py * W + px  # (N,)

    # Accumulate colors and counts
    color_sum = torch.zeros(H * W, 3, device=device, dtype=dtype)
    counts = torch.zeros(H * W, device=device, dtype=dtype)

    # Scatter add for each color channel
    for c in range(3):
        color_sum[:, c].scatter_add_(0, flat_idx, colors[:, c])
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))

    # Average where count > 0
    mask = counts > 0
    color_avg = torch.zeros_like(color_sum)
    color_avg[mask] = color_sum[mask] / counts[mask].unsqueeze(-1)

    # Combine with background
    if background is None:
        background = torch.ones(H, W, 3, device=device, dtype=dtype)

    img = background.clone().view(-1, 3)
    img[mask] = color_avg[mask]

    return img.view(H, W, 3)


def rasterize_splat(
    coords: torch.Tensor,       # (N, 2) as (x, y)
    colors: torch.Tensor,       # (N, 3) RGB in [0, 1]
    H: int, W: int,
    radius: float = 1.5,        # splat radius in pixels
    background: Optional[torch.Tensor] = None
) -> torch.Tensor:              # (H, W, 3)
    """
    Rasterize with Gaussian splatting: each point contributes to nearby pixels.

    Soft, antialiased rendering. More expensive but smoother.
    """
    device = coords.device
    dtype = colors.dtype

    if background is None:
        img = torch.ones(H, W, 3, device=device, dtype=dtype)
    else:
        img = background.clone()

    # Create pixel coordinate grids
    py_grid = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
    px_grid = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)  # (1, W)

    # Accumulate weighted colors
    weight_sum = torch.zeros(H, W, device=device, dtype=dtype)
    color_sum = torch.zeros(H, W, 3, device=device, dtype=dtype)

    # Process points in batches to manage memory
    batch_size = 1000
    N = coords.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_coords = coords[start:end]  # (B, 2)
        batch_colors = colors[start:end]  # (B, 3)
        B = batch_coords.shape[0]

        # Distance from each point to each pixel
        # coords[:, 0] is x, coords[:, 1] is y
        dx = px_grid.unsqueeze(0) - batch_coords[:, 0].view(B, 1, 1)  # (B, 1, W)
        dy = py_grid.unsqueeze(0) - batch_coords[:, 1].view(B, 1, 1)  # (B, H, 1)

        dist_sq = dx**2 + dy**2  # (B, H, W)

        # Gaussian weights
        weights = torch.exp(-dist_sq / (2 * radius**2))  # (B, H, W)

        # Zero out weights beyond 3*radius for efficiency
        weights = weights * (dist_sq < (3 * radius)**2)

        # Accumulate
        weight_sum += weights.sum(dim=0)  # (H, W)
        for c in range(3):
            color_sum[:, :, c] += (weights * batch_colors[:, c].view(B, 1, 1)).sum(dim=0)

    # Normalize and blend with background
    mask = weight_sum > 1e-6
    img[mask] = color_sum[mask] / weight_sum[mask].unsqueeze(-1)

    return img


def demo_rasterization():
    """
    Demonstrate rasterization methods on simple test data.
    """
    print("=" * 60)
    print("RASTERIZATION DEMO")
    print("=" * 60)

    device = torch.device('cpu')
    H, W = 120, 200

    # Create test points: a few colored shapes
    # Circle of red points
    n_circle = 30
    theta = torch.linspace(0, 2 * np.pi, n_circle, device=device)
    circle_x = 50 + 25 * torch.cos(theta)
    circle_y = 60 + 25 * torch.sin(theta)
    circle_coords = torch.stack([circle_x, circle_y], dim=-1)
    circle_colors = torch.tensor([[1.0, 0.2, 0.2]], device=device).expand(n_circle, -1)

    # Line of green points
    n_line = 20
    line_t = torch.linspace(0, 1, n_line, device=device)
    line_x = 100 + line_t * 60
    line_y = 30 + line_t * 60
    line_coords = torch.stack([line_x, line_y], dim=-1)
    line_colors = torch.tensor([[0.2, 0.8, 0.2]], device=device).expand(n_line, -1)

    # Cluster of blue points
    n_cluster = 40
    torch.manual_seed(42)
    cluster_x = 150 + torch.randn(n_cluster, device=device) * 10
    cluster_y = 40 + torch.randn(n_cluster, device=device) * 10
    cluster_coords = torch.stack([cluster_x, cluster_y], dim=-1)
    cluster_colors = torch.tensor([[0.2, 0.2, 1.0]], device=device).expand(n_cluster, -1)

    # Combine all
    coords = torch.cat([circle_coords, line_coords, cluster_coords], dim=0)
    colors = torch.cat([circle_colors, line_colors, cluster_colors], dim=0)
    N = coords.shape[0]

    print(f"\nPoints: {N} total")
    print(f"  - Red circle: {n_circle} points")
    print(f"  - Green line: {n_line} points")
    print(f"  - Blue cluster: {n_cluster} points")
    print(f"\nOutput size: {H}x{W}")

    # Method 1: Nearest neighbor
    img_nearest = rasterize_nearest(coords, colors, H, W)
    print("\nMethod 1: Nearest neighbor (last write wins)")

    # Method 2: Accumulate/average
    img_accum = rasterize_accumulate(coords, colors, H, W)
    print("Method 2: Accumulate and average")

    # Method 3: Gaussian splatting
    img_splat = rasterize_splat(coords, colors, H, W, radius=2.0)
    print("Method 3: Gaussian splatting (radius=2.0)")

    # Visualize all three
    visualize_rasterization(img_nearest, img_accum, img_splat, coords, H, W)

    return coords, colors, img_nearest, img_accum, img_splat


def visualize_rasterization(img_nearest, img_accum, img_splat, coords, H, W):
    """Create comparison image of all three methods."""

    def to_pil(tensor):
        arr = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # Create side-by-side comparison
    panel_w = W
    panel_h = H
    margin = 5
    label_h = 20

    total_w = 3 * panel_w + 4 * margin
    total_h = panel_h + 2 * margin + label_h

    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))

    # Paste each image
    img1 = to_pil(img_nearest)
    img2 = to_pil(img_accum)
    img3 = to_pil(img_splat)

    canvas.paste(img1, (margin, margin + label_h))
    canvas.paste(img2, (2 * margin + panel_w, margin + label_h))
    canvas.paste(img3, (3 * margin + 2 * panel_w, margin + label_h))

    # Labels
    draw = ImageDraw.Draw(canvas)
    draw.text((margin + 5, 3), "Nearest (last wins)", fill=(0, 0, 0))
    draw.text((2 * margin + panel_w + 5, 3), "Accumulate (average)", fill=(0, 0, 0))
    draw.text((3 * margin + 2 * panel_w + 5, 3), "Gaussian splat", fill=(0, 0, 0))

    canvas.save('demo_output/rasterization.png')
    print(f"\nSaved: demo_output/rasterization.png")


if __name__ == "__main__":
    demo_rasterization()
