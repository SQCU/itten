"""
Translation field: compute where to move each block.

Given block centroids and some guiding field (e.g., distance field,
density gradient, or spectral flow), compute translation vectors.

The translation pushes fragments "outward" - away from dense structure
into void regions.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional


def gradient_2d(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of a 2D field.

    Returns (grad_y, grad_x) with same shape as input.
    """
    # Forward differences with zero-padding
    grad_y = torch.zeros_like(field)
    grad_x = torch.zeros_like(field)

    grad_y[:-1, :] = field[1:, :] - field[:-1, :]
    grad_x[:, :-1] = field[:, 1:] - field[:, :-1]

    return grad_y, grad_x


def sample_field_at_points(
    field: torch.Tensor,    # (H, W) or (H, W, d)
    points: torch.Tensor,   # (N, 2) as (x, y) coordinates
    H: int, W: int
) -> torch.Tensor:          # (N,) or (N, d)
    """
    Sample a field at given continuous coordinates via bilinear interpolation.
    """
    # Normalize to [-1, 1] for grid_sample
    x_norm = 2 * points[:, 0] / (W - 1) - 1
    y_norm = 2 * points[:, 1] / (H - 1) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

    if field.dim() == 2:
        field_4d = field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        field_4d = field.permute(2, 0, 1).unsqueeze(0)  # (1, d, H, W)

    sampled = torch.nn.functional.grid_sample(
        field_4d, grid, mode='bilinear', padding_mode='border', align_corners=True
    )

    if field.dim() == 2:
        return sampled.squeeze()  # (N,)
    else:
        return sampled.squeeze(0).squeeze(-1).T  # (N, d)


def compute_translation_from_field_gradient(
    centroids: torch.Tensor,    # (K, 2) block centers as (x, y)
    field: torch.Tensor,        # (H, W) scalar field (e.g., distance field)
    strength: float,            # translation magnitude
    H: int, W: int
) -> torch.Tensor:              # (K, 2) translation vectors
    """
    Compute translation as negative gradient of field (move toward increasing field).

    For a distance field: gradient points away from boundaries, so
    negative gradient points toward boundaries. We want the opposite -
    to push INTO void, so we use positive gradient.
    """
    grad_y, grad_x = gradient_2d(field)

    # Stack gradients: (H, W, 2)
    grad_field = torch.stack([grad_x, grad_y], dim=-1)

    # Sample at centroids
    grad_at_centroids = sample_field_at_points(grad_field, centroids, H, W)  # (K, 2)

    # Normalize and scale
    norm = torch.norm(grad_at_centroids, dim=-1, keepdim=True).clamp(min=1e-8)
    direction = grad_at_centroids / norm

    translation = direction * strength

    return translation


def compute_translation_perpendicular_to_axis(
    centroids: torch.Tensor,    # (K, 2)
    axes: torch.Tensor,         # (K, 2) unit vectors of block orientation
    strength: float,
    outward_sign: torch.Tensor = None  # (K,) +1 or -1 to choose perpendicular direction
) -> torch.Tensor:              # (K, 2)
    """
    Compute translation perpendicular to each block's axis.

    Perpendicular to (ax, ay) is (-ay, ax) or (ay, -ax).
    """
    K = centroids.shape[0]
    device = centroids.device

    # Perpendicular direction: rotate axis by 90°
    perp = torch.stack([-axes[:, 1], axes[:, 0]], dim=-1)  # (K, 2)

    # Optionally flip direction per block
    if outward_sign is None:
        outward_sign = torch.ones(K, device=device)

    translation = perp * outward_sign.unsqueeze(-1) * strength

    return translation


def demo_translation_field():
    """
    Demonstrate translation field computation on simple test case.
    """
    print("=" * 60)
    print("TRANSLATION FIELD DEMO")
    print("=" * 60)

    device = torch.device('cpu')
    H, W = 100, 150

    # Create a simple "density" field - higher in center, lower at edges
    y_coords = torch.linspace(0, 1, H, device=device).unsqueeze(1)
    x_coords = torch.linspace(0, 1, W, device=device).unsqueeze(0)

    # Gaussian blob in center
    cx, cy = 0.4, 0.5
    density = torch.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / 0.05)

    # Distance field: inverse of density (high = far from center)
    # This makes gradient point AWAY from center
    distance_field = 1.0 - density

    # Create some test blocks (centroids + axes)
    centroids = torch.tensor([
        [30.0, 50.0],   # near center
        [100.0, 30.0],  # upper right
        [80.0, 70.0],   # lower right
        [50.0, 20.0],   # upper middle
    ], device=device)

    axes = torch.tensor([
        [1.0, 0.0],     # horizontal
        [0.0, 1.0],     # vertical
        [0.707, 0.707], # diagonal
        [0.6, 0.8],     # angled
    ], device=device)

    K = centroids.shape[0]

    print(f"\nField: {H}x{W} distance field (high = far from density center)")
    print(f"Blocks: {K} test blocks with different orientations")

    # Method 1: Translation from field gradient
    trans_gradient = compute_translation_from_field_gradient(
        centroids, distance_field, strength=20.0, H=H, W=W
    )

    print("\nMethod 1: Follow field gradient (push away from density)")
    for k in range(K):
        t = trans_gradient[k]
        print(f"  Block {k}: centroid=({centroids[k,0]:.0f}, {centroids[k,1]:.0f}) "
              f"→ translation=({t[0]:.1f}, {t[1]:.1f})")

    # Method 2: Translation perpendicular to axis
    trans_perp = compute_translation_perpendicular_to_axis(
        centroids, axes, strength=15.0
    )

    print("\nMethod 2: Perpendicular to block axis")
    for k in range(K):
        t = trans_perp[k]
        print(f"  Block {k}: axis=({axes[k,0]:.2f}, {axes[k,1]:.2f}) "
              f"→ translation=({t[0]:.1f}, {t[1]:.1f})")

    # Visualize
    visualize_translations(
        density, centroids, axes,
        trans_gradient, trans_perp,
        H, W
    )

    return centroids, axes, trans_gradient, trans_perp


def visualize_translations(density, centroids, axes, trans_grad, trans_perp, H, W):
    """Draw the field and translation vectors."""

    # Create image from density field
    density_np = density.numpy()
    img_array = (density_np * 255).astype(np.uint8)
    img_array = np.stack([img_array, img_array, img_array], axis=-1)

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    centroids_np = centroids.numpy()
    axes_np = axes.numpy()
    trans_grad_np = trans_grad.numpy()
    trans_perp_np = trans_perp.numpy()

    K = centroids_np.shape[0]
    colors = [(255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 200, 60)]

    for k in range(K):
        cx, cy = centroids_np[k]
        ax, ay = axes_np[k] * 15  # scale for visibility

        # Draw centroid
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=colors[k], outline='black')

        # Draw axis (thin line)
        draw.line([cx - ax, cy - ay, cx + ax, cy + ay],
                  fill=colors[k], width=2)

        # Draw gradient translation (solid arrow)
        tg = trans_grad_np[k]
        end_g = (cx + tg[0], cy + tg[1])
        draw.line([cx, cy, end_g[0], end_g[1]], fill='white', width=3)
        draw.ellipse([end_g[0]-3, end_g[1]-3, end_g[0]+3, end_g[1]+3],
                     fill='white', outline='black')

        # Draw perpendicular translation (dashed-ish, offset slightly)
        tp = trans_perp_np[k]
        start_p = (cx + 2, cy + 2)
        end_p = (cx + tp[0] + 2, cy + tp[1] + 2)
        draw.line([start_p[0], start_p[1], end_p[0], end_p[1]],
                  fill=colors[k], width=2)
        draw.ellipse([end_p[0]-3, end_p[1]-3, end_p[0]+3, end_p[1]+3],
                     fill=colors[k], outline='white')

    # Legend
    draw.text((5, 5), "Translation Field Demo", fill='white')
    draw.text((5, H-35), "White arrow: gradient direction", fill='white')
    draw.text((5, H-20), "Colored arrow: perpendicular to axis", fill='white')

    img.save('demo_output/translation_field.png')
    print(f"\nSaved: demo_output/translation_field.png")


if __name__ == "__main__":
    demo_translation_field()
