"""
Block rotation operations: functional tensor transformations.

Rotation is expressed as weighted combinations of coordinate columns,
not explicit rotation matrices. This form generalizes to higher dimensions
(3D spatial, 3D+curl, 3D+curl+velocity) by changing the orientation
parameterization and coefficient functions.

Core pattern:
    rotation_transform(coords, selection, found_angles, target_angles, pivots)

The rotation angle per block is (target - found). Transform is applied as
column-wise operations weighted by angle-dependent coefficients.
"""
import torch
from typing import Tuple, Optional


# =============================================================================
# AXIS COMPUTATION: find current orientation of each block
# =============================================================================

def compute_block_stats(
    coords: torch.Tensor,      # (N, d)
    selection: torch.Tensor,   # (N,) int in [0, K) or -1
    K: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute centroid and covariance statistics per block via scatter.

    Returns:
        centroids: (K, d)
        cov_matrices: (K, d, d) - covariance per block
        counts: (K,) - points per block
    """
    device = coords.device
    dtype = coords.dtype
    N, d = coords.shape

    # Valid mask
    valid = selection >= 0
    safe_sel = selection.clone()
    safe_sel[~valid] = 0

    # Counts per block
    counts = torch.zeros(K, device=device, dtype=dtype)
    counts.scatter_add_(0, safe_sel, valid.float())
    counts_safe = counts.clamp(min=1.0)

    # Centroids via scatter
    coord_sums = torch.zeros(K, d, device=device, dtype=dtype)
    for i in range(d):
        coord_sums[:, i].scatter_add_(0, safe_sel, coords[:, i] * valid.float())
    centroids = coord_sums / counts_safe.unsqueeze(-1)

    # Covariance via scatter (for axis computation)
    # cov[k] = E[(x - mu)(x - mu)^T] for block k
    centered = coords - centroids[safe_sel]  # (N, d)
    centered = centered * valid.unsqueeze(-1).float()

    # Compute outer products and scatter
    cov_matrices = torch.zeros(K, d, d, device=device, dtype=dtype)
    for i in range(d):
        for j in range(d):
            prod = centered[:, i] * centered[:, j]  # (N,)
            cov_matrices[:, i, j].scatter_add_(0, safe_sel, prod)
    cov_matrices = cov_matrices / counts_safe.unsqueeze(-1).unsqueeze(-1)

    return centroids, cov_matrices, counts


def principal_angle_2d(cov: torch.Tensor) -> torch.Tensor:
    """
    Extract principal axis angle from 2D covariance matrices.

    Args:
        cov: (K, 2, 2) covariance matrices

    Returns:
        angles: (K,) principal axis angle in radians

    Uses closed-form 2x2 eigenvector solution.
    """
    # For 2x2 symmetric [[a, b], [b, c]]:
    # Principal eigenvector direction: atan2(2b, a-c) / 2 + adjustment
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    c = cov[:, 1, 1]

    # Angle of principal axis
    # theta = 0.5 * atan2(2b, a - c)
    angles = 0.5 * torch.atan2(2 * b, a - c)

    return angles


def compute_block_axes_2d(
    coords: torch.Tensor,      # (N, 2)
    selection: torch.Tensor,   # (N,)
    K: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute centroid and principal axis angle for each 2D block.

    Returns:
        centroids: (K, 2)
        angles: (K,) axis angle in radians
        counts: (K,)
    """
    centroids, cov_matrices, counts = compute_block_stats(coords, selection, K)
    angles = principal_angle_2d(cov_matrices)
    return centroids, angles, counts


# =============================================================================
# ROTATION TRANSFORM: functional form with column-wise operations
# =============================================================================

def rotation_transform_2d(
    coords: torch.Tensor,           # (N, 2)
    selection: torch.Tensor,        # (N,) block labels
    found_angles: torch.Tensor,     # (K,) current axis angles
    target_angles: torch.Tensor,    # (K,) target axis angles
    pivots: torch.Tensor            # (K, 2) rotation centers
) -> torch.Tensor:                  # (N, 2)
    """
    Rotate 2D coordinates from found_angles to target_angles around pivots.

    The rotation angle per block is (target - found).
    Transform is column-wise operations weighted by cos/sin coefficients.

    No explicit rotation matrices are constructed.
    """
    N = coords.shape[0]
    device = coords.device

    # Rotation angle per block, then gather to per-node
    delta_angles = target_angles - found_angles  # (K,)

    # Handle invalid selections
    valid = selection >= 0
    safe_sel = selection.clone()
    safe_sel[~valid] = 0

    delta = delta_angles[safe_sel]  # (N,)
    pivot = pivots[safe_sel]        # (N, 2)

    # Center coordinates
    centered = coords - pivot  # (N, 2)
    x, y = centered[:, 0], centered[:, 1]

    # Rotation coefficients
    c = torch.cos(delta)  # (N,)
    s = torch.sin(delta)  # (N,)

    # Apply rotation as weighted column combination
    x_rot = c * x - s * y
    y_rot = s * x + c * y

    rotated = torch.stack([x_rot, y_rot], dim=-1) + pivot

    # Preserve unselected points
    result = torch.where(valid.unsqueeze(-1), rotated, coords)

    return result


# =============================================================================
# HIGH-LEVEL: rotate blocks to target orientation
# =============================================================================

def rotate_blocks_to_target_2d(
    coords: torch.Tensor,           # (N, 2)
    selection: torch.Tensor,        # (N,) block labels, -1 for unselected
    K: int,
    target_angles: Optional[torch.Tensor] = None,  # (K,) or None for +90°
    relative_angle: Optional[float] = None         # if target_angles is None, rotate by this
) -> Tuple[torch.Tensor, dict]:
    """
    Rotate each block around its centroid to target orientation.

    If target_angles is None and relative_angle is given, target = found + relative_angle.
    Default: rotate 90° (perpendicular to current orientation).

    Returns:
        rotated_coords: (N, 2)
        info: dict with computed intermediates
    """
    device = coords.device

    # Compute current block statistics
    centroids, found_angles, counts = compute_block_axes_2d(coords, selection, K)

    # Determine target angles
    if target_angles is None:
        if relative_angle is None:
            relative_angle = torch.pi / 2  # default: 90°
        target_angles = found_angles + relative_angle

    # Apply rotation
    rotated = rotation_transform_2d(
        coords, selection, found_angles, target_angles, centroids
    )

    info = {
        'centroids': centroids,
        'found_angles': found_angles,
        'target_angles': target_angles,
        'counts': counts,
    }

    return rotated, info


# =============================================================================
# EXTENSION STUBS: higher dimensional cases
# =============================================================================

def rotation_transform_3d(
    coords: torch.Tensor,           # (N, 3)
    selection: torch.Tensor,        # (N,)
    found_orientation: torch.Tensor,  # (K, 3) axis-angle or (K, 4) quaternion
    target_orientation: torch.Tensor, # (K, 3) or (K, 4)
    pivots: torch.Tensor            # (K, 3)
) -> torch.Tensor:
    """
    Rotate 3D coordinates. Orientation parameterized as axis-angle or quaternion.

    The transform is still column-wise: x, y, z weighted by orientation-dependent
    coefficients (derived from quaternion multiplication or Rodrigues formula).

    TODO: implement
    """
    raise NotImplementedError("3D rotation transform not yet implemented")


def rotation_transform_6d(
    coords: torch.Tensor,           # (N, 6) = (position, curl)
    selection: torch.Tensor,        # (N,)
    found_orientation: torch.Tensor,  # (K, ??) position + curl orientation
    target_orientation: torch.Tensor, # (K, ??)
    pivots: torch.Tensor            # (K, 6)
) -> torch.Tensor:
    """
    Rotate 6D coordinates (3D position + 3D curl/rotation).

    Position and curl may transform with coupled or independent rotations.
    The structure depends on the physical meaning of the curl dimensions.

    TODO: implement based on specific curl semantics
    """
    raise NotImplementedError("6D rotation transform not yet implemented")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_rotation_2d():
    """Numerical verification of 2D rotation correctness."""
    device = torch.device('cpu')

    # Simple test: diagonal line
    t = torch.linspace(0, 1, 5, device=device)
    coords = torch.stack([t, t * 1.5], dim=-1)  # slope 1.5, angle ~56°

    selection = torch.zeros(5, dtype=torch.long, device=device)
    K = 1

    rotated, info = rotate_blocks_to_target_2d(coords, selection, K, relative_angle=torch.pi/2)

    print("=" * 60)
    print("2D ROTATION VERIFICATION")
    print("=" * 60)

    centroid = info['centroids'][0]
    found = info['found_angles'][0] * 180 / torch.pi
    target = info['target_angles'][0] * 180 / torch.pi

    print(f"\nCentroid: ({centroid[0]:.4f}, {centroid[1]:.4f})")
    print(f"Found axis angle: {found:.1f}°")
    print(f"Target axis angle: {target:.1f}°")
    print(f"Rotation applied: {target - found:.1f}°")

    # Check perpendicularity via direction vectors
    orig_dir = coords[-1] - coords[0]
    rot_dir = rotated[-1] - rotated[0]

    orig_angle = torch.atan2(orig_dir[1], orig_dir[0]) * 180 / torch.pi
    rot_angle = torch.atan2(rot_dir[1], rot_dir[0]) * 180 / torch.pi

    print(f"\nOriginal segment direction: {orig_angle:.1f}°")
    print(f"Rotated segment direction: {rot_angle:.1f}°")
    print(f"Difference: {rot_angle - orig_angle:.1f}° (should be 90°)")

    # Dot product check
    orig_norm = orig_dir / torch.norm(orig_dir)
    rot_norm = rot_dir / torch.norm(rot_dir)
    dot = torch.dot(orig_norm, rot_norm)
    print(f"Dot product: {dot:.6f} (should be 0 for perpendicular)")

    # Length preservation
    orig_len = torch.norm(orig_dir)
    rot_len = torch.norm(rot_dir)
    print(f"\nLength: original={orig_len:.4f}, rotated={rot_len:.4f}, ratio={rot_len/orig_len:.4f}")

    # Point spacing
    print("\nPoint spacing preservation:")
    for i in range(4):
        d_orig = torch.norm(coords[i+1] - coords[i])
        d_rot = torch.norm(rotated[i+1] - rotated[i])
        print(f"  {i}-{i+1}: original={d_orig:.4f}, rotated={d_rot:.4f}")

    return coords, rotated, info


def demo_rotation_visual():
    """Visual demo with proper aspect ratio."""
    from PIL import Image, ImageDraw
    import numpy as np

    device = torch.device('cpu')

    # Three blocks: horizontal, vertical, diagonal
    block0 = torch.stack([torch.linspace(0, 2, 20, device=device),
                          torch.zeros(20, device=device) + 0.5], dim=-1)

    block1 = torch.stack([torch.zeros(15, device=device) + 3.0,
                          torch.linspace(0, 1.5, 15, device=device)], dim=-1)

    t = torch.linspace(0, 1, 25, device=device)
    block2 = torch.stack([4.0 + t, t * 1.5], dim=-1)

    coords = torch.cat([block0, block1, block2], dim=0)
    selection = torch.cat([
        torch.zeros(20, dtype=torch.long, device=device),
        torch.ones(15, dtype=torch.long, device=device),
        torch.full((25,), 2, dtype=torch.long, device=device)
    ])
    K = 3

    rotated, info = rotate_blocks_to_target_2d(coords, selection, K, relative_angle=torch.pi/2)

    print("\nBlock info:")
    for k in range(K):
        c = info['centroids'][k]
        fa = info['found_angles'][k] * 180 / torch.pi
        ta = info['target_angles'][k] * 180 / torch.pi
        print(f"  Block {k}: centroid=({c[0]:.2f}, {c[1]:.2f}), "
              f"found={fa:.1f}°, target={ta:.1f}°")

    # Visualization with EQUAL ASPECT RATIO
    coords_np = coords.numpy()
    rotated_np = rotated.numpy()
    selection_np = selection.numpy()
    centroids_np = info['centroids'].numpy()

    all_x = np.concatenate([coords_np[:, 0], rotated_np[:, 0]])
    all_y = np.concatenate([coords_np[:, 1], rotated_np[:, 1]])

    x_min, x_max = all_x.min() - 0.5, all_x.max() + 0.5
    y_min, y_max = all_y.min() - 0.5, all_y.max() + 0.5

    # Equal scale: pixels per data unit
    ppu = 60  # pixels per unit
    panel_w = int((x_max - x_min) * ppu)
    panel_h = int((y_max - y_min) * ppu)
    margin = 20

    img_w = 2 * panel_w + 3 * margin
    img_h = panel_h + 2 * margin

    img = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(img)

    def to_px(x, y, panel=0):
        px = margin + (x - x_min) * ppu + panel * (panel_w + margin)
        py = margin + (y_max - y) * ppu  # flip y
        return int(px), int(py)

    colors = [(220, 60, 60), (60, 180, 60), (60, 60, 220)]

    # Left panel: original
    draw.text((margin, 2), "Original", fill=(0, 0, 0))
    for k in range(K):
        mask = selection_np == k
        for p in coords_np[mask]:
            px, py = to_px(p[0], p[1], 0)
            draw.ellipse([px-3, py-3, px+3, py+3], fill=colors[k])
        # Centroid
        cx, cy = to_px(centroids_np[k, 0], centroids_np[k, 1], 0)
        draw.line([cx-5, cy-5, cx+5, cy+5], fill=colors[k], width=2)
        draw.line([cx-5, cy+5, cx+5, cy-5], fill=colors[k], width=2)

    # Right panel: rotated
    draw.text((margin + panel_w + margin, 2), "After 90° rotation", fill=(0, 0, 0))
    for k in range(K):
        mask = selection_np == k
        for p in rotated_np[mask]:
            px, py = to_px(p[0], p[1], 1)
            draw.ellipse([px-3, py-3, px+3, py+3], fill=colors[k])
        # Centroid (same position)
        cx, cy = to_px(centroids_np[k, 0], centroids_np[k, 1], 1)
        draw.line([cx-5, cy-5, cx+5, cy+5], fill=colors[k], width=2)
        draw.line([cx-5, cy+5, cx+5, cy-5], fill=colors[k], width=2)

    # Divider
    div_x = margin + panel_w + margin // 2
    draw.line([div_x, 0, div_x, img_h], fill=(180, 180, 180), width=1)

    img.save('demo_output/block_rotation_functional.png')
    print(f"\nSaved: demo_output/block_rotation_functional.png")

    return coords, rotated, info


if __name__ == "__main__":
    verify_rotation_2d()
    print("\n")
    demo_rotation_visual()
