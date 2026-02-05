"""
Construct abstract intermediate images that satisfy filter constraints.

Pipeline:
  1. Bitmap line art → SDF (creates gradients from binary)
  2. SDF → segment into bands → abstract regions
  3. Regions → sweep radial gradient / geometry → abstract_rich_image
  4. abstract_rich_image → heat diffusion / bandpass (now works!)
  5. filtered_abstract → control signal for pixel shader

The key: don't complain filters need gradients, MAKE gradients.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from spectral_ops_fast import (
    build_weighted_image_laplacian, heat_diffusion_sparse,
    chebyshev_filter, estimate_lambda_max, lanczos_k_eigenvectors,
    DEVICE
)


# =============================================================================
# STAGE 1: Bitmap → Rich Abstract Intermediates
# =============================================================================

def bitmap_to_sdf(img: np.ndarray) -> np.ndarray:
    """Convert bitmap to signed distance field - instant gradients."""
    # Threshold to binary
    binary = img > 0.5
    dist_outside = distance_transform_edt(binary)
    dist_inside = distance_transform_edt(~binary)
    sdf = dist_outside - dist_inside
    # Normalize to [0, 1]
    sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-8)
    return sdf


def sdf_to_banded_regions(sdf: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """Quantize SDF into bands - creates staircase of values."""
    bands = np.floor(sdf * n_bands) / n_bands
    return bands


def sweep_radial_gradient(img: np.ndarray, n_centers: int = 12) -> np.ndarray:
    """
    Place radial gradients at contour points, blend additively.
    Creates rich tonal variation from sparse seeds.
    """
    H, W = img.shape

    # Find contour points
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    ys, xs = np.where(grad_mag > np.percentile(grad_mag, 90))

    if len(xs) < n_centers:
        # Fallback to random if not enough contour
        xs = np.random.randint(0, W, n_centers)
        ys = np.random.randint(0, H, n_centers)
    else:
        indices = np.random.choice(len(xs), n_centers, replace=False)
        xs, ys = xs[indices], ys[indices]

    # Create radial gradients
    result = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]

    for cx, cy in zip(xs, ys):
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        # Radial falloff
        radius = min(H, W) * 0.3
        gradient = np.clip(1 - dist / radius, 0, 1)
        result += gradient

    # Normalize
    result = result / (result.max() + 1e-8)
    return result


def sweep_along_sdf_contours(sdf: np.ndarray, n_levels: int = 6) -> np.ndarray:
    """
    Create abstract image by varying intensity along SDF iso-contours.
    Each contour level gets a different base intensity + local modulation.
    """
    H, W = sdf.shape
    result = np.zeros((H, W), dtype=np.float32)

    levels = np.linspace(0.1, 0.9, n_levels)
    for i, level in enumerate(levels):
        # Mask for this level band
        if i < n_levels - 1:
            mask = (sdf >= level) & (sdf < levels[i+1])
        else:
            mask = sdf >= level

        # Base intensity for this band
        base = (i + 1) / n_levels

        # Local modulation: distance within band
        band_sdf = np.abs(sdf - level)
        band_sdf = 1 - np.clip(band_sdf * 10, 0, 1)  # Sharp falloff from level

        result[mask] = base * (0.5 + 0.5 * band_sdf[mask])

    return result


def create_abstract_from_bitmap(img: np.ndarray) -> dict:
    """
    Create multiple abstract intermediate representations from bitmap.
    Returns dict of named abstract images, all with rich gradients.
    """
    sdf = bitmap_to_sdf(img)

    return {
        'sdf': sdf,
        'sdf_banded': sdf_to_banded_regions(sdf, n_bands=8),
        'radial_sweep': sweep_radial_gradient(img, n_centers=15),
        'contour_sweep': sweep_along_sdf_contours(sdf, n_levels=8),
        'sdf_x_radial': sdf * sweep_radial_gradient(img, n_centers=10),
    }


# =============================================================================
# STAGE 2: Filters on Abstract Images (now they have gradients!)
# =============================================================================

def filter_heat_diffusion(abstract_img: np.ndarray, t: float = 0.2) -> np.ndarray:
    """Heat diffusion on abstract image - now has gradients to diffuse."""
    H, W = abstract_img.shape
    carrier = torch.tensor(abstract_img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    signal = carrier.flatten()

    diffused = heat_diffusion_sparse(L, signal, alpha=t, iterations=25)
    result = diffused.cpu().numpy().reshape(H, W)
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def filter_bandpass(abstract_img: np.ndarray, center: float = 0.5,
                    width: float = 0.25) -> np.ndarray:
    """Bandpass filter on abstract image."""
    H, W = abstract_img.shape
    carrier = torch.tensor(abstract_img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    signal = carrier.flatten()

    lambda_max = estimate_lambda_max(L)
    filtered = chebyshev_filter(L, signal,
                                center=center * lambda_max,
                                width=width * lambda_max,
                                order=20, lambda_max=lambda_max)
    result = filtered.cpu().numpy().reshape(H, W)
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def filter_eigenvector_phase(abstract_img: np.ndarray) -> np.ndarray:
    """Extract eigenvector phase from abstract image."""
    H, W = abstract_img.shape
    carrier = torch.tensor(abstract_img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)

    evs, _ = lanczos_k_eigenvectors(L, num_eigenvectors=2)
    if hasattr(evs, 'cpu'):
        evs = evs.cpu().numpy()

    ev0 = evs[:, 0].reshape(H, W)
    ev1 = evs[:, 1].reshape(H, W)

    phase = np.arctan2(ev1, ev0)
    return (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]


# =============================================================================
# STAGE 3: Combine filtered abstracts as control signals
# =============================================================================

def apply_as_gate(original: np.ndarray, control: np.ndarray,
                  strength: float = 0.8) -> np.ndarray:
    """Use control signal to gate original multiplicatively."""
    return original * (1 - strength + strength * control)


def apply_as_blend_selector(original: np.ndarray, alt: np.ndarray,
                            control: np.ndarray) -> np.ndarray:
    """Control selects between original and alt."""
    return control * alt + (1 - control) * original


def run_pipeline(input_path: str, output_dir: str):
    """
    Full pipeline: bitmap → abstract intermediates → filters → control → output
    """
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # === STAGE 1: Create abstract intermediates ===
    abstracts = create_abstract_from_bitmap(img)

    # === STAGE 2: Filter the abstracts ===
    filtered = {}

    # Heat diffusion on SDF (smooth structure-preserving)
    filtered['heat_sdf'] = filter_heat_diffusion(abstracts['sdf'], t=0.25)

    # Bandpass on radial sweep (isolate mid-frequencies)
    filtered['band_radial'] = filter_bandpass(abstracts['radial_sweep'],
                                               center=0.4, width=0.3)

    # Phase on contour sweep (topological structure)
    filtered['phase_contour'] = filter_eigenvector_phase(abstracts['contour_sweep'])

    # Heat diffusion on SDF×radial product
    filtered['heat_product'] = filter_heat_diffusion(abstracts['sdf_x_radial'], t=0.3)

    # === STAGE 3: Use as control signals ===
    outputs = {}

    # Heat-diffused SDF gates the original
    outputs['heat_gate'] = apply_as_gate(img, filtered['heat_sdf'], strength=0.85)

    # Bandpass-filtered radial gates original
    outputs['band_gate'] = apply_as_gate(img, filtered['band_radial'], strength=0.8)

    # Phase selects between original and inverted
    outputs['phase_select'] = apply_as_blend_selector(img, 1 - img,
                                                       filtered['phase_contour'])

    # Combine: phase controls blend between heat-gated and band-gated
    outputs['combined'] = apply_as_blend_selector(
        outputs['heat_gate'],
        outputs['band_gate'],
        filtered['phase_contour']
    )

    # === BUILD OUTPUT GRID ===
    # Row 1: Original + Abstract intermediates
    row1 = [img, abstracts['sdf'], abstracts['radial_sweep'], abstracts['contour_sweep']]

    # Row 2: Filtered abstracts
    row2 = [abstracts['sdf_x_radial'], filtered['heat_sdf'],
            filtered['band_radial'], filtered['phase_contour']]

    # Row 3: Final outputs using controls
    row3 = [img, outputs['heat_gate'], outputs['band_gate'], outputs['phase_select']]

    # Row 4: Combined and variations
    row4 = [outputs['combined'],
            apply_as_gate(img, filtered['heat_product'], 0.9),
            apply_as_gate(outputs['heat_gate'], filtered['phase_contour'], 0.7),
            apply_as_blend_selector(outputs['band_gate'], outputs['heat_gate'],
                                    filtered['heat_sdf'])]

    def make_row(imgs):
        return np.concatenate([np.stack([x]*3, axis=-1) for x in imgs], axis=1)

    grid = np.concatenate([make_row(r) for r in [row1, row2, row3, row4]], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"abstract_pipeline_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Labeled
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["Original", "SDF", "Radial Sweep", "Contour Sweep"],
        ["SDF×Radial", "Heat(SDF)", "Band(Radial)", "Phase(Contour)"],
        ["Original", "Heat⊗Orig", "Band⊗Orig", "Phase→Sel"],
        ["Combined", "Heat(Prod)⊗", "Heat⊗Phase⊗", "Nested Blend"],
    ]
    for row_idx, row_labels in enumerate(labels):
        for col_idx, lbl in enumerate(row_labels):
            draw.text((col_idx * W + 3, row_idx * H + 3), lbl, fill=(255, 50, 50))

    labeled.save(Path(output_dir) / f"abstract_pipeline_{stem}_labeled.png")
    print(f"Saved labeled")


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_pipeline(str(inp), out_dir)
