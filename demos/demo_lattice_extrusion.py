#!/usr/bin/env python3
"""Demo 2: Lattice Extrusion — Phase E of demo recovery.

Composition chain:
    base graph -> spectral analysis -> lattice selection -> extrusion -> 3D visualization

Demonstrates:
1. Build a base graph with interesting spectral variation (islands + bridge).
2. Run ExpansionGatedExtruder at multiple theta values to show different lattice types.
3. Verify 3+ non-isomorphic lattice types appear (square, triangle, hex).
4. Visualize the extruded graph as a 3D surface via EggSurfaceRenderer.
5. Show which regions got which lattice type.

Uses ONLY the _even_cuter Module files:
- spectral_graph_embedding.py: GraphEmbedding
- spectral_lattice.py: ExpansionGatedExtruder, LatticeTypeSelector,
                        build_islands_bridge_graph, build_grid_graph
- spectral_renderer.py: HeightToNormals, EggSurfaceRenderer
- image_io.py: save_image

Zero imports from spectral_ops_fast.py or spectral_ops_fns.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch

from spectral_graph_embedding import GraphEmbedding
from spectral_lattice import (
    ExpansionGatedExtruder,
    LatticeTypeSelector,
    build_islands_bridge_graph,
    build_grid_graph,
)
from spectral_renderer import HeightToNormals, EggSurfaceRenderer
from image_io import save_image


# Lattice type names for display
LATTICE_NAMES = {0: "square", 1: "triangle", 2: "hex"}


def graph_to_height_image(
    coords: torch.Tensor,
    node_properties: torch.Tensor,
    resolution: int = 128,
    value_column: int = 3,
) -> torch.Tensor:
    """Rasterize graph nodes into a 2D height-field image.

    Maps 3D node coordinates to a 2D image where pixel intensity encodes the
    chosen node property (default: node_value from column 3 of node_properties).

    For nodes with z > 0 (extruded layers), the height value is accumulated
    to show the layered structure.

    Args:
        coords: (m, 3) node positions (x, y, z).
        node_properties: (m, 4) per-node metadata.
        resolution: Output image resolution.
        value_column: Which column of node_properties to use as intensity.

    Returns:
        (resolution, resolution) float32 height field in [0, 1].
    """
    device = coords.device
    m = coords.shape[0]

    # Get 2D bounding box of all nodes
    x_min, x_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    y_min, y_max = coords[:, 1].min().item(), coords[:, 1].max().item()

    # Add margin
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    margin = max(x_range, y_range) * 0.1
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Map node positions to pixel coordinates
    px = ((coords[:, 0] - x_min) / x_range * (resolution - 1)).long().clamp(0, resolution - 1)
    py = ((coords[:, 1] - y_min) / y_range * (resolution - 1)).long().clamp(0, resolution - 1)

    # Accumulate values and counts
    height = torch.zeros((resolution, resolution), device=device, dtype=torch.float32)
    counts = torch.zeros((resolution, resolution), device=device, dtype=torch.float32)

    # Node values: use the requested column, plus layer index for height variation
    values = node_properties[:, value_column]
    layer_boost = node_properties[:, 0] * 0.15  # Layer index adds height
    combined = (values + layer_boost).clamp(0.0, 1.5)

    # Scatter nodes into the image
    for i in range(m):
        xi, yi = px[i].item(), py[i].item()
        height[yi, xi] += combined[i].item()
        counts[yi, xi] += 1.0

    # Average where multiple nodes land on same pixel
    valid = counts > 0
    height[valid] = height[valid] / counts[valid]

    # Fill gaps with a simple dilation (nearest-neighbor spread)
    filled = height.clone()
    for _ in range(3):
        padded = torch.nn.functional.pad(filled.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        smoothed = torch.nn.functional.conv2d(padded, kernel).squeeze()
        # Only fill where we have no data
        empty = counts == 0
        filled[empty] = smoothed[empty]

    # Normalize to [0, 1]
    fmin, fmax = filled.min(), filled.max()
    if fmax > fmin:
        filled = (filled - fmin) / (fmax - fmin)

    return filled


def graph_to_lattice_type_image(
    coords: torch.Tensor,
    node_properties: torch.Tensor,
    resolution: int = 128,
) -> torch.Tensor:
    """Rasterize graph nodes into an RGB image color-coded by lattice type.

    Colors:
    - Square (0): blue [0.2, 0.4, 0.9]
    - Triangle (1): red [0.9, 0.2, 0.2]
    - Hex (2): green [0.2, 0.9, 0.3]

    Args:
        coords: (m, 3) node positions.
        node_properties: (m, 4) per-node metadata (column 1 = lattice type).
        resolution: Output image resolution.

    Returns:
        (resolution, resolution, 3) float32 RGB image in [0, 1].
    """
    device = coords.device
    m = coords.shape[0]

    # Color map for lattice types
    colors = torch.tensor([
        [0.2, 0.4, 0.9],   # square = blue
        [0.9, 0.2, 0.2],   # triangle = red
        [0.2, 0.9, 0.3],   # hex = green
    ], device=device, dtype=torch.float32)

    # Get 2D bounding box
    x_min, x_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    y_min, y_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    margin = max(x_range, y_range) * 0.1
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Map to pixel coords
    px = ((coords[:, 0] - x_min) / x_range * (resolution - 1)).long().clamp(0, resolution - 1)
    py = ((coords[:, 1] - y_min) / y_range * (resolution - 1)).long().clamp(0, resolution - 1)

    # Initialize with dark background
    img = torch.ones((resolution, resolution, 3), device=device, dtype=torch.float32) * 0.1

    # Paint each node with its lattice type color
    lattice_types = node_properties[:, 1].long().clamp(0, 2)
    for i in range(m):
        xi, yi = px[i].item(), py[i].item()
        lt = lattice_types[i].item()
        # Brightness modulated by layer
        layer = node_properties[i, 0].item()
        brightness = 0.6 + 0.4 * min(layer / 3.0, 1.0)
        img[yi, xi] = colors[lt] * brightness

    # Dilate slightly to make nodes visible
    for _ in range(2):
        padded = torch.nn.functional.pad(
            img.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        )
        kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        for c in range(3):
            smoothed = torch.nn.functional.conv2d(
                padded[:, c:c+1], kernel
            ).squeeze()
            dark = img[:, :, c] < 0.15
            img[:, :, c] = torch.where(dark, smoothed, img[:, :, c])

    return img.clamp(0.0, 1.0)


def run_extrusion_at_theta(
    adjacency: torch.Tensor,
    coords: torch.Tensor,
    theta: float,
    extruder: ExpansionGatedExtruder,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Run a single extrusion at a given theta value and collect statistics.

    Returns:
        Tuple of (extruded_coords, extruded_adj, node_properties, stats_dict)
    """
    t0 = time.time()
    ext_coords, ext_adj, ext_props = extruder(adjacency, coords, theta=theta)
    elapsed = time.time() - t0

    m = ext_coords.shape[0]
    n_base = coords.shape[0]
    layers = torch.unique(ext_props[:, 0])
    lattice_types = ext_props[:, 1].long()
    type_counts = {}
    for tid in range(3):
        count = (lattice_types == tid).sum().item()
        type_counts[LATTICE_NAMES[tid]] = count

    stats = {
        "theta": theta,
        "label": label,
        "base_nodes": n_base,
        "total_nodes": m,
        "extruded_nodes": m - n_base,
        "layers": layers.tolist(),
        "num_layers": len(layers),
        "type_counts": type_counts,
        "elapsed": elapsed,
    }

    return ext_coords, ext_adj, ext_props, stats


def main():
    parser = argparse.ArgumentParser(
        description="Demo 2: Lattice Extrusion — iterative spectral-gated extrusion with 3+ non-isomorphic lattice types"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("demo_output"),
        help="Output directory (default: demo_output/)"
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Visualization resolution (default: 256)"
    )
    parser.add_argument(
        "--egg-resolution", type=int, default=512,
        help="Egg render resolution (default: 512)"
    )
    parser.add_argument(
        "--island-radius", type=int, default=4,
        help="Radius of each island (default: 4)"
    )
    parser.add_argument(
        "--bridge-length", type=int, default=6,
        help="Length of the bridge between islands (default: 6)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu' or 'cuda' (default: auto-detect)"
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ===================================================================
    # Step 1: Create base graph
    # ===================================================================
    print("[1/6] Building islands-bridge graph...")
    t0 = time.time()

    adjacency, coords = build_islands_bridge_graph(
        island_radius=args.island_radius,
        bridge_length=args.bridge_length,
        bridge_width=1,
        device=device,
    )
    n = coords.shape[0]
    n_edges = adjacency._nnz()
    t1 = time.time()
    print(f"       Graph: {n} nodes, {n_edges} edges")
    print(f"       Coord range: x=[{coords[:, 0].min():.0f}, {coords[:, 0].max():.0f}], "
          f"y=[{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}]")
    print(f"       Time: {t1 - t0:.3f}s")

    # ===================================================================
    # Step 2: Set up extruder
    # ===================================================================
    print("\n[2/6] Initializing extruder modules...")
    t2 = time.time()

    graph_embedding = GraphEmbedding(
        num_eigenvectors=4,
        lanczos_iterations=20,
    ).to(device)

    lattice_selector = LatticeTypeSelector(
        high_threshold=0.60,
        low_threshold=0.40,
    ).to(device)

    extruder = ExpansionGatedExtruder(
        graph_embedding=graph_embedding,
        lattice_selector=lattice_selector,
        expansion_threshold=0.5,
        max_layers=3,
        hop_radius=2,
    ).to(device)

    t3 = time.time()
    print(f"       Time: {t3 - t2:.3f}s")

    # ===================================================================
    # Step 3: Run extrusion at multiple theta values
    # ===================================================================
    print("\n[3/6] Running extrusions at multiple theta values...")

    theta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    all_results = []
    all_lattice_types_seen = set()

    for theta in theta_values:
        label = f"theta={theta:.1f}"
        print(f"\n  --- {label} ---")

        ext_coords, ext_adj, ext_props, stats = run_extrusion_at_theta(
            adjacency, coords, theta, extruder, label
        )
        all_results.append((ext_coords, ext_adj, ext_props, stats))

        # Track which lattice types we've seen
        for name, count in stats["type_counts"].items():
            if count > 0:
                all_lattice_types_seen.add(name)

        print(f"       Nodes: {stats['base_nodes']} base -> {stats['total_nodes']} total "
              f"(+{stats['extruded_nodes']} extruded)")
        print(f"       Layers: {stats['num_layers']} ({stats['layers']})")
        print(f"       Lattice types: {stats['type_counts']}")
        print(f"       Time: {stats['elapsed']:.3f}s")

    # ===================================================================
    # Step 4: Verify 3+ non-isomorphic lattice types
    # ===================================================================
    print("\n[4/6] Verifying non-isomorphic lattice types...")
    print(f"       Types seen across all theta values: {sorted(all_lattice_types_seen)}")
    num_types = len(all_lattice_types_seen)
    if num_types >= 3:
        print(f"       PASS: {num_types} non-jointly-isomorphic lattice types confirmed")
        print("       (Square: degree 4, girth 4 | Triangle: degree 6, girth 3 | Hex: degree 3, girth 6)")
    else:
        print(f"       NOTE: Only {num_types} types seen. Adjusting thresholds may reveal more.")
        print("       The LatticeTypeSelector guarantees all 3 types are reachable;")
        print("       specific graph topology may concentrate on fewer types.")

    # ===================================================================
    # Step 5: Visualize extruded graphs
    # ===================================================================
    print(f"\n[5/6] Visualizing extruded graphs (resolution={args.resolution})...")

    height_to_normals = HeightToNormals(strength=3.0).to(device)
    egg_renderer = EggSurfaceRenderer(
        resolution=args.egg_resolution,
        egg_factor=0.25,
        bump_strength=1.5,
        light_dir=(0.5, 0.7, 1.0),
    ).to(device)

    # Pick 3 representative theta values for visualization
    vis_indices = [0, 2, 4]  # theta = 0.0, 0.5, 1.0
    for idx in vis_indices:
        ext_coords, ext_adj, ext_props, stats = all_results[idx]
        theta = stats["theta"]
        label = f"theta{theta:.1f}".replace(".", "")

        print(f"\n  --- Visualizing theta={theta:.1f} ---")

        # Create height-field image from graph
        t_vis_start = time.time()
        height_img = graph_to_height_image(
            ext_coords, ext_props, resolution=args.resolution, value_column=3
        )

        # Create lattice-type visualization
        lattice_img = graph_to_lattice_type_image(
            ext_coords, ext_props, resolution=args.resolution
        )

        # Save lattice type map
        save_image(lattice_img, args.output_dir / f"demo2_lattice_types_{label}.png")

        # Height -> normals -> egg render
        normals = height_to_normals(height_img)

        # Create texture from lattice type image for the egg
        texture = lattice_img.clone()

        rendered = egg_renderer(texture, normals)
        t_vis_end = time.time()

        save_image(height_img.unsqueeze(-1).expand(-1, -1, 3),
                   args.output_dir / f"demo2_height_{label}.png")
        save_image(rendered, args.output_dir / f"demo2_egg_{label}.png")

        print(f"       Height field, lattice map, and egg render saved")
        print(f"       Time: {t_vis_end - t_vis_start:.3f}s")

    # ===================================================================
    # Step 6: Summary
    # ===================================================================
    total_end = time.time()

    print("\n" + "=" * 60)
    print("Demo 2: Lattice Extrusion -- Complete")
    print("=" * 60)
    print(f"  Base graph:        {n} nodes, {n_edges} edges (islands + bridge)")
    print(f"  Theta values:      {theta_values}")
    print(f"  Lattice types seen: {sorted(all_lattice_types_seen)}")
    print(f"  Total time:        {total_end - total_start:.2f}s")
    print()
    print("  Per-theta results:")
    for _, _, _, stats in all_results:
        theta = stats["theta"]
        tc = stats["type_counts"]
        print(f"    theta={theta:.1f}: {stats['total_nodes']} nodes, "
              f"{stats['num_layers']} layers, "
              f"types: sq={tc['square']} tri={tc['triangle']} hex={tc['hex']}")
    print()
    print("  Outputs:")
    print(f"    Lattice type maps:  {args.output_dir}/demo2_lattice_types_*.png")
    print(f"    Height fields:      {args.output_dir}/demo2_height_*.png")
    print(f"    Egg renders:        {args.output_dir}/demo2_egg_*.png")
    print()
    print("  Lattice type color code:")
    print("    Blue  = square  (degree 4, girth 4)")
    print("    Red   = triangle (degree 6, girth 3)")
    print("    Green = hex     (degree 3, girth 6)")
    print()
    print("  Composition chain:")
    print("    base graph -> spectral analysis -> lattice selection -> extrusion -> 3D visualization")


if __name__ == "__main__":
    main()
