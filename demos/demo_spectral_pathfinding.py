#!/usr/bin/env python3
"""Demo 3: Spectral Pathfinding -- Phase E of TODO_DEMO_RECOVERY.md.

Demonstrates the spectral pathfinder using local graph spectra to approximate
Dijkstra without O(n^2) accesses. The key idea: accept a "Dijkstra-but-kinda-wrong"
which iterates toward the same targets with empirically measurable low transport
distance from optimal full-graph-access Dijkstra.

Composition chain:
    grid + obstacles -> spectral pathfinding -> path quality comparison -> visualization

Uses ONLY the _even_cuter Module files:
    - spectral_graph_embedding.py: LocalSpectralProbe
    - spectral_pathfinder.py: SpectralPathfinder, PathQualityEstimator,
      build_grid_adjacency, dijkstra_shortest_path

Zero imports from spectral_ops_fast.py or spectral_ops_fns.py.
All code pure PyTorch -- no numpy (except final PIL save via image_io).

Usage:
    uv run demos/demo_spectral_pathfinding.py
    uv run demos/demo_spectral_pathfinding.py --grid-size 48 --output-dir demo_output
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spectral_graph_embedding import LocalSpectralProbe
from spectral_pathfinder import (
    PathQualityEstimator,
    SpectralPathfinder,
    build_grid_adjacency,
    dijkstra_shortest_path,
)
from image_io import save_image


# ======================================================================
# Obstacle pattern generators (pure PyTorch)
# ======================================================================


def create_wall_with_gap(
    H: int, W: int, wall_col: int, gap_start: int, gap_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Vertical wall at wall_col with a gap from gap_start to gap_end (rows)."""
    blocked = torch.zeros(H, W, dtype=torch.bool, device=device)
    for r in range(H):
        if r < gap_start or r > gap_end:
            blocked[r, wall_col] = True
    return blocked


def create_double_wall(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Two vertical walls with gaps at opposite ends.

    Wall 1 at W/3 with gap at bottom.
    Wall 2 at 2W/3 with gap at top.
    The pathfinder must navigate down through gap 1, across, then up through gap 2.

    Gap size is generous (half the height) so the local-only pathfinder
    can find its way through with geometric + spectral guidance.
    """
    blocked = torch.zeros(H, W, dtype=torch.bool, device=device)
    gap_size = H // 2  # Very wide gaps for local-only navigation

    # Wall 1 at W/3, gap at bottom rows
    col1 = W // 3
    for r in range(H - gap_size):
        if col1 < W:
            blocked[r, col1] = True

    # Wall 2 at 2W/3, gap at top rows
    col2 = 2 * W // 3
    for r in range(gap_size, H):
        if col2 < W:
            blocked[r, col2] = True

    return blocked


def create_scattered_obstacles(
    H: int, W: int, device: torch.device,
) -> torch.Tensor:
    """Scattered circular obstacles along the diagonal."""
    blocked = torch.zeros(H, W, dtype=torch.bool, device=device)
    radius = max(2, H // 16)
    num_obstacles = 6

    for i in range(num_obstacles):
        frac = (i + 1) / (num_obstacles + 1)
        cr = int(H * frac)
        cc = int(W * frac)
        # Offset alternate obstacles to avoid blocking ALL diagonal paths
        if i % 2 == 0:
            cc = min(W - radius - 1, cc + radius * 2)
        else:
            cc = max(radius, cc - radius * 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        blocked[nr, nc] = True
    return blocked


def create_narrow_passage(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Two large blocks with a narrow 2-cell passage between them."""
    blocked = torch.zeros(H, W, dtype=torch.bool, device=device)
    mid_row = H // 2
    passage_width = 2

    # Top block: rows from H//4 to mid-passage, cols from W//4 to 3*W//4
    for r in range(H // 4, mid_row - passage_width // 2):
        for c in range(W // 4, 3 * W // 4):
            blocked[r, c] = True

    # Bottom block: rows from mid+passage to 3*H//4, same cols
    for r in range(mid_row + passage_width // 2 + 1, 3 * H // 4):
        for c in range(W // 4, 3 * W // 4):
            blocked[r, c] = True

    return blocked


# ======================================================================
# Visualization (pure PyTorch -> image tensor)
# ======================================================================


def visualize_paths(
    H: int,
    W: int,
    blocked: torch.Tensor,
    spectral_path: torch.Tensor,
    dijkstra_path: torch.Tensor,
    coords: torch.Tensor,
    start: int,
    goal: int,
) -> torch.Tensor:
    """Create (H, W, 3) float32 visualization image.

    - White = free space
    - Dark gray = obstacles
    - Red = spectral path
    - Blue = Dijkstra path
    - Green = start marker
    - Yellow = goal marker
    """
    # Base image: white for free, dark gray for blocked
    img = torch.ones(H, W, 3, dtype=torch.float32)
    blocked_2d = blocked.float()
    for c in range(3):
        img[:, :, c] = torch.where(blocked, torch.full_like(blocked_2d, 0.25), img[:, :, c])

    # Draw Dijkstra path first (blue, slightly wider) so spectral can overlay
    if dijkstra_path.numel() > 0:
        for idx in range(dijkstra_path.shape[0]):
            node = dijkstra_path[idx].item()
            r = int(coords[node, 0].item())
            c_val = int(coords[node, 1].item())
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr, cc = r + dr, c_val + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        img[rr, cc, 0] = 0.2
                        img[rr, cc, 1] = 0.3
                        img[rr, cc, 2] = 0.9

    # Draw spectral path (red)
    if spectral_path.numel() > 0:
        for idx in range(spectral_path.shape[0]):
            node = spectral_path[idx].item()
            r = int(coords[node, 0].item())
            c_val = int(coords[node, 1].item())
            if 0 <= r < H and 0 <= c_val < W:
                img[r, c_val, 0] = 0.9
                img[r, c_val, 1] = 0.2
                img[r, c_val, 2] = 0.2

    # Start marker (green, 3x3)
    sr = int(coords[start, 0].item())
    sc = int(coords[start, 1].item())
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr, cc = sr + dr, sc + dc
            if 0 <= rr < H and 0 <= cc < W:
                img[rr, cc] = torch.tensor([0.0, 0.9, 0.0])

    # Goal marker (yellow, 3x3)
    gr = int(coords[goal, 0].item())
    gc = int(coords[goal, 1].item())
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr, cc = gr + dr, gc + dc
            if 0 <= rr < H and 0 <= cc < W:
                img[rr, cc] = torch.tensor([0.9, 0.9, 0.0])

    return img


# ======================================================================
# Scenario runner
# ======================================================================


def run_scenario(
    name: str,
    H: int,
    W: int,
    blocked: torch.Tensor,
    start_node: int,
    goal_node: int,
    device: torch.device,
    output_dir: str,
) -> Dict:
    """Run one pathfinding scenario, returning metrics dict."""
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")

    n = H * W
    blocked_count = int(blocked.sum().item())
    print(f"  Grid: {H}x{W} = {n} nodes, {blocked_count} blocked ({100*blocked_count/n:.1f}%)")

    # Build graph
    t0 = time.time()
    adjacency, coords = build_grid_adjacency(H, W, blocked=blocked, connectivity=4, device=device)
    build_time = time.time() - t0
    nnz = adjacency._nnz()
    print(f"  Graph built: {nnz} edges in {build_time:.3f}s")

    start_coord = coords[start_node]
    goal_coord = coords[goal_node]
    print(f"  Start: node {start_node} at ({start_coord[0].item():.0f}, {start_coord[1].item():.0f})")
    print(f"  Goal:  node {goal_node} at ({goal_coord[0].item():.0f}, {goal_coord[1].item():.0f})")

    # --- Dijkstra baseline ---
    print("\n  --- Dijkstra (optimal) ---")
    t0 = time.time()
    dij_path, dij_success, dij_cost = dijkstra_shortest_path(adjacency, start_node, goal_node)
    dij_time = time.time() - t0
    print(f"  Success: {dij_success}")
    if dij_success:
        print(f"  Path length: {dij_path.shape[0]} nodes")
        print(f"  Cost: {dij_cost:.2f}")
    print(f"  Time: {dij_time:.4f}s")

    if not dij_success:
        print("  Dijkstra failed -- no path exists. Skipping scenario.")
        return {"name": name, "success": False, "reason": "no_path"}

    # --- Spectral pathfinder ---
    print("\n  --- Spectral Pathfinder ---")
    probe = LocalSpectralProbe(hop_radius=2, lanczos_iterations=15)
    # Higher exploration for scenarios with more blocked cells --
    # the local-only pathfinder needs stochastic escape from dead ends
    blocked_frac = blocked_count / n
    exploration = 0.03 if blocked_frac < 0.1 else 0.10
    pathfinder = SpectralPathfinder(
        local_probe=probe,
        spectral_weight=0.3,
        heuristic_weight=0.7,
        exploration_probability=exploration,
    )

    t0 = time.time()
    spec_path, diag = pathfinder(adjacency, coords, start_node, goal_node, max_steps=n * 2)
    spec_time = time.time() - t0

    print(f"  Success: {diag['success']}")
    print(f"  Path length: {diag['path_length']} nodes")
    print(f"  Steps: {diag['steps']}")
    print(f"  Avg confidence: {diag['avg_confidence']:.4f}")
    print(f"  Nodes visited: {diag['nodes_visited']}")
    print(f"  Time: {spec_time:.4f}s")

    # --- Quality comparison ---
    quality = {}
    if dij_success and diag["success"]:
        print("\n  --- Quality Metrics ---")
        estimator = PathQualityEstimator()
        quality = estimator(spec_path, dij_path, coords, adjacency)
        print(f"  Length ratio: {quality['length_ratio'].item():.2f}x")
        print(f"  Cost ratio:   {quality['cost_ratio'].item():.2f}x")
        print(f"  Wasserstein:  {quality['wasserstein_approx'].item():.2f}")
        print(f"  Hausdorff:    {quality['hausdorff'].item():.2f}")

    # --- Visualization ---
    print("\n  Generating visualization...")
    vis = visualize_paths(H, W, blocked, spec_path, dij_path, coords, start_node, goal_node)
    out_path = os.path.join(output_dir, f"demo3_pathfinding_{name}.png")
    save_image(vis, out_path, timestamp=True)

    return {
        "name": name,
        "grid": f"{H}x{W}",
        "blocked_pct": f"{100*blocked_count/n:.1f}%",
        "dijkstra_success": dij_success,
        "dijkstra_length": dij_path.shape[0] if dij_success else 0,
        "dijkstra_cost": dij_cost,
        "dijkstra_time": dij_time,
        "spectral_success": diag["success"],
        "spectral_length": diag["path_length"],
        "spectral_time": spec_time,
        "length_ratio": quality["length_ratio"].item() if quality else 0,
        "cost_ratio": quality["cost_ratio"].item() if quality else 0,
        "wasserstein": quality["wasserstein_approx"].item() if quality else 0,
        "hausdorff": quality["hausdorff"].item() if quality else 0,
    }


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Demo 3: Spectral Pathfinding -- local spectra approximate Dijkstra"
    )
    parser.add_argument("--grid-size", type=int, default=32,
                        help="Grid dimension (NxN). Default 32.")
    parser.add_argument("--output-dir", type=str, default="demo_output",
                        help="Output directory for images. Default demo_output.")
    args = parser.parse_args()

    N = args.grid_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")  # Pathfinding is sequential; CPU is fine

    print("=" * 60)
    print("DEMO 3: Spectral Pathfinding")
    print("  Local graph spectra approximate Dijkstra")
    print("  without O(n^2) full-graph access")
    print("=" * 60)
    print(f"Grid size: {N}x{N}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    # Define scenarios
    scenarios = []

    # Scenario 1: Wall with gap
    blocked_1 = create_wall_with_gap(
        N, N,
        wall_col=N // 2,
        gap_start=N * 3 // 8,
        gap_end=N * 5 // 8,
        device=device,
    )
    scenarios.append(("wall_with_gap", blocked_1, 0, N * N - 1))

    # Scenario 2: Double wall
    blocked_2 = create_double_wall(N, N, device=device)
    scenarios.append(("double_wall", blocked_2, 0, N * N - 1))

    # Scenario 3: Scattered obstacles
    blocked_3 = create_scattered_obstacles(N, N, device=device)
    scenarios.append(("scattered_obstacles", blocked_3, 0, N * N - 1))

    # Scenario 4: Narrow passage
    blocked_4 = create_narrow_passage(N, N, device=device)
    # Start at top-left free area, goal at bottom-right free area
    start_4 = 0
    goal_4 = N * N - 1
    scenarios.append(("narrow_passage", blocked_4, start_4, goal_4))

    # Run all scenarios
    results = []
    for name, blocked, start, goal in scenarios:
        res = run_scenario(name, N, N, blocked, start, goal, device, output_dir)
        results.append(res)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (
        f"{'Scenario':<20} {'Grid':<8} {'Blocked':<8} "
        f"{'Dij Len':>8} {'Spec Len':>9} {'Len Ratio':>10} "
        f"{'Cost Ratio':>11} {'Wasser.':>8} {'Hausdorff':>10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        if not r.get("spectral_success", False):
            print(f"{r['name']:<20} {r.get('grid',''):<8} {r.get('blocked_pct',''):<8} "
                  f"{'FAILED':>8}")
            continue
        print(
            f"{r['name']:<20} {r['grid']:<8} {r['blocked_pct']:<8} "
            f"{r['dijkstra_length']:>8} {r['spectral_length']:>9} "
            f"{r['length_ratio']:>10.2f} {r['cost_ratio']:>11.2f} "
            f"{r['wasserstein']:>8.2f} {r['hausdorff']:>10.2f}"
        )

    print("\n" + "=" * 100)
    print("TIMING")
    print("=" * 100)
    for r in results:
        dtime = r.get("dijkstra_time", 0)
        stime = r.get("spectral_time", 0)
        print(f"  {r['name']:<20}  Dijkstra: {dtime:.4f}s  Spectral: {stime:.4f}s")

    # Overall assessment
    successful = [r for r in results if r.get("spectral_success", False)]
    print(f"\n{len(successful)}/{len(results)} scenarios completed successfully.")
    if successful:
        avg_length_ratio = sum(r["length_ratio"] for r in successful) / len(successful)
        avg_cost_ratio = sum(r["cost_ratio"] for r in successful) / len(successful)
        avg_wasserstein = sum(r["wasserstein"] for r in successful) / len(successful)
        print(f"  Average length ratio:  {avg_length_ratio:.2f}x")
        print(f"  Average cost ratio:    {avg_cost_ratio:.2f}x")
        print(f"  Average Wasserstein:   {avg_wasserstein:.2f}")

    print("\nDemo 3 complete.")


if __name__ == "__main__":
    main()
