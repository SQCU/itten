#!/usr/bin/env python3
"""
Pathfinding on Texture Surfaces

Demonstrates the integration of pathfinding with texture synthesis:
1. Generates texture with nodal line "walls" using spectral synthesis
2. Creates blocking mask from height threshold
3. Finds paths using spectral, A*, and Dijkstra methods
4. Compares and visualizes results

This demo shows how pathfinding can navigate around the nodal line
structures created by spectral texture synthesis.

Output saved to: outputs/pathfinding_demo/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

# Import texture synthesis from unified module
from texture import (
    synthesize as synthesize_texture,
    generate_amongus,
    generate_checkerboard,
)

# Import pathfinding module
from pathfinding import (
    find_surface_path,
    compare_surface_paths,
    create_blocking_mask,
    find_valid_endpoints,
    visualize_path,
    visualize_blocking,
    create_summary_image,
    save_image,
    PathResult,
)


def generate_noise(size: int, scale: float = 8.0, seed: int = 42) -> np.ndarray:
    """
    Generate smooth noise pattern for operand.

    Args:
        size: Output size (square)
        scale: Frequency scale (higher = more detail)
        seed: Random seed

    Returns:
        Noise pattern normalized to [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Multi-scale noise
    noise = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0

    freq = scale
    while freq >= 1.0:
        # Generate noise at this scale
        small_size = int(size / freq) + 1
        small_noise = rng.random((small_size, small_size)).astype(np.float32)

        # Upsample to full size using linear interpolation
        from scipy import ndimage
        upsampled = ndimage.zoom(small_noise, freq, order=1)
        upsampled = upsampled[:size, :size]

        noise += amplitude * upsampled
        amplitude *= 0.5
        freq /= 2

    # Normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


def run_demo():
    """Run the pathfinding demo."""
    print("=" * 60)
    print("PATHFINDING ON TEXTURE SURFACES")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "pathfinding_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Parameters
    size = 256  # Image size
    theta = 0.3  # Low theta = prominent Fiedler/nodal lines
    blocking_threshold = 0.7  # Block heights above this

    print(f"\nGenerating texture ({size}x{size})...")
    print(f"  theta = {theta} (low = prominent nodal lines)")
    print(f"  blocking threshold = {blocking_threshold}")

    # Generate carrier (Among Us pattern, scaled up)
    carrier = generate_amongus(size)

    # Generate operand (noise pattern)
    operand = generate_noise(size, scale=8.0, seed=42)

    # Synthesize texture
    print("\nSynthesizing texture with spectral methods...")
    heightfield = synthesize_texture(
        carrier=carrier,
        operand=operand,
        theta=theta,
        gamma=0.3,
        num_eigenvectors=8,
        edge_threshold=0.1
    )

    print(f"  Heightfield range: [{heightfield.min():.3f}, {heightfield.max():.3f}]")

    # Create blocking mask
    blocking = create_blocking_mask(heightfield, threshold=blocking_threshold, above=True)
    blocked_percentage = 100.0 * blocking.sum() / blocking.size
    print(f"  Blocked pixels: {blocked_percentage:.1f}%")

    # Save heightfield and blocking visualization
    print("\nSaving heightfield and blocking...")

    # Convert heightfield to image
    from pathfinding.visualization import heightfield_to_rgb
    heightfield_img = heightfield_to_rgb(heightfield, colormap='viridis')
    save_image(heightfield_img, str(output_dir / "01_heightfield.png"))

    # Save blocking overlay
    blocking_img = visualize_blocking(
        heightfield, blocking,
        blocked_color=(200, 50, 50),
        blocked_alpha=0.6
    )
    save_image(blocking_img, str(output_dir / "02_blocking.png"))

    # Find valid start and goal
    print("\nFinding valid start/goal points...")
    try:
        start, goal = find_valid_endpoints(blocking, margin=20)
    except ValueError:
        # If too much blocking, lower threshold
        print("  Too much blocking, adjusting threshold...")
        blocking = create_blocking_mask(heightfield, threshold=0.8, above=True)
        start, goal = find_valid_endpoints(blocking, margin=20)

    print(f"  Start: {start}")
    print(f"  Goal: {goal}")

    # Run pathfinding comparison
    print("\nRunning pathfinding algorithms...")
    print("-" * 40)

    # Run comparison (Dijkstra, A*, Spectral)
    results = compare_surface_paths(
        heightfield=heightfield,
        start=start,
        goal=goal,
        blocking_mask=blocking,
        elevation_cost_scale=1.0,
        connectivity="8",  # 8-connectivity allows diagonal moves
        max_steps=5000,    # More steps for spectral
        max_nodes=200000   # More nodes for classical
    )

    # Print results
    method_order = ['dijkstra', 'astar', 'spectral']
    idx = 0
    while idx < len(method_order):
        method = method_order[idx]
        if method in results:
            result = results[method]
            print(f"\n{method.upper()}:")
            print(f"  Success: {result.success}")
            if result.success:
                print(f"  Path length: {result.path_length} steps")
                print(f"  Total cost: {result.total_cost:.2f}")
                print(f"  Nodes visited: {result.nodes_visited}")
                print(f"  Time: {result.computation_time:.3f}s")
                if result.cost_ratio is not None:
                    print(f"  Cost ratio: {result.cost_ratio:.2f}x optimal")
        idx += 1

    # Save individual path visualizations
    print("\nSaving path visualizations...")

    method_colors = {
        'dijkstra': (50, 200, 50),   # Green
        'astar': (50, 50, 255),       # Blue
        'spectral': (255, 50, 50),    # Red
    }

    idx = 0
    while idx < len(method_order):
        method = method_order[idx]
        if method in results:
            result = results[method]
            if result.success:
                path_img = visualize_path(
                    heightfield, result,
                    path_color=method_colors[method],
                    line_width=2,
                    colormap='viridis'
                )
                save_image(path_img, str(output_dir / f"03_path_{method}.png"))
        idx += 1

    # Save comparison summary
    print("\nSaving comparison summary...")
    try:
        summary_img = create_summary_image(
            heightfield=heightfield,
            results=results,
            blocking_mask=blocking,
            output_path=str(output_dir / "04_comparison.png"),
            title=f"Pathfinding Comparison (theta={theta})"
        )
    except Exception as e:
        print(f"  Could not create summary image: {e}")
        # Fallback to simple comparison
        from pathfinding.visualization import visualize_comparison
        comparison_img = visualize_comparison(
            heightfield, results, blocking,
            output_path=str(output_dir / "04_comparison.png")
        )

    # Print metrics summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'dijkstra' in results and results['dijkstra'].success:
        optimal_cost = results['dijkstra'].total_cost
        optimal_length = results['dijkstra'].path_length
        optimal_time = results['dijkstra'].computation_time

        print(f"\nOptimal (Dijkstra):")
        print(f"  Cost: {optimal_cost:.2f}")
        print(f"  Length: {optimal_length} steps")
        print(f"  Time: {optimal_time:.3f}s")

        if 'astar' in results and results['astar'].success:
            astar = results['astar']
            print(f"\nA*:")
            print(f"  Cost ratio: {astar.total_cost / optimal_cost:.3f}x")
            print(f"  Length ratio: {astar.path_length / optimal_length:.3f}x")
            if optimal_time > 0:
                print(f"  Time ratio: {astar.computation_time / optimal_time:.3f}x")
            else:
                print(f"  Time: {astar.computation_time:.3f}s")
            print(f"  Nodes ratio: {astar.nodes_visited / results['dijkstra'].nodes_visited:.3f}x")

        if 'spectral' in results and results['spectral'].success:
            spectral = results['spectral']
            print(f"\nSpectral:")
            print(f"  Cost ratio: {spectral.total_cost / optimal_cost:.3f}x")
            print(f"  Length ratio: {spectral.path_length / optimal_length:.3f}x")
            if optimal_time > 0:
                print(f"  Time ratio: {spectral.computation_time / optimal_time:.3f}x")
            else:
                print(f"  Time: {spectral.computation_time:.3f}s")
            print(f"  Nodes ratio: {spectral.nodes_visited / results['dijkstra'].nodes_visited:.3f}x")

    print(f"\nOutput files saved to: {output_dir}")
    print("  01_heightfield.png - Raw heightfield")
    print("  02_blocking.png - Heightfield with blocking overlay")
    print("  03_path_*.png - Individual path visualizations")
    print("  04_comparison.png - Side-by-side comparison")

    return results


def run_simple_test():
    """Run a simple test with basic heightfield."""
    print("\n" + "=" * 60)
    print("SIMPLE PATHFINDING TEST")
    print("=" * 60)

    # Create simple heightfield with a wall
    size = 64
    heightfield = np.zeros((size, size), dtype=np.float32)

    # Add a vertical wall in the middle
    wall_x = size // 2
    wall_height = int(size * 0.4)
    heightfield[wall_height:-wall_height, wall_x-2:wall_x+2] = 1.0

    # Blocking where height > 0.5
    blocking = heightfield > 0.5

    print(f"Created {size}x{size} heightfield with vertical wall")
    print(f"Blocked pixels: {blocking.sum()}")

    # Path should go around the wall
    start = (size // 2, 10)
    goal = (size // 2, size - 10)

    print(f"Start: {start}")
    print(f"Goal: {goal}")

    # Find paths
    print("\nFinding paths...")
    results = compare_surface_paths(
        heightfield, start, goal, blocking,
        elevation_cost_scale=0.5
    )

    # Print results
    method_order = ['dijkstra', 'astar', 'spectral']
    idx = 0
    while idx < len(method_order):
        method = method_order[idx]
        if method in results:
            result = results[method]
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  {method}: {status}, length={result.path_length}, cost={result.total_cost:.2f}")
        idx += 1

    return results


if __name__ == "__main__":
    # Run simple test first
    run_simple_test()

    # Run full demo
    run_demo()
