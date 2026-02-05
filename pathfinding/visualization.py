"""
Path Visualization - Render paths on texture surfaces.

Provides functions to visualize paths, blocking regions, and comparisons.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
from pathlib import Path

from .result import PathResult


def heightfield_to_rgb(
    heightfield: np.ndarray,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Convert heightfield to RGB image using matplotlib colormap.

    Args:
        heightfield: 2D array normalized to [0, 1]
        colormap: Matplotlib colormap name

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        cmap = cm.get_cmap(colormap)
        rgba = cmap(heightfield)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        # Fallback: grayscale
        gray = (heightfield * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def visualize_path(
    heightfield: np.ndarray,
    result: PathResult,
    output_path: Optional[str] = None,
    path_color: Tuple[int, int, int] = (255, 50, 50),
    start_color: Tuple[int, int, int] = (50, 255, 50),
    goal_color: Tuple[int, int, int] = (50, 50, 255),
    line_width: int = 3,
    colormap: str = 'viridis',
    show_stats: bool = True
) -> np.ndarray:
    """
    Visualize path on heightfield.

    Args:
        heightfield: 2D height array
        result: PathResult from pathfinding
        output_path: Optional path to save image
        path_color: RGB color for path
        start_color: RGB color for start point
        goal_color: RGB color for goal point
        line_width: Width of path line
        colormap: Matplotlib colormap for heightfield
        show_stats: Add text overlay with statistics

    Returns:
        RGB image with path overlay (H, W, 3) as uint8
    """
    height, width = heightfield.shape

    # Convert heightfield to RGB
    base_image = heightfield_to_rgb(heightfield, colormap)

    # Create path overlay
    overlay = result.to_colored_overlay(
        height, width,
        path_color=path_color,
        start_color=start_color,
        goal_color=goal_color,
        line_width=line_width
    )

    # Composite overlay onto base
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    result_image = (
        base_image.astype(np.float32) * (1 - alpha) +
        overlay[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)

    # Save if path provided
    if output_path:
        save_image(result_image, output_path)

    return result_image


def visualize_blocking(
    heightfield: np.ndarray,
    blocking_mask: np.ndarray,
    output_path: Optional[str] = None,
    blocked_color: Tuple[int, int, int] = (200, 50, 50),
    blocked_alpha: float = 0.7,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Visualize blocking regions on heightfield.

    Args:
        heightfield: 2D height array
        blocking_mask: Boolean mask where True = blocked
        output_path: Optional path to save image
        blocked_color: RGB color for blocked regions
        blocked_alpha: Opacity of blocked overlay
        colormap: Matplotlib colormap for heightfield

    Returns:
        RGB image with blocking overlay (H, W, 3) as uint8
    """
    # Convert heightfield to RGB
    base_image = heightfield_to_rgb(heightfield, colormap)

    # Create blocking overlay
    result_image = base_image.astype(np.float32)

    r = 0
    while r < heightfield.shape[0]:
        c = 0
        while c < heightfield.shape[1]:
            if blocking_mask[r, c]:
                result_image[r, c, 0] = (
                    result_image[r, c, 0] * (1 - blocked_alpha) +
                    blocked_color[0] * blocked_alpha
                )
                result_image[r, c, 1] = (
                    result_image[r, c, 1] * (1 - blocked_alpha) +
                    blocked_color[1] * blocked_alpha
                )
                result_image[r, c, 2] = (
                    result_image[r, c, 2] * (1 - blocked_alpha) +
                    blocked_color[2] * blocked_alpha
                )
            c += 1
        r += 1

    result_image = result_image.astype(np.uint8)

    if output_path:
        save_image(result_image, output_path)

    return result_image


def visualize_comparison(
    heightfield: np.ndarray,
    results: Dict[str, PathResult],
    blocking_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Create comparison grid of different pathfinding methods.

    Args:
        heightfield: 2D height array
        results: Dict mapping method name to PathResult
        blocking_mask: Optional blocking mask to show
        output_path: Optional path to save image
        colormap: Matplotlib colormap

    Returns:
        RGB image grid (H, W*n, 3) as uint8 where n is number of methods
    """
    height, width = heightfield.shape
    method_names = list(results.keys())
    n_methods = len(method_names)

    # Colors for each method
    method_colors = {
        'dijkstra': (50, 200, 50),   # Green (optimal)
        'astar': (50, 50, 255),       # Blue
        'spectral': (255, 50, 50),    # Red
    }

    panels = []

    # First panel: heightfield with blocking
    if blocking_mask is not None:
        first_panel = visualize_blocking(heightfield, blocking_mask, colormap=colormap)
    else:
        first_panel = heightfield_to_rgb(heightfield, colormap)
    panels.append(first_panel)

    # One panel per method
    idx = 0
    while idx < n_methods:
        method = method_names[idx]
        result = results[method]
        color = method_colors.get(method, (255, 255, 0))

        panel = visualize_path(
            heightfield, result,
            path_color=color,
            colormap=colormap,
            line_width=2
        )
        panels.append(panel)
        idx += 1

    # Stack panels horizontally
    comparison = np.concatenate(panels, axis=1)

    if output_path:
        save_image(comparison, output_path)

    return comparison


def save_image(image: np.ndarray, path: str):
    """
    Save RGB image to file.

    Uses PIL if available, falls back to matplotlib.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        Image.fromarray(image).save(str(output_path))
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(str(output_path), image)
        except ImportError:
            raise RuntimeError("Neither PIL nor matplotlib available for saving images")


def create_summary_image(
    heightfield: np.ndarray,
    results: Dict[str, PathResult],
    blocking_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Pathfinding Comparison"
) -> np.ndarray:
    """
    Create annotated summary image with statistics.

    Args:
        heightfield: 2D height array
        results: Dict mapping method name to PathResult
        blocking_mask: Optional blocking mask
        output_path: Optional path to save image
        title: Title for the summary

    Returns:
        Annotated RGB image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        height, width = heightfield.shape
        method_names = list(results.keys())
        n_methods = len(method_names)

        # Create figure with subplots
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

        if n_methods == 0:
            axes = [axes]

        # Colors
        method_colors = {
            'dijkstra': 'green',
            'astar': 'blue',
            'spectral': 'red',
        }

        # First subplot: heightfield with blocking
        ax = axes[0]
        ax.imshow(heightfield, cmap='viridis')
        if blocking_mask is not None:
            blocked_overlay = np.zeros((height, width, 4))
            blocked_overlay[:, :, 0] = blocking_mask.astype(float)  # Red channel
            blocked_overlay[:, :, 3] = blocking_mask.astype(float) * 0.5  # Alpha
            ax.imshow(blocked_overlay)
        ax.set_title("Heightfield + Blocking")
        ax.axis('off')

        # One subplot per method
        idx = 0
        while idx < n_methods:
            method = method_names[idx]
            result = results[method]
            color = method_colors.get(method, 'yellow')
            ax = axes[idx + 1]

            ax.imshow(heightfield, cmap='viridis')

            if result.success and result.path:
                # Plot path
                rows = [p[0] for p in result.path]
                cols = [p[1] for p in result.path]
                ax.plot(cols, rows, color=color, linewidth=2, label=method)

                # Mark start and goal
                ax.plot(cols[0], rows[0], 'go', markersize=8, label='Start')
                ax.plot(cols[-1], rows[-1], 'bo', markersize=8, label='Goal')

            # Stats text
            stats_text = f"Cost: {result.total_cost:.1f}\n"
            stats_text += f"Steps: {result.path_length}\n"
            stats_text += f"Time: {result.computation_time:.3f}s\n"
            if result.cost_ratio is not None:
                stats_text += f"Ratio: {result.cost_ratio:.2f}x"

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f"{method.upper()}")
            ax.axis('off')
            idx += 1

        fig.suptitle(title)
        plt.tight_layout()

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')

        # Convert to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return image

    except ImportError:
        # Fallback to simple grid
        return visualize_comparison(heightfield, results, blocking_mask, output_path)
