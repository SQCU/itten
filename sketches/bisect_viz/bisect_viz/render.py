"""
Rendering functions for bisection visualization.

Uses PIL for image generation. No GUI dependencies.
"""

from typing import Dict, List, Tuple, Optional
import math
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("PIL (pillow) is required for rendering. Install with: uv add pillow")

from .core import VisualizationState, GraphState, BisectionState


def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def fiedler_to_color(
    value: float,
    partition: str,
    intensity: float = 1.0,
) -> Tuple[int, int, int]:
    """
    Convert Fiedler value to color.

    Partition A: red spectrum
    Partition B: blue spectrum
    Intensity based on absolute Fiedler value.
    """
    # Base colors
    red_dark = (100, 30, 30)
    red_bright = (255, 80, 80)
    blue_dark = (30, 30, 100)
    blue_bright = (80, 80, 255)

    # Neutral for near-zero values
    neutral = (128, 128, 128)

    abs_val = abs(value)
    # Normalize intensity (Fiedler values are typically small)
    t = min(1.0, abs_val * 5) * intensity

    if partition == "a":
        return lerp_color(red_dark, red_bright, t)
    elif partition == "b":
        return lerp_color(blue_dark, blue_bright, t)
    else:
        return neutral


def world_to_screen(
    x: float,
    y: float,
    width: int,
    height: int,
    margin: int = 50,
) -> Tuple[int, int]:
    """Convert world coordinates [-1, 1] to screen coordinates."""
    screen_x = int(margin + (x + 1) / 2 * (width - 2 * margin))
    screen_y = int(margin + (1 - y) / 2 * (height - 2 * margin))  # Flip y
    return screen_x, screen_y


def render_frame(
    state: VisualizationState,
    width: int = 800,
    height: int = 600,
    bg_color: Tuple[int, int, int] = (20, 20, 30),
    edge_color: Tuple[int, int, int] = (60, 60, 70),
    cut_edge_color: Tuple[int, int, int] = (200, 200, 50),
    node_min_radius: int = 5,
    node_max_radius: int = 15,
    show_labels: bool = False,
    show_title: bool = True,
    show_stats: bool = True,
) -> Image.Image:
    """
    Render a single frame of the visualization.

    Args:
        state: VisualizationState to render
        width: Image width in pixels
        height: Image height in pixels
        bg_color: Background color RGB
        edge_color: Default edge color RGB
        cut_edge_color: Color for edges crossing partition boundary
        node_min_radius: Minimum node radius
        node_max_radius: Maximum node radius
        show_labels: Whether to show node IDs
        show_title: Whether to show title
        show_stats: Whether to show statistics

    Returns:
        PIL Image object
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    graph = state.graph
    bisection = state.bisection

    # Get positions
    positions = graph.positions
    if not positions:
        # Fallback: arrange in circle
        n = len(graph.nodes)
        for i, node in enumerate(graph.nodes):
            angle = 2 * math.pi * i / max(n, 1)
            positions[node] = (math.cos(angle) * 0.8, math.sin(angle) * 0.8)

    # Compute Fiedler value range for normalization
    fiedler_vals = list(bisection.fiedler_values.values())
    if fiedler_vals:
        max_fiedler = max(abs(v) for v in fiedler_vals) or 1.0
    else:
        max_fiedler = 1.0

    # Draw edges first (under nodes)
    for a, b in graph.edges:
        if a not in positions or b not in positions:
            continue

        x1, y1 = world_to_screen(*positions[a], width, height)
        x2, y2 = world_to_screen(*positions[b], width, height)

        # Check if edge crosses partition
        is_cut = (
            (a in bisection.partition_a and b in bisection.partition_b) or
            (a in bisection.partition_b and b in bisection.partition_a)
        )

        color = cut_edge_color if is_cut else edge_color
        line_width = 2 if is_cut else 1

        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

    # Draw nodes
    for node in graph.nodes:
        if node not in positions:
            continue

        x, y = world_to_screen(*positions[node], width, height)

        # Determine partition
        if node in bisection.partition_a:
            partition = "a"
        elif node in bisection.partition_b:
            partition = "b"
        else:
            partition = "neutral"

        # Get Fiedler value
        fiedler_val = bisection.fiedler_values.get(node, 0.0)
        normalized_val = abs(fiedler_val) / max_fiedler if max_fiedler > 0 else 0

        # Node color based on partition and Fiedler magnitude
        color = fiedler_to_color(fiedler_val, partition, intensity=0.5 + normalized_val * 0.5)

        # Node size based on Fiedler magnitude
        radius = int(node_min_radius + normalized_val * (node_max_radius - node_min_radius))

        # Draw node
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=(255, 255, 255) if node in graph.expanded else (100, 100, 100),
            width=2 if node in graph.expanded else 1,
        )

        # Draw label if requested
        if show_labels:
            draw.text(
                (x, y - radius - 10),
                str(node),
                fill=(200, 200, 200),
                anchor="mb",
            )

    # Draw title
    if show_title:
        draw.text(
            (width // 2, 20),
            state.title,
            fill=(255, 255, 255),
            anchor="mt",
        )

    # Draw statistics
    if show_stats:
        stats_lines = [
            f"Nodes: {len(graph.nodes)}",
            f"Edges: {len(graph.edges)}",
            f"Lambda2: {bisection.lambda2:.4f}",
            f"Cut edges: {bisection.cut_edges}",
            f"Partition A: {len(bisection.partition_a)}",
            f"Partition B: {len(bisection.partition_b)}",
        ]

        y_offset = height - 20 - len(stats_lines) * 15
        for i, line in enumerate(stats_lines):
            draw.text(
                (10, y_offset + i * 15),
                line,
                fill=(180, 180, 180),
            )

    # Draw legend
    legend_y = 50
    legend_x = width - 120

    # Partition A
    draw.ellipse(
        [(legend_x, legend_y), (legend_x + 10, legend_y + 10)],
        fill=(200, 80, 80),
    )
    draw.text((legend_x + 15, legend_y), "Partition A", fill=(200, 200, 200))

    # Partition B
    draw.ellipse(
        [(legend_x, legend_y + 20), (legend_x + 10, legend_y + 30)],
        fill=(80, 80, 200),
    )
    draw.text((legend_x + 15, legend_y + 20), "Partition B", fill=(200, 200, 200))

    # Cut edge
    draw.line(
        [(legend_x, legend_y + 45), (legend_x + 10, legend_y + 45)],
        fill=cut_edge_color,
        width=2,
    )
    draw.text((legend_x + 15, legend_y + 40), "Cut edge", fill=(200, 200, 200))

    return img


def render_animation(
    frames: List[VisualizationState],
    output_path: str,
    width: int = 800,
    height: int = 600,
    fps: int = 5,
    loop: int = 0,
    **render_kwargs,
) -> str:
    """
    Render multiple frames to an animated GIF.

    Args:
        frames: List of VisualizationState objects
        output_path: Output file path (should end in .gif)
        width: Frame width
        height: Frame height
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        **render_kwargs: Additional arguments passed to render_frame

    Returns:
        Output file path
    """
    if not frames:
        raise ValueError("No frames to render")

    images = []
    for state in frames:
        img = render_frame(state, width=width, height=height, **render_kwargs)
        images.append(img)

    # Calculate duration in milliseconds
    duration = int(1000 / fps)

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )

    return output_path


def save_frames(
    frames: List[VisualizationState],
    output_dir: str,
    prefix: str = "frame",
    width: int = 800,
    height: int = 600,
    format: str = "png",
    **render_kwargs,
) -> List[str]:
    """
    Save multiple frames as individual images.

    Args:
        frames: List of VisualizationState objects
        output_dir: Directory to save frames
        prefix: Filename prefix
        width: Frame width
        height: Frame height
        format: Image format (png, jpg, etc.)
        **render_kwargs: Additional arguments passed to render_frame

    Returns:
        List of saved file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, state in enumerate(frames):
        img = render_frame(state, width=width, height=height, **render_kwargs)
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.{format}")
        img.save(path)
        paths.append(path)

    return paths
