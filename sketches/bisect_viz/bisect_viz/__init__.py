"""
Infinite Bisection Visualizer

Visualizes graph growth and spectral bisection on GraphViews.
Follows GUI-optional, pipe-friendly architecture.
"""

from .core import (
    GraphState,
    BisectionState,
    VisualizationState,
    grow_graph_step,
    compute_bisection,
    state_to_json,
    state_from_json,
)
from .layout import (
    force_directed_layout,
    spectral_layout,
    compute_layout,
)
from .render import (
    render_frame,
    render_animation,
)

__all__ = [
    "GraphState",
    "BisectionState",
    "VisualizationState",
    "grow_graph_step",
    "compute_bisection",
    "state_to_json",
    "state_from_json",
    "force_directed_layout",
    "spectral_layout",
    "compute_layout",
    "render_frame",
    "render_animation",
]
