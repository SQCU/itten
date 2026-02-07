"""
Optional interactive GUI for bisection visualization.

This module provides an optional interactive viewer using pygame or tkinter.
It is NOT required for core functionality - the CLI and render modules
work independently.

GUI follows the same architecture:
- Reads VisualizationState JSON
- Displays visualization
- User interactions produce new state JSON
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Callable
from .core import VisualizationState, state_to_json, state_from_json


# GUI implementation status
GUI_AVAILABLE = False
GUI_BACKEND = None

# Try to import pygame
try:
    import pygame
    GUI_AVAILABLE = True
    GUI_BACKEND = "pygame"
except ImportError:
    pass

# Try tkinter as fallback
if not GUI_AVAILABLE:
    try:
        import tkinter as tk
        GUI_AVAILABLE = True
        GUI_BACKEND = "tkinter"
    except ImportError:
        pass


def check_gui_available() -> bool:
    """Check if any GUI backend is available."""
    return GUI_AVAILABLE


def get_gui_backend() -> Optional[str]:
    """Get name of available GUI backend, or None."""
    return GUI_BACKEND


class InteractiveViewer:
    """
    Interactive visualization viewer.

    Displays the bisection visualization and allows:
    - Pan/zoom navigation
    - Click to select nodes
    - Step forward/backward through animation
    - Export current frame

    This is a placeholder implementation. Full implementation would
    depend on available GUI library (pygame, tkinter, or web-based).
    """

    def __init__(
        self,
        initial_state: Optional[VisualizationState] = None,
        width: int = 800,
        height: int = 600,
        on_state_change: Optional[Callable[[VisualizationState], None]] = None,
    ):
        self.state = initial_state or VisualizationState()
        self.width = width
        self.height = height
        self.on_state_change = on_state_change
        self.running = False

    def load_state(self, state: VisualizationState):
        """Load a new visualization state."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(self.state)

    def load_state_json(self, json_str: str):
        """Load state from JSON string."""
        self.load_state(state_from_json(json_str))

    def get_state_json(self) -> str:
        """Get current state as JSON string."""
        return state_to_json(self.state)

    def run(self):
        """
        Run the interactive viewer.

        This is a placeholder that prints a message.
        Full implementation would start the GUI event loop.
        """
        if not GUI_AVAILABLE:
            print(
                "GUI not available. Install pygame or use a system with tkinter.",
                file=sys.stderr,
            )
            print(
                "Use --render to generate static images instead.",
                file=sys.stderr,
            )
            return

        print(f"GUI would start here using {GUI_BACKEND}", file=sys.stderr)
        print("GUI implementation is a placeholder.", file=sys.stderr)
        print("Use --render for image output.", file=sys.stderr)

        # In full implementation, this would:
        # 1. Create window
        # 2. Render current state
        # 3. Handle input events
        # 4. Update state on interaction
        # 5. Re-render

    def quit(self):
        """Stop the viewer."""
        self.running = False


def run_gui(
    state: Optional[VisualizationState] = None,
    width: int = 800,
    height: int = 600,
):
    """
    Convenience function to run the GUI viewer.

    Args:
        state: Initial visualization state
        width: Window width
        height: Window height
    """
    viewer = InteractiveViewer(
        initial_state=state,
        width=width,
        height=height,
    )
    viewer.run()


if __name__ == "__main__":
    # Quick test
    print(f"GUI available: {GUI_AVAILABLE}")
    print(f"Backend: {GUI_BACKEND}")
