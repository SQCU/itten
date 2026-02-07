"""
TUI session state management.

Maintains the current state of geometry, carriers, operands, and synthesis
parameters. Supports cloning for undo functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import copy


@dataclass
class TUIState:
    """
    Complete state of a TUI session.

    Attributes:
        geometry: Current 3D mesh geometry
        carrier: Current carrier pattern input
        operand: Current operand pattern input
        theta: Spectral rotation angle [0, 1]
        gamma: Etch/synthesis strength
        render_mode: Rendering mode ('dichromatic', 'raking', etc.)
        output_dir: Directory for rendered outputs
        step_count: Number of commands executed
    """
    geometry: Any = None           # Mesh object
    carrier: Any = None            # CarrierInput object
    operand: Any = None            # OperandInput object
    theta: float = 0.5             # Spectral rotation angle
    gamma: float = 0.3             # Etch strength
    render_mode: str = 'dichromatic'
    output_dir: Path = field(default_factory=lambda: Path('outputs'))
    step_count: int = 0

    def __post_init__(self):
        """Ensure output_dir is a Path."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def clone(self) -> 'TUIState':
        """
        Create a deep copy of the state for undo operations.

        Returns:
            New TUIState with copied data
        """
        return TUIState(
            geometry=copy.deepcopy(self.geometry) if self.geometry else None,
            carrier=copy.deepcopy(self.carrier) if self.carrier else None,
            operand=copy.deepcopy(self.operand) if self.operand else None,
            theta=self.theta,
            gamma=self.gamma,
            render_mode=self.render_mode,
            output_dir=self.output_dir,
            step_count=self.step_count
        )

    def increment_step(self) -> int:
        """Increment step count and return the new value."""
        self.step_count += 1
        return self.step_count

    def has_geometry(self) -> bool:
        """Check if geometry is set."""
        return self.geometry is not None

    def has_carrier(self) -> bool:
        """Check if carrier is set."""
        return self.carrier is not None

    def has_operand(self) -> bool:
        """Check if operand is set."""
        return self.operand is not None

    def summary(self) -> str:
        """Return a summary of current state."""
        lines = []
        lines.append(f"Step: {self.step_count}")
        lines.append(f"Geometry: {self.geometry if self.geometry else 'None'}")
        lines.append(f"Carrier: {type(self.carrier).__name__ if self.carrier else 'None'}")
        lines.append(f"Operand: {type(self.operand).__name__ if self.operand else 'None'}")
        lines.append(f"Theta: {self.theta:.3f}")
        lines.append(f"Gamma: {self.gamma:.3f}")
        lines.append(f"Render mode: {self.render_mode}")
        return "\n".join(lines)
