"""
Command executor for TUI.

Executes parsed commands by modifying state and triggering renders.
Uses the core geometry, inputs, render, and synthesis modules.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .state import TUIState
from .parser import (
    Command, GeometryCommand, TransformCommand,
    TextureCommand, ParamCommand, MetaCommand
)


class ExecutionError(Exception):
    """Raised when command execution fails."""
    pass


class CommandExecutor:
    """
    Execute parsed commands and trigger renders.

    Every successful command execution produces a rendered output.
    """

    def __init__(self, state: TUIState, trace):
        """
        Args:
            state: TUI session state
            trace: RenderTrace for saving outputs
        """
        self.state = state
        self.trace = trace

    def execute(self, command: Command) -> Path:
        """
        Execute command, render result, return image path.

        Args:
            command: Parsed Command object

        Returns:
            Path to rendered output image

        Raises:
            ExecutionError: If execution fails
        """
        if isinstance(command, GeometryCommand):
            self._execute_geometry(command)
        elif isinstance(command, TransformCommand):
            self._execute_transform(command)
        elif isinstance(command, TextureCommand):
            self._execute_texture(command)
        elif isinstance(command, ParamCommand):
            self._execute_param(command)
        elif isinstance(command, MetaCommand):
            return self._execute_meta(command)
        else:
            raise ExecutionError(f"Unknown command type: {type(command)}")

        # ALWAYS render after execution
        return self._render_current_state(command.raw_text)

    def _execute_geometry(self, cmd: GeometryCommand):
        """Execute geometry creation/modification command."""
        from ...geometry import Icosahedron, Sphere, Egg, fuse

        # Create meshes based on primitive type
        meshes = []
        for i in range(cmd.count):
            if cmd.primitive == 'icosahedron':
                mesh = Icosahedron(radius=1.0, subdivisions=cmd.subdivisions)
            elif cmd.primitive == 'sphere':
                mesh = Sphere(radius=1.0)
            elif cmd.primitive == 'egg':
                mesh = Egg(radius=1.0)
            else:
                mesh = Icosahedron(radius=1.0)

            # Offset each mesh for fusing
            if i > 0:
                offset = 1.5 * i
                mesh = mesh.translate(offset, 0, 0)

            meshes.append(mesh)

        # Fuse if requested
        if cmd.fuse and len(meshes) > 1:
            self.state.geometry = fuse(*meshes)
        elif meshes:
            if len(meshes) > 1:
                self.state.geometry = fuse(*meshes)
            else:
                self.state.geometry = meshes[0]

        # Apply texture wrap if specified
        if cmd.wrap:
            self._apply_carrier(cmd.wrap)

    def _apply_carrier(self, wrap_type: str, size: int = 128):
        """Apply carrier pattern based on wrap type."""
        from ...carriers import (
            AmongusCarrier, CheckerboardCarrier,
            DragonCurveCarrier, NoiseCarrier
        )

        if wrap_type == 'amongus':
            self.state.carrier = AmongusCarrier(size)
        elif wrap_type == 'checkerboard':
            self.state.carrier = CheckerboardCarrier(size)
        elif wrap_type == 'dragon':
            self.state.carrier = DragonCurveCarrier(size)
        elif wrap_type == 'noise':
            self.state.carrier = NoiseCarrier(size)

    def _execute_transform(self, cmd: TransformCommand):
        """Execute transformation command."""
        from ...geometry import squash, chop

        target = cmd.target

        if cmd.action == 'rotate':
            if target == 'carrier' and self.state.carrier:
                self.state.carrier = self.state.carrier.rotate(cmd.angle)
            elif target == 'geometry' and self.state.geometry:
                # Rotate geometry
                angle_rad = np.radians(cmd.angle)
                self.state.geometry = self.state.geometry.rotate('y', angle_rad)

        elif cmd.action == 'stretch':
            if target == 'carrier' and self.state.carrier:
                sx = cmd.factor if cmd.axis == 'x' else 1.0
                sy = cmd.factor if cmd.axis == 'y' else 1.0
                self.state.carrier = self.state.carrier.stretch(sx, sy)
            elif target == 'geometry' and self.state.geometry:
                # Stretch geometry
                self.state.geometry = self.state.geometry.scale(
                    cmd.factor if cmd.axis == 'x' else 1.0,
                    cmd.factor if cmd.axis == 'y' else 1.0,
                    cmd.factor if cmd.axis == 'z' else 1.0
                )

        elif cmd.action == 'squash':
            if target == 'geometry' and self.state.geometry:
                self.state.geometry = squash(
                    self.state.geometry,
                    axis=cmd.axis,
                    factor=cmd.factor
                )
            elif target == 'carrier' and self.state.carrier:
                # For carrier, use stretch with inverse
                sx = cmd.factor if cmd.axis == 'x' else 1.0
                sy = cmd.factor if cmd.axis == 'y' else 1.0
                self.state.carrier = self.state.carrier.stretch(sx, sy)

        elif cmd.action == 'chop':
            if self.state.geometry:
                self.state.geometry = chop(
                    self.state.geometry,
                    plane_normal=cmd.plane_normal,
                    plane_origin=cmd.plane_origin
                )

        elif cmd.action == 'translate':
            offset = cmd.plane_origin  # Reusing this field for offset
            if target == 'geometry' and self.state.geometry:
                self.state.geometry = self.state.geometry.translate(
                    offset[0], offset[1], offset[2]
                )
            elif target == 'carrier' and self.state.carrier:
                self.state.carrier = self.state.carrier.translate(
                    int(offset[0]), int(offset[1])
                )

    def _execute_texture(self, cmd: TextureCommand):
        """Execute texture/pattern command."""
        from ...carriers import (
            AmongusCarrier, CheckerboardCarrier,
            DragonCurveCarrier, NoiseCarrier, GradientCarrier,
            SVGCarrier
        )
        from ...operands import CheckerboardOperand, NoiseOperand, SolidOperand

        size = cmd.size

        if cmd.action == 'set_carrier':
            if cmd.pattern == 'dragon':
                self.state.carrier = DragonCurveCarrier(
                    size, iterations=cmd.iterations
                )
            elif cmd.pattern == 'amongus':
                self.state.carrier = AmongusCarrier(size)
            elif cmd.pattern == 'checkerboard':
                self.state.carrier = CheckerboardCarrier(size)
            elif cmd.pattern == 'noise':
                self.state.carrier = NoiseCarrier(size)
            elif cmd.pattern == 'gradient':
                self.state.carrier = GradientCarrier(size)
            elif cmd.pattern == 'svg' and cmd.path:
                self.state.carrier = SVGCarrier.from_file(cmd.path, size)

        elif cmd.action == 'set_operand':
            if cmd.pattern == 'checkerboard':
                self.state.operand = CheckerboardOperand(size)
            elif cmd.pattern == 'noise':
                self.state.operand = NoiseOperand(size)
            else:
                self.state.operand = SolidOperand(size)

    def _execute_param(self, cmd: ParamCommand):
        """Execute parameter setting command."""
        if cmd.param == 'theta':
            self.state.theta = float(cmd.value)
        elif cmd.param == 'gamma':
            self.state.gamma = float(cmd.value)
        elif cmd.param == 'render_mode':
            self.state.render_mode = str(cmd.value)

    def _execute_meta(self, cmd: MetaCommand) -> Optional[Path]:
        """Execute meta command (status, help, etc.)."""
        # Meta commands don't trigger renders, return None
        return None

    def _render_current_state(self, command_str: str) -> Path:
        """Synthesize texture and render mesh."""
        from ...core import synthesize as synthesize_texture

        # If no geometry, create placeholder
        if not self.state.has_geometry():
            return self.trace.render_placeholder(
                command_str, "No geometry - use 'icosahedron' first"
            )

        # Get carrier/operand images
        if self.state.carrier:
            carrier_img = self.state.carrier.render()
        else:
            # Default carrier: gradient
            size = 128
            y, x = np.ogrid[:size, :size]
            carrier_img = ((x + y) / (2 * size)).astype(np.float32)

        if self.state.operand:
            operand_img = self.state.operand.render()
        else:
            # Default operand: solid white
            operand_img = np.ones_like(carrier_img)

        # Synthesize height field
        height = synthesize_texture(
            carrier_img, operand_img,
            theta=self.state.theta,
            gamma=self.state.gamma
        )

        # Render and save
        path = self.trace.render_and_save(
            self.state.geometry,
            height,
            command_str
        )

        # Print render notification
        print(f"[RENDER] {path}")

        # Increment step count
        self.state.increment_step()

        return path
