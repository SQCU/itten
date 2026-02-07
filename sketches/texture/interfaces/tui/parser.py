"""
Natural language command parser for TUI.

Parses user commands like:
- "two icosahedrons fused, amonguswrapped"
- "squash vertically 30%"
- "rotate the carrier 45 degrees"
- "set theta to 0.7"
- "make carrier a dragon curve"
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union


class ParseError(Exception):
    """Raised when a command cannot be parsed."""
    pass


# ============================================================
# Command Data Classes
# ============================================================

@dataclass
class Command:
    """Base class for all commands."""
    raw_text: str = ""


@dataclass
class GeometryCommand(Command):
    """Command to create or modify geometry."""
    action: str = 'create'              # 'create', 'modify'
    primitive: str = 'icosahedron'      # 'icosahedron', 'sphere', 'egg'
    count: int = 1
    fuse: bool = False
    wrap: Optional[str] = None          # 'amongus', 'checkerboard', etc.
    subdivisions: int = 0
    radius: float = 1.0


@dataclass
class TransformCommand(Command):
    """Command to transform geometry or carrier."""
    action: str = 'rotate'              # 'rotate', 'stretch', 'squash', 'chop', 'translate'
    target: str = 'geometry'            # 'geometry', 'carrier', 'operand'
    angle: float = 0.0                  # For rotate
    axis: str = 'y'                     # For squash/stretch: 'x', 'y', 'z'
    factor: float = 1.0                 # Scale factor
    plane_origin: Tuple[float, float, float] = (0, 0, 0)  # For chop
    plane_normal: Tuple[float, float, float] = (0, 1, 0)  # For chop


@dataclass
class TextureCommand(Command):
    """Command to set carrier or operand pattern."""
    action: str = 'set_carrier'         # 'set_carrier', 'set_operand'
    pattern: str = 'checkerboard'       # 'amongus', 'checkerboard', 'dragon', 'noise', 'svg'
    path: Optional[str] = None          # For SVG patterns
    size: int = 128
    iterations: int = 10                # For dragon curve


@dataclass
class ParamCommand(Command):
    """Command to set synthesis parameters."""
    param: str = 'theta'                # 'theta', 'gamma', 'render_mode'
    value: Union[float, str] = 0.5


@dataclass
class MetaCommand(Command):
    """Meta commands like status, undo, help."""
    action: str = 'status'              # 'status', 'undo', 'help', 'quit'


# ============================================================
# Command Parser
# ============================================================

class CommandParser:
    """
    Parse natural language commands to structured Command objects.

    Handles flexible natural language variations:
    - "two icosahedrons fused, amonguswrapped"
    - "2 icosahedrons, fused together with amongus"
    - "make 2 icosahedrons and fuse them"
    """

    # Number words
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1, 'single': 1
    }

    def parse(self, text: str) -> Command:
        """
        Parse natural language command to Command object.

        Args:
            text: Raw user input

        Returns:
            Parsed Command object

        Raises:
            ParseError: If command cannot be parsed
        """
        text = text.strip()
        if not text:
            raise ParseError("Empty command")

        original_text = text
        text_lower = text.lower()

        # Meta commands
        if text_lower in ('quit', 'exit', 'q'):
            return MetaCommand(action='quit', raw_text=original_text)
        if text_lower in ('status', 'state', 'info'):
            return MetaCommand(action='status', raw_text=original_text)
        if text_lower in ('help', 'h', '?'):
            return MetaCommand(action='help', raw_text=original_text)
        if text_lower in ('undo', 'back'):
            return MetaCommand(action='undo', raw_text=original_text)
        if text_lower == 'history':
            return MetaCommand(action='history', raw_text=original_text)

        # Geometry commands (icosahedron, sphere, egg)
        if any(prim in text_lower for prim in ['icosahedron', 'sphere', 'egg']):
            return self._parse_geometry_command(text_lower, original_text)

        # Transform commands
        if any(w in text_lower for w in ['rotate', 'stretch', 'squash', 'chop', 'translate', 'move', 'shift']):
            return self._parse_transform_command(text_lower, original_text)

        # Texture/carrier commands
        if any(w in text_lower for w in ['carrier', 'operand', 'pattern', 'texture', 'make', 'dragon', 'amongus', 'checkerboard']):
            return self._parse_texture_command(text_lower, original_text)

        # Parameter commands
        if any(w in text_lower for w in ['theta', 'gamma', 'set', 'render']):
            return self._parse_param_command(text_lower, original_text)

        raise ParseError(f"Could not parse: {original_text}")

    def _extract_number(self, text: str, default: float = 1.0) -> float:
        """
        Extract a number from text.

        Handles: "45 degrees", "30%", "0.7", "two", etc.
        """
        # Try to find a decimal/integer number
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))

        # Try number words
        for word, num in self.NUMBER_WORDS.items():
            if word in text.lower():
                return float(num)

        return default

    def _extract_count(self, text: str) -> int:
        """Extract count from text (for "two icosahedrons")."""
        for word, num in self.NUMBER_WORDS.items():
            if word in text:
                return num

        # Try numeric
        match = re.search(r'^(\d+)\s', text)
        if match:
            return int(match.group(1))

        return 1

    def _parse_geometry_command(self, text: str, original: str) -> GeometryCommand:
        """Parse geometry creation commands."""
        # Detect primitive type
        primitive = 'icosahedron'
        if 'sphere' in text:
            primitive = 'sphere'
        elif 'egg' in text:
            primitive = 'egg'

        # Count instances
        count = self._extract_count(text)

        # Check for fuse
        fused = any(w in text for w in ['fuse', 'fused', 'merge', 'merged', 'combine', 'combined', 'join', 'joined'])

        # Check for texture wrapping
        wrap = None
        if 'amongus' in text or 'among us' in text:
            wrap = 'amongus'
        elif 'checkerboard' in text or 'checker' in text:
            wrap = 'checkerboard'
        elif 'dragon' in text:
            wrap = 'dragon'
        elif 'noise' in text:
            wrap = 'noise'

        # Check for subdivisions
        subdivisions = 0
        sub_match = re.search(r'subdiv\w*\s*(\d+)', text)
        if sub_match:
            subdivisions = int(sub_match.group(1))

        return GeometryCommand(
            action='create',
            primitive=primitive,
            count=count,
            fuse=fused,
            wrap=wrap,
            subdivisions=subdivisions,
            raw_text=original
        )

    def _parse_transform_command(self, text: str, original: str) -> TransformCommand:
        """Parse transformation commands."""
        # Determine target (geometry, carrier, or operand)
        target = 'geometry'
        if 'carrier' in text or 'pattern' in text or 'texture' in text:
            target = 'carrier'
        elif 'operand' in text:
            target = 'operand'

        # Rotate
        if 'rotate' in text:
            angle = self._extract_number(text, default=45.0)
            return TransformCommand(
                action='rotate',
                target=target,
                angle=angle,
                raw_text=original
            )

        # Stretch
        if 'stretch' in text:
            factor = self._extract_number(text, default=2.0)
            axis = 'x' if 'horizontal' in text else 'y'
            return TransformCommand(
                action='stretch',
                target=target,
                axis=axis,
                factor=factor,
                raw_text=original
            )

        # Squash
        if 'squash' in text:
            percent = self._extract_number(text, default=30.0)
            # "squash 30%" means scale to 70%
            factor = 1.0 - percent / 100.0
            axis = 'y' if 'vertical' in text else 'x' if 'horizontal' in text else 'y'
            return TransformCommand(
                action='squash',
                target=target,
                axis=axis,
                factor=factor,
                raw_text=original
            )

        # Chop
        if 'chop' in text:
            plane_origin = (0.0, 0.0, 0.0)
            if 'half' in text:
                plane_origin = (0.0, 0.0, 0.0)
            elif 'top' in text:
                plane_origin = (0.0, 0.5, 0.0)
            elif 'bottom' in text:
                plane_origin = (0.0, -0.5, 0.0)

            plane_normal = (0.0, 1.0, 0.0)
            if 'horizontal' in text or 'side' in text:
                plane_normal = (1.0, 0.0, 0.0)

            return TransformCommand(
                action='chop',
                target='geometry',
                plane_origin=plane_origin,
                plane_normal=plane_normal,
                raw_text=original
            )

        # Translate / move / shift
        if any(w in text for w in ['translate', 'move', 'shift']):
            # Try to extract x, y, z values
            x_match = re.search(r'x\s*[=:]?\s*([-\d.]+)', text)
            y_match = re.search(r'y\s*[=:]?\s*([-\d.]+)', text)
            z_match = re.search(r'z\s*[=:]?\s*([-\d.]+)', text)

            dx = float(x_match.group(1)) if x_match else 0.0
            dy = float(y_match.group(1)) if y_match else 0.0
            dz = float(z_match.group(1)) if z_match else 0.0

            # If just a single number, use it for primary axis
            if not any([x_match, y_match, z_match]):
                offset = self._extract_number(text, default=1.0)
                dx = offset

            return TransformCommand(
                action='translate',
                target=target,
                plane_origin=(dx, dy, dz),  # Reuse this field for offset
                raw_text=original
            )

        raise ParseError(f"Unknown transform: {original}")

    def _parse_texture_command(self, text: str, original: str) -> TextureCommand:
        """Parse texture/pattern commands."""
        # Determine if carrier or operand
        action = 'set_carrier'
        if 'operand' in text:
            action = 'set_operand'

        # Determine pattern type
        pattern = 'checkerboard'  # default

        if 'dragon' in text:
            pattern = 'dragon'
        elif 'amongus' in text or 'among us' in text:
            pattern = 'amongus'
        elif 'checkerboard' in text or 'checker' in text:
            pattern = 'checkerboard'
        elif 'noise' in text or 'perlin' in text:
            pattern = 'noise'
        elif 'svg' in text:
            pattern = 'svg'
            # Try to extract path
            path_match = re.search(r'["\']([^"\']+\.svg)["\']', text)
            if path_match:
                return TextureCommand(
                    action=action,
                    pattern='svg',
                    path=path_match.group(1),
                    raw_text=original
                )

        # Extract size if specified
        size = 128
        size_match = re.search(r'(\d+)\s*(?:px|pixels?|size)', text)
        if size_match:
            size = int(size_match.group(1))

        # Extract iterations for dragon curve
        iterations = 10
        iter_match = re.search(r'(\d+)\s*iter', text)
        if iter_match:
            iterations = int(iter_match.group(1))

        return TextureCommand(
            action=action,
            pattern=pattern,
            size=size,
            iterations=iterations,
            raw_text=original
        )

    def _parse_param_command(self, text: str, original: str) -> ParamCommand:
        """Parse parameter setting commands."""
        # Theta
        if 'theta' in text:
            value = self._extract_number(text, default=0.5)
            # Clamp to [0, 1]
            value = max(0.0, min(1.0, value))
            return ParamCommand(param='theta', value=value, raw_text=original)

        # Gamma
        if 'gamma' in text:
            value = self._extract_number(text, default=0.3)
            value = max(0.0, min(1.0, value))
            return ParamCommand(param='gamma', value=value, raw_text=original)

        # Render mode
        if 'render' in text:
            if 'dichromatic' in text:
                return ParamCommand(param='render_mode', value='dichromatic', raw_text=original)
            elif 'raking' in text:
                return ParamCommand(param='render_mode', value='raking', raw_text=original)
            elif 'anisotropic' in text:
                return ParamCommand(param='render_mode', value='anisotropic', raw_text=original)

        # Generic "set X to Y"
        set_match = re.search(r'set\s+(\w+)\s+(?:to\s+)?(\d*\.?\d+)', text)
        if set_match:
            param = set_match.group(1)
            value = float(set_match.group(2))
            if param in ('theta', 'gamma'):
                return ParamCommand(param=param, value=value, raw_text=original)

        raise ParseError(f"Unknown parameter command: {original}")
