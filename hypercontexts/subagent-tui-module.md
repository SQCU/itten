# Hypercontext: TUI Module

## Mission
Create `texture_synth/tui/` module that:
1. Parses natural language commands
2. Maintains session state (geometry, texture, params)
3. Executes commands and triggers render at EVERY atomic edit
4. Works both interactively and programmatically

## Files to Create

### `texture_synth/tui/__init__.py`
```python
from .session import TUISession
from .parser import CommandParser
from .executor import CommandExecutor
```

### `texture_synth/tui/state.py`
Session state:
```python
@dataclass
class TUIState:
    geometry: Mesh = None           # Current 3D geometry
    carrier: CarrierInput = None    # Current carrier pattern
    operand: OperandInput = None    # Current operand pattern
    theta: float = 0.5              # Spectral rotation angle
    gamma: float = 0.3              # Etch strength
    render_mode: str = 'dichromatic'
    output_dir: Path = Path('outputs')
    step_count: int = 0

    def clone(self) -> 'TUIState':
        """Deep copy for undo."""
        ...
```

### `texture_synth/tui/parser.py`
Natural language to commands:
```python
class CommandParser:
    """Parse natural language to structured commands."""

    def parse(self, text: str) -> Command:
        """
        Parse command like:
        - "two icosahedrons fused together, amonguswrapped"
        - "rotate the amongus 45 degrees"
        - "squash all hedrons vertically by 30%"
        - "set theta to 0.7"
        - "make the carrier a dragon curve"
        """
        text = text.lower().strip()

        # Geometry commands
        if 'icosahedron' in text:
            return self._parse_geometry_command(text)

        # Transform commands
        if any(w in text for w in ['rotate', 'stretch', 'squash', 'chop', 'translate']):
            return self._parse_transform_command(text)

        # Texture commands
        if any(w in text for w in ['carrier', 'operand', 'pattern', 'texture']):
            return self._parse_texture_command(text)

        # Parameter commands
        if any(w in text for w in ['theta', 'gamma', 'set']):
            return self._parse_param_command(text)

        raise ParseError(f"Could not parse: {text}")

    def _parse_geometry_command(self, text: str) -> GeometryCommand:
        # Count icosahedrons
        count = 1
        if 'two' in text or '2' in text:
            count = 2
        elif 'three' in text or '3' in text:
            count = 3

        # Check for fuse
        fused = 'fuse' in text

        # Check for texture wrapping
        wrap = None
        if 'amongus' in text:
            wrap = 'amongus'
        elif 'checkerboard' in text:
            wrap = 'checkerboard'

        return GeometryCommand(
            action='create',
            primitive='icosahedron',
            count=count,
            fuse=fused,
            wrap=wrap
        )

    def _parse_transform_command(self, text: str) -> TransformCommand:
        # Extract transform type
        if 'rotate' in text:
            # Find angle: "rotate 45 degrees" or "rotate by 45°"
            angle = self._extract_number(text, default=45)
            return TransformCommand(action='rotate', angle=angle)

        elif 'stretch' in text:
            # "stretch horizontally 2x" or "stretch 2x horizontal"
            factor = self._extract_number(text, default=2)
            axis = 'x' if 'horizontal' in text else 'y'
            return TransformCommand(action='stretch', axis=axis, factor=factor)

        elif 'squash' in text:
            # "squash vertically by 30%"
            percent = self._extract_number(text, default=30)
            axis = 'y' if 'vertical' in text else 'x'
            factor = 1.0 - percent / 100.0
            return TransformCommand(action='squash', axis=axis, factor=factor)

        elif 'chop' in text:
            # "chop in half" or "chop the top"
            if 'half' in text:
                plane_origin = (0, 0, 0)
            elif 'top' in text:
                plane_origin = (0, 0.5, 0)
            else:
                plane_origin = (0, 0, 0)
            return TransformCommand(action='chop', plane_origin=plane_origin)

    def _parse_texture_command(self, text: str) -> TextureCommand:
        if 'dragon' in text:
            return TextureCommand(action='set_carrier', pattern='dragon')
        elif 'amongus' in text:
            return TextureCommand(action='set_carrier', pattern='amongus')
        elif 'checkerboard' in text:
            return TextureCommand(action='set_carrier', pattern='checkerboard')
        elif 'svg' in text:
            # Extract path
            path = self._extract_path(text)
            return TextureCommand(action='set_carrier', pattern='svg', path=path)

    def _parse_param_command(self, text: str) -> ParamCommand:
        if 'theta' in text:
            value = self._extract_number(text, default=0.5)
            return ParamCommand(param='theta', value=value)
        elif 'gamma' in text:
            value = self._extract_number(text, default=0.3)
            return ParamCommand(param='gamma', value=value)
```

### `texture_synth/tui/executor.py`
Execute commands and render:
```python
class CommandExecutor:
    """Execute parsed commands and trigger renders."""

    def __init__(self, state: TUIState, trace: RenderTrace):
        self.state = state
        self.trace = trace

    def execute(self, command: Command) -> Path:
        """Execute command, render result, return image path."""

        if isinstance(command, GeometryCommand):
            self._execute_geometry(command)
        elif isinstance(command, TransformCommand):
            self._execute_transform(command)
        elif isinstance(command, TextureCommand):
            self._execute_texture(command)
        elif isinstance(command, ParamCommand):
            self._execute_param(command)

        # ALWAYS render after execution
        return self._render_current_state(str(command))

    def _execute_geometry(self, cmd: GeometryCommand):
        from ..geometry import Icosahedron, fuse

        meshes = []
        for i in range(cmd.count):
            mesh = Icosahedron()
            # Offset each icosahedron
            mesh = mesh.translate(i * 1.5, 0, 0)
            meshes.append(mesh)

        if cmd.fuse and len(meshes) > 1:
            self.state.geometry = fuse(*meshes)
        elif meshes:
            self.state.geometry = meshes[0]

        # Apply texture wrap if specified
        if cmd.wrap:
            if cmd.wrap == 'amongus':
                self.state.carrier = AmongusCarrier(128)
            elif cmd.wrap == 'checkerboard':
                self.state.carrier = CheckerboardCarrier(128)

    def _execute_transform(self, cmd: TransformCommand):
        if cmd.action == 'rotate' and self.state.carrier:
            self.state.carrier = self.state.carrier.rotate(cmd.angle)

        elif cmd.action == 'stretch' and self.state.carrier:
            sx = cmd.factor if cmd.axis == 'x' else 1.0
            sy = cmd.factor if cmd.axis == 'y' else 1.0
            self.state.carrier = self.state.carrier.stretch(sx, sy)

        elif cmd.action == 'squash' and self.state.geometry:
            from ..geometry import squash
            self.state.geometry = squash(
                self.state.geometry,
                axis=cmd.axis,
                factor=cmd.factor
            )

        elif cmd.action == 'chop' and self.state.geometry:
            from ..geometry import chop
            self.state.geometry = chop(
                self.state.geometry,
                plane_normal=(0, 1, 0),
                plane_origin=cmd.plane_origin
            )

    def _execute_texture(self, cmd: TextureCommand):
        if cmd.pattern == 'dragon':
            self.state.carrier = DragonCurveCarrier(128)
        elif cmd.pattern == 'amongus':
            self.state.carrier = AmongusCarrier(128)
        elif cmd.pattern == 'checkerboard':
            self.state.carrier = CheckerboardCarrier(128)
        elif cmd.pattern == 'svg':
            self.state.carrier = SVGCarrier.from_file(cmd.path, 128)

    def _execute_param(self, cmd: ParamCommand):
        if cmd.param == 'theta':
            self.state.theta = cmd.value
        elif cmd.param == 'gamma':
            self.state.gamma = cmd.value

    def _render_current_state(self, command_str: str) -> Path:
        """Synthesize texture and render mesh."""
        from ..synthesis import synthesize_texture
        from ..render import render_mesh_dichromatic

        # Get carrier/operand images
        carrier_img = self.state.carrier.render() if self.state.carrier else np.ones((128, 128))
        operand_img = self.state.operand.render() if self.state.operand else np.ones((128, 128))

        # Synthesize height field
        height = synthesize_texture(
            carrier_img, operand_img,
            theta=self.state.theta,
            gamma=self.state.gamma
        )

        # Render
        return self.trace.render_and_save(
            self.state.geometry,
            height,
            command_str
        )
```

### `texture_synth/tui/session.py`
Main session interface:
```python
class TUISession:
    """Interactive TUI session."""

    def __init__(self, output_dir: str = 'outputs'):
        self.state = TUIState(output_dir=Path(output_dir))
        self.trace = RenderTrace(self.state.output_dir)
        self.parser = CommandParser()
        self.executor = CommandExecutor(self.state, self.trace)
        self.history = []

    def execute(self, command: str) -> Path:
        """Execute natural language command, return rendered image path."""
        parsed = self.parser.parse(command)
        path = self.executor.execute(parsed)
        self.history.append((command, path))
        return path

    def run_interactive(self):
        """Run interactive REPL."""
        print("Texture Synthesis TUI")
        print("=" * 50)
        print("Commands: geometry, transforms, textures, params")
        print("Type 'quit' to exit, 'history' to see past commands")
        print()

        while True:
            try:
                command = input("> ").strip()

                if command.lower() == 'quit':
                    break
                elif command.lower() == 'history':
                    for cmd, path in self.history:
                        print(f"  {cmd} -> {path}")
                    continue
                elif not command:
                    continue

                path = self.execute(command)
                print(f"[RENDER] {path}")

            except ParseError as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")

        self.trace.save_manifest()
        print(f"Saved {len(self.history)} renders to {self.state.output_dir}")
```

### `texture_tui.py` (Main Entry Point)
```python
#!/usr/bin/env python3
"""Texture Synthesis TUI - main entry point."""

import argparse
from texture_synth.tui import TUISession

def main():
    parser = argparse.ArgumentParser(description='Texture Synthesis TUI')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive REPL')
    parser.add_argument('--command', '-c', type=str,
                        help='Execute single command')
    parser.add_argument('--config', type=str,
                        help='Load commands from YAML config')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='Output directory')
    args = parser.parse_args()

    session = TUISession(output_dir=args.output)

    if args.interactive:
        session.run_interactive()
    elif args.command:
        path = session.execute(args.command)
        print(path)
    elif args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for cmd in config.get('commands', []):
            session.execute(cmd)
    else:
        # Default: show help
        parser.print_help()

if __name__ == '__main__':
    main()
```

## Usage Examples

### Interactive
```bash
$ python texture_tui.py -i
> two icosahedrons fused, amonguswrapped
[RENDER] outputs/step_0000_two_icosahedrons_fused.png
> squash vertically 30%
[RENDER] outputs/step_0001_squash_vertically.png
> rotate amongus 45 degrees
[RENDER] outputs/step_0002_rotate_amongus.png
> set theta to 0.8
[RENDER] outputs/step_0003_set_theta.png
```

### Programmatic (Claude calling)
```python
from texture_synth.tui import TUISession

session = TUISession()
session.execute("two icosahedrons fused, amonguswrapped")
session.execute("chop one in half")
session.execute("make carrier a dragon curve")
```

### Config file
```yaml
# scene.yaml
commands:
  - "three icosahedrons fused, checkerboard wrapped"
  - "squash vertically 20%"
  - "set theta to 0.6"
```

## Deliverables
1. `texture_synth/tui/state.py`
2. `texture_synth/tui/parser.py`
3. `texture_synth/tui/executor.py`
4. `texture_synth/tui/session.py`
5. `texture_synth/tui/__init__.py`
6. `texture_tui.py` (main entry)
7. Working demo showing interactive session with renders at each step
