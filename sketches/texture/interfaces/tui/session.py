"""
TUI Session management.

Provides both programmatic and interactive REPL interfaces.
"""

from pathlib import Path
from typing import List, Tuple, Optional

from .state import TUIState
from .parser import CommandParser, ParseError, MetaCommand
from .executor import CommandExecutor, ExecutionError
from ...render import RenderTrace


class TUISession:
    """
    Interactive TUI session.

    Supports both programmatic use and interactive REPL mode.
    """

    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize TUI session.

        Args:
            output_dir: Directory for rendered outputs
        """
        self.state = TUIState(output_dir=Path(output_dir))
        self.trace = RenderTrace(self.state.output_dir)
        self.parser = CommandParser()
        self.executor = CommandExecutor(self.state, self.trace)
        self.history: List[Tuple[str, Optional[Path]]] = []
        self._undo_stack: List[TUIState] = []

    def execute(self, command: str) -> Optional[Path]:
        """
        Execute natural language command, return rendered image path.

        Args:
            command: Natural language command string

        Returns:
            Path to rendered image, or None for meta commands
        """
        # Save state for undo
        self._undo_stack.append(self.state.clone())
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

        try:
            parsed = self.parser.parse(command)

            # Handle meta commands specially
            if isinstance(parsed, MetaCommand):
                return self._handle_meta(parsed)

            path = self.executor.execute(parsed)
            self.history.append((command, path))
            return path

        except ParseError as e:
            # Parsing failed, restore state
            if self._undo_stack:
                self._undo_stack.pop()
            raise e

        except ExecutionError as e:
            # Execution failed, restore state
            if self._undo_stack:
                self._undo_stack.pop()
            raise e

    def _handle_meta(self, cmd: MetaCommand) -> Optional[Path]:
        """Handle meta commands."""
        if cmd.action == 'status':
            print("\n" + self.state.summary() + "\n")
            return None

        elif cmd.action == 'help':
            self._print_help()
            return None

        elif cmd.action == 'history':
            print("\nCommand history:")
            for i, (cmd_str, path) in enumerate(self.history):
                print(f"  {i}: {cmd_str} -> {path}")
            print()
            return None

        elif cmd.action == 'undo':
            if self._undo_stack:
                self.state = self._undo_stack.pop()
                print("Undone last command")
            else:
                print("Nothing to undo")
            return None

        elif cmd.action == 'quit':
            # Will be handled by run_interactive
            return None

        return None

    def _print_help(self):
        """Print help message."""
        help_text = """
Texture Synthesis TUI - Commands

GEOMETRY:
  "icosahedron"                    Create single icosahedron
  "two icosahedrons fused"         Create and fuse multiple
  "sphere", "egg"                  Other primitives
  "..., amonguswrapped"            Apply Among Us texture

TRANSFORMS:
  "squash vertically 30%"          Squash geometry
  "rotate 45 degrees"              Rotate geometry
  "rotate the carrier 45 degrees"  Rotate carrier pattern
  "stretch horizontally 2x"        Stretch
  "chop in half"                   Cut with plane

PATTERNS:
  "make carrier a dragon curve"    Set carrier pattern
  "make carrier checkerboard"      Set carrier pattern
  "set operand to noise"           Set operand pattern

PARAMETERS:
  "set theta to 0.7"               Spectral rotation angle
  "set gamma to 0.5"               Etch strength

META:
  "status"                         Show current state
  "history"                        Show command history
  "undo"                           Undo last command
  "help"                           Show this message
  "quit"                           Exit session

Every command produces a rendered PNG output.
"""
        print(help_text)

    def run_interactive(self):
        """Run interactive REPL."""
        print("=" * 60)
        print("Texture Synthesis TUI")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit")
        print()

        while True:
            try:
                command = input("> ").strip()

                if not command:
                    continue

                if command.lower() in ('quit', 'exit', 'q'):
                    break

                path = self.execute(command)
                # Note: [RENDER] is printed by executor

            except ParseError as e:
                print(f"Parse error: {e}")

            except ExecutionError as e:
                print(f"Execution error: {e}")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")

            except EOFError:
                break

        # Save manifest on exit
        self.trace.save_manifest()
        print(f"\nSaved {len(self.history)} renders to {self.state.output_dir}")

    def run_batch(self, commands: List[str]) -> List[Path]:
        """
        Run batch of commands.

        Args:
            commands: List of command strings

        Returns:
            List of output paths
        """
        paths = []
        for cmd in commands:
            try:
                path = self.execute(cmd)
                if path:
                    paths.append(path)
            except (ParseError, ExecutionError) as e:
                print(f"Error on '{cmd}': {e}")

        self.trace.save_manifest()
        return paths

    def get_state(self) -> TUIState:
        """Get current state."""
        return self.state

    def get_history(self) -> List[Tuple[str, Optional[Path]]]:
        """Get command history."""
        return self.history

    def get_last_render(self) -> Optional[Path]:
        """Get path to most recent render."""
        return self.trace.get_last_render()
