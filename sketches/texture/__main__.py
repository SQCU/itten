"""
Entry point for running texture module as CLI.

Usage:
    python -m texture --demo
    python -m texture --carrier amongus --operand noise
    python -m texture --help
"""

from .interfaces.cli import main

if __name__ == "__main__":
    main()
