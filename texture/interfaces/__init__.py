"""
Interface implementations for the unified texture module.

Provides:
- cli: Command-line interface
- headless: Batch processing from config files
"""

from .cli import main as cli_main
from .headless import process_config, batch_process

__all__ = ['cli_main', 'process_config', 'batch_process']
