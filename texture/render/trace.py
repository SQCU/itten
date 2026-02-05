"""
Render trace tracking.

Tracks every rendered output for review and debugging.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def slugify(text: str) -> str:
    """
    Convert text to a safe filename slug.

    Args:
        text: Input text

    Returns:
        Slugified string safe for filenames
    """
    # Lowercase
    text = text.lower()
    # Replace spaces and punctuation with underscores
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    # Limit length
    text = text[:50]
    return text.strip('_')


class RenderTrace:
    """
    Track every rendered output.

    Saves images with incrementing step numbers and maintains
    a manifest of all renders.
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save rendered outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.history: List[Dict] = []
        self.session_start = datetime.now().isoformat()

    def render_and_save(
        self,
        mesh,
        height_field,
        command: str,
        **render_kwargs
    ) -> Path:
        """
        Render current state and save to trace.

        Args:
            mesh: Mesh object to render
            height_field: Height field texture
            command: Command description for filename
            **render_kwargs: Additional arguments for renderer

        Returns:
            Path to saved image
        """
        from .pbr import render_mesh_dichromatic, height_to_normals
        from PIL import Image
        import numpy as np

        # Generate normal map from height field
        normal_map = height_to_normals(height_field)

        # Render
        img = render_mesh_dichromatic(
            mesh, height_field, normal_map,
            **render_kwargs
        )

        # Generate filename
        slug = slugify(command)
        filename = f"step_{self.step:04d}_{slug}.png"
        path = self.output_dir / filename

        # Save image
        Image.fromarray(img).save(path)

        # Record in history
        self.history.append({
            'step': self.step,
            'command': command,
            'path': str(path),
            'timestamp': datetime.now().isoformat()
        })
        self.step += 1

        return path

    def render_placeholder(self, command: str, message: str = "No geometry") -> Path:
        """
        Create a placeholder image when geometry is missing.

        Args:
            command: Command that triggered render
            message: Message to display

        Returns:
            Path to placeholder image
        """
        from PIL import Image, ImageDraw
        import numpy as np

        # Create placeholder image
        size = 512
        img = Image.new('RGB', (size, size), (40, 40, 45))
        draw = ImageDraw.Draw(img)

        # Draw message
        text = f"{message}\n{command}"
        draw.text((size // 4, size // 2), text, fill=(180, 180, 180))

        # Generate filename
        slug = slugify(command)
        filename = f"step_{self.step:04d}_{slug}.png"
        path = self.output_dir / filename

        # Save
        img.save(path)

        # Record
        self.history.append({
            'step': self.step,
            'command': command,
            'path': str(path),
            'timestamp': datetime.now().isoformat(),
            'placeholder': True,
            'message': message
        })
        self.step += 1

        return path

    def save_manifest(self) -> Path:
        """
        Save JSON manifest of all renders.

        Returns:
            Path to manifest file
        """
        manifest_path = self.output_dir / "manifest.json"

        manifest = {
            'session_start': self.session_start,
            'total_steps': self.step,
            'renders': self.history
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def get_last_render(self) -> Optional[Path]:
        """Get path to most recent render."""
        if self.history:
            return Path(self.history[-1]['path'])
        return None

    def get_render_by_step(self, step: int) -> Optional[Path]:
        """Get render path by step number."""
        for entry in self.history:
            if entry['step'] == step:
                return Path(entry['path'])
        return None
