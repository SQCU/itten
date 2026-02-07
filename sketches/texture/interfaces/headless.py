"""
Headless (batch) processing interface for texture synthesis.

Processes textures from JSON configuration files without GUI.
Suitable for automated pipelines and batch processing.
"""

import os
import json
from typing import Dict, Any, List, Optional

from ..core import synthesize, TextureResult
from ..io import export_all, load_image_as_array, load_config


def resolve_input_from_config(
    input_config: Dict[str, Any],
    base_dir: str = "."
) -> Any:
    """
    Resolve input specification from config to actual input.

    Config format:
        {"type": "pattern", "name": "amongus", "size": 64}
        {"type": "file", "path": "carrier.png"}
        {"type": "noise", "seed": 42}

    Args:
        input_config: Input configuration dict
        base_dir: Base directory for relative paths

    Returns:
        Resolved input (array or string pattern name)
    """
    input_type = input_config.get("type", "pattern")

    if input_type == "file":
        path = input_config["path"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        return load_image_as_array(path)

    elif input_type == "pattern":
        return input_config.get("name", "amongus")

    elif input_type == "noise":
        from ..patterns import generate_noise
        size = input_config.get("size", 64)
        seed = input_config.get("seed", 42)
        return generate_noise(size, seed=seed)

    elif input_type == "checkerboard":
        from ..patterns import generate_checkerboard
        size = input_config.get("size", 64)
        tile_size = input_config.get("tile_size", 8)
        return generate_checkerboard(size, tile_size)

    else:
        raise ValueError(f"Unknown input type: {input_type}")


def process_config(
    config: Dict[str, Any],
    base_dir: str = ".",
    output_dir: Optional[str] = None
) -> TextureResult:
    """
    Process a single texture synthesis configuration.

    Config format:
    {
        "carrier": {"type": "pattern", "name": "amongus"},
        "operand": {"type": "noise", "seed": 42},
        "theta": 0.5,
        "gamma": 0.3,
        "output": {
            "dir": "output/",
            "name": "texture",
            "height": "height.png",
            "normal": "normal.png"
        }
    }

    Args:
        config: Configuration dict
        base_dir: Base directory for relative paths
        output_dir: Override output directory

    Returns:
        TextureResult from synthesis
    """
    # Resolve carrier
    carrier_config = config.get("carrier", {"type": "pattern", "name": "amongus"})
    carrier = resolve_input_from_config(carrier_config, base_dir)

    # Resolve operand
    operand_config = config.get("operand", None)
    operand = None
    if operand_config:
        operand = resolve_input_from_config(operand_config, base_dir)

    # Get synthesis parameters
    theta = config.get("theta", 0.5)
    gamma = config.get("gamma", 0.3)
    num_eigenvectors = config.get("num_eigenvectors", 8)
    edge_threshold = config.get("edge_threshold", 0.1)
    output_size = config.get("output_size", None)
    normal_strength = config.get("normal_strength", 2.0)
    mode = config.get("mode", "spectral")
    preset = config.get("preset", None)

    # Synthesize
    result = synthesize(
        carrier,
        operand,
        theta=theta,
        gamma=gamma,
        num_eigenvectors=num_eigenvectors,
        edge_threshold=edge_threshold,
        output_size=output_size,
        normal_strength=normal_strength,
        mode=mode,
        preset=preset
    )

    # Export if output config present
    output_config = config.get("output", {})
    if output_config or output_dir:
        out_dir = output_dir or output_config.get("dir", ".")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(base_dir, out_dir)

        base_name = output_config.get("name", "texture")
        export_obj = output_config.get("obj", False)

        export_all(
            result.height_field,
            result.normal_map,
            out_dir,
            base_name=base_name,
            export_obj=export_obj,
            metadata=result.metadata
        )

    return result


def batch_process(
    config_path: str,
    output_dir: Optional[str] = None
) -> List[TextureResult]:
    """
    Process multiple textures from a batch config file.

    Batch config format:
    {
        "batch": [
            {"carrier": {...}, "operand": {...}, ...},
            {"carrier": {...}, "operand": {...}, ...}
        ],
        "defaults": {
            "theta": 0.5,
            "gamma": 0.3
        }
    }

    Args:
        config_path: Path to batch config JSON
        output_dir: Override output directory for all textures

    Returns:
        List of TextureResults
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, 'r') as f:
        batch_config = json.load(f)

    # Get defaults
    defaults = batch_config.get("defaults", {})

    # Process each item
    items = batch_config.get("batch", [])
    if not items:
        # Single item config
        items = [batch_config]

    results = []
    for i, item_config in enumerate(items):
        # Merge with defaults
        merged_config = {**defaults, **item_config}

        # Add index to output name if not specified
        if "output" not in merged_config:
            merged_config["output"] = {"name": f"texture_{i:03d}"}
        elif "name" not in merged_config["output"]:
            merged_config["output"]["name"] = f"texture_{i:03d}"

        print(f"Processing item {i+1}/{len(items)}...")
        result = process_config(merged_config, base_dir, output_dir)
        results.append(result)

    return results


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration template.

    Returns:
        Default config dict
    """
    return {
        "carrier": {
            "type": "pattern",
            "name": "amongus"
        },
        "operand": {
            "type": "noise",
            "seed": 42
        },
        "theta": 0.5,
        "gamma": 0.3,
        "num_eigenvectors": 8,
        "edge_threshold": 0.1,
        "normal_strength": 2.0,
        "mode": "spectral",
        "output": {
            "dir": "output",
            "name": "texture",
            "obj": False
        }
    }


def main():
    """Headless processing entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Headless texture synthesis from config files"
    )
    parser.add_argument(
        "config",
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--create-template", action="store_true",
        help="Create template config file instead of processing"
    )

    args = parser.parse_args()

    if args.create_template:
        config = create_default_config()
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created template config: {args.config}")
        return

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    results = batch_process(args.config, args.output)
    print(f"\nProcessed {len(results)} texture(s)")


if __name__ == "__main__":
    main()
