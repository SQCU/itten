"""
Command-line interface for unified texture synthesis.

Usage:
    python -m texture --demo            # Run demo with amongus carrier
    python -m texture --carrier amongus --operand checkerboard
    python -m texture --carrier image.png --output result/
    python -m texture --help
"""

import argparse
import sys
import os


def run_demo(output_dir: str, size: int = 64, preset: str = 'balanced'):
    """Run demo synthesis with amongus carrier and checkerboard operand."""
    from ..core import synthesize
    from ..io import export_all

    print("=" * 60)
    print("Texture Synthesis Demo")
    print("=" * 60)

    print(f"\n[1/3] Synthesizing texture...")
    print(f"  Carrier: amongus ({size}x{size})")
    print(f"  Operand: checkerboard")
    print(f"  Preset: {preset}")

    result = synthesize(
        'amongus',
        'checkerboard',
        output_size=size,
        preset=preset,
        return_diagnostics=True
    )

    print(f"\n[2/3] Synthesis complete:")
    print(f"  Height field: {result.height_field.shape}")
    print(f"  Normal map: {result.normal_map.shape}")
    print(f"  Theta: {result.metadata['theta']}")
    print(f"  Gamma: {result.metadata['gamma']}")

    print(f"\n[3/3] Exporting...")
    os.makedirs(output_dir, exist_ok=True)
    results = export_all(
        result.height_field,
        result.normal_map,
        output_dir,
        base_name="demo",
        metadata=result.metadata
    )

    print("\nExported files:")
    for asset_type, path in results.items():
        print(f"  {asset_type}: {path}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return result


def run_synthesis(args):
    """Run synthesis with CLI arguments."""
    from ..core import synthesize
    from ..io import export_all, load_image_as_array

    print("=" * 60)
    print("Texture Synthesis")
    print("=" * 60)

    # Resolve carrier
    carrier = args.carrier
    if os.path.isfile(carrier):
        print(f"Loading carrier from: {carrier}")
        carrier = load_image_as_array(carrier)
    else:
        print(f"Using pattern: {carrier}")

    # Resolve operand
    operand = args.operand
    if operand and os.path.isfile(operand):
        print(f"Loading operand from: {operand}")
        operand = load_image_as_array(operand)
    elif operand:
        print(f"Using pattern: {operand}")

    # Synthesize
    print("\nSynthesizing...")
    result = synthesize(
        carrier,
        operand,
        theta=args.theta,
        gamma=args.gamma,
        num_eigenvectors=args.num_eigenvectors,
        edge_threshold=args.edge_threshold,
        output_size=args.size,
        normal_strength=args.normal_strength,
        mode=args.mode,
        preset=args.preset
    )

    print(f"\nResult:")
    print(f"  Height field: {result.height_field.shape}")
    print(f"  Normal map: {result.normal_map.shape}")

    # Export
    os.makedirs(args.output, exist_ok=True)
    results = export_all(
        result.height_field,
        result.normal_map,
        args.output,
        base_name=args.name,
        export_obj=args.obj,
        metadata=result.metadata
    )

    print("\nExported files:")
    for asset_type, path in results.items():
        print(f"  {asset_type}: {path}")

    return result


def run_theta_comparison(output_dir: str, size: int = 64):
    """Generate comparison grid showing different theta values."""
    from ..core import synthesize
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        print("PIL required for comparison grid")
        return

    print("=" * 60)
    print("Theta Comparison")
    print("=" * 60)

    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for theta in thetas:
        print(f"  Synthesizing at theta={theta}...")
        result = synthesize('amongus', 'checkerboard', theta=theta, output_size=size)
        results.append(result.height_field)

    # Create grid
    from scipy.ndimage import zoom
    cell_size = 128
    scale = cell_size / size

    grid = np.zeros((cell_size * 2, cell_size * len(thetas)), dtype=np.uint8)

    # Top row: carrier
    carrier_result = synthesize('amongus', None, mode='simple', output_size=size)
    carrier_resized = zoom(carrier_result.height_field, scale, order=1)
    carrier_uint8 = (carrier_resized * 255).astype(np.uint8)

    for i in range(len(thetas)):
        grid[0:cell_size, i*cell_size:(i+1)*cell_size] = carrier_uint8

    # Bottom row: results
    for i, height in enumerate(results):
        height_resized = zoom(height, scale, order=1)
        height_uint8 = (height_resized * 255).astype(np.uint8)
        grid[cell_size:, i*cell_size:(i+1)*cell_size] = height_uint8

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "theta_comparison.png")
    Image.fromarray(grid).save(path)
    print(f"\nSaved: {path}")
    print(f"Top row: carrier")
    print(f"Bottom row: theta = {thetas}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Unified Texture Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m texture --demo                    # Quick demo
  python -m texture --demo --preset fine      # Demo with fine detail
  python -m texture --compare                 # Theta comparison grid
  python -m texture --carrier amongus --operand noise
  python -m texture --carrier input.png --output results/
        """
    )

    # Demo modes
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with amongus carrier and checkerboard operand"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate theta comparison grid"
    )

    # Inputs
    parser.add_argument(
        "--carrier", "-c", type=str,
        help="Carrier: pattern name (amongus, checkerboard, dragon, noise) or image path"
    )
    parser.add_argument(
        "--operand", "-p", type=str, default=None,
        help="Operand: pattern name (noise, checkerboard, solid) or image path"
    )

    # Parameters
    parser.add_argument(
        "--theta", "-t", type=float, default=0.5,
        help="Rotation angle [0-1], 0=coarse, 1=fine (default: 0.5)"
    )
    parser.add_argument(
        "--gamma", "-g", type=float, default=0.3,
        help="Etch strength [0-1] (default: 0.3)"
    )
    parser.add_argument(
        "--num-eigenvectors", type=int, default=8,
        help="Number of eigenvectors (default: 8)"
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=0.1,
        help="Carrier edge sensitivity (default: 0.1)"
    )
    parser.add_argument(
        "--normal-strength", type=float, default=2.0,
        help="Normal map strength (default: 2.0)"
    )
    parser.add_argument(
        "--mode", type=str, default="spectral",
        choices=["spectral", "blend", "simple"],
        help="Synthesis mode (default: spectral)"
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=["coarse", "balanced", "fine", "etch_heavy"],
        help="Parameter preset (overrides theta/gamma)"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--name", "-n", type=str, default="texture",
        help="Output base filename (default: texture)"
    )
    parser.add_argument(
        "--size", "-s", type=int, default=64,
        help="Output size (default: 64)"
    )
    parser.add_argument(
        "--obj", action="store_true",
        help="Also export OBJ mesh"
    )

    args = parser.parse_args()

    # Handle modes
    if args.demo:
        run_demo(args.output, args.size, args.preset or 'balanced')
    elif args.compare:
        run_theta_comparison(args.output, args.size)
    elif args.carrier:
        run_synthesis(args)
    else:
        parser.print_help()
        print("\nRun with --demo for quick start, or specify --carrier")


if __name__ == "__main__":
    main()
