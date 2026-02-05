"""
Validation harness for spectral_shader_ops refactoring.

Captures intermediate tensors at key checkpoints to verify that refactored
code produces identical (or intentionally different) outputs.

Usage:
    # Before refactoring: capture baseline
    python test_spectral_shader_validation.py --capture baseline

    # After each refactoring step: compare to baseline
    python test_spectral_shader_validation.py --compare baseline

    # Compare two different captures
    python test_spectral_shader_validation.py --diff capture_a capture_b
"""
import torch
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Checkpoint storage directory
CHECKPOINT_DIR = Path("validation_checkpoints")


@dataclass
class ValidationCheckpoint:
    """Container for intermediate tensor values at a checkpoint."""
    name: str
    tensors: Dict[str, torch.Tensor]

    def save(self, base_path: Path):
        """Save checkpoint tensors to disk."""
        checkpoint_path = base_path / self.name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        for tensor_name, tensor in self.tensors.items():
            torch.save(tensor.cpu(), checkpoint_path / f"{tensor_name}.pt")

    @classmethod
    def load(cls, base_path: Path, name: str) -> "ValidationCheckpoint":
        """Load checkpoint tensors from disk."""
        checkpoint_path = base_path / name
        tensors = {}
        for pt_file in checkpoint_path.glob("*.pt"):
            tensor_name = pt_file.stem
            tensors[tensor_name] = torch.load(pt_file)
        return cls(name=name, tensors=tensors)


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6
) -> Tuple[bool, str]:
    """Compare two tensors, return (match, message)."""
    if a.shape != b.shape:
        return False, f"{name}: shape mismatch {a.shape} vs {b.shape}"

    # Move both to CPU for comparison
    a = a.cpu()
    b = b.cpu()

    if a.dtype != b.dtype:
        # Allow dtype differences if values match
        a = a.float()
        b = b.float()

    if torch.allclose(a, b, rtol=rtol, atol=atol):
        return True, f"{name}: MATCH"

    # Compute difference statistics
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if difference is meaningful
    relative_diff = diff / (a.abs() + 1e-8)
    max_rel = relative_diff.max().item()

    return False, (
        f"{name}: MISMATCH - max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
        f"max_rel={max_rel:.2e}"
    )


def run_validation_capture(
    image_rgb: torch.Tensor,
    capture_name: str = "baseline",
    config: Optional[Dict] = None
) -> ValidationCheckpoint:
    """
    Run spectral shader and capture all intermediate tensors.

    Checkpoints (pure tensor API - no segments):
    - grayscale: (H, W) luminance
    - contours: (H, W) bool mask
    - fiedler: (H, W) Fiedler vector
    - gate: (H, W) gated values
    - high_gate_output: (H, W, 3) after dilation
    - low_gate_output: (H, W, 3) after low-gate transform
    - final_output: (H, W, 3) complete result
    """
    from spectral_shader_ops import (
        fiedler_gate, adaptive_threshold, dilate_high_gate_regions,
        apply_low_gate_transform, to_grayscale, detect_contours,
        _compute_fiedler_from_tensor, spectral_shader_pass
    )

    if config is None:
        config = {
            'effect_strength': 1.0,
            'gate_sharpness': 10.0,
            'translation_strength': 15.0,
            'shadow_offset': 5.0,
            'dilation_radius': 2,
        }

    tensors = {}

    # 1. Grayscale
    gray = to_grayscale(image_rgb)
    tensors['grayscale'] = gray

    # 2. Contours
    contours = detect_contours(gray)
    tensors['contours'] = contours.float()

    # 3. Fiedler
    fiedler = _compute_fiedler_from_tensor(image_rgb)
    tensors['fiedler'] = fiedler

    # 4. Gate
    threshold = adaptive_threshold(fiedler, 40.0)
    gate = fiedler_gate(fiedler, threshold, config['gate_sharpness'])
    tensors['gate'] = gate
    tensors['gate_threshold'] = torch.tensor([threshold])

    # 5. High-gate dilation output
    effect_strength = config.get('effect_strength', 1.0)
    dilation_radius = int(config.get('dilation_radius', 2) * effect_strength)

    high_gate_output = dilate_high_gate_regions(
        image_rgb, gate, fiedler,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        modulation_strength=config.get('thicken_modulation', 0.3),
        gray=gray,
        contours=contours
    )
    tensors['high_gate_output'] = high_gate_output

    # 6. Low-gate transform output (pure tensor, no segments)
    low_gate_output = apply_low_gate_transform(
        high_gate_output, gate, fiedler, contours,
        shadow_offset=config.get('shadow_offset', 7.0) * effect_strength,
        translation_strength=config.get('translation_strength', 20.0) * effect_strength,
        effect_strength=effect_strength
    )
    tensors['low_gate_output'] = low_gate_output

    # 7. Final output via full pass
    config_with_threshold = {**config, 'gate_threshold': threshold}
    final_output = spectral_shader_pass(image_rgb, fiedler, config_with_threshold)
    tensors['final_output'] = final_output

    checkpoint = ValidationCheckpoint(name=capture_name, tensors=tensors)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint.save(CHECKPOINT_DIR)
    print(f"Saved checkpoint '{capture_name}' to {CHECKPOINT_DIR}")

    return checkpoint


def compare_checkpoints(
    checkpoint_a: ValidationCheckpoint,
    checkpoint_b: ValidationCheckpoint,
    rtol: float = 1e-5,
    atol: float = 1e-6
) -> List[Tuple[bool, str]]:
    """Compare two checkpoints tensor-by-tensor."""
    results = []

    # Get all tensor names from both
    all_names = set(checkpoint_a.tensors.keys()) | set(checkpoint_b.tensors.keys())

    for name in sorted(all_names):
        if name not in checkpoint_a.tensors:
            results.append((False, f"{name}: missing in {checkpoint_a.name}"))
            continue
        if name not in checkpoint_b.tensors:
            results.append((False, f"{name}: missing in {checkpoint_b.name}"))
            continue

        match, msg = compare_tensors(
            checkpoint_a.tensors[name],
            checkpoint_b.tensors[name],
            name,
            rtol=rtol,
            atol=atol
        )
        results.append((match, msg))

    return results


def load_test_image(device: torch.device) -> torch.Tensor:
    """Load a consistent test image."""
    # Try several possible locations
    test_paths = [
        "demo_output/inputs/1bit redraw.png",
        "test_images/test.png",
        "demo_output/inputs/snek-heavy.png",
    ]

    for path in test_paths:
        if os.path.exists(path):
            from PIL import Image
            import numpy as np
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.tensor(arr, device=device)

    # Fallback: generate synthetic test image
    print("No test image found, generating synthetic test pattern")
    H, W = 256, 256
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    # Checkerboard + gradient
    checker = ((x * 8).floor() + (y * 8).floor()) % 2
    gradient = torch.stack([x, y, 0.5 * torch.ones_like(x)], dim=-1)
    return gradient * 0.7 + checker.unsqueeze(-1) * 0.3


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validation harness for spectral_shader_ops")
    parser.add_argument("--capture", type=str, help="Capture checkpoint with given name")
    parser.add_argument("--compare", type=str, help="Compare current code against checkpoint")
    parser.add_argument("--diff", nargs=2, help="Compare two checkpoints")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.capture:
        print(f"\n=== CAPTURING CHECKPOINT: {args.capture} ===")
        image = load_test_image(device)
        print(f"Test image shape: {image.shape}")
        checkpoint = run_validation_capture(image, args.capture)
        print(f"\nCaptured {len(checkpoint.tensors)} tensors:")
        for name, tensor in sorted(checkpoint.tensors.items()):
            print(f"  {name}: {tensor.shape} ({tensor.dtype})")

    elif args.compare:
        print(f"\n=== COMPARING AGAINST CHECKPOINT: {args.compare} ===")
        # Load baseline
        baseline = ValidationCheckpoint.load(CHECKPOINT_DIR, args.compare)
        print(f"Loaded baseline with {len(baseline.tensors)} tensors")

        # Run current code
        image = load_test_image(device)
        current = run_validation_capture(image, "current_run")

        # Compare
        print("\n--- COMPARISON RESULTS ---")
        results = compare_checkpoints(baseline, current, args.rtol, args.atol)

        n_match = sum(1 for match, _ in results if match)
        n_total = len(results)

        for match, msg in results:
            status = "OK" if match else "FAIL"
            print(f"[{status}] {msg}")

        print(f"\n{n_match}/{n_total} tensors match")

        if n_match == n_total:
            print("SUCCESS: All tensors match!")
        else:
            print("FAILURE: Some tensors differ")
            return 1

    elif args.diff:
        name_a, name_b = args.diff
        print(f"\n=== COMPARING {name_a} vs {name_b} ===")

        checkpoint_a = ValidationCheckpoint.load(CHECKPOINT_DIR, name_a)
        checkpoint_b = ValidationCheckpoint.load(CHECKPOINT_DIR, name_b)

        results = compare_checkpoints(checkpoint_a, checkpoint_b, args.rtol, args.atol)

        for match, msg in results:
            status = "OK" if match else "DIFF"
            print(f"[{status}] {msg}")

    else:
        # Default: capture baseline if none exists, otherwise compare
        baseline_path = CHECKPOINT_DIR / "baseline"
        if baseline_path.exists():
            print("Baseline exists, running comparison...")
            image = load_test_image(device)
            baseline = ValidationCheckpoint.load(CHECKPOINT_DIR, "baseline")
            current = run_validation_capture(image, "current")
            results = compare_checkpoints(baseline, current, args.rtol, args.atol)

            for match, msg in results:
                print(f"[{'OK' if match else 'FAIL'}] {msg}")
        else:
            print("No baseline found, capturing initial baseline...")
            image = load_test_image(device)
            run_validation_capture(image, "baseline")

    return 0


if __name__ == "__main__":
    exit(main())
