"""
STACKED RESNET DEMONSTRATION

Applies the same spectral shader rule 1x, 2x, 4x, 8x times in sequence.
Each deeper pass has ATTENUATED gating - requires higher activation energy
to trigger edits.

This demonstrates true ResNet-like behavior:
- Same operation applied repeatedly
- Output of pass N feeds into pass N+1
- Deeper passes are more selective (higher activation threshold)
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Import the v4 shader components
from resnet_spectral_shader_v4 import (
    compute_fiedler_gate,
    thicken_contours_gated,
    extract_contour_segments,
    compute_translation_direction,
    draw_segment_with_drop_shadow,
)


def single_pass(img_rgb, pass_number, total_passes, base_config):
    """
    Apply one pass of the shader with attenuated gating.

    Attenuation: deeper passes require higher activation to trigger edits.
    Pass 1: gate_threshold = base (permissive)
    Pass N: gate_threshold = base - (N-1) * attenuation_step (stricter)
    """
    # Convert RGB to grayscale for processing
    if img_rgb.ndim == 3:
        img_gray = img_rgb.mean(axis=2)
    else:
        img_gray = img_rgb

    H, W = img_gray.shape

    # Attenuate gate threshold: deeper passes are stricter
    # Start at 0.65, decrease by 0.08 per pass
    attenuation_step = 0.08
    gate_threshold = base_config['base_gate_threshold'] - (pass_number - 1) * attenuation_step
    gate_threshold = max(gate_threshold, 0.25)  # Floor at 0.25

    # Also attenuate effect strength for deeper passes
    effect_strength = 1.0 - (pass_number - 1) * 0.15
    effect_strength = max(effect_strength, 0.4)

    print(f"    Pass {pass_number}: gate_threshold={gate_threshold:.2f}, effect_strength={effect_strength:.2f}")

    # Compute Fiedler gate
    fiedler, gate = compute_fiedler_gate(img_gray, edge_threshold=base_config['edge_threshold'])

    # Start with input as output
    output_rgb = img_rgb.copy() if img_rgb.ndim == 3 else np.stack([img_rgb]*3, axis=-1)

    # [BRANCH A] Thicken contours in high-gate regions
    # Scale thickness by effect strength
    thickness = int(base_config['max_thickness'] * effect_strength)
    thickness = max(thickness, 1)
    thickened = thicken_contours_gated(img_gray, gate, max_thickness=thickness)

    # Apply thickening
    high_gate = gate > (1 - gate_threshold)  # Inverted: high gate = above threshold
    thicken_region = (thickened > 0.5) & high_gate
    output_rgb[thicken_region, 0] = 0.02
    output_rgb[thicken_region, 1] = 0.02
    output_rgb[thicken_region, 2] = 0.05

    # [BRANCH B] Extract and rotate segments in low-gate regions
    segments = extract_contour_segments(
        img_gray, gate,
        min_pixels=base_config['min_segment_pixels'],
        max_segments=int(base_config['max_segments'] * effect_strength),
        gate_threshold=gate_threshold
    )

    # Process segments with attenuated translation
    translation_distance = int(base_config['translation_distance'] * effect_strength)

    for segment in segments:
        cy, cx = segment['center']
        ty, tx = compute_translation_direction(img_gray, cy, cx, offset_distance=translation_distance)

        draw_segment_with_drop_shadow(
            output_rgb, segment, img_gray,
            translation=(ty, tx),
            rotation_k=1,
            front_color_rotation=55,
            shadow_color_rotation=100,
            shadow_offset=(int(6 * effect_strength), int(6 * effect_strength)),
            front_contrast=1.4,
            shadow_contrast=1.5
        )

    return output_rgb, len(segments)


def stacked_resnet_shader(img_np, num_passes, base_config=None):
    """
    Apply shader num_passes times in sequence.
    Each pass feeds its output to the next pass.
    """
    if base_config is None:
        base_config = {
            'edge_threshold': 0.10,
            'max_thickness': 5,
            'min_segment_pixels': 10,
            'max_segments': 40,
            'base_gate_threshold': 0.65,
            'translation_distance': 25,
        }

    # Start with grayscale -> RGB
    if img_np.ndim == 2:
        current = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        current = img_np.copy()

    total_segments = 0

    for pass_num in range(1, num_passes + 1):
        current, num_segments = single_pass(current, pass_num, num_passes, base_config)
        total_segments += num_segments

    return current, total_segments


def run_stacked_demo(image_path, output_dir):
    """
    Run demo showing 1x, 2x, 4x, 8x stacked passes.
    """
    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    print(f"  Image: {W}x{H}")

    pass_counts = [1, 2, 4, 8]
    outputs = []

    for num_passes in pass_counts:
        print(f"\n  Running {num_passes}x stacked passes:")
        output, total_segments = stacked_resnet_shader(img, num_passes)
        outputs.append((num_passes, output, total_segments))
        print(f"    Total segments processed: {total_segments}")

    # Build comparison grid
    # Row 1: Original, 1x, 2x
    # Row 2: 4x, 8x, difference (8x vs original)

    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    original_rgb = to_rgb(img)

    # Pad to same size if needed
    row1 = np.concatenate([
        original_rgb,
        outputs[0][1],  # 1x
        outputs[1][1],  # 2x
    ], axis=1)

    # Difference: 8x vs original (amplified)
    diff = np.abs(outputs[3][1] - original_rgb)
    diff = diff / (diff.max() + 1e-8) * 2  # Amplify
    diff = np.clip(diff, 0, 1)

    row2 = np.concatenate([
        outputs[2][1],  # 4x
        outputs[3][1],  # 8x
        diff,
    ], axis=1)

    grid = np.concatenate([row1, row2], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_stacked_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"\n  Saved: {out_path}")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["ORIGINAL", "1x PASS", "2x PASSES"],
        ["4x PASSES", "8x PASSES", "DIFFERENCE (8x)"],
    ]
    for ri, row_labels in enumerate(labels):
        for ci, lbl in enumerate(row_labels):
            x, y = ci * W + 5, ri * H + 5
            draw.text((x+1, y+1), lbl, fill=(0, 0, 0))
            draw.text((x, y), lbl, fill=(255, 255, 100))

    labeled.save(Path(output_dir) / f"resnet_stacked_{stem}_labeled.png")

    return outputs


def run_attenuation_visualization(image_path, output_dir):
    """
    Visualize how attenuation affects each pass.
    Shows the gate threshold and effect strength per pass.
    """
    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # Run 8 passes and capture intermediate states
    base_config = {
        'edge_threshold': 0.10,
        'max_thickness': 5,
        'min_segment_pixels': 10,
        'max_segments': 40,
        'base_gate_threshold': 0.65,
        'translation_distance': 25,
    }

    current = np.stack([img, img, img], axis=-1)
    intermediates = [current.copy()]

    print(f"\n  Attenuation visualization for 8 passes:")
    for pass_num in range(1, 9):
        current, _ = single_pass(current, pass_num, 8, base_config)
        intermediates.append(current.copy())

    # Build grid: passes 1, 2, 3, 4 on row 1; passes 5, 6, 7, 8 on row 2
    row1 = np.concatenate([intermediates[i] for i in [1, 2, 3, 4]], axis=1)
    row2 = np.concatenate([intermediates[i] for i in [5, 6, 7, 8]], axis=1)

    grid = np.concatenate([row1, row2], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_attenuation_{stem}.png"
    Image.fromarray(grid).save(out_path)

    # Labels with attenuation info
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)

    for row in range(2):
        for col in range(4):
            pass_num = row * 4 + col + 1
            gate_thresh = 0.65 - (pass_num - 1) * 0.08
            gate_thresh = max(gate_thresh, 0.25)
            effect = 1.0 - (pass_num - 1) * 0.15
            effect = max(effect, 0.4)

            x = col * W + 5
            y = row * H + 5
            label = f"Pass {pass_num}"
            sublabel = f"gate={gate_thresh:.2f}"

            draw.text((x+1, y+1), label, fill=(0, 0, 0))
            draw.text((x, y), label, fill=(255, 255, 100))
            draw.text((x+1, y+16), sublabel, fill=(0, 0, 0))
            draw.text((x, y+15), sublabel, fill=(200, 200, 255))

    labeled.save(Path(output_dir) / f"resnet_attenuation_{stem}_labeled.png")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    test_images = [
        "demo_output/inputs/snek-heavy.png",
        "demo_output/inputs/toof.png",
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n{'='*60}")
            print(f"Processing: {img_path}")
            print('='*60)

            # Main stacked demo
            run_stacked_demo(img_path, str(output_dir))

            # Attenuation visualization
            run_attenuation_visualization(img_path, str(output_dir))
