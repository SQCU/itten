# Data Flow Documentation: `two_image_shader_pass`

This document traces the complete data flow of the `two_image_shader_pass` function in `spectral_shader_ops.py`, documenting every function called (and functions those call) with exactly 3 sentences describing inputs, computation, and outputs.

---

## Overview

The `two_image_shader_pass` function performs a two-image spectral shader operation where:
- **Image A** (target): defines WHERE segments go and WHAT topology we seek
- **Image B** (source): provides WHAT content (actual segment pixels, colors, shape)

Images A and B can have different sizes. No rescaling is performed because graphs have spectra, and rescaling destroys spectra.

---

## Call Order and Data Flow

### 1. `two_image_shader_pass` (Entry Point)

**Inputs:** Takes `image_A` (H_A, W_A, 3) as the target image defining topology and placement locations, `image_B` (H_B, W_B, 3) as the source image providing content, `fiedler_A` (H_A, W_A) and `fiedler_B` (H_B, W_B) as precomputed Fiedler vectors for each image, and a `config` dictionary containing effect parameters.

**Computation:** Orchestrates the full pipeline: (1) computes gate for image A, (2) applies high-gate thickening to A, (3) finds matching segments between A and B via spectral signatures, (4) transplants B's segment content to A's locations, and (5) draws the transplanted segments onto the output.

**Outputs:** Returns a tensor of shape (H_A, W_A, 3) containing image A with B's matched segment content composited at A's query locations, with shadow and front layers applied.

---

### 2. `adaptive_threshold` (Called by `two_image_shader_pass`)

**Inputs:** Takes a `fiedler` tensor of any shape (typically H, W) containing Fiedler vector values, and a `percentile` float (default 50.0) specifying which percentile to use as threshold.

**Computation:** Flattens the Fiedler tensor, sorts the values, computes the index corresponding to the requested percentile, and extracts that value as a scalar threshold.

**Outputs:** Returns a single float representing the Fiedler value at the specified percentile, useful for adaptive gating when Fiedler ranges vary across images.

---

### 3. `fiedler_gate` (Called by `two_image_shader_pass` and `transfer_segments_A_to_B`)

**Inputs:** Takes a `fiedler` tensor (N,) or (H, W) containing Fiedler vector values, a `threshold` float (default 0.0) as the sigmoid center point, and a `sharpness` float (default 10.0) controlling sigmoid steepness.

**Computation:** Applies the formula `gate = sigmoid((fiedler - threshold) * sharpness)` to produce soft gating values where high gate (near 1) indicates strong spectral resonance suitable for thickening, and low gate (near 0) indicates weak resonance suitable for shadow creation.

**Outputs:** Returns a tensor of the same shape as input with values in [0, 1] representing the gating strength at each pixel.

---

### 4. `dilate_high_gate_regions` (Called by `two_image_shader_pass`)

**Inputs:** Takes `image_rgb` (H, W, 3) as source image, `gate` (H, W) as gate values, `gate_threshold` float for high/low split, `dilation_radius` int for thickening extent, optional `fiedler` tensor for modulation, and `modulation_strength` float controlling how much complexity affects thickening.

**Computation:** Performs structure-preserving thickening in two stages: (1) creates a Gaussian splat selection mask via conv2d indicating WHERE to thicken, with inverse complexity modulation so high-complexity areas (kinks, corners) get LESS thickening for stability, (2) fills selected locations by sampling from original image with tiny transform (translate + rotation) using grid_sample with NEAREST mode to preserve exact values without interpolation artifacts.

**Outputs:** Returns an (H, W, 3) tensor where high-gate contour regions have been thickened while preserving their internal structure and periodicity, using a SwiGLU-style hard gate to decide which pixels receive copied values.

---

### 5. `compute_local_spectral_complexity` (Called by `dilate_high_gate_regions`)

**Inputs:** Takes a `fiedler` tensor (H, W) representing the Fiedler vector field, and a `window_size` int (default 5) for local variance computation.

**Computation:** Computes Sobel-style gradients via conv2d to find gradient magnitude, computes local variance via conv2d using the formula var = E[x^2] - E[x]^2, combines gradient magnitude and sqrt of local variance to produce a complexity measure, and normalizes to [0, 1].

**Outputs:** Returns an (H, W) tensor where high values indicate kinks, corners, and pinches (interesting structure), and low values indicate straight lines and smooth gradients.

---

### 6. `transfer_segments_A_to_B` (Called by `two_image_shader_pass`)

**Inputs:** Takes `image_A` (H_A, W_A, 3) as target image, `image_B` (H_B, W_B, 3) as source image (can have different size), `fiedler_A` and `fiedler_B` as their respective Fiedler fields, and a `config` dictionary with segmentation parameters.

**Computation:** Extracts segments from both images using their respective gates and Fiedler fields, computes spectral signatures for all segments (scale-invariant features: mean Fiedler, std Fiedler, log size, aspect ratio), and matches A's query segments to B's source segments by spectral similarity using L2 distance in normalized signature space.

**Outputs:** Returns a tuple of (query_segments from A defining WHERE, matched_segments from B providing CONTENT, match_indices tensor indicating which B segment matches each A segment).

---

### 7. `extract_segments_from_contours` (Called by `transfer_segments_A_to_B`)

**Inputs:** Takes `image_rgb` (H, W, 3) as source image, `gate` (H, W) as gate values, `gate_threshold` float for region selection, `contour_threshold` float (default 1.0) as std devs from mean for contour detection, `min_pixels` int for minimum segment size, and `max_segments` int as maximum count.

**Computation:** Converts image to grayscale, normalizes via LayerNorm style (subtract mean, divide by std), finds contours where absolute normalized value exceeds threshold, intersects with low-gate regions, and runs connected components via `_torch_connected_components` followed by conversion to Segment objects via `_labels_to_segments`.

**Outputs:** Returns a list of Segment dataclass objects, each containing mask, bbox, centroid, coords, colors, and label for a contiguous region.

---

### 8. `_torch_connected_components` (Called by `extract_segments_from_contours`)

**Inputs:** Takes a `mask` tensor (H, W) as a binary mask where True indicates foreground pixels to be labeled.

**Computation:** Initializes each foreground pixel with a unique label (its flattened index), then iteratively propagates minimum labels in 4 directions (up, down, left, right) until convergence (no changes), effectively merging connected regions to share the same minimum label, then relabels to consecutive integers.

**Outputs:** Returns an (H, W) tensor of integer labels where -1 indicates background and non-negative integers indicate distinct connected component IDs.

---

### 9. `_labels_to_segments` (Called by `extract_segments_from_contours`)

**Inputs:** Takes `image_rgb` (H, W, 3) as source for colors, `labels` (H, W) as integer component labels, `min_pixels` int as minimum segment size threshold, and `max_segments` int as maximum count to return.

**Computation:** Iterates through unique non-negative labels, for each computes the mask coordinates via torch.where, calculates bounding box (y0, y1, x0, x1), centroid as mean of coordinates, stacks coordinates as (x, y) pairs, and extracts RGB colors from source image at those positions.

**Outputs:** Returns a list of Segment dataclass objects, filtered to include only segments with at least min_pixels and capped at max_segments total.

---

### 10. `compute_segment_spectral_signature` (Called by `transfer_segments_A_to_B`)

**Inputs:** Takes a `segment` Segment dataclass containing coords, bbox, and other attributes, plus a `fiedler` tensor (H, W) representing the Fiedler field.

**Computation:** Samples Fiedler values at the segment's coordinate locations, computes mean and standard deviation of these values, calculates log of segment size, and computes aspect ratio from the bounding box (width/height).

**Outputs:** Returns a (4,) tensor containing [mean_fiedler, std_fiedler, log(size+1), aspect_ratio] as a scale-invariant spectral signature suitable for cross-image matching.

---

### 11. `match_segments_by_topology` (Called by `transfer_segments_A_to_B`)

**Inputs:** Takes `query_signatures` tensor (Q, 4) containing spectral signatures from image A, `source_signatures` tensor (S, 4) containing signatures from image B, and `top_k` int (default 1) specifying how many matches to return per query.

**Computation:** Normalizes both signature sets to zero mean and unit variance for fair comparison, computes pairwise L2 distance via broadcasting: (Q, 1, 4) - (1, S, 4) -> squared sum -> (Q, S), then uses torch.topk to find the top_k smallest distances (closest matches) for each query.

**Outputs:** Returns a (Q, top_k) tensor of indices into source_signatures indicating which source segments best match each query segment.

---

### 12. `draw_all_segments_batched` (Called by `two_image_shader_pass`)

**Inputs:** Takes `output` tensor (H, W, 3) as base image to draw onto, `segments` list of Segment objects, `translation_strength` float controlling displacement, `shadow_offset` float for additional shadow displacement, and `effect_strength` float scaling the overall effect.

**Computation:** Prepares all segment data in parallel via `prepare_batched_segment_data` (one concat + vectorized transforms), scatters shadows to one layer and fronts to another layer via `scatter_to_layer`, composites the two layers using Hadamard masking via `composite_layers_hadamard`, then blends the composite with base output where segments exist.

**Outputs:** Returns an (H, W, 3) tensor with all segments rendered as shadow+front pairs, composited onto the base image.

---

### 13. `prepare_batched_segment_data` (Called by `draw_all_segments_batched`)

**Inputs:** Takes `segments` list of Segment objects, `translation_strength` float, `shadow_offset` float, and `effect_strength` float controlling transformation magnitudes.

**Computation:** Concatenates all segment coords, colors, and centroids into batched tensors, rotates all points 90 degrees CCW around their respective centroids via vectorized formula (rel = coords - centroids; rotated = [-rel_y, rel_x] + centroids), computes translations for front and shadow positions, and computes shadow/front colors via `compute_shadow_colors` and `compute_front_colors`.

**Outputs:** Returns a tuple of (shadow_coords, shadow_colors, front_coords, front_colors), each as tensors with total_points rows, ready for scatter operations.

---

### 14. `compute_shadow_colors` (Called by `prepare_batched_segment_data`)

**Inputs:** Takes `rgb` tensor (N, 3) containing RGB colors and `effect_strength` float (default 1.0) scaling the transformation intensity.

**Computation:** Applies cyclic color transform with moderate rotation (0.3 * effect_strength) and strong contrast (0.8 * effect_strength), then applies a blue bias by boosting blue channel (0.7 * blue + 0.3) and reducing red channel (0.7 * red).

**Outputs:** Returns an (N, 3) tensor of shadow-tinted colors with blue bias, clamped to [0, 1].

---

### 15. `compute_front_colors` (Called by `prepare_batched_segment_data`)

**Inputs:** Takes `rgb` tensor (N, 3) containing RGB colors and `effect_strength` float (default 1.0) scaling the transformation intensity.

**Computation:** Applies cyclic color transform with light rotation (0.2 * effect_strength) and moderate contrast (0.6 * effect_strength), then applies a teal/cyan bias by boosting green (0.8 * green + 0.2), boosting blue (0.8 * blue + 0.15), and reducing red (0.6 * red).

**Outputs:** Returns an (N, 3) tensor of front-tinted colors with teal/cyan bias, clamped to [0, 1].

---

### 16. `cyclic_color_transform` (Called by `compute_shadow_colors` and `compute_front_colors`)

**Inputs:** Takes `rgb` tensor (N, 3) or (H, W, 3) with RGB values in [0, 1], `rotation_strength` float (default 0.3) controlling color wheel rotation amount, `contrast_strength` float (default 0.5) as sinusoid amplitude, and `phase_offset` float (default 0.0) for additional phase rotation.

**Computation:** Computes luminance as weighted sum (0.299R + 0.587G + 0.114B), maps luminance through sinusoidal color wheel with three phases 120 degrees apart for R/G/B channels using formula new_c = 0.5 + amplitude * sin(freq * luminance + phase), then blends the transformed colors with original based on rotation_strength.

**Outputs:** Returns a tensor of same shape as input containing cyclically transformed RGB colors, clamped to [0, 1].

---

### 17. `scatter_to_layer` (Called by `draw_all_segments_batched`)

**Inputs:** Takes `coords` tensor (N, 2) as (x, y) positions, `colors` tensor (N, 3) as RGB values, and `H`, `W` ints specifying output layer dimensions.

**Computation:** Creates empty layer buffer (H, W, 3) and occupancy mask (H, W), rounds coordinates to pixel positions and clamps to valid range, then scatters colors to the layer and 1.0 values to the mask at those positions (last-write-wins for overlapping points).

**Outputs:** Returns a tuple of (layer tensor (H, W, 3) containing scattered colors, mask tensor (H, W) with 1.0 where written and 0.0 elsewhere).

---

### 18. `composite_layers_hadamard` (Called by `draw_all_segments_batched`)

**Inputs:** Takes `layers` list of (H, W, 3) color buffer tensors, `masks` list of corresponding (H, W) occupancy mask tensors, and optional `priorities` list of floats (higher wins, default: later = higher).

**Computation:** For 2 layers (optimized case): computes output = shadow * shadow_mask * (1 - front_mask) + front * front_mask, implementing "front overwrites shadow" rule via pure Hadamard (elementwise) operations; for N layers: processes in reverse priority order, accumulating a visibility mask so each layer is visible only where it exists AND no higher-priority layer exists.

**Outputs:** Returns an (H, W, 3) tensor containing the composited result where each pixel shows the highest-priority layer that has content at that location.

---

## Complete Data Flow Diagram

```
two_image_shader_pass(image_A, image_B, fiedler_A, fiedler_B, config)
    |
    +---> adaptive_threshold(fiedler_A, 40) --> threshold_A
    |
    +---> fiedler_gate(fiedler_A, threshold_A, 10.0) --> gate_A
    |
    +---> dilate_high_gate_regions(image_A, gate_A, fiedler_A, ...)
    |         |
    |         +---> compute_local_spectral_complexity(fiedler_A) --> complexity
    |         |
    |         +---> [Gaussian splat via conv2d] --> selection_mask
    |         |
    |         +---> [grid_sample with NEAREST mode] --> thickened output
    |
    +---> transfer_segments_A_to_B(image_A, image_B, fiedler_A, fiedler_B, config)
    |         |
    |         +---> adaptive_threshold(fiedler_A/B) --> thresholds
    |         |
    |         +---> fiedler_gate(fiedler_A/B, ...) --> gate_A, gate_B
    |         |
    |         +---> extract_segments_from_contours(image_A, gate_A, ...) --> segments_A
    |         |         |
    |         |         +---> _torch_connected_components(mask) --> labels
    |         |         |
    |         |         +---> _labels_to_segments(image, labels, ...) --> segments
    |         |
    |         +---> extract_segments_from_contours(image_B, gate_B, ...) --> segments_B
    |         |
    |         +---> compute_segment_spectral_signature(seg, fiedler) --> signatures (for each)
    |         |
    |         +---> match_segments_by_topology(sigs_A, sigs_B) --> match_indices
    |         |
    |         +---> [gather matched segments] --> matched_segments
    |
    +---> [Create transplanted segments: B's content at A's locations]
    |
    +---> draw_all_segments_batched(output, transplanted_segments, ...)
              |
              +---> prepare_batched_segment_data(segments, ...)
              |         |
              |         +---> compute_shadow_colors(colors, ...)
              |         |         |
              |         |         +---> cyclic_color_transform(rgb, ...) --> transformed
              |         |
              |         +---> compute_front_colors(colors, ...)
              |                   |
              |                   +---> cyclic_color_transform(rgb, ...) --> transformed
              |
              +---> scatter_to_layer(shadow_coords, shadow_colors, H, W) --> shadow_layer, shadow_mask
              |
              +---> scatter_to_layer(front_coords, front_colors, H, W) --> front_layer, front_mask
              |
              +---> composite_layers_hadamard([shadow, front], [sm, fm]) --> segment_composite
              |
              +---> [Blend with base: output * (1-mask) + composite * mask] --> final_output
```

---

## Summary

The `two_image_shader_pass` function implements a sophisticated image composition pipeline that:

1. **Gates** the target image A using its Fiedler vector to identify high/low spectral resonance regions
2. **Thickens** high-gate regions in A while preserving structure through complexity-modulated Gaussian splatting
3. **Extracts** segments from both images via contour detection and connected components
4. **Matches** A's segments to B's segments using scale-invariant spectral signatures
5. **Transplants** B's matched segment content (shape, colors) to A's query locations (centroids)
6. **Transforms** transplanted segments with 90-degree rotation and translation
7. **Renders** shadow and front layers via parallel scatter operations
8. **Composites** layers using Hadamard masking where front overwrites shadow

The entire pipeline is designed for tensor operations without sequential pixel loops, enabling efficient parallel computation on GPU.
