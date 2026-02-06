# Segment-Based Cross-Attention Restore Notes

## What was done

Restored segment-based cross-attention transfer across all four shader implementation files.
The prior state had broken cross-attention that operated on individual pixels rather than
coherent spectral segments, producing salt-and-pepper artifacts instead of meaningful
structure transfer.

## Files modified

1. **`spectral_shader_ops_cuter.py`** -- Rewrote `cross_attention_transfer()` with full
   segment infrastructure: `extract_segments`, `compute_segment_signature`, `match_segments`,
   `scatter_to_layer`, `composite_layers_hadamard`. Also added the `Segment` dataclass
   (coords, colors, centroid, bbox).

2. **`spectral_shader_layers.py`** -- `SpectralCrossAttention.forward()` delegates to
   the _cuter `cross_attention_transfer()` function. Fixed device-mismatch bug in
   `SpectralThickening._ensure_kernel()` (`.to(dtype=dtype)` -> `.to(device=device, dtype=dtype)`).

3. **`spectral_shader_model.py`** -- `SpectralCrossAttentionBlock` passes `min_segment_pixels`
   and `max_segments` config keys through to `SpectralCrossAttention`.

4. **`spectral_shader_ops.py`** -- Same segment pipeline in verbose style (the original
   implementation file, retained for backward compatibility and e2e regression tests).

5. **`spectral_shader_main.py`** -- Rewired from direct `spectral_shader_ops` imports to
   `SpectralShader.from_config()` from `spectral_shader_model.py`. CLI interface unchanged.

6. **`demo_spectral_shader.py`** -- Rewired from `shader_forwards`/`two_image_shader_pass`
   imports to `SpectralShader.from_config()`. ADSR envelope preserved by constructing a
   fresh model per pass with the appropriate config.

## Segment pipeline

The cross-attention transfer pipeline processes spectral segments, not individual pixels:

1. **Extract** -- Connected components in the low-gate region of image A (below Fiedler
   threshold). Each segment is a contiguous blob with its pixel coordinates, colors,
   centroid, and bounding box.

2. **Signature** -- Each segment gets a 4D spectral signature:
   `[mean_fiedler, std_fiedler, log_area, aspect_ratio]`. This captures the segment's
   spectral character independent of its spatial position.

3. **Match** -- Segments from A are matched to segments from B by L2 distance in
   z-normalized signature space. The normalization ensures all four dimensions contribute
   equally regardless of scale.

4. **Transplant** -- Each matched B segment is shifted so its centroid aligns with the
   corresponding A segment's centroid. The B segment's pixel colors travel with it.

5. **Rotate** -- Transplanted pixels are rotated 90 degrees around each A segment's
   centroid. This prevents trivial self-copies in self-attention mode and creates the
   characteristic visual displacement.

6. **Scatter** -- Rotated pixels are scattered into shadow and front layers with
   displacement offsets derived from `shadow_offset` and `translation_strength` config.

7. **Composite** -- Shadow and front layers are composited via Hadamard multiplication
   onto the thickened image. Front pixels overwrite shadow pixels where both exist.

## Why the old global-pixel approach was wrong

The prior (broken) cross-attention operated on individual pixels: it flattened the Fiedler
field into a 1D histogram, binned pixels by Fiedler value, and transferred colors between
bins. This had several problems:

- **No spatial coherence**: Pixels with similar Fiedler values but in different spatial
  regions were mixed together, destroying spatial structure.
- **Salt-and-pepper artifacts**: The bin-based transfer produced scattered single-pixel
  changes rather than coherent region transfers.
- **No structural matching**: The transfer was purely value-based (histogram equalization)
  rather than structure-based (segment matching by spectral signature).
- **Scale-invariance lost**: The approach was sensitive to image resolution because bin
  widths were fixed, not adaptive to the spectral structure.

The segment-based approach preserves spatial coherence because segments are contiguous
spatial regions, and matching operates on aggregate signatures rather than individual
pixel values.

## Test results

### E2E regression (4/4 PASS)
- `tonegraph_x_snek/cuter_vs_evencuter_pass1`: 0.000000 mean_abs (bit-exact)
- `tonegraph_x_snek/cuter_vs_evencuter_all`: 0.000000 mean_abs across all 4 passes
- `tonegraph_x_enso/cuter_vs_evencuter_pass1`: 0.000000 mean_abs (bit-exact)
- `tonegraph_x_enso/cuter_vs_evencuter_all`: 0.000000 mean_abs across all 4 passes

### Demo verification
- `demo_spectral_shader.py --cross` (3 cross-attention pairs): PASS
- `demo_spectral_shader.py --self` (2 self-attention images): PASS
- `spectral_shader_main.py` single-image mode: PASS
- `spectral_shader_main.py` two-image mode: PASS

## Known limitations

- **torch.compile fullgraph=False**: `SpectralEmbedding` requires `fullgraph=False` due
  to `sparse_coo_tensor` graph break in the Lanczos eigenvector computation. The shader
  layers themselves (gate, thickening, shadow, cross-attention) are fullgraph-compatible.

- **ADSR per-pass config**: The `SpectralShader.forward()` API supports `decay` (scalar
  effect-strength decay per pass) but not arbitrary per-pass config changes. The demo
  script works around this by constructing a fresh `SpectralShader.from_config()` per
  ADSR pass. This is correct but allocates a new model each pass.

- **CUDA non-determinism**: Cross-run comparison (same code, different executions) shows
  ~2-4% mean_abs diff due to CUDA floating-point non-determinism in the Lanczos iteration.
  Within a single run, _cuter and _even_cuter are bit-exact (0.000000 diff).
