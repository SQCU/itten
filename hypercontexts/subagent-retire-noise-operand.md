# Subagent Handoff: Retire Noise Operand, Create CG-Derived Alternatives

## Problem
Uniform random noise as operand produces "mushy" images that don't show features from both carrier AND operand. No spatial covariance relating to both images.

The old reference used amongus × checkerboard because ANY transform matrix applied to either produces OBVIOUS variations - good covariance metrics.

## Mission
1. Retire/deprecate `generate_noise()` as a go-to operand
2. Create new "noisy-like" operands derived from CG operations on natural images

## New Operand Types to Create

Add to `/home/bigboi/itten/texture/patterns.py`:

### 1. Edge-Detected Natural Image
```python
def generate_edges_operand(natural_image, method='sobel'):
    """Extract edges from natural image as operand."""
    # Sobel/Canny edge detection
    # Result has structure but isn't uniform
```

### 2. Quantized/Posterized Natural Image
```python
def generate_posterized_operand(natural_image, levels=4):
    """Quantize natural image to N levels."""
    # Creates bands/regions with sharp boundaries
```

### 3. Morphological Operations
```python
def generate_morpho_operand(natural_image, operation='gradient'):
    """Apply morphological operations (dilate-erode, gradient, etc.)"""
    # Creates structure-aware "noise-like" patterns
```

### 4. Frequency-Shifted Natural Image
```python
def generate_frequency_shift_operand(natural_image, shift='high'):
    """High-pass or band-pass filtered version."""
    # Keeps spatial structure but changes frequency content
```

### 5. Thresholded Natural Image
```python
def generate_threshold_operand(natural_image, percentile=50):
    """Binary threshold at percentile."""
    # Creates silhouette-like pattern
```

### 6. Self-Similarity Tiled
```python
def generate_tiled_operand(natural_image, tile_size=16):
    """Tile a cropped region of the image."""
    # Creates periodic structure from natural content
```

## Test Configuration

For each new operand type:
1. Apply to `snek-heavy.png` as source
2. Use as operand with `amongus` as carrier
3. Compute PSNR(carrier, result) and PSNR(operand, result)
4. Verify BOTH show meaningful theta dependence (not just one)

## Success Criteria
- New operands produce visible features from BOTH carrier and operand
- PSNR covariance exists on both dimensions
- Results are NOT "mushy"

## Output
- Updated `/home/bigboi/itten/texture/patterns.py` with new operand generators
- Test in `/home/bigboi/itten/tests/test_cg_operands.py`
- Results in `/home/bigboi/itten/hypercontexts/cg-operands-results.md`
- Demo images in `/home/bigboi/itten/demo_output/cg_operands/`
