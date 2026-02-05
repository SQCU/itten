# Lattice Spectral Decisions Results

## Overview

This report documents the results of testing theta-dependent lattice type selection
in the `ExpansionGatedExtruder`. The theta parameter controls the spectral emphasis:

- **theta=0.1**: Expansion-dominant (high expansion -> hex in open regions)
- **theta=0.5**: Balanced between expansion and gradient
- **theta=0.9**: Gradient-dominant (high gradient -> triangle at bottlenecks)

## Test Configuration

- **Territory**: 4 islands connected by 3 bridges
- **Island radius**: 6 nodes
- **Bridge width**: 2 nodes
- **Spacing**: 20 units between island centers
- **Expansion threshold**: 1.0
- **Lanczos iterations**: 15
- **Hop radius**: 3

## Results

### Lattice Type Distribution

| theta | square | triangle | hex | total |
|-------|--------|----------|-----|-------|
| 0.1 | 0 | 0 | 590 | 590 |
| 0.5 | 85 | 0 | 505 | 590 |
| 0.9 | 378 | 128 | 84 | 590 |

### Node Value Statistics

| theta | mean | std | min | max |
|-------|------|-----|-----|-----|
| 0.1 | 0.7901 | 0.0063 | 0.7688 | 0.8045 |
| 0.5 | 0.5000 | 0.0206 | 0.4496 | 0.5521 |
| 0.9 | 0.5431 | 0.0848 | 0.2809 | 0.6732 |

## Success Criteria

1. **Lattice type distribution changes with theta**: PASS
2. **Not all geometry_type == 'square'**: PASS
3. **Node values show theta dependence**: PASS

**Overall Result**: PASS

## Implementation Details

### select_lattice_type() Function

The `select_lattice_type()` function determines lattice geometry based on:

```python
def select_lattice_type(expansion, fiedler_gradient_mag, theta):
    # Use log-scale normalization for expansion (handles wide range 0.1-10+)
    log_expansion = log(max(expansion, 0.1))
    norm_expansion = clip((log_expansion + 1.0) / 3.5, 0, 1)

    # Use log-scale normalization for gradient (handles small values)
    log_gradient = log(max(fiedler_gradient_mag, 0.0001))
    norm_gradient = clip((log_gradient + 9.0) / 8.3, 0, 1)

    # Compute spectral score
    spectral_score = (1 - theta) * norm_expansion + theta * (1 - norm_gradient)

    if spectral_score > 0.60:
        return 'hex'       # Open region
    elif spectral_score < 0.40:
        return 'triangle'  # Bottleneck
    else:
        return 'square'    # Neutral
```

### Node Value Computation

Node values for visualization are computed as theta-weighted blend:

- theta < 0.5: Blend between expansion and Fiedler value
- theta >= 0.5: Blend between Fiedler value and gradient magnitude

This ensures visual properties change continuously with theta.

## Interpretation

The test demonstrates that lattice extrusion is now spectrally-determined:

1. **Spectral properties drive geometry selection**: The combination of local
   expansion (lambda_2) and Fiedler gradient determines lattice type.

2. **Theta provides user control**: Adjusting theta changes the emphasis
   between expansion-based and gradient-based geometry selection.

3. **Visual properties vary**: Node values change with theta, enabling
   theta-dependent visualization in downstream rendering.

This satisfies the goal of making lattice extrusion spectrally-determined
with theta interactions, rather than always using 'square' geometry.