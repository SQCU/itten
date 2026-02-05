# Hypercontext: Iterative Spectral Transform Approximation

## Mission
Derive how to approximate a large rotation angle partial spectral transform by iterating small rotation angle partial spectral transforms, achieving FFT-style O(n log n) complexity.

## Background: Current Spectral Operations

### The θ Parameter
In `texture_synth_v2/synthesize.py`:
- θ=0: Vertex domain (carrier edges, Fiedler dominant)
- θ=1: Spectral domain (higher eigenvectors dominant)
- Intermediate θ: Gaussian-weighted blend of eigenvectors

Current formula (from `spectral_etch_residual`):
```python
weights = np.exp(-((indices - theta * actual_k) ** 2) / 2.0)
spectral_field = sum(|eigenvector_k| * weight_k)
```

### The Problem
Computing eigenvectors requires:
- Dense solver: O(n³)
- Lanczos: O(k·n·iterations) but still needs full matrix access

For sparse/infinite graphs, we cannot:
- Materialize the full Laplacian
- Access arbitrary matrix elements
- Store all eigenvectors

## Key Insight Needed
How to decompose:
```
spectral_transform(θ=0.8) ≈ iterate(spectral_transform(θ=0.1), 8 times)
```

Without computing eigenvectors explicitly.

## Mathematical Framework

### 1. Heat Kernel Perspective
Heat diffusion: `u(t) = exp(-tL) @ u(0)`
- Small t = local smoothing (small θ)
- Large t = global equilibrium (large θ)
- Composable: `exp(-t₁L) @ exp(-t₂L) = exp(-(t₁+t₂)L)`

Current heat diffusion (from `spectral_ops_fast.py`):
```python
def heat_diffusion_sparse(L, signal, alpha=0.1, iterations=10):
    x = signal
    for _ in range(iterations):
        x = x - alpha * L @ x  # Explicit Euler: x' = -Lx
    return x
```

This is already iterative! But it's smoothing, not rotation.

### 2. Rotation in Spectral Space
True rotation between vertex/spectral domain involves:
```
R(θ) = cos(θ)I + sin(θ)·(something involving eigenvectors)
```

The "something" is the hard part.

### 3. Chebyshev Polynomial Approach
Chebyshev polynomials approximate functions on [-1,1]:
```
f(L) ≈ sum_k c_k T_k(L̃)
```
where L̃ = 2L/λ_max - I

Properties:
- T_k(L̃) only requires matrix-vector products
- Recurrence: T_k(x) = 2x·T_{k-1}(x) - T_{k-2}(x)
- O(k·n) per polynomial evaluation

### 4. Lanczos as Implicit Spectral Rotation
Lanczos builds Krylov subspace: {v, Lv, L²v, ..., L^k v}
- This is implicitly the spectral basis restricted to starting vector
- k iterations ≈ "resolution" in spectral domain
- More iterations = access to higher eigenvector components

### 5. Polynomial Filtering (Key Technique)
To emphasize eigenvectors with eigenvalue λ ≈ λ_target:
```
filter(L) @ v ≈ projection onto eigenvectors near λ_target
```

Gaussian filter in spectral domain:
```
g(λ) = exp(-(λ - λ_target)² / 2σ²)
```

Can be approximated by polynomial:
```
g(L) ≈ sum_k c_k L^k
```

Each L^k only needs matrix-vector products!

## Proposed Algorithm

### Small-θ Step
A single small rotation step:
```python
def small_theta_step(L, signal, delta_theta, current_theta):
    # Polynomial approximation of spectral filter
    # centered at current_theta, width delta_theta

    # Chebyshev coefficients for Gaussian filter
    target_lambda = current_theta * lambda_max
    sigma = delta_theta * lambda_max

    # Apply polynomial filter using Chebyshev recurrence
    result = chebyshev_filter(L, signal, target_lambda, sigma, order=10)
    return result
```

### Iterative Large-θ Transform
```python
def iterative_spectral_transform(L, signal, theta, num_steps=8):
    delta_theta = theta / num_steps
    current = signal

    for step in range(num_steps):
        current_theta = step * delta_theta
        current = small_theta_step(L, current, delta_theta, current_theta)

    return current
```

### Complexity Analysis
- Each step: O(order · n) matrix-vector products
- Total steps: O(log(1/ε)) for ε precision
- Total: O(n log(1/ε) · order) ≈ O(n log n) for fixed precision

## Key Files to Modify/Create

```
spectral_ops_fast.py:
  - Add chebyshev_filter(L, signal, center, width, order)
  - Add iterative_spectral_transform(L, signal, theta, num_steps)
  - Add polynomial_spectral_field(L, carrier, theta) using Chebyshev

texture_synth_v2/synthesize.py:
  - Option to use iterative transform instead of explicit eigenvectors
```

## Research Questions
1. What polynomial order is needed for acceptable approximation?
2. How does error accumulate over iterations?
3. Can we use adaptive step sizes (smaller at sharp transitions)?
4. How does this relate to multi-scale spectral graph wavelets?

## References (Conceptual)
- Chebyshev polynomial filtering for graph signals
- Fast spectral graph wavelets (Hammond et al.)
- Lanczos-based spectral approximation
- Krylov subspace methods

## Deliverables
1. Mathematical derivation of the iterative scheme
2. Implementation in spectral_ops_fast.py
3. Comparison: explicit eigenvectors vs iterative approximation
4. Complexity analysis confirming O(n log n) scaling
