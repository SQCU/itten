# Distance Field via Heat Kernel

## The Computation

```
INPUT: image gray, full Laplacian L_B

1. boundary_signal = |layernorm(gray)| > threshold
   - 1 where content edges exist, 0 in flat regions (void or interior)

2. distance_field = heat_kernel(L_B, t) @ boundary_signal
   - heat_kernel(L, t) ≈ (I + tL)^{-1}
   - approximated by polynomial: Σ_k a_k L^k (Chebyshev)
   - or by repeated (I - αL)^k for small α

3. toward_void = -∇(distance_field)
   - gradient points toward high values (toward boundary)
   - negative gradient points toward low values (toward void)

OUTPUT: toward_void_y, toward_void_x (unit vectors)
```

## Why This Works

- boundary_signal is source of "heat"
- heat diffuses through L_B, decays with graph distance
- far from boundary (deep in void): low temperature
- near boundary: high temperature
- gradient of temperature points toward hot (boundary)
- negative gradient points toward cold (void)

## What This Replaces

OLD (wrong): diffuse "void indicator" through L, hope gradient points right
- confused because void regions have high edge weights (uniform = easy diffusion)
- diffusion spread everywhere, gradient was noisy

NEW (correct): diffuse from BOUNDARY, gradient points toward/away from boundary
- boundary is well-defined (edge detection)
- heat kernel gives distance-like decay
- gradient is clean: toward boundary or away from it

## Polynomial Approximation of Heat Kernel

heat_kernel(λ) = 1/(1 + tλ)

For λ ∈ [0, λ_max], approximate with Chebyshev polynomial of degree K.
Or simpler: (I - αL)^k has eigenvalue response (1-αλ)^k ≈ e^{-kαλ} ≈ heat kernel.

Choose α = t/k for stability (need αλ_max < 1).
