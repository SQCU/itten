# What a Boring Neural Network Actually Is

## The ResNet Pattern

A resnet is not "output = input + delta at every step." It is nested scopes with forced combination points:

```
f(x, h):
    h1 = h                              # stash hidden state
    h2 = f1(norm(x))                    # compute in isolation
    x = x + h2                          # forced combine
    h2 = f2(norm(x) + f3(h1))           # use BOTH current x AND stashed h1
    x = norm(x + h2)                    # forced combine with postnorm
    return x, h2                        # return data AND hidden state
```

The pattern: each local syntactical scope computes an ABSTRACT VALUE before being FORCED to COMBINE it with whatever its input was. This is S-expression style composition — function application like you're living in Lisp.

## Prenorm and Postnorm

The norms appear at structural boundaries because that's what the composition demands. prenorm before computing, postnorm after combining. They are not arbitrary choices or hyperparameters to tune — they are where the S-expression structure puts them.

## Hidden State

Resnets return hidden states in parallel to transformed data. The hidden state `h` flows alongside `x`, gets used in intermediate computations (`f3(h1)`), and gets returned for downstream consumption. This is real. Two outputs, not one.

## What This Means for Tensor Code

Every operation should be expressible as:
- A function application to normalized input
- Followed by forced addition to the input stream
- With hidden state available for cross-scope communication

NOT:
- Per-pixel iteration with conditionals
- Unpacking tensors into Python control flow
- Scipy function calls with 23 keyword arguments
- For loops that "iterate over segments"

## Gates Are Selection, Not Partition

A gate in this context means truncation and selection. `gate(x)` produces a SPARSE subset of x — the values that pass. The values that don't pass are excluded entirely.

- High gate: `(gamma1 * x + bias1) > mean(x)` selects positive outliers
- Low gate: `(gamma2 * x + bias2) < mean(x)` selects negative outliers
- Middle: excluded, not processed, gated out

This is symmetric. The two gates do not partition the input into two exhaustive sets. The excluded middle is the point — most values are filtered out, creating sparsity.

The sparse outputs are OPERANDS for later functions. The gate creates the sparsity pattern. What happens to those sparse elements is a separate computation downstream.

## Tensor Operations Only

When you have prepared data — a Laplacian L, spectral coefficients phi, an activation field act — the algebraic operations on these tensors ARE the semantics you want.

- `L @ signal` IS the graph diffusion step
- `phi_t @ phi_s.T` IS the attention/affinity matrix
- `(I - alpha * L) @ x` repeated k times IS the heat kernel
- `act * sparse_mask` IS the gated selection

There is no need to unpack these into scipy function calls. The type signature is already satisfied by the tensor shapes. The matmul IS the operation with 23 implicit parameters — they're encoded in how L was constructed.

Adding library calls means:
- 30 lines of argument marshalling per call
- Algorithms that compute the wrong thing
- Runtime type coercion overhead
- String-typed tuple input patterns

Direct fmadd means:
- The loop variable IS the state
- The sparse matmul IS the action
- Numbers multiplying numbers that already mean what they need to mean

## Code Shape

A boring neural network looks like:
- 5-10 helper functions defining primitives (build laplacian, spectral basis, heat kernel, gradient)
- 8-12 lines in the main body composing these primitives
- Total maybe 100-150 LOC including helpers

NOT:
- 800 LOC of numpy iteration
- Per-pixel for loops
- scipy.ndimage calls wrapped in conditionals
- Random subsampling to manage complexity that shouldn't exist

The code is simple because the math is simple. Bloat comes from refusing to let tensor shapes do the work.
