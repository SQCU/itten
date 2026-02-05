# Shader Operations Specification

## Context

Two images exist: a target and a sample. Both are graphs embedded in pixel grids — edges where there's ink, vertices at junctions and endpoints, the grid topology connecting neighboring pixels. Spectral methods compare these graphs the way attention compares token sequences: by projecting into a shared basis where structural similarity becomes inner product.

## Activation and Gating

This comparison produces an activation field over the target. The activation measures how much each target location resonates with sample structure.

Gating separates this field into two sparse operand sets:

- **High gate**: locations where `(gamma * activation + bias) > activation` — positive outliers, places of unusually strong resonance
- **Low gate**: locations where `(gamma * activation + bias) < activation` — negative outliers, symmetric to high gate

Everything in between is excluded. This is what "gate" means: truncation, selection, filtering. The excluded middle is the point. The gates are sparse masks, mostly zeros. Multiplying by them keeps all subsequent operations tensor-to-tensor.

These sparse operand sets are INPUTS to downstream functions. They are not immediate pixel-copy destinations. They are arguments for later scatter, gather, reduce operations.

## Thicken (High Gate Operands)

The high-gated operands feed the thicken function. These operands happen to land on contiguous segments of drawn lines, because high spectral resonance concentrates on coherent graph structure.

Thicken copies these segments and displaces the copies orthogonally to the segment's own span — perpendicular to the direction the line travels. The displacement distance is at most the segment's width.

The effect is what happens when you trace over a pen stroke a second time, slightly offset: the line gets bolder, thicker, more emphatic. This works on curves and self-intersections because orthogonality is computed locally from the segment's own tangent, not globally.

## Drop Shadow (Low Gate Operands)

The low-gated operands feed the drop shadow function. These operands mark transitional or ambiguous structure — places that weakly echo the sample without strongly matching it.

Drop shadow soft-matches these locations back to the sample to find corresponding segments, then copies entire matched segments and transforms them. The transformation has three parts:

1. **Rotation**: align the copied segment normal to the eliciting location's local orientation
2. **Translation**: displace by more than half the segment's length, in the normal direction
3. **Color rotation**: shift in defined color space for visual distinction

The normal direction must point away from image density — outward into void, not inward toward existing ink. This constraint extends to avoiding collision with disconnected segments that might be enclosed by the eliciting structure.

The direction and distance emerge from the density field: heat kernel diffusion from boundaries produces a potential whose negative gradient points toward emptiness. Constraints are satisfied by vector arithmetic on this field, not by conditionals or search.

## Preservation Constraints

Neither operation should obliterate the image.

- **Copied-structure preserving**: the source segment retains its integrity after being copied
- **Orthogonal**: displacement directions are perpendicular to source tangents
- **Pasted-structure preserving**: destinations with existing graph-like qualities are avoided, not overwritten

Sparsity ensures most pixels pass through unchanged. The residual structure means modifications accumulate rather than replace.

## Repeated Application

Repeated application causes halos and auras of geometry to accrete around the original forms. The central image acquires bold strokes at selected contours while retaining its legibility. The displaced shadow fragments themselves become candidates for future thickening and shadowing, picking up colors and textures from whatever reference image is active. The system feeds back on itself, exaggerating and elaborating without erasing.

## Implementation Scope

The entire mechanism is:
- Graph comparison (one matmul)
- Sparse gating (Hadamard with mostly-zeros masks)
- Local normal computation (gradient of activation or distance field)
- Density-aware displacement (heat kernel gradient)
- Scattered writes (index_put or equivalent)

Each step is a tensor operation. The specification asks for approximately ten lines of arithmetic in the main body, with helper functions for spectral and geometric primitives defined above. This is not an 800 LOC problem.
