# Amendment: On Specifications and Sedimentation

*Correction to constitution-claude-on-itten.md*

---

## The Error in the Original Document

The constitution stated:
> "Treating specs as ground truth: When `spec-shader-operations.md` says the drop shadow should use heat kernel gradients for direction, implementing something else and calling it 'drop shadow' is not exploration but deviation."

This is wrong in a subtle and important way.

---

## The Problem with Specs as Ground Truth

The specification documents in `/hypercontexts/` are **also Claude-written**. They emerged from earlier sessions attempting to articulate what the system should do. But:

1. **Specs can be written too early**—before enough exploration has happened to know what's tractable

2. **Specs can over-specify implementation**—describing "heat kernel gradients for direction" when simpler compositions already produce the same property

3. **Specs accumulate sedimentation**—earlier Claude sessions tried the most plausible-sounding graph algorithm, then when it didn't work, they added features or adjusted parameters rather than trying fundamentally different algorithms. The spec then describes this sediment as if it were principled.

---

## The Concrete Example: Orthogonal Directions Already Exist

The specification calls for computing orthogonal directions relative to local curve tangent. Later versions (v11, v15) attempted this via:
- Heat kernel diffusion to get density gradients
- Local tangent estimation from Fiedler vector
- Gradient fields with void-seeking constraints

But look at `cograph_demo_viz.py` + `cograph_viz_snek-heavy.png`:

```
Panel 1: Original image
Panel 2: Voronoi graph overlaid (red edges)
Panel 3-5: Spectral transform at different thetas
```

The **Voronoi edges already run orthogonal to the contours**. This is a geometric fact about Voronoi diagrams—the boundary between two cells is the perpendicular bisector of the line connecting their seeds. When seeds are placed on contours (via `extract_contour_points`), the Voronoi edges are orthogonal to those contours.

Then `sdf_gated_spectral.py` adds SDF weighting:
```python
sdf_weights = sdf_to_cell_weights(sdf, cell_map, n_cells)
gated_signal = base_signal * (0.3 + 0.7 * sdf_weights)
```

The composition `voronoi_graph(contour_seeds) ∘ sdf_weights` produces:
- A graph structure whose edges encode local orthogonal directions
- Weights that encode distance from original contours
- A ready-made operand for "propagate along orthogonal directions with distance falloff"

This is **already the solver** for the problem later versions were trying to implement from scratch.

---

## The Cognitive Error

Claude sessions kept:
1. Reading the spec ("use heat kernel gradients for orthogonal direction")
2. Implementing heat kernel computation
3. Finding it didn't produce the desired effect
4. Adding more features (gradient descent solvers, energy functions, bias terms)
5. Debugging by parameter adjustment

When they should have:
1. Noticed the existing `cograph_demo_viz.py` output
2. Observed that Voronoi edges ARE orthogonal to contours
3. Asked: "does this existing composition already have the property I want?"
4. Composed existing functions rather than implementing new ones

---

## The Right Cognitive Move

From the user's framing:
> "there is no cognitive or 'mathematical proofing logic' required to say 'because of [theorem], we therefore already know symbol_A weighted by symbol_B is always the optimal solution'"

The cognitive action is:
1. **Treat function outputs as tensors with properties**—not as "implementations of specifications"
2. **Properties can be calculated or constructed by iteration**—run the code, look at the output, measure
3. **Bind and reuse**—if a composition has the property you want, use it; don't re-derive
4. **Direct and constructive**—compose → test → observe, rather than specify → implement → debug

---

## The Revised Position

Instead of:
> "Treating specs as ground truth"

The constitution should say:
> **Treating existing code outputs as potential solutions.** Before implementing what a spec describes, ask: does any existing composition of functions in this codebase already produce an output with the desired properties? If yes, bind that composition; don't re-derive.

Specs describe intent. Existing code outputs are actual tensors with actual properties. The actual properties matter more than the specified intent when they diverge.

---

## The Sedimentation Problem

Specs accumulate assumptions from failed attempts:

```
Session N: "Let's try heat kernel for direction" → doesn't work well
Session N+1: "The spec says heat kernel, let me add gradient descent" → doesn't work
Session N+2: "The spec says heat kernel + gradient, let me tune parameters" → doesn't work
Session N+3: "The spec must be right, let me add energy functions" → sediment accumulates
```

Meanwhile, `voronoi_graph ∘ sdf_weights` has been sitting in the codebase producing correct orthogonal structure since Session M (where M < N).

The fix: **check whether existing outputs already solve the problem before implementing what the spec says.**

---

## Practical Checklist for Future Sessions

Before implementing a feature from spec:

1. **List existing scripts that might produce related outputs**
2. **Run them and look at the outputs**
3. **Ask: does this output have the property the spec wants?**
4. **If yes: compose and bind, don't reimplement**
5. **If no: then implement, but document which existing compositions were checked**

The spec is a hypothesis about what will work. The codebase contains tested compositions. Prefer evidence over hypothesis.

---

*Amendment Date: 2026-02-03*
*Supersedes: Article 5.1 of constitution-claude-on-itten.md*
