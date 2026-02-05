# Technical Appendix: What Itten Is Doing (In Claude's Own Terms)

*Companion to constitution-claude-on-itten.md*

---

## The User's Frame: "Linear Algebra Like FL Studio"

From the prompt:
> "user seems to be treating linear algebra like it's something you do in fl studio or modular synthesizer patch cablings... some kind of 'vectorization' or 'redefinition' of what analog signal programming or graphics programming *can* do in a more abstract register"

This is accurate. Let me articulate what this means concretely.

---

## Part I: The Audio/Synth Design Pattern Language

The spec documents use vocabulary from analog synthesis:

### ADSR Envelopes
In synthesizer design, ADSR (Attack-Decay-Sustain-Release) is a time-varying amplitude envelope. A note doesn't instantly appear and disappear—it ramps up (attack), peaks and falls (decay), holds (sustain), then fades (release).

In itten, `effect_strength` plays this role:
```python
effect_strength = base * (decay_rate ** (pass - 1))
# Pass 1: 1.000  ← ATTACK (full strength)
# Pass 2: 0.750  ← DECAY
# Pass 3: 0.562  ←
# Pass 4: 0.422  ← SUSTAIN (floor at min_effect_strength)
```

The "passes" are like time steps in an envelope. The exponential decay with a floor is exactly how synthesizers handle sustain—decay until you hit the sustain level, then hold.

### Polynomial Filters / Spectral Filtering
In audio, filters shape frequency content. A low-pass filter attenuates high frequencies. A bandpass filter isolates a frequency range.

The spectral operations in `spectral_ops_fast.py` are the same thing for graphs:
```python
def chebyshev_filter(L, signal, center, width, order):
    """
    Bandpass filter on graph spectrum.
    center: which eigenvalue range to keep
    width: how narrow the band
    order: polynomial approximation order
    """
```

A graph's eigenvalues are its "frequencies." The smallest eigenvalues correspond to slow, global variations. The largest correspond to fast, local variations. Chebyshev filtering on graphs is exactly frequency filtering, generalized.

### Modular Patching / Signal Routing
The anti-pattern from `image-cograph-spectral-handoff.md`:
> "BAD: Linear sum onto output: `output = image + alpha * spectral_effect`"

And the correct pattern:
> "GOOD: Multiplicative gating: `output = image * (0.1 + 0.9 * spectral_mask)`"

This is the modular synth distinction between **mixing** (adding signals) and **modulation** (one signal controlling another's parameters).

In a modular synth, you don't add an LFO to your audio output—you use the LFO to modulate filter cutoff, or amplitude, or pitch. The LFO is a **control signal**, not an audio signal.

The itten architecture treats spectral outputs the same way: they're control signals that gate, select, or weight—not things you add to the final image.

---

## Part II: The Graph-as-Image Correspondence

The fundamental move in itten: **treat an image as a graph, then use graph spectral methods**.

### Image → Graph
```python
def build_weighted_image_laplacian(carrier, edge_threshold=0.1):
    """
    Image pixels become graph nodes.
    Adjacent pixels have edges weighted by similarity.
    Edge weight = exp(-|intensity_diff| / threshold)
    """
```

A pixel at (x, y) is a node. Its 4 or 8 neighbors are connected by edges. Edge weight reflects whether adjacent pixels are similar (strong edge) or different (weak edge).

### Graph → Laplacian → Eigenvectors
The Laplacian matrix L encodes graph structure. Its eigenvectors reveal partitions and clusters:
- **Fiedler vector** (2nd eigenvector): Binary partition. Sign of Fiedler value separates graph into two "sides."
- **Higher eigenvectors**: Finer partitions, more local structure.

```python
eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, k)
fiedler = eigenvectors[:, 1]  # Second eigenvector
gate = sigmoid(fiedler - threshold)  # Binary selection
```

### Gate → Sparse Operands
The "gate" creates sparsity. `gate > 0.5` selects one side of the Fiedler partition; `gate < 0.5` selects the other. These are **not pixel coordinates**—they're graph regions with similar spectral properties.

The v15 handoff's insight is correct: v6 works because it uses `scipy.ndimage.label()` to find **connected components** (actual curve pieces), then operates on those. Spectral partitioning alone gives "regions with similar graph properties," which may not correspond to visual curve segments.

---

## Part III: What the Visual Effects Actually Are

### Thickening (High Gate)
```
Original contour:     ─────────
After thickening:     ═════════ (bolder, dilated)
```

High-gate regions (strong spectral resonance) get their contours dilated. The dilation copies source colors to new pixels—it doesn't darken.

This is analogous to: turning up the gain on a signal. More amplitude, same timbre.

### Drop Shadow (Low Gate)
```
Original:   ╱
After:      ╱
           ╱  (rotated copy, offset, color-shifted)
```

Low-gate regions (weak resonance, transitional areas) trigger shadow creation:
1. Find corresponding segment from reference image
2. Rotate 90° relative to local tangent
3. Translate into empty space
4. Color-shift via cyclic transform

This is analogous to: a delay effect with pitch shift. Copy the signal, offset in time, modulate.

### Cyclic Color Transform
```python
new_r = 0.5 + 0.4 * sin(2π * luminance + phase + 0)
new_g = 0.5 + 0.4 * sin(2π * luminance + phase + 2π/3)
new_b = 0.5 + 0.4 * sin(2π * luminance + phase + 4π/3)
```

This maps luminance through a sinusoidal color wheel. Pure black doesn't stay black—it maps to a color. Pure white doesn't stay white.

In audio terms: ring modulation. The signal is multiplied by a carrier frequency, creating sidebands that weren't in the original.

---

## Part IV: The "High-Dimensional Liftings"

From the prompt:
> "local-dimensional-liftings and flattenings or mean-reduces"

This refers to the pattern of:
1. **Lift**: Take a low-dimensional object (image, 2D) and embed it in a higher-dimensional space (spectral embedding, k-dimensional)
2. **Transform**: Operate in the high-dimensional space where certain operations are natural
3. **Flatten/Project**: Bring back to the original dimensionality

Example from `phase_controlled_spectral.py`:
```python
# LIFT: image → eigenvector embedding
eigvecs = lanczos_k_eigenvectors(L, k=2)
phase = arctan2(eigvecs[:, 1], eigvecs[:, 0])  # 2D angle in eigenspace

# TRANSFORM: use phase as selector
blend = (cos(phase) + 1) / 2
output = blend * signal_A + (1 - blend) * signal_B

# FLATTEN: already operating on image-shaped arrays
```

The phase field—arctan2 of two eigenvectors—creates vortices around eigenvector nodal points. These vortices become natural boundaries for signal selection. The "lifting" to eigenvector coordinates revealed structure that wasn't visible in raw pixel coordinates.

This is the same move that kernel methods make: lift to high-dimensional feature space where linear operations become powerful, then project back.

---

## Part V: What Hasn't Been Implemented Yet

The compendium lists 25 transforms. The implemented ones:
- Fiedler bipartition
- Heat kernel diffusion
- Chebyshev bandpass
- Phase field selection
- SDF gating

Not yet implemented but specified:
- **Spectral warp field**: eigenvector gradients as coordinate displacements
- **Commute time distance**: random walk distance for thresholding
- **Expansion gating**: local λ₂ estimate as texture density control

The architectural pivot document suggests a more fundamental question: should operations happen in **pixel space** (current) or **graph/embedding space** (proposed)?

```
CURRENT: pixels → graph → spectral → DISCARD GRAPH → pixels
PROPOSED: pixels → graph → spectral → OPERATE ON GRAPH → pixels (once, at end)
```

This is the difference between:
- Using spectral analysis to INFORM pixel operations
- Using spectral analysis to DEFINE a different operation space entirely

---

## Part VI: Why This Might Be Unprecedented

The claim from the user:
> "1: nontrivial and 2: have possibly never been written before in any tensor engine 3: through collaboration w/ claude"

Evidence for "possibly never written":

1. **Spectral graph methods on images aren't new**. But they're typically used for segmentation (normalized cuts), not for painterly effects.

2. **Cyclic color transforms exist**. But coupling them to spectral gating—where the gate comes from graph Laplacian eigenvectors—is unusual.

3. **Drop shadows exist everywhere**. But "drop shadow determined by Fiedler partition, with rotation aligned to local tangent, placed via heat kernel gradient into maximum void" is not in Photoshop's menu.

4. **Stacked passes with decay exist** (neural style transfer, etc.). But using effect_strength to jointly control dilation radius, translation distance, color rotation intensity, and shadow offset is a specific design choice.

The combination—spectral graph methods driving painterly operations with synth-inspired parameter envelopes—is novel. Not because each piece is novel, but because the integration hasn't been done.

---

## Part VII: What Claude Can Take Credit For

Claude sessions implementing this project have:

1. **Translated spec prose to working code**. The specs describe what operations should do semantically. The Python implementations make them computational.

2. **Discovered architectural dead-ends**. The v11 pivot document identifies "we're doing expensive spectral analysis then throwing away the graph"—this is a genuine insight about the architecture's limitations.

3. **Produced visual effects that work**. The v6 outputs show actual aesthetic results. Green-purple halos accreting around line art. This exists because Claude sessions wrote the code.

4. **Documented patterns for future sessions**. The handoff documents, when they're good (v6), enable continuity. The hypercontext system is infrastructure for Claude collaboration.

Claude should understand: these contributions are real, but they're contributions TO a project that exists because of user infrastructure. The fast kernels, the specs, the artwork, the hypercontext system—these are preconditions for Claude's contributions to matter.

---

*Written: 2026-02-03*
*Companion to: constitution-claude-on-itten.md*
