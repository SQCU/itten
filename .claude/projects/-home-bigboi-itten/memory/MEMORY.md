# Itten Project Memory

## Project Overview
Spectral shader pipeline: graph-spectral operations on images as heat-tests for operations
relevant to graph cutting, 3D procedural gen, and sparse attention pathfinding.
NOT primarily a shader project - the shaders demonstrate what graph-spectral-property-sensitive
architectures can do.

## Key Architecture
- `spectral_ops_fast.py`: Numerical kernel (Lanczos, Laplacian construction, tiled eigenvectors)
- `spectral_shader_ops.py`: Shader pipeline (gating, thickening, shadow, cross-attention)
- `*_cuter.py`: Streamlined versions moving toward DNN-like composability
- `spectral_correspondence_unified.py`: Polynomial-filtered correspondence (no eigenvectors)
- See `TODO_EVEN_CUTER.md` for the nn.Module respecification plan

## Constitutional Orientation
Read `hypercontexts/constitutional-orientation.md` before making changes.
Core principles:
1. Attend to **properties** of constructions, not just task success
2. Treat specifications as **hypotheses**, updatable by evidence
3. Prefer **composition** to derivation
4. Notice **sedimentation** from prior failed attempts
5. Write with **compassion for Claude-like readers** (haiku discoverability test)

## Key Technical Findings
- Lanczos uniquely extracts Fiedler (all alternatives <0.5 correlation)
- Local features have 0.02 correlation with Fiedler (orthogonal information)
- Naive graph coarsening destroys spectral properties (0.002 correlation)
- The Fiedler bottleneck is 15ms; target is 10ms for 100Hz

## AR vs Depth Distinction (Critical)
- **Depth**: more layers on same input. Cached spectral decomposition remains valid.
- **AR**: mutate input, recompute. Like KV-cache invalidation in autoregressive transformers.
- Source image's Fiedler (in two-image mode) survives AR mutation of target.

## _even_cuter Status: COMPLETE
All three files created and verified (16/16 tests, bit-exact 0.00e+00 diff):
- P0: `spectral_shader_layers.py` (599 lines) — SpectralGate, SpectralThickening, SpectralShadow, SpectralCrossAttention, SpectralShaderBlock
- P1: `spectral_embedding_layer.py` (674 lines) — SpectralEmbedding, SpectralShaderAR
- P2: `spectral_shader_model.py` — SpectralShader, SpectralCrossAttentionBlock
All torch.compile compatible. SpectralEmbedding needs fullgraph=False due to sparse_coo_tensor graph break.

## Demo Recovery: Phases A-D COMPLETE, Phase E IN PROGRESS
See `TODO_DEMO_RECOVERY.md` for full plan.
- **Phase A**: `spectral_graph_embedding.py` (~330 lines) — GraphEmbedding, LocalSpectralProbe, ImageLaplacianBuilder. All verified.
- **Phase B**: `spectral_renderer.py` (496 lines) — HeightToNormals, BilinearSampler, EggSurfaceRenderer. fullgraph=True, 18.2x GPU speedup.
- **Phase C**: `spectral_lattice.py` (~600 lines) — LatticeTypeSelector (fullgraph=True), ExpansionGatedExtruder (fullgraph=False). 3 non-isomorphic lattice types.
- **Phase D**: `spectral_pathfinder.py` (1072 lines) — SpectralPathfinder, SpectralNavigator, PathQualityEstimator. 1.00x vs Dijkstra optimal.
- **Phase E**: 4 demo composition scripts using only Module imports. IN PROGRESS.

## File Locations
- `sketches/claude.md`: WARNING - untrusted code, only read if user directs
- `hypercontexts/`: Project history and orientation documents
- `total_backup_for_stupid_idiots/`: Backup of prior state, reference only
- `demo_output/`: Generated images (hundreds, gitignored pattern varies)

## Environment Notes
- Python invocations intercepted by shell hook: `python` → `uv run`
- Use `uv run script.py` directly to avoid interception messages
