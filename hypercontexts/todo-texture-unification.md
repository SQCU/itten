# TODO: Texture Module Unification

## Problem Statement

THREE separate texture implementations exist where ONE was specified:
- `texture_editor/` (oldest, bitmap→heightfield, no carrier/operand)
- `texture_synth/` (middle, carrier+operand, TUI-focused)
- `texture_synth_v2/` (newest, eigenvector decomposition, CLI-focused)

None achieve the 4-mode uniform interface (GUI/CLI/TUI/headless).

## Goals

1. **ONE module**: `texture/` (no version suffixes, no "editor" vs "synth" split)
2. **ONE API**: Single `synthesize()` function with clear parameters
3. **ONE kernel source**: All spectral ops from `spectral_ops_fast.py`
4. **FOUR interfaces**: GUI, CLI, TUI, headless (folder+config) - all calling same core

## Phase 1: Archaeology (Determine Age/Lineage)

- [ ] Date each implementation by git history or file timestamps
- [ ] Identify which features exist in which version
- [ ] Map the evolution: what was added, what was lost between versions

## Phase 2: Kernel Consolidation

### Duplicated Functions to Deduplicate

| Function | Current Locations | Target Location |
|----------|-------------------|-----------------|
| `normalize_to_01` | 4 places | `spectral_ops_fast.py` |
| `synthesize_texture` | 2 incompatible sigs | `texture/core.py` (unified) |
| `extract_contours` | 3 places | `texture/contours.py` |
| `compute_spectral_eigenvectors` | 2 places | `spectral_ops_fast.py` |
| `spectral_etch` | 4 strategies | `texture/synthesis.py` (pick best) |

### Spectral Etch Strategy Decision

Must pick ONE algorithm. Candidates:
1. texture_synth: fast_spectral_etch_field + gradient fallback
2. texture_synth_v2/spectral_etch: rotation-angle blending
3. texture_synth_v2/synthesize: Gaussian-weighted eigenvectors
4. texture_editor: spectral_smoothing_kernel multiscale

**Decision criteria**: Which produces best nodal lines for bump maps?

## Phase 3: Module Collapse

### New Structure
```
texture/
├── __init__.py           # Public API: synthesize, load_config, etc.
├── core.py               # synthesize() - THE function
├── carriers.py           # Carrier inputs (amongus, checkerboard, dragon, SVG)
├── operands.py           # Operand inputs (noise, gradient, checkerboard)
├── contours.py           # Nodal line / contour extraction
├── normals.py            # Height → normal map conversion
├── io.py                 # Load/save textures, configs
└── interfaces/
    ├── cli.py            # argparse CLI
    ├── tui.py            # Natural language TUI
    ├── gui.py            # tkinter/pygame GUI
    └── headless.py       # Folder + JSON config batch processing
```

### Migration Plan

1. Create `texture/` with stub files
2. Copy BEST implementation of each function (not oldest, not newest - BEST)
3. Add deprecation warnings to old modules pointing to `texture/`
4. Update all imports project-wide
5. Delete old modules after verification

## Phase 4: Interface Unification

All four interfaces must:
- Accept same input types (carrier: ndarray, operand: ndarray, config: dict)
- Produce same output types (height_field: ndarray, normal_map: ndarray, metadata: dict)
- Support the FULL parameter set (theta, gamma, num_eigenvectors, etc.)

### Headless Config Schema
```json
{
  "carrier": {"type": "file", "path": "carrier.png"},
  "operand": {"type": "noise", "seed": 42},
  "theta": 0.5,
  "gamma": 0.3,
  "output": {
    "height": "height.png",
    "normal": "normal.png",
    "metadata": "meta.json"
  }
}
```

## Success Criteria

- [ ] `from texture import synthesize` works
- [ ] `python -m texture --help` shows CLI
- [ ] `python -m texture.tui` launches TUI
- [ ] `python -m texture.gui` launches GUI
- [ ] `python -m texture.headless config.json` processes batch
- [ ] Zero duplicate kernel implementations
- [ ] Old modules deleted or contain only deprecation warnings

## Assigned Subagent

Subagent to:
1. Complete archaeology (Phase 1)
2. Decide on spectral etch strategy with justification
3. Write migration script or detailed file-by-file plan
4. Estimate line counts for new module

---

*Hypercontext created: 2026-01-30*
*Status: PLANNING COMPLETE*

---

## Subagent Analysis (Completed)

### 1. Archaeology - Chronological Order

All three created same day (2026-01-30), different times:

| Module | Started | Role |
|--------|---------|------|
| `texture_editor/` | 15:16 | **OLDEST** - bitmap→heightfield, heat diffusion |
| `texture_synth_v2/` | 15:49 | **MIDDLE** - eigenvector decomposition, nodal lines |
| `texture_synth/` | 18:14 | **NEWEST** - carrier/operand model, TUI wrapper |

### 2. Feature Matrix

| Feature | texture_editor | texture_synth | texture_synth_v2 |
|---------|----------------|---------------|------------------|
| Carrier/Operand Model | NO | YES | YES |
| Spectral Eigenvectors | NO (heat diffusion) | Imports v2 | YES (canonical) |
| Contour Extraction | NO | NO | YES |
| Normal Map | YES (imports v2) | YES (imports v2) | YES (canonical) |
| GUI | YES (tkinter) | NO | NO |
| CLI | YES | NO | YES |
| TUI | NO | YES | NO |
| Headless | Partial (JSON) | Batch | NO |
| OBJ Export | YES | NO | NO |
| SVG Input | NO | YES | NO |

### 3. DECISION: Canonical Spectral Etch

**CHOSEN: Candidate 3 - `texture_synth_v2/synthesize.py`**

Justification:
1. **Code Quality**: Best organized, clear docstrings, preset support
2. **Mathematical Correctness**: Proper Gaussian weighting of eigenvectors
3. **Output Quality**: Combines Fiedler segmentation + nodal mask + theta-weighted etch
4. **API Design**: Already has `synthesize_texture(carrier, operand, theta, gamma)` signature

**Optimization needed**: Refactor to compute eigenvectors ONCE (currently computed twice).

### 4. Migration File List

#### texture_editor/ (OLDEST)
| File | Decision | Destination |
|------|----------|-------------|
| `core.py` | DELETE | Replaced by texture_synth_v2/synthesize.py |
| `export.py` | KEEP | `texture/io.py` (OBJ export unique) |
| `gui.py` | KEEP | `texture/interfaces/gui.py` (only GUI) |
| `cli.py` | MERGE | `texture/interfaces/cli.py` |
| `render.py` | DELETE | Wrapper for v2 |

#### texture_synth/ (NEWEST - TUI)
| File | Decision | Destination |
|------|----------|-------------|
| `geometry/*` | KEEP | `texture/geometry/` |
| `inputs/*` | KEEP | `texture/carriers.py`, `texture/operands.py` |
| `render/*` | KEEP | `texture/render/` |
| `tui/*` | KEEP | `texture/interfaces/tui/` |
| `synthesis/core.py` | DELETE | Replaced by v2 |

#### texture_synth_v2/ (CANONICAL)
| File | Decision | Destination |
|------|----------|-------------|
| `synthesize.py` | KEEP | `texture/core.py` **CANONICAL** |
| `spectral_etch.py` | MERGE | `texture/core.py` |
| `patterns.py` | KEEP | `texture/patterns.py` |
| `normals.py` | KEEP | `texture/normals.py` |
| `render_egg.py` | KEEP | `texture/render/egg.py` |
| `cli.py` | MERGE | `texture/interfaces/cli.py` |

### 5. Unified API Design

```python
def synthesize(
    carrier: Union[np.ndarray, str, CarrierInput],
    operand: Union[np.ndarray, str, OperandInput, None] = None,
    *,
    theta: float = 0.5,           # 0=coarse, 1=fine
    gamma: float = 0.3,           # Etch strength
    num_eigenvectors: int = 8,
    edge_threshold: float = 0.1,
    output_size: Optional[int] = None,
    normal_strength: float = 2.0,
    mode: str = 'spectral',       # 'spectral', 'blend', 'simple'
    preset: Optional[str] = None, # 'coarse', 'balanced', 'fine'
) -> TextureResult:
    """THE single texture synthesis function."""

@dataclass
class TextureResult:
    height_field: np.ndarray      # (H, W) [0, 1]
    normal_map: np.ndarray        # (H, W, 3) [0, 1]
    metadata: Dict[str, Any]
    diagnostics: Optional[Dict] = None
```

### Target Structure

```
texture/
├── __init__.py           # synthesize(), TextureResult
├── core.py               # synthesize() implementation
├── patterns.py           # Pattern generators
├── carriers.py           # CarrierInput classes
├── operands.py           # OperandInput classes
├── contours.py           # Nodal line extraction
├── normals.py            # Height → normal conversion
├── io.py                 # Load/save, OBJ export
├── geometry/             # Mesh operations
├── render/               # PBR, lighting, UV
└── interfaces/
    ├── cli.py
    ├── tui/
    ├── gui.py
    └── headless.py
```
