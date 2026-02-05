# Hypercontext: Unified Texture Synthesis TUI Architecture

## Vision
A single modular system where:
1. **Kernels** are imported from `spectral_ops_fast.py` (never reimplemented)
2. **Data inputs** (carriers, operands) are composable modules
3. **3D geometry** is composable (fuse icosahedrons, chop, squash)
4. **Rendering** wraps textures onto arbitrary convex surfaces
5. **TUI** accepts natural language commands, produces rendered output at EVERY atomic edit
6. **No logic duplication** between TUI and CLI

## Module Architecture

```
itten/
├── spectral_ops_fast.py          # CANONICAL kernels (never duplicated)
├── texture_synth/                 # UNIFIED module (replaces texture_synth_v2/)
│   ├── __init__.py               # Clean exports
│   ├── inputs/                   # Composable data inputs
│   │   ├── __init__.py
│   │   ├── carriers.py           # Carrier generators (amongus, checkerboard, svg_to_carrier)
│   │   ├── operands.py           # Operand generators
│   │   └── svg_parser.py         # SVG → binary mask (dragon curves, etc.)
│   ├── synthesis/                # Core synthesis (imports kernels)
│   │   ├── __init__.py
│   │   ├── spectral_etch.py      # Eigenvector nodal lines
│   │   ├── synthesize.py         # synthesize_texture() main entry
│   │   └── normals.py            # Height → normal map
│   ├── geometry/                 # 3D geometry primitives
│   │   ├── __init__.py
│   │   ├── primitives.py         # Icosahedron, sphere, egg, etc.
│   │   ├── operations.py         # Fuse, chop, squash, boolean ops
│   │   └── mesh.py               # Mesh data structure
│   ├── render/                   # PBR rendering
│   │   ├── __init__.py
│   │   ├── pbr.py                # Unified PBR (anisotropic, raking, iridescent)
│   │   ├── uv_mapping.py         # UV unwrap for arbitrary convex surfaces
│   │   └── trace.py              # Render trace (saves each step)
│   └── tui/                      # Terminal User Interface
│       ├── __init__.py
│       ├── parser.py             # Natural language → commands
│       ├── state.py              # Session state (current geometry, texture, etc.)
│       ├── executor.py           # Execute commands, trigger renders
│       └── history.py            # Edit history with rendered outputs
└── texture_tui.py                # Main TUI entry point
```

## TUI Command Examples

### Geometry Commands
```
> give me two icosahedrons fused together, amonguswrapped
[RENDER] outputs/step_001_fused_icosahedrons.png

> three icosahedrons
[RENDER] outputs/step_002_three_icosahedrons.png

> chop one of the icosahedrons in half
[RENDER] outputs/step_003_chopped.png

> squash all hedrons vertically by 30%
[RENDER] outputs/step_004_squashed.png
```

### Texture Commands
```
> rotate the amongus 45 degrees
[RENDER] outputs/step_005_rotated_carrier.png

> stretch the carrier horizontally 2x
[RENDER] outputs/step_006_stretched.png

> make the carrier a dragon curve from dragon.svg
[RENDER] outputs/step_007_dragon_carrier.png

> set theta to 0.7
[RENDER] outputs/step_008_theta_07.png
```

## Key Design Principles

### 1. Every Edit Renders
No staging. Every atomic command produces a PNG in the trace directory.
```python
class TUIExecutor:
    def execute(self, command: str) -> Path:
        """Execute command, return path to rendered output."""
        parsed = self.parser.parse(command)
        self.state.apply(parsed)
        output_path = self.render_current_state()
        self.history.append(command, output_path)
        return output_path
```

### 2. Kernels Are Never Reimplemented
All spectral operations import from `spectral_ops_fast.py`:
```python
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_fiedler_gpu,
    heat_diffusion_sparse,
    chebyshev_filter,
    iterative_spectral_transform,
)
```

### 3. Geometry Is Composable
```python
from texture_synth.geometry import Icosahedron, fuse, chop, squash

mesh = Icosahedron()
mesh = fuse(mesh, Icosahedron().translate(2, 0, 0))
mesh = chop(mesh, plane_normal=(0, 1, 0), plane_origin=(0, 0.5, 0))
mesh = squash(mesh, axis='y', factor=0.7)
```

### 4. Textures Are Composable
```python
from texture_synth.inputs import AmongusCarrier, SVGCarrier, CheckerboardOperand

carrier = AmongusCarrier(size=128).rotate(45).stretch(2, 1)
# OR
carrier = SVGCarrier.from_file("dragon.svg", size=128)

operand = CheckerboardOperand(size=128, cells=8)

height = synthesize_texture(carrier.render(), operand.render(), theta=0.5)
```

### 5. TUI and CLI Share Core
```python
# CLI usage (config file)
python texture_tui.py --config scene.yaml

# TUI usage (interactive)
python texture_tui.py --interactive

# Programmatic usage (Claude calling)
from texture_synth.tui import TUISession
session = TUISession()
session.execute("two icosahedrons fused, amonguswrapped")
session.execute("squash 30% vertically")
```

## State Machine

```
TUIState:
  geometry: Mesh           # Current 3D geometry
  carrier: CarrierInput    # Current carrier pattern
  operand: OperandInput    # Current operand pattern
  theta: float             # Spectral rotation angle
  gamma: float             # Etch strength
  render_mode: str         # 'anisotropic', 'raking', 'iridescent'
  output_dir: Path         # Trace output directory
  step_count: int          # For sequential filenames
```

## PBR Rendering with Dichromatic Rule

The dichromatic reflection model separates:
1. **Interface reflection** (specular) - depends on viewing angle, light angle
2. **Body reflection** (diffuse) - depends on material color

For normal map differentiation:
- Use HIGH anisotropy so specular follows bump direction
- Use CONTRASTING colors for diffuse vs specular
- Example: warm body (orange/copper), cool specular (blue/white)

```python
def render_dichromatic(height_field, normal_map, ...):
    # Body reflection: warm, follows base color
    body_color = np.array([200, 120, 80])  # copper

    # Interface reflection: cool, follows normal perturbation
    interface_color = np.array([180, 200, 255])  # blue-white

    # Fresnel determines mix
    fresnel = schlick_fresnel(ndotv, F0=0.04)

    color = (1 - fresnel) * body_color * diffuse + fresnel * interface_color * specular
```

## Files to Create

1. `texture_synth/` - New unified module structure
2. `texture_synth/inputs/carriers.py` - Carrier generators
3. `texture_synth/inputs/svg_parser.py` - SVG to mask
4. `texture_synth/geometry/primitives.py` - Icosahedron, etc.
5. `texture_synth/geometry/operations.py` - Fuse, chop, squash
6. `texture_synth/render/pbr.py` - Unified dichromatic PBR
7. `texture_synth/tui/parser.py` - Natural language parser
8. `texture_synth/tui/executor.py` - Command executor with render
9. `texture_tui.py` - Main entry point

## Success Criteria

1. `python texture_tui.py --interactive` launches TUI
2. Every command produces a rendered PNG
3. Geometry operations (fuse, chop, squash) work
4. Texture operations (rotate, stretch, SVG import) work
5. No kernel code duplicated - all imports from spectral_ops_fast.py
6. Claude can call the TUI programmatically
