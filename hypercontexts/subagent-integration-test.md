# Hypercontext: Integration Test & Demo

## Mission
Verify the unified texture synthesis TUI works end-to-end. Run a complete demo session that exercises all modules.

## Test Sequence

### Phase 1: Import Verification
```python
# All modules should import cleanly
from texture_synth.geometry import Icosahedron, Sphere, Egg, fuse, chop, squash
from texture_synth.inputs import AmongusCarrier, CheckerboardCarrier, DragonCurveCarrier
from texture_synth.render import render_mesh_dichromatic, RenderTrace
from texture_synth.synthesis import synthesize_texture
from texture_synth.tui import TUISession, CommandParser
```

### Phase 2: Programmatic API Test
```python
from texture_synth.tui import TUISession

session = TUISession(output_dir='demo_outputs')

# Geometry commands
session.execute("two icosahedrons fused, amonguswrapped")
session.execute("three icosahedrons")
session.execute("chop one in half")
session.execute("squash all hedrons vertically by 30%")

# Texture commands
session.execute("rotate the amongus 45 degrees")
session.execute("stretch the carrier horizontally 2x")
session.execute("make the carrier a dragon curve")

# Parameter commands
session.execute("set theta to 0.3")
session.execute("set theta to 0.7")
session.execute("set gamma to 0.5")
```

### Phase 3: Verify Outputs
Check that:
1. Each command produced a PNG in demo_outputs/
2. Files are named sequentially: step_0000_*.png, step_0001_*.png, etc.
3. manifest.json exists and lists all renders
4. Images are valid (non-zero size, loadable)

### Phase 4: CLI Test
```bash
# Single command
python texture_tui.py -c "icosahedron amonguswrapped" -o cli_test/

# Multiple commands
python texture_tui.py -c "two icosahedrons fused" -c "squash 20%" -o cli_test/
```

### Phase 5: Module Integration Test
Test each module independently then together:

```python
# Geometry alone
from texture_synth.geometry import Icosahedron, fuse, squash
mesh = fuse(Icosahedron(), Icosahedron().translate(1.5, 0, 0))
mesh = squash(mesh, 'y', 0.7)
print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Inputs alone
from texture_synth.inputs import DragonCurveCarrier
carrier = DragonCurveCarrier(128, iterations=10)
img = carrier.render()
print(f"Dragon curve: {img.shape}, range [{img.min():.2f}, {img.max():.2f}]")

# Synthesis with spectral_ops_fast kernels
from texture_synth.synthesis import synthesize_texture
from texture_synth.inputs import AmongusCarrier, CheckerboardOperand
carrier = AmongusCarrier(64).render()
operand = CheckerboardOperand(64).render()
height = synthesize_texture(carrier, operand, theta=0.5)
print(f"Height field: {height.shape}, range [{height.min():.2f}, {height.max():.2f}]")

# Render
from texture_synth.render import render_mesh_dichromatic, height_to_normals
from texture_synth.geometry import Icosahedron
mesh = Icosahedron()
normal_map = height_to_normals(height)
img = render_mesh_dichromatic(mesh, height, normal_map, output_size=256)
print(f"Rendered image: {img.shape}")
```

## Expected Outputs

```
demo_outputs/
├── step_0000_two_icosahedrons_fused_amonguswrapped.png
├── step_0001_three_icosahedrons.png
├── step_0002_chop_one_in_half.png
├── step_0003_squash_all_hedrons_vertically_by_30.png
├── step_0004_rotate_the_amongus_45_degrees.png
├── step_0005_stretch_the_carrier_horizontally_2x.png
├── step_0006_make_the_carrier_a_dragon_curve.png
├── step_0007_set_theta_to_03.png
├── step_0008_set_theta_to_07.png
├── step_0009_set_gamma_to_05.png
└── manifest.json
```

## Debugging

If imports fail:
1. Check __init__.py files export the right names
2. Check relative imports use correct paths
3. Check spectral_ops_fast.py is importable from texture_synth/

If rendering fails:
1. Check mesh has valid vertices/faces/uvs
2. Check height field is 2D numpy array
3. Check normal map is (H, W, 3)

If TUI parsing fails:
1. Check CommandParser handles the command format
2. Check executor has handlers for all command types

## Deliverables
1. Run full test sequence
2. Report any import/runtime errors
3. Fix any issues found
4. Confirm all renders produced
5. Show sample output images exist
