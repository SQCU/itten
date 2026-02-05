# Hypercontext: Inputs Module

## Mission
Create `texture_synth/inputs/` module with composable carrier and operand generators.

## Files to Create

### `texture_synth/inputs/__init__.py`
```python
from .carriers import (
    AmongusCarrier, CheckerboardCarrier, SVGCarrier,
    NoiseCarrier, GradientCarrier
)
from .operands import (
    CheckerboardOperand, NoiseOperand, SolidOperand
)
from .svg_parser import svg_to_mask
```

### `texture_synth/inputs/base.py`
Base classes with transform chain:
```python
class CarrierInput:
    """Base class for carrier patterns."""
    def __init__(self, size: int):
        self.size = size
        self._transforms = []

    def rotate(self, degrees: float) -> 'CarrierInput':
        self._transforms.append(('rotate', degrees))
        return self

    def stretch(self, sx: float, sy: float) -> 'CarrierInput':
        self._transforms.append(('stretch', sx, sy))
        return self

    def translate(self, dx: int, dy: int) -> 'CarrierInput':
        self._transforms.append(('translate', dx, dy))
        return self

    def render(self) -> np.ndarray:
        """Generate base pattern, apply transforms, return (H,W) array."""
        img = self._generate_base()
        for transform in self._transforms:
            img = self._apply_transform(img, transform)
        return img

    def _generate_base(self) -> np.ndarray:
        raise NotImplementedError

    def _apply_transform(self, img, transform):
        from scipy.ndimage import rotate as nd_rotate, zoom, shift
        if transform[0] == 'rotate':
            return nd_rotate(img, transform[1], reshape=False, mode='wrap')
        elif transform[0] == 'stretch':
            zoomed = zoom(img, (transform[2], transform[1]), mode='wrap')
            # Crop/pad to original size
            return self._crop_or_pad(zoomed, self.size)
        elif transform[0] == 'translate':
            return shift(img, (transform[2], transform[1]), mode='wrap')
```

### `texture_synth/inputs/carriers.py`
```python
class AmongusCarrier(CarrierInput):
    """Among Us crewmate silhouette with variations."""
    def __init__(self, size: int = 64, variations: int = 4):
        super().__init__(size)
        self.variations = variations

    def _generate_base(self) -> np.ndarray:
        # Use existing generate_varied_amongus logic
        ...

class CheckerboardCarrier(CarrierInput):
    """Checkerboard pattern."""
    def __init__(self, size: int = 64, cells: int = 8):
        super().__init__(size)
        self.cells = cells

    def _generate_base(self) -> np.ndarray:
        cell_size = self.size // self.cells
        pattern = np.zeros((self.size, self.size))
        for i in range(self.cells):
            for j in range(self.cells):
                if (i + j) % 2 == 0:
                    pattern[i*cell_size:(i+1)*cell_size,
                            j*cell_size:(j+1)*cell_size] = 1.0
        return pattern

class SVGCarrier(CarrierInput):
    """Carrier from SVG file (dragon curve, etc.)."""
    def __init__(self, svg_path: str, size: int = 64):
        super().__init__(size)
        self.svg_path = svg_path

    def _generate_base(self) -> np.ndarray:
        from .svg_parser import svg_to_mask
        return svg_to_mask(self.svg_path, self.size)

    @classmethod
    def from_file(cls, path: str, size: int = 64):
        return cls(path, size)

class NoiseCarrier(CarrierInput):
    """Perlin/simplex noise carrier."""
    def __init__(self, size: int = 64, scale: float = 4.0, seed: int = 42):
        super().__init__(size)
        self.scale = scale
        self.seed = seed

    def _generate_base(self) -> np.ndarray:
        np.random.seed(self.seed)
        # Simple multi-octave noise
        ...
```

### `texture_synth/inputs/svg_parser.py`
Parse SVG paths to binary masks:
```python
def svg_to_mask(svg_path: str, size: int) -> np.ndarray:
    """Convert SVG file to binary mask."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get viewBox for scaling
    viewbox = root.get('viewBox', f'0 0 {size} {size}')
    vb = [float(x) for x in viewbox.split()]

    mask = np.zeros((size, size), dtype=np.float32)

    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path.get('d', '')
        points = parse_svg_path(d)
        # Rasterize path to mask
        rasterize_path_to_mask(mask, points, vb, size)

    return mask

def parse_svg_path(d: str) -> list:
    """Parse SVG path 'd' attribute to list of points."""
    # Handle M, L, C, Z commands
    ...

def rasterize_path_to_mask(mask, points, viewbox, size):
    """Draw path onto mask."""
    # Scale points from viewbox to image coords
    # Use Bresenham or similar for lines
    ...
```

### Dragon Curve Generator
If no SVG available, generate procedurally:
```python
def dragon_curve(iterations: int = 10) -> list:
    """Generate dragon curve as list of points."""
    # L-system: F -> F+G, G -> F-G
    sequence = 'F'
    for _ in range(iterations):
        sequence = sequence.replace('F', 'F+G').replace('G', 'f-G')
        sequence = sequence.replace('f', 'F')

    # Convert to points
    x, y = 0, 0
    angle = 0
    points = [(x, y)]
    for char in sequence:
        if char == 'F' or char == 'G':
            x += np.cos(np.radians(angle))
            y += np.sin(np.radians(angle))
            points.append((x, y))
        elif char == '+':
            angle += 90
        elif char == '-':
            angle -= 90

    return points

class DragonCurveCarrier(CarrierInput):
    """Procedurally generated dragon curve."""
    def __init__(self, size: int = 64, iterations: int = 10, line_width: float = 2.0):
        super().__init__(size)
        self.iterations = iterations
        self.line_width = line_width

    def _generate_base(self) -> np.ndarray:
        points = dragon_curve(self.iterations)
        return rasterize_curve(points, self.size, self.line_width)
```

## Test Cases

```python
# Basic carriers
carrier = AmongusCarrier(128).rotate(45).stretch(2, 1)
img = carrier.render()

# SVG carrier
carrier = SVGCarrier.from_file("dragon.svg", 128)
img = carrier.render()

# Procedural dragon
carrier = DragonCurveCarrier(128, iterations=12)
img = carrier.render()

# Composable operands
operand = CheckerboardOperand(128, cells=16)
```

## Deliverables
1. `texture_synth/inputs/base.py`
2. `texture_synth/inputs/carriers.py`
3. `texture_synth/inputs/operands.py`
4. `texture_synth/inputs/svg_parser.py`
5. `texture_synth/inputs/__init__.py`
6. Test showing rotation, stretching, SVG import working
