# Subagent Handoff: Lattice Extrusion Spectral Decisions

## Problem
Current `ExpansionGatedExtruder` in `/home/bigboi/itten/lattice/extrude.py`:
- Uses expansion to decide WHETHER to extrude (threshold comparison)
- But makes NO spectral decisions about:
  - WHAT lattice type (always "square", never triangle/hex/parallelogram)
  - WHICH spatial dimension to project into
  - HOW to weight edges/nodes for visualization
  - NO theta parameter for sweeping spectral blend

## What Needs To Change

### 1. Spectral Lattice Type Selection
Based on local spectral properties, choose lattice geometry:

```python
def select_lattice_type(expansion, fiedler_gradient_magnitude, theta):
    """
    Spectrally-determined lattice type selection.

    - High expansion + low gradient → "hex" (open region)
    - Low expansion + high gradient → "triangle" (bottleneck)
    - Medium values → "square" (neutral)
    - theta rotates between geometry preferences
    """
    # theta controls spectral emphasis
    spectral_score = (1 - theta) * expansion + theta * fiedler_gradient_magnitude

    if spectral_score > high_threshold:
        return "hex"
    elif spectral_score < low_threshold:
        return "triangle"
    else:
        return "square"
```

### 2. Projection Dimension Selection
Graph dimensions don't need to match 3D surface:

```python
def select_projection_dimension(fiedler_value, eigenvalue_ratio, theta):
    """
    Choose which 3D axis the node extrudes along.

    - Positive fiedler → extrude in +Z
    - Negative fiedler → extrude in -Z or XY plane
    - theta controls projection blend
    """
    pass
```

### 3. Edge/Node Value Assignment
Add scalar values for visualization:

```python
@dataclass
class ExtrudedNode:
    # ... existing fields ...
    node_value: float = 0.0      # For coloring
    edge_weights: Dict = None     # Weights to neighbors
```

### 4. Theta Parameter
Add theta throughout the extrusion pipeline:

```python
class ExpansionGatedExtruder:
    def __init__(self, ..., theta: float = 0.5):
        self.theta = theta

    def step(self):
        # Use theta in lattice type selection
        # Use theta in projection dimension
        # Use theta in edge weighting
```

## Test Configuration

1. Create TerritoryGraph with varied structure (islands + bridges)
2. Run extrusion at theta = [0.1, 0.5, 0.9]
3. Measure:
   - Distribution of lattice types across theta
   - Variance in node_value across theta
   - Visual difference in renders

## Success Criteria
- At theta=0.1: Mostly one lattice type (expansion-dominant)
- At theta=0.9: Different distribution (gradient-dominant)
- Measurable PSNR difference between theta renders

## Reference Files
- `/home/bigboi/itten/lattice/extrude.py` - current implementation
- `/home/bigboi/itten/spectral_ops_fast.py` - spectral functions
- `/home/bigboi/itten/hypercontexts/transform-psnr-results.md` - PSNR methodology

## Output
- Modified `/home/bigboi/itten/lattice/extrude.py`
- Test script `/home/bigboi/itten/tests/test_lattice_theta.py`
- Results in `/home/bigboi/itten/hypercontexts/lattice-spectral-decisions-results.md`
