# Subagent Handoff: Pathfinding with Blocking and Graph-Local Decisions

## Problem
Current spectral pathfinding in `/home/bigboi/itten/pathfinding/spectral.py`:
- Uses `heuristic_weight=0.7` → 70% Manhattan distance dominates
- Uses `spectral_weight=0.3` → spectral guidance is only 30%
- Result: straight line on unblocked grids
- Blocking support EXISTS but isn't exercised in demos

## The Point of Graph-Local Pathfinding
The spec says:
> "graph spectra (hence graph local) pathfinding decision logic which cannot make globally correct decisions for minimizing distance on all possible graphs, since it *doesn't have access* to interactions from anything but a graph-local direction at each polling / movement step"

This is INTENTIONAL. The pathfinder should:
- Only see local neighborhood (not full graph)
- Make locally-optimal decisions that may be globally suboptimal
- Demonstrate emergent behavior around obstacles
- Show theta affecting decision quality/style

## What Needs To Change

### 1. Create Meaningful Blocking Masks
Use natural images to create non-trivial blocking:

```python
def create_blocking_from_image(image, threshold_percentile=70):
    """
    High-intensity regions become blocked.
    Natural images have varied structure → interesting obstacles.
    """
    threshold = np.percentile(image, threshold_percentile)
    return image > threshold
```

### 2. Adjust Spectral/Heuristic Weights
When blocking is present, spectral should matter MORE:

```python
# With blocking: spectral guidance helps navigate around obstacles
spectral_weight = 0.6
heuristic_weight = 0.4
```

Or make it theta-dependent:
```python
spectral_weight = 0.3 + 0.4 * theta  # theta=0.9 → 0.66 spectral
heuristic_weight = 1.0 - spectral_weight
```

### 3. Demo with Natural Images
Use images from `/home/bigboi/itten/demo_output/inputs/`:
- `snek-heavy.png` - has varied structure
- `toof.png` - has edges/boundaries
- `mspaint-enso-*.png` - has curves

Create blocking from these, then pathfind.

### 4. Measure Path Non-Straightness
```python
def path_curvature(path):
    """Measure how much path deviates from straight line."""
    direct = euclidean(path[0], path[-1])
    actual = sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))
    return actual / direct  # 1.0 = straight, >1.0 = curved
```

### 5. Theta Interaction
Different theta should produce different path styles:
- theta=0.1: Fiedler-dominant → follows spectral "valleys"
- theta=0.9: Higher eigenvectors → more erratic/exploratory

## Test Configuration

1. Load natural image from `demo_output/inputs/`
2. Create blocking mask (threshold high-intensity regions)
3. Find path from corner to corner
4. Sweep theta = [0.1, 0.5, 0.9]
5. Measure:
   - Path curvature (should be >1.0 with blocking)
   - Path length difference across theta
   - Nodes visited difference across theta

## Success Criteria
- Path is NOT a straight line when blocking present
- Path curvature varies with theta
- Visual shows path navigating around obstacles

## Reference Files
- `/home/bigboi/itten/pathfinding/spectral.py` - SpectralPathfinder
- `/home/bigboi/itten/pathfinding/graph.py` - HeightfieldGraph
- `/home/bigboi/itten/demo_output/inputs/` - natural images

## Output
- Modified `/home/bigboi/itten/pathfinding/spectral.py` (if needed)
- Demo script `/home/bigboi/itten/demos/pathfinding_blocked.py`
- Results in `/home/bigboi/itten/hypercontexts/pathfinding-blocking-results.md`
- Visualization renders in `/home/bigboi/itten/demo_output/pathfinding_blocked/`
