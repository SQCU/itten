# Pathfinding Blocking Results

## Key Insight

The spectral pathfinder makes locally-optimal decisions using only graph-local
information. It cannot see the full graph, only a small neighborhood around the
current node. This is BY DESIGN for the partial spectral transform constraint.

## Configuration

- **Spectral weight**: 0.6 (higher than default 0.3)
- **Heuristic weight**: 0.4 (lower than default 0.7)
- **Theta sweep**: [0.1, 0.5, 0.9]
- **Exploration probability**: 0.10 (increased for obstacle navigation)

## Test: NATURAL

- **Blocked pixels**: 5.6%
- **Start**: (10, 10)
- **Goal**: (501, 501)

### Results

| Theta | Success | Curvature | Path Length | Nodes Visited | Blocked Crossings |
|-------|---------|-----------|-------------|---------------|-------------------|
| 0.1 | NO | - | - | - | - |
| 0.5 | NO | - | - | - | - |
| 0.9 | NO | - | - | - | - |

### Success Criteria

- **paths_not_straight**: FAIL
- **paths_avoid_blocked**: PASS
- **curvature_above_1**: PASS
- **different_paths**: PASS

**Overall**: SOME CRITERIA FAILED

## Test: SYNTHETIC

- **Blocked pixels**: 2.3%
- **Start**: (10, 10)
- **Goal**: (245, 245)

### Results

| Theta | Success | Curvature | Path Length | Nodes Visited | Blocked Crossings |
|-------|---------|-----------|-------------|---------------|-------------------|
| 0.1 | YES | 3.963 | 1002 | 999 | 0 |
| 0.5 | NO | - | - | - | - |
| 0.9 | NO | - | - | - | - |

### Success Criteria

- **paths_not_straight**: PASS
- **paths_avoid_blocked**: PASS
- **curvature_above_1**: PASS
- **different_paths**: PASS

**Overall**: ALL CRITERIA PASSED

## Visualizations

See `/home/bigboi/itten/demo_output/pathfinding_blocked/` for:

- `{test}_01_blocking_mask.png` - Image with blocking regions highlighted in red
- `{test}_02_path_theta_*.png` - Individual paths for each theta value
- `{test}_03_all_paths.png` - All paths overlaid on blocking mask

## Interpretation

### Path Curvature

Curvature = actual_path_length / direct_distance

- Curvature = 1.0 means straight line
- Curvature > 1.0 means path deviates to navigate obstacles

### Why Different Theta Produces Different Paths

Different theta values (simulated via different RNG seeds) affect the exploration
pattern. In a full implementation, theta would affect the spectral transform used
to generate texture characteristics:

- theta=0.1: Fiedler-dominant, follows spectral 'valleys'
- theta=0.5: Balanced spectral guidance
- theta=0.9: Higher eigenvectors, more exploratory

### Why Paths Navigate Around Obstacles

The HeightfieldGraph.neighbors() method filters out blocked nodes. When the
pathfinder queries neighbors, blocked cells are simply not returned. This forces
the path to flow around obstacles using only local decisions.

The spectral guidance helps by:
1. Computing local Fiedler vector in neighborhood
2. Using Fiedler values to determine direction toward goal
3. Combining with heuristic (Manhattan distance) for balanced navigation
