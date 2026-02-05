# Subagent Handoff: 2D Covariance Validation

## Problem
Surface matrix shows low theta variance (0.003-0.008) using basic synthesize().
New transforms show higher theta variance (2.6-4.6 dB) but weren't tested against surfaces.

## Mission
Test new transforms (Phase Field, Nodal Lines) against 3 surfaces to get proper 2D covariance.

## Steps
1. Load transforms from `/home/bigboi/itten/texture/transforms.py`
2. Load surfaces from `/home/bigboi/itten/texture/surfaces.py`
3. For each (transform, surface, theta) tuple:
   - Apply transform at theta
   - Compute PSNR(surface, transformed)
4. Build 2 matrices (one per transform), each 3x5 (surfaces × theta=[0.1,0.3,0.5,0.7,0.9])
5. Compute covariance metrics:
   - row_var: variance across theta for each surface
   - col_var: variance across surfaces for each theta
   - We want BOTH to be non-trivial

## Success Criteria
- row_var (theta sensitivity) > 0.5 for at least 2 surfaces
- col_var (surface sensitivity) > 0.5 for at least 3 theta values
- Total covariance showing non-separable interaction

## Output
Write to `/home/bigboi/itten/hypercontexts/covariance-validation-results.md`
Include both matrices and computed variances
