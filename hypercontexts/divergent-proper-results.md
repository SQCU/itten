# Divergent Transforms Proper Test Results

## Test Configuration

This test fixes issues from the previous divergent transforms test:

| Issue | Previous | Fixed |
|-------|----------|-------|
| Theta values | Only 0.1 and 0.9 | All 5: [0.1, 0.3, 0.5, 0.7, 0.9] |
| Carriers | Noise fields | Natural images + checkerboard + scattered amongus |
| Operands | Noise fields | Natural images + checkerboard + scattered amongus |
| Amongus | Dense tiling | Scattered with random transforms |

## Carriers Used

- **amongus_scattered_3**: Scattered amongus (3 copies, random transforms)
- **checkerboard_16**: Checkerboard with block_size=16
- **snek_heavy**: Natural image (snek-heavy.png)

## Operands Used

- **amongus_scattered_4**: Scattered amongus (4 copies, seed=42, different from carrier)
- **checkerboard_8**: Checkerboard with block_size=8 (different from carrier)
- **toof**: Natural image (toof.png)

## Summary Statistics

- **Total tests**: 225
- **Full divergence** (both PSNR < 15dB): 196 (87.1%)
- **Partial divergence** (one PSNR < 15dB): 29 (12.9%)
- **Failed** (both PSNR >= 15dB): 0 (0.0%)

**Success criteria:**
- PSNR(carrier, result) < 15dB means result diverged from carrier
- PSNR(operand, result) < 15dB means result diverged from operand
- Full success requires BOTH conditions

## Results by Transform

| Transform | Success | Partial | Failed | Avg PSNR(carrier) | Avg PSNR(operand) |
|-----------|---------|---------|--------|-------------------|-------------------|
| commute_time_distance_field | 45 | 0 | 0 | 5.3 dB | 5.2 dB |
| eigenvector_phase_field | 45 | 0 | 0 | 4.7 dB | 4.5 dB |
| spectral_contour_sdf | 45 | 0 | 0 | 4.2 dB | 0.8 dB |
| spectral_subdivision_blend | 45 | 0 | 0 | 4.7 dB | 4.3 dB |
| spectral_warp | 16 | 29 | 0 | 5.7 dB | 15.2 dB |

## Results by Theta

| Theta | Success | Partial | Failed | Avg PSNR(carrier) | Avg PSNR(operand) |
|-------|---------|---------|--------|-------------------|-------------------|
| 0.1 | 39 | 6 | 0 | 4.8 dB | 6.3 dB |
| 0.3 | 39 | 6 | 0 | 5.0 dB | 5.9 dB |
| 0.5 | 39 | 6 | 0 | 4.9 dB | 6.1 dB |
| 0.7 | 40 | 5 | 0 | 4.9 dB | 5.9 dB |
| 0.9 | 39 | 6 | 0 | 5.0 dB | 6.0 dB |

## Full Results Table

| Carrier | Operand | Transform | Theta | PSNR(c) | PSNR(o) | Status |
|---------|---------|-----------|-------|---------|---------|--------|
| snek_heavy | toof | spectral_subdivision_blend | 0.3 | 0.4 dB | 0.0 dB | OK |
| snek_heavy | toof | spectral_subdivision_blend | 0.5 | 0.4 dB | 0.0 dB | OK |
| snek_heavy | toof | spectral_subdivision_blend | 0.7 | 0.4 dB | 0.0 dB | OK |
| snek_heavy | toof | spectral_subdivision_blend | 0.9 | 0.4 dB | 0.0 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_contour_sdf | 0.5 | 0.8 dB | 0.2 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_contour_sdf | 0.7 | 0.8 dB | 0.2 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_contour_sdf | 0.9 | 0.8 dB | 0.3 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_contour_sdf | 0.1 | 0.8 dB | 0.3 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_contour_sdf | 0.3 | 0.8 dB | 0.3 dB | OK |
| snek_heavy | toof | commute_time_distance_field | 0.9 | 1.0 dB | 0.6 dB | OK |
| snek_heavy | toof | commute_time_distance_field | 0.7 | 1.0 dB | 0.7 dB | OK |
| snek_heavy | toof | commute_time_distance_field | 0.5 | 1.1 dB | 0.7 dB | OK |
| snek_heavy | toof | commute_time_distance_field | 0.3 | 1.1 dB | 0.8 dB | OK |
| snek_heavy | toof | commute_time_distance_field | 0.1 | 1.2 dB | 0.9 dB | OK |
| snek_heavy | toof | eigenvector_phase_field | 0.5 | 1.8 dB | 1.4 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_contour_sdf | 0.1 | 3.1 dB | 0.1 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_contour_sdf | 0.9 | 3.1 dB | 0.1 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_contour_sdf | 0.3 | 3.1 dB | 0.1 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_contour_sdf | 0.5 | 3.1 dB | 0.1 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_contour_sdf | 0.7 | 3.1 dB | 0.1 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_contour_sdf | 0.1 | 3.2 dB | 0.2 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_contour_sdf | 0.3 | 3.2 dB | 0.2 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_contour_sdf | 0.9 | 3.2 dB | 0.2 dB | OK |
| snek_heavy | toof | eigenvector_phase_field | 0.1 | 1.9 dB | 1.5 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_contour_sdf | 0.5 | 3.2 dB | 0.2 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_contour_sdf | 0.7 | 3.2 dB | 0.2 dB | OK |
| snek_heavy | toof | eigenvector_phase_field | 0.7 | 1.9 dB | 1.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_contour_sdf | 0.1 | 3.1 dB | 0.4 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_contour_sdf | 0.3 | 3.1 dB | 0.4 dB | OK |
| snek_heavy | toof | eigenvector_phase_field | 0.3 | 2.0 dB | 1.6 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_contour_sdf | 0.5 | 3.1 dB | 0.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_contour_sdf | 0.7 | 3.1 dB | 0.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_contour_sdf | 0.9 | 3.1 dB | 0.5 dB | OK |
| snek_heavy | toof | eigenvector_phase_field | 0.9 | 2.2 dB | 1.9 dB | OK |
| checkerboard_16 | toof | spectral_contour_sdf | 0.1 | 3.7 dB | 0.7 dB | OK |
| checkerboard_16 | toof | spectral_contour_sdf | 0.9 | 3.7 dB | 0.8 dB | OK |
| snek_heavy | checkerboard_8 | spectral_contour_sdf | 0.3 | 4.1 dB | 0.4 dB | OK |
| checkerboard_16 | toof | spectral_contour_sdf | 0.3 | 3.7 dB | 0.8 dB | OK |
| snek_heavy | checkerboard_8 | spectral_contour_sdf | 0.9 | 4.1 dB | 0.4 dB | OK |
| snek_heavy | checkerboard_8 | spectral_contour_sdf | 0.5 | 4.2 dB | 0.4 dB | OK |
| snek_heavy | checkerboard_8 | spectral_contour_sdf | 0.7 | 4.2 dB | 0.4 dB | OK |
| checkerboard_16 | toof | spectral_contour_sdf | 0.5 | 3.8 dB | 0.8 dB | OK |
| checkerboard_16 | toof | spectral_contour_sdf | 0.7 | 3.8 dB | 0.8 dB | OK |
| snek_heavy | checkerboard_8 | spectral_contour_sdf | 0.1 | 4.2 dB | 0.5 dB | OK |
| snek_heavy | checkerboard_8 | commute_time_distance_field | 0.9 | 1.1 dB | 3.8 dB | OK |
| snek_heavy | toof | spectral_contour_sdf | 0.3 | 2.7 dB | 2.2 dB | OK |
| snek_heavy | toof | spectral_contour_sdf | 0.1 | 2.7 dB | 2.2 dB | OK |
| snek_heavy | checkerboard_8 | commute_time_distance_field | 0.7 | 1.1 dB | 3.8 dB | OK |
| snek_heavy | checkerboard_8 | commute_time_distance_field | 0.5 | 1.2 dB | 3.9 dB | OK |
| snek_heavy | toof | spectral_contour_sdf | 0.9 | 2.8 dB | 2.4 dB | OK |
| snek_heavy | checkerboard_8 | spectral_subdivision_blend | 0.1 | 1.4 dB | 3.9 dB | OK |
| snek_heavy | checkerboard_8 | commute_time_distance_field | 0.3 | 1.3 dB | 4.0 dB | OK |
| snek_heavy | toof | spectral_contour_sdf | 0.5 | 2.9 dB | 2.5 dB | OK |
| snek_heavy | toof | spectral_contour_sdf | 0.7 | 2.9 dB | 2.5 dB | OK |
| snek_heavy | checkerboard_8 | spectral_subdivision_blend | 0.3 | 1.5 dB | 4.0 dB | OK |
| snek_heavy | checkerboard_8 | commute_time_distance_field | 0.1 | 1.4 dB | 4.1 dB | OK |
| checkerboard_16 | toof | eigenvector_phase_field | 0.9 | 4.1 dB | 1.6 dB | OK |
| checkerboard_16 | toof | eigenvector_phase_field | 0.3 | 4.2 dB | 1.5 dB | OK |
| snek_heavy | checkerboard_8 | eigenvector_phase_field | 0.3 | 1.6 dB | 4.1 dB | OK |
| snek_heavy | checkerboard_8 | spectral_subdivision_blend | 0.5 | 1.8 dB | 4.1 dB | OK |
| checkerboard_16 | toof | eigenvector_phase_field | 0.5 | 4.1 dB | 1.9 dB | OK |
| snek_heavy | checkerboard_8 | eigenvector_phase_field | 0.1 | 1.9 dB | 4.2 dB | OK |
| checkerboard_16 | toof | eigenvector_phase_field | 0.7 | 4.3 dB | 1.9 dB | OK |
| checkerboard_16 | toof | eigenvector_phase_field | 0.1 | 4.1 dB | 2.1 dB | OK |
| snek_heavy | checkerboard_8 | eigenvector_phase_field | 0.5 | 2.0 dB | 4.2 dB | OK |
| snek_heavy | checkerboard_8 | eigenvector_phase_field | 0.9 | 2.0 dB | 4.3 dB | OK |
| snek_heavy | toof | spectral_subdivision_blend | 0.1 | 3.4 dB | 3.1 dB | OK |
| snek_heavy | checkerboard_8 | eigenvector_phase_field | 0.7 | 2.2 dB | 4.4 dB | OK |
| snek_heavy | checkerboard_8 | spectral_subdivision_blend | 0.7 | 2.2 dB | 4.4 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_subdivision_blend | 0.9 | 3.9 dB | 3.3 dB | OK |
| checkerboard_16 | toof | spectral_subdivision_blend | 0.5 | 3.8 dB | 3.4 dB | OK |
| checkerboard_16 | toof | spectral_subdivision_blend | 0.9 | 3.8 dB | 3.4 dB | OK |
| checkerboard_16 | toof | spectral_subdivision_blend | 0.3 | 3.8 dB | 3.4 dB | OK |
| checkerboard_16 | toof | spectral_subdivision_blend | 0.7 | 3.8 dB | 3.4 dB | OK |
| checkerboard_16 | toof | spectral_subdivision_blend | 0.1 | 4.0 dB | 3.4 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_subdivision_blend | 0.7 | 4.1 dB | 3.4 dB | OK |
| snek_heavy | checkerboard_8 | spectral_subdivision_blend | 0.9 | 3.2 dB | 4.3 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_subdivision_blend | 0.5 | 4.3 dB | 3.7 dB | OK |
| checkerboard_16 | checkerboard_8 | eigenvector_phase_field | 0.3 | 4.2 dB | 4.0 dB | OK |
| checkerboard_16 | checkerboard_8 | eigenvector_phase_field | 0.9 | 4.2 dB | 4.2 dB | OK |
| checkerboard_16 | checkerboard_8 | eigenvector_phase_field | 0.1 | 4.1 dB | 4.4 dB | OK |
| checkerboard_16 | checkerboard_8 | eigenvector_phase_field | 0.7 | 4.2 dB | 4.3 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_contour_sdf | 0.3 | 8.4 dB | 0.2 dB | OK |
| checkerboard_16 | checkerboard_8 | eigenvector_phase_field | 0.5 | 4.4 dB | 4.2 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_contour_sdf | 0.1 | 8.4 dB | 0.2 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_contour_sdf | 0.9 | 8.4 dB | 0.2 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_contour_sdf | 0.5 | 8.4 dB | 0.2 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_contour_sdf | 0.7 | 8.4 dB | 0.2 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_subdivision_blend | 0.3 | 4.7 dB | 4.1 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_subdivision_blend | 0.1 | 1.0 dB | 7.8 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_subdivision_blend | 0.3 | 1.0 dB | 7.9 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_subdivision_blend | 0.5 | 1.1 dB | 7.9 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_subdivision_blend | 0.7 | 1.2 dB | 7.9 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_subdivision_blend | 0.9 | 4.6 dB | 4.5 dB | OK |
| amongus_scattered_3 | toof | eigenvector_phase_field | 0.1 | 6.3 dB | 2.9 dB | OK |
| snek_heavy | amongus_scattered_4 | commute_time_distance_field | 0.9 | 0.8 dB | 8.4 dB | OK |
| snek_heavy | amongus_scattered_4 | commute_time_distance_field | 0.7 | 0.9 dB | 8.4 dB | OK |
| snek_heavy | amongus_scattered_4 | spectral_subdivision_blend | 0.9 | 1.3 dB | 8.0 dB | OK |
| checkerboard_16 | checkerboard_8 | spectral_subdivision_blend | 0.1 | 4.9 dB | 4.4 dB | OK |
| snek_heavy | amongus_scattered_4 | commute_time_distance_field | 0.5 | 0.9 dB | 8.5 dB | OK |
| amongus_scattered_3 | toof | eigenvector_phase_field | 0.7 | 7.0 dB | 2.4 dB | OK |
| amongus_scattered_3 | toof | eigenvector_phase_field | 0.5 | 7.0 dB | 2.5 dB | OK |
| snek_heavy | amongus_scattered_4 | commute_time_distance_field | 0.3 | 1.0 dB | 8.6 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_subdivision_blend | 0.7 | 4.8 dB | 4.9 dB | OK |
| amongus_scattered_3 | toof | eigenvector_phase_field | 0.3 | 8.5 dB | 1.2 dB | OK |
| snek_heavy | amongus_scattered_4 | commute_time_distance_field | 0.1 | 1.1 dB | 8.7 dB | OK |
| snek_heavy | amongus_scattered_4 | eigenvector_phase_field | 0.3 | 1.5 dB | 8.3 dB | OK |
| checkerboard_16 | toof | commute_time_distance_field | 0.9 | 5.5 dB | 4.3 dB | OK |
| snek_heavy | amongus_scattered_4 | eigenvector_phase_field | 0.5 | 1.7 dB | 8.2 dB | OK |
| snek_heavy | amongus_scattered_4 | eigenvector_phase_field | 0.7 | 2.2 dB | 7.7 dB | OK |
| amongus_scattered_3 | toof | spectral_subdivision_blend | 0.1 | 7.7 dB | 2.2 dB | OK |
| snek_heavy | amongus_scattered_4 | eigenvector_phase_field | 0.1 | 2.0 dB | 7.9 dB | OK |
| snek_heavy | amongus_scattered_4 | eigenvector_phase_field | 0.9 | 2.0 dB | 7.9 dB | OK |
| checkerboard_16 | toof | commute_time_distance_field | 0.7 | 5.5 dB | 4.4 dB | OK |
| amongus_scattered_3 | toof | spectral_subdivision_blend | 0.3 | 7.9 dB | 2.1 dB | OK |
| amongus_scattered_3 | toof | spectral_subdivision_blend | 0.5 | 8.0 dB | 2.1 dB | OK |
| amongus_scattered_3 | toof | spectral_subdivision_blend | 0.7 | 8.0 dB | 2.1 dB | OK |
| amongus_scattered_3 | toof | spectral_subdivision_blend | 0.9 | 8.1 dB | 2.0 dB | OK |
| checkerboard_16 | toof | commute_time_distance_field | 0.5 | 5.6 dB | 4.6 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_subdivision_blend | 0.5 | 4.9 dB | 5.4 dB | OK |
| amongus_scattered_3 | toof | commute_time_distance_field | 0.9 | 9.5 dB | 0.8 dB | OK |
| amongus_scattered_3 | toof | commute_time_distance_field | 0.7 | 9.5 dB | 0.9 dB | OK |
| checkerboard_16 | toof | commute_time_distance_field | 0.3 | 5.6 dB | 4.7 dB | OK |
| amongus_scattered_3 | toof | eigenvector_phase_field | 0.9 | 8.7 dB | 1.6 dB | OK |
| amongus_scattered_3 | toof | commute_time_distance_field | 0.5 | 9.5 dB | 0.9 dB | OK |
| amongus_scattered_3 | toof | commute_time_distance_field | 0.3 | 9.5 dB | 1.0 dB | OK |
| amongus_scattered_3 | toof | commute_time_distance_field | 0.1 | 9.5 dB | 1.1 dB | OK |
| checkerboard_16 | toof | commute_time_distance_field | 0.1 | 5.7 dB | 4.9 dB | OK |
| amongus_scattered_3 | checkerboard_8 | eigenvector_phase_field | 0.1 | 6.8 dB | 4.3 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_subdivision_blend | 0.3 | 4.9 dB | 6.2 dB | OK |
| amongus_scattered_3 | toof | spectral_contour_sdf | 0.5 | 8.9 dB | 2.4 dB | OK |
| amongus_scattered_3 | toof | spectral_contour_sdf | 0.7 | 8.9 dB | 2.4 dB | OK |
| checkerboard_16 | amongus_scattered_4 | eigenvector_phase_field | 0.1 | 4.0 dB | 7.3 dB | OK |
| amongus_scattered_3 | toof | spectral_contour_sdf | 0.9 | 8.9 dB | 2.4 dB | OK |
| amongus_scattered_3 | toof | spectral_contour_sdf | 0.1 | 8.7 dB | 2.7 dB | OK |
| amongus_scattered_3 | toof | spectral_contour_sdf | 0.3 | 8.6 dB | 2.8 dB | OK |
| checkerboard_16 | amongus_scattered_4 | spectral_subdivision_blend | 0.1 | 4.4 dB | 7.1 dB | OK |
| checkerboard_16 | checkerboard_8 | commute_time_distance_field | 0.9 | 5.3 dB | 6.3 dB | OK |
| checkerboard_16 | amongus_scattered_4 | eigenvector_phase_field | 0.5 | 4.8 dB | 6.9 dB | OK |
| checkerboard_16 | amongus_scattered_4 | eigenvector_phase_field | 0.3 | 4.3 dB | 7.4 dB | OK |
| checkerboard_16 | checkerboard_8 | commute_time_distance_field | 0.7 | 5.3 dB | 6.4 dB | OK |
| checkerboard_16 | amongus_scattered_4 | eigenvector_phase_field | 0.7 | 4.1 dB | 7.8 dB | OK |
| checkerboard_16 | checkerboard_8 | commute_time_distance_field | 0.5 | 5.4 dB | 6.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | eigenvector_phase_field | 0.5 | 7.5 dB | 4.4 dB | OK |
| checkerboard_16 | amongus_scattered_4 | eigenvector_phase_field | 0.9 | 4.1 dB | 7.8 dB | OK |
| checkerboard_16 | checkerboard_8 | commute_time_distance_field | 0.3 | 5.4 dB | 6.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_subdivision_blend | 0.9 | 8.6 dB | 3.4 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_subdivision_blend | 0.7 | 8.6 dB | 3.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_subdivision_blend | 0.5 | 8.6 dB | 3.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_subdivision_blend | 0.3 | 8.6 dB | 3.5 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_subdivision_blend | 0.1 | 8.5 dB | 3.6 dB | OK |
| checkerboard_16 | checkerboard_8 | commute_time_distance_field | 0.1 | 5.5 dB | 6.7 dB | OK |
| amongus_scattered_3 | checkerboard_8 | eigenvector_phase_field | 0.3 | 8.5 dB | 4.0 dB | OK |
| amongus_scattered_3 | checkerboard_8 | eigenvector_phase_field | 0.7 | 8.7 dB | 4.2 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | eigenvector_phase_field | 0.1 | 7.1 dB | 5.8 dB | OK |
| amongus_scattered_3 | checkerboard_8 | eigenvector_phase_field | 0.9 | 8.7 dB | 4.2 dB | OK |
| amongus_scattered_3 | toof | spectral_warp | 0.5 | 5.3 dB | 7.8 dB | OK |
| amongus_scattered_3 | toof | spectral_warp | 0.7 | 5.3 dB | 7.9 dB | OK |
| amongus_scattered_3 | toof | spectral_warp | 0.3 | 5.3 dB | 7.8 dB | OK |
| amongus_scattered_3 | toof | spectral_warp | 0.9 | 5.2 dB | 8.0 dB | OK |
| amongus_scattered_3 | checkerboard_8 | commute_time_distance_field | 0.9 | 9.5 dB | 4.1 dB | OK |
| amongus_scattered_3 | checkerboard_8 | commute_time_distance_field | 0.7 | 9.5 dB | 4.1 dB | OK |
| checkerboard_16 | amongus_scattered_4 | commute_time_distance_field | 0.9 | 4.8 dB | 8.9 dB | OK |
| amongus_scattered_3 | checkerboard_8 | commute_time_distance_field | 0.5 | 9.5 dB | 4.2 dB | OK |
| checkerboard_16 | amongus_scattered_4 | commute_time_distance_field | 0.7 | 4.9 dB | 8.9 dB | OK |
| amongus_scattered_3 | checkerboard_8 | commute_time_distance_field | 0.3 | 9.5 dB | 4.3 dB | OK |
| checkerboard_16 | amongus_scattered_4 | commute_time_distance_field | 0.5 | 4.9 dB | 8.9 dB | OK |
| amongus_scattered_3 | toof | spectral_warp | 0.1 | 4.3 dB | 9.5 dB | OK |
| checkerboard_16 | amongus_scattered_4 | commute_time_distance_field | 0.3 | 5.0 dB | 9.0 dB | OK |
| amongus_scattered_3 | checkerboard_8 | commute_time_distance_field | 0.1 | 9.5 dB | 4.4 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | eigenvector_phase_field | 0.7 | 7.3 dB | 6.7 dB | OK |
| checkerboard_16 | amongus_scattered_4 | commute_time_distance_field | 0.1 | 5.0 dB | 9.0 dB | OK |
| checkerboard_16 | toof | spectral_warp | 0.7 | 6.9 dB | 8.3 dB | OK |
| checkerboard_16 | toof | spectral_warp | 0.3 | 7.0 dB | 8.6 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | eigenvector_phase_field | 0.5 | 7.9 dB | 7.7 dB | OK |
| checkerboard_16 | toof | spectral_warp | 0.9 | 6.8 dB | 8.8 dB | OK |
| checkerboard_16 | toof | spectral_warp | 0.5 | 6.6 dB | 9.0 dB | OK |
| checkerboard_16 | toof | spectral_warp | 0.1 | 5.6 dB | 10.4 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_subdivision_blend | 0.3 | 8.5 dB | 7.8 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_subdivision_blend | 0.1 | 8.6 dB | 7.9 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_subdivision_blend | 0.5 | 8.8 dB | 7.8 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_subdivision_blend | 0.7 | 8.8 dB | 7.8 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | spectral_subdivision_blend | 0.9 | 8.8 dB | 7.8 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | eigenvector_phase_field | 0.3 | 8.9 dB | 7.8 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | eigenvector_phase_field | 0.9 | 8.8 dB | 8.1 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | commute_time_distance_field | 0.9 | 9.2 dB | 8.3 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | commute_time_distance_field | 0.7 | 9.2 dB | 8.3 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | commute_time_distance_field | 0.5 | 9.2 dB | 8.3 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | commute_time_distance_field | 0.3 | 9.2 dB | 8.4 dB | OK |
| amongus_scattered_3 | amongus_scattered_4 | commute_time_distance_field | 0.1 | 9.3 dB | 8.4 dB | OK |
| snek_heavy | toof | spectral_warp | 0.7 | 9.7 dB | 9.2 dB | OK |
| snek_heavy | toof | spectral_warp | 0.3 | 9.7 dB | 9.3 dB | OK |
| snek_heavy | toof | spectral_warp | 0.5 | 10.0 dB | 9.6 dB | OK |
| amongus_scattered_3 | checkerboard_8 | spectral_warp | 0.7 | 5.5 dB | 14.3 dB | OK |
| snek_heavy | toof | spectral_warp | 0.1 | 10.2 dB | 9.8 dB | OK |
| snek_heavy | toof | spectral_warp | 0.9 | 10.2 dB | 9.8 dB | OK |
| snek_heavy | checkerboard_8 | spectral_warp | 0.5 | 4.1 dB | 16.0 dB | PARTIAL |
| snek_heavy | checkerboard_8 | spectral_warp | 0.3 | 4.1 dB | 16.1 dB | PARTIAL |
| checkerboard_16 | checkerboard_8 | spectral_warp | 0.3 | 4.5 dB | 15.8 dB | PARTIAL |
| amongus_scattered_3 | checkerboard_8 | spectral_warp | 0.3 | 5.4 dB | 15.1 dB | PARTIAL |
| checkerboard_16 | checkerboard_8 | spectral_warp | 0.9 | 4.5 dB | 16.1 dB | PARTIAL |
| snek_heavy | checkerboard_8 | spectral_warp | 0.9 | 4.1 dB | 16.5 dB | PARTIAL |
| checkerboard_16 | checkerboard_8 | spectral_warp | 0.5 | 4.4 dB | 16.3 dB | PARTIAL |
| snek_heavy | amongus_scattered_4 | spectral_warp | 0.3 | 2.0 dB | 18.8 dB | PARTIAL |
| snek_heavy | amongus_scattered_4 | spectral_warp | 0.1 | 2.1 dB | 19.1 dB | PARTIAL |
| amongus_scattered_3 | checkerboard_8 | spectral_warp | 0.1 | 5.0 dB | 16.2 dB | PARTIAL |
| snek_heavy | checkerboard_8 | spectral_warp | 0.7 | 4.1 dB | 17.1 dB | PARTIAL |
| snek_heavy | amongus_scattered_4 | spectral_warp | 0.7 | 2.1 dB | 19.2 dB | PARTIAL |
| checkerboard_16 | checkerboard_8 | spectral_warp | 0.1 | 4.1 dB | 17.4 dB | PARTIAL |
| snek_heavy | checkerboard_8 | spectral_warp | 0.1 | 4.1 dB | 17.5 dB | PARTIAL |
| snek_heavy | amongus_scattered_4 | spectral_warp | 0.9 | 2.1 dB | 19.6 dB | PARTIAL |
| snek_heavy | amongus_scattered_4 | spectral_warp | 0.5 | 2.1 dB | 19.6 dB | PARTIAL |
| amongus_scattered_3 | checkerboard_8 | spectral_warp | 0.9 | 4.9 dB | 16.9 dB | PARTIAL |
| checkerboard_16 | checkerboard_8 | spectral_warp | 0.7 | 4.4 dB | 17.6 dB | PARTIAL |
| amongus_scattered_3 | checkerboard_8 | spectral_warp | 0.5 | 4.6 dB | 18.5 dB | PARTIAL |
| checkerboard_16 | amongus_scattered_4 | spectral_warp | 0.3 | 4.0 dB | 19.3 dB | PARTIAL |
| checkerboard_16 | amongus_scattered_4 | spectral_warp | 0.5 | 3.9 dB | 20.5 dB | PARTIAL |
| checkerboard_16 | amongus_scattered_4 | spectral_warp | 0.7 | 4.1 dB | 20.5 dB | PARTIAL |
| checkerboard_16 | amongus_scattered_4 | spectral_warp | 0.9 | 4.0 dB | 20.8 dB | PARTIAL |
| checkerboard_16 | amongus_scattered_4 | spectral_warp | 0.1 | 3.8 dB | 21.8 dB | PARTIAL |
| amongus_scattered_3 | amongus_scattered_4 | spectral_warp | 0.3 | 10.5 dB | 19.7 dB | PARTIAL |
| amongus_scattered_3 | amongus_scattered_4 | spectral_warp | 0.9 | 10.5 dB | 20.0 dB | PARTIAL |
| amongus_scattered_3 | amongus_scattered_4 | spectral_warp | 0.7 | 10.4 dB | 20.3 dB | PARTIAL |
| amongus_scattered_3 | amongus_scattered_4 | spectral_warp | 0.1 | 10.1 dB | 21.5 dB | PARTIAL |
| amongus_scattered_3 | amongus_scattered_4 | spectral_warp | 0.5 | 10.2 dB | 23.3 dB | PARTIAL |

## Key Insights

### Why Checkerboard is Better than Noise

Checkerboards have:
- Sharp edges that spectral methods can identify
- Regular periodicity that shows up in eigenvalues
- Clear visual features for PSNR comparison

Noise has:
- No coherent structure
- Random features that don't survive spectral operations
- Misleading PSNR (noise vs noise is always low PSNR)

### Why Scattered Amongus is Better than Dense Tiling

Scattered (random transforms):
- Each copy has unique position, rotation, scale, shear
- Recognizable features without dense periodicity
- Tests how transforms handle isolated features

Dense tiling:
- Creates strong periodic signal that dominates spectrum
- Hides subtle divergent effects
- Not realistic test case

### Transform Behavior Analysis

**commute_time_distance_field** (100% success)
- Uses eigenvalue-weighted sum of squared eigenvector differences
- Creates organic distance fields respecting graph topology

**eigenvector_phase_field** (100% success)
- Uses arctan2 of eigenvector pairs to create spiral patterns
- Creates topological defects (vortices) at eigenvector zeros

**spectral_contour_sdf** (100% success)
- Computes distance to eigenvector iso-contours
- Creates smooth gradients not in discrete inputs

**spectral_subdivision_blend** (100% success)
- Recursively subdivides by Fiedler vector sign
- Creates stained-glass patterns with operand statistics

**spectral_warp** (36% success)
- Uses eigenvector gradients for displacement field
- Coordinate remapping is inherently non-linear


## Output Files

Results saved to: `demo_output/divergent_proper/`

- `inputs/`: Carrier and operand textures
- `*_texture.png`: Raw transform outputs
- `*_egg.png`: Bump-mapped 3D renders (for successful/partial cases)
