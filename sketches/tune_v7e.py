"""
Numerical tuning for v7e fuzz annihilator parameters.
Grid search over temp, threshold, strength to find sweet spot.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import math

DEVICE = torch.device('cpu')

def laplacian(gray, edge_thresh=0.1):
    H, W = gray.shape; n = H * W; dev = gray.device
    dh, dv = (gray[:, 1:] - gray[:, :-1]).abs(), (gray[1:, :] - gray[:-1, :]).abs()
    wh, wv = (-dh / edge_thresh).exp(), (-dv / edge_thresh).exp()
    yh, xh = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W-1, device=dev), indexing='ij')
    yv, xv = torch.meshgrid(torch.arange(H-1, device=dev), torch.arange(W, device=dev), indexing='ij')
    il, ir = (yh * W + xh).flatten(), (yh * W + xh).flatten() + 1
    it, ib = (yv * W + xv).flatten(), (yv * W + xv).flatten() + W
    rows = torch.cat([il, ir, it, ib]); cols = torch.cat([ir, il, ib, it])
    vals = torch.cat([-wh.flatten(), -wh.flatten(), -wv.flatten(), -wv.flatten()])
    deg = torch.zeros(n, device=dev)
    deg.scatter_add_(0, il, wh.flatten()); deg.scatter_add_(0, ir, wh.flatten())
    deg.scatter_add_(0, it, wv.flatten()); deg.scatter_add_(0, ib, wv.flatten())
    diag = torch.arange(n, device=dev)
    return torch.sparse_coo_tensor(torch.stack([torch.cat([rows, diag]), torch.cat([cols, diag])]),
                                   torch.cat([vals, deg]), (n, n)).coalesce()

def lanczos(L, k=8):
    n, dev = L.shape[0], L.device; k = min(k, n - 1)
    torch.manual_seed(42)
    v = torch.randn(n, device=dev); v = (v - v.mean()) / v.norm()
    V, alphas, betas = torch.zeros(n, k+1, device=dev), torch.zeros(k, device=dev), torch.zeros(k, device=dev)
    V[:, 0] = v
    for i in range(k):
        w = torch.sparse.mm(L, V[:, i:i+1]).squeeze(1)
        alphas[i] = (V[:, i] * w).sum()
        w = w - alphas[i] * V[:, i] - (betas[i-1] * V[:, i-1] if i > 0 else 0)
        w = w - V[:, :i+1] @ (V[:, :i+1].T @ w) - w.mean()
        beta = w.norm()
        if beta < 1e-10: break
        betas[i], V[:, i+1] = beta, w / beta
    T = torch.diag(alphas)
    if k > 1: off = torch.arange(k-1, device=dev); T[off, off+1] = betas[:-1]; T[off+1, off] = betas[:-1]
    evals, evecs = torch.linalg.eigh(T)
    idx = torch.where(evals > 1e-6)[0][:k]
    return V[:, :k] @ evecs[:, idx] if len(idx) > 0 else torch.zeros(n, k, device=dev)

def heat(L, sig, t=2.0, steps=20):
    alpha, x = min(t / steps, 0.1), sig.clone()
    for _ in range(steps): x = x - alpha * torch.sparse.mm(L, x.unsqueeze(1)).squeeze(1)
    return x

def annihilate_fuzz(x, L, temp=5.0, use_global=True):
    local_fuzz = torch.sparse.mm(L, x.unsqueeze(1)).squeeze(1).abs()
    gate = torch.sigmoid(-local_fuzz * temp)
    x_gated = x * gate
    if use_global:
        cut = x_gated @ torch.sparse.mm(L, x_gated.unsqueeze(1)).squeeze(1)
        mass = (x_gated @ x_gated) + 1e-8
        coherence = torch.exp(-cut / mass)
        return x_gated * coherence
    return x_gated

def grad2d(f, H, W):
    f = f.reshape(H, W); gy, gx = torch.zeros_like(f), torch.zeros_like(f)
    gy[:-1, :], gx[:, :-1] = f[1:, :] - f[:-1, :], f[:, 1:] - f[:, :-1]
    return gy, gx

def cyclic_color(rgb, phase=0.5):
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    theta = 2 * math.pi * lum + phase * math.pi
    return torch.stack([0.5 + 0.4 * torch.sin(theta + i * 2 * math.pi / 3) for i in range(3)], dim=-1)

def segment_properties(seg_id, y_coords, x_coords, num_segs):
    ones = torch.ones_like(seg_id, dtype=torch.float32)
    count = torch.zeros(num_segs, device=seg_id.device).scatter_add_(0, seg_id, ones).clamp(min=1)
    sum_y = torch.zeros(num_segs, device=seg_id.device).scatter_add_(0, seg_id, y_coords.float())
    sum_x = torch.zeros(num_segs, device=seg_id.device).scatter_add_(0, seg_id, x_coords.float())
    cent_y, cent_x = sum_y / count, sum_x / count
    return cent_y, cent_x, count

def shader(tgt, smp, strength=0.7, num_segs=16, fuzz_temp=1.0, use_global_fuzz=False):
    H, W, _ = tgt.shape; n = H * W
    gt, gs = tgt.mean(2), smp.mean(2)
    dev = tgt.device

    L_t = laplacian(gt); phi_t = lanczos(L_t, k=8)
    L_s = laplacian(gs); phi_s = lanczos(L_s, k=8)

    act = phi_t @ (phi_s.T @ gs.flatten()) / (phi_s.abs().sum() + 1e-8)
    act_clean = annihilate_fuzz(act, L_t, temp=fuzz_temp, use_global=use_global_fuzz)

    # Gating
    gamma_h, bias_h = 1.4, 0.05
    gamma_l, bias_l = 0.7, -0.05
    high = (gamma_h * act_clean + bias_h) > act_clean
    low = (gamma_l * act_clean + bias_l) < act_clean

    # Segments
    fiedler = phi_t[:, min(1, phi_t.shape[1]-1)] if phi_t.shape[1] > 1 else phi_t[:, 0]
    fiedler_norm = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
    seg_id = (fiedler_norm * num_segs).floor().long().clamp(0, num_segs - 1)

    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    yf, xf = yy.flatten(), xx.flatten()
    cent_y, cent_x, count = segment_properties(seg_id, yf, xf, num_segs)

    bnd = ((gt.flatten() - gt.mean()).abs() > gt.std() * 0.5).float()
    dist = heat(L_t, bnd, t=3.0); gy, gx = grad2d(dist, H, W)
    mag = (gy**2 + gx**2).sqrt().flatten() + 1e-8
    vy, vx = -gy.flatten() / mag, -gx.flatten() / mag

    out = tgt.clone()

    # THICKEN via heat kernel
    high2d = high.reshape(H, W) & (bnd.reshape(H, W) > 0.5)
    if high2d.any():
        ink = high2d.float().flatten()
        diffused = heat(L_t, ink, t=2.0 * strength, steps=20)
        diffused = diffused / (diffused.max() + 1e-8)
        thickened = (diffused > 0.2).reshape(H, W)
        add_mask = thickened & (bnd.reshape(H, W) < 0.5)
        if add_mask.any():
            high_colors = tgt[high2d].mean(0)
            out[add_mask] = high_colors

    # SHADOW via segment scatter (simplified - not full rigid transform to avoid loops)
    low2d = low.reshape(H, W) & (bnd.reshape(H, W) > 0.5)
    if low2d.any():
        lo_idx = torch.where(low2d.flatten())[0]
        lo_seg = seg_id[lo_idx]
        lo_cy, lo_cx = cent_y[lo_seg], cent_x[lo_seg]

        # 90-degree rotation around segment centroid
        local_y = yf[lo_idx].float() - lo_cy
        local_x = xf[lo_idx].float() - lo_cx
        rot_y, rot_x = -local_x, local_y

        void_dy, void_dx = vy[lo_idx], vx[lo_idx]
        disp = 8 * strength * count[lo_seg].sqrt().clamp(min=1, max=10)
        new_cy = lo_cy + void_dy * disp
        new_cx = lo_cx + void_dx * disp

        dst_y = (new_cy + rot_y).long().clamp(0, H-1)
        dst_x = (new_cx + rot_x).long().clamp(0, W-1)
        dst_flat = dst_y * W + dst_x

        valid = bnd[dst_flat] < 0.5
        if valid.any():
            out.reshape(-1, 3)[dst_flat[valid]] = cyclic_color(tgt.reshape(-1, 3)[lo_idx[valid]], 0.6)

    return out

def shade(tgt_np, smp_np, pattern='cs', strength=0.7, fuzz_temp=1.0, use_global_fuzz=False):
    t = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)
    s = torch.tensor(smp_np, dtype=torch.float32, device=DEVICE)
    for i, m in enumerate(pattern):
        t = shader(t, s if m == 'c' else t, strength * (0.75 ** i), fuzz_temp=fuzz_temp, use_global_fuzz=use_global_fuzz)
    return t.cpu().numpy()

def measure_diff(before, after):
    diff = np.abs(after - before)
    return {
        'total': diff.sum(),
        'pixels': (diff.max(axis=2) > 0.01).sum(),
        'mean': diff.mean(),
    }

def score(before, after):
    d = measure_diff(before, after)
    n_pix = before.shape[0] * before.shape[1]
    # Want: meaningful change (>100 pixels), not destruction (<30% pixels)
    if d['pixels'] < 100: return -1000
    if d['pixels'] > n_pix * 0.3: return -500
    if d['mean'] > 0.2: return -300
    # Score: number of changed pixels, penalized by mean intensity of change
    return d['pixels'] / 100 - d['mean'] * 50

def grid_search():
    inp = Path("demo_output/inputs")
    tgt = np.array(Image.open(inp / "toof.png").convert('RGB')).astype(np.float32) / 255.0
    smp = np.array(Image.open(inp / "red-tonegraph.png").convert('RGB')).astype(np.float32) / 255.0

    temps = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    strengths = [0.5, 0.7, 1.0, 1.5]
    globals_fuzz = [False, True]

    best_score = -float('inf')
    best_params = None
    best_result = None

    print(f"{'temp':>6} {'str':>5} {'glob':>5} {'pixels':>8} {'mean':>8} {'score':>8}")
    print("-" * 50)

    for temp in temps:
        for strength in strengths:
            for use_g in globals_fuzz:
                try:
                    result = shade(tgt, smp, pattern='c', strength=strength, fuzz_temp=temp, use_global_fuzz=use_g)
                    s = score(tgt, result)
                    d = measure_diff(tgt, result)
                    print(f"{temp:>6.1f} {strength:>5.1f} {str(use_g):>5} {d['pixels']:>8.0f} {d['mean']:>8.4f} {s:>8.1f}")

                    if s > best_score:
                        best_score = s
                        best_params = {'temp': temp, 'strength': strength, 'use_global': use_g}
                        best_result = result
                except Exception as e:
                    print(f"{temp:>6.1f} {strength:>5.1f} {str(use_g):>5} ERROR: {e}")

    print("-" * 50)
    if best_params:
        print(f"Best: temp={best_params['temp']}, strength={best_params['strength']}, global={best_params['use_global']}, score={best_score:.1f}")
        Image.fromarray((best_result * 255).clip(0, 255).astype(np.uint8)).save("demo_output/v7e_tuned_c.png")
        print("Saved: demo_output/v7e_tuned_c.png")

        # cccc
        result_cccc = shade(tgt, smp, pattern='cccc', strength=best_params['strength'], fuzz_temp=best_params['temp'], use_global_fuzz=best_params['use_global'])
        Image.fromarray((result_cccc * 255).clip(0, 255).astype(np.uint8)).save("demo_output/v7e_tuned_cccc.png")
        print("Saved: demo_output/v7e_tuned_cccc.png")
    else:
        print("No valid params found")

if __name__ == "__main__":
    grid_search()
