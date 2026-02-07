#!/usr/bin/env python3
"""Demo: Lattice Extrusion as Edge-Based Height Field on Egg Surface.

Three lattice types (triangular, square, hexagonal) wrapped on an egg surface,
extruded iteratively to create high-dimensional structure, rendered as:
  - Positive bumps at node positions
  - Negative channels along edge paths
  - Height field -> normal map -> lit 3D egg surface

The key insight: edges are lines (many pixels), nodes are points (one pixel).
Map edge directions to tangent-plane directions, render as bump/channel texture.
Different lattice types produce visibly different channel hatching patterns.

Uses ONLY the _even_cuter Module files:
- spectral_lattice.py: build_egg_lattice, edges_to_height_field, lattice_type_texture
- spectral_renderer.py: HeightToNormals, EggSurfaceRenderer
- image_io.py: save_image
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import torch

from spectral_lattice import (
    build_egg_lattice,
    edges_to_height_field,
    lattice_type_texture,
    compute_degrees,
)
from spectral_renderer import HeightToNormals, EggSurfaceRenderer
from image_io import save_image


def main():
    parser = argparse.ArgumentParser(
        description="Lattice Extrusion: Edge-Based Height Field on Egg Surface"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    parser.add_argument("--tex-res", type=int, default=256,
                        help="Texture resolution (default: 256)")
    parser.add_argument("--egg-res", type=int, default=512,
                        help="Egg render resolution (default: 512)")
    parser.add_argument("--grid-size", type=int, default=14,
                        help="Lattice grid size per patch (default: 14)")
    parser.add_argument("--layers", type=int, default=6,
                        help="Extrusion layers per lattice type (default: 6)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tex_res = args.tex_res
    egg_res = args.egg_res

    ts = time.strftime("%Y%m%d_%H%M%S")
    stats = {}

    print(f"Device: {device}")
    print(f"Texture: {tex_res}x{tex_res}, Egg: {egg_res}x{egg_res}")
    print(f"Grid: {args.grid_size}, Layers: {args.layers}")
    print()

    # ================================================================
    # Phase 1: Build lattice graph on egg surface
    # ================================================================
    print("Phase 1: Building lattice graph...")
    t0 = time.time()

    lattice = build_egg_lattice(
        grid_size=args.grid_size,
        n_extrusion_layers=args.layers,
        jitter_scale=0.012,
        device=device,
    )

    t1 = time.time()
    stats["phase1_build"] = {
        "n_nodes": lattice["n"],
        "n_edges": lattice["n_edges"],
        "seed_counts": {k: v for k, v in lattice["seed_counts"].items()},
        "extruded_counts": {k: v for k, v in lattice["extruded_counts"].items()},
        "n_extrusion_layers": lattice["n_extrusion_layers"],
        "n_distinct_edge_directions": lattice["n_distinct_directions"],
        "degree_range": [int(lattice["degrees"].min().item()),
                         int(lattice["degrees"].max().item())],
        "degree_mean": round(lattice["degrees"].mean().item(), 1),
        "time_s": round(t1 - t0, 3),
    }
    print(f"  {lattice['n']} nodes, {lattice['n_edges']} edges")
    print(f"  Seeds: tri={lattice['seed_counts']['tri']}, "
          f"sq={lattice['seed_counts']['sq']}, hex={lattice['seed_counts']['hex']}")
    print(f"  Extruded: tri={lattice['extruded_counts']['tri']}, "
          f"sq={lattice['extruded_counts']['sq']}, hex={lattice['extruded_counts']['hex']}")
    print(f"  Distinct edge directions: {lattice['n_distinct_directions']}")
    print(f"  Degree range: [{int(lattice['degrees'].min())}, {int(lattice['degrees'].max())}], "
          f"mean={lattice['degrees'].mean():.1f}")
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Phase 2: Generate height field from edges
    # ================================================================
    print("Phase 2: Generating height field...")
    t0 = time.time()

    height_field = edges_to_height_field(
        lattice["adj"], lattice["uv"], lattice["n"],
        tex_res=tex_res,
        bump_height=1.5,
        bump_sigma=2.0,
        channel_depth=-0.5,
        device=device,
    )

    t1 = time.time()
    stats["phase2_height_field"] = {
        "resolution": tex_res,
        "height_min": round(height_field.min().item(), 4),
        "height_max": round(height_field.max().item(), 4),
        "height_mean": round(height_field.mean().item(), 4),
        "height_std": round(height_field.std().item(), 4),
        "nonzero_frac": round((height_field.abs() > 1e-4).float().mean().item(), 4),
        "time_s": round(t1 - t0, 3),
    }
    print(f"  Height range: [{height_field.min():.3f}, {height_field.max():.3f}]")
    print(f"  Nonzero: {(height_field.abs() > 1e-4).float().mean() * 100:.1f}%")
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # Save height field as grayscale image
    h_norm = height_field - height_field.min()
    h_norm = h_norm / h_norm.max().clamp(min=1e-6)
    h_rgb = h_norm.unsqueeze(-1).expand(-1, -1, 3)
    save_image(h_rgb, out / f"lattice_height_field_{ts}.png", timestamp=False)

    # ================================================================
    # Phase 3: Generate color texture from lattice types
    # ================================================================
    print("Phase 3: Generating type texture...")
    t0 = time.time()

    color_tex = lattice_type_texture(
        lattice["uv"], lattice["types"], lattice["n"],
        tex_res=tex_res, device=device,
    )

    t1 = time.time()
    save_image(color_tex, out / f"lattice_type_texture_{ts}.png", timestamp=False)
    stats["phase3_type_texture"] = {"time_s": round(t1 - t0, 3)}
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Phase 4: Height -> Normals
    # ================================================================
    print("Phase 4: Computing normal map...")
    t0 = time.time()

    h2n = HeightToNormals(strength=3.0).to(device)
    normal_map = h2n(height_field)  # (tex_res, tex_res, 3)

    t1 = time.time()
    save_image(normal_map, out / f"lattice_normal_map_{ts}.png", timestamp=False)
    stats["phase4_normals"] = {"time_s": round(t1 - t0, 3)}
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Phase 5: Egg surface render with lattice texture
    # ================================================================
    print("Phase 5: Rendering egg surface...")
    t0 = time.time()

    egg = EggSurfaceRenderer(
        resolution=egg_res,
        egg_factor=0.25,
        bump_strength=1.5,
        light_dir=(0.5, 0.7, 1.0),
        specular_power=32.0,
        fresnel_ior=1.5,
        ambient=0.08,
    ).to(device)

    # Render with the color texture and normal map
    egg_render = egg(color_tex, normal_map)

    t1 = time.time()
    save_image(egg_render, out / f"lattice_egg_render_{ts}.png", timestamp=False)
    stats["phase5_egg_render"] = {
        "egg_resolution": egg_res,
        "render_std": round(egg_render.std().item(), 4),
        "render_nonzero": round((egg_render.abs() > 0.01).float().mean().item(), 4),
        "time_s": round(t1 - t0, 3),
    }
    print(f"  Render std: {egg_render.std():.4f}")
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Phase 6: Render egg with height-only texture (channels visible as geometry)
    # ================================================================
    print("Phase 6: Rendering egg with height-only texture...")
    t0 = time.time()

    # Use height field as a grayscale "material" texture
    # Bumps (positive) = light areas, channels (negative) = dark grooves
    h_tex = h_norm.unsqueeze(-1).expand(-1, -1, 3).contiguous()
    # Warm tint: multiply by a warm color to make it look like carved stone
    stone_color = torch.tensor([0.85, 0.75, 0.60], device=device)
    h_tex_tinted = h_tex * stone_color

    egg_height_render = egg(h_tex_tinted, normal_map)
    t1 = time.time()
    save_image(egg_height_render, out / f"lattice_egg_carved_{ts}.png", timestamp=False)
    stats["phase6_carved"] = {"time_s": round(t1 - t0, 3)}
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Phase 7: Render egg with combined type color + height modulation
    # ================================================================
    print("Phase 7: Rendering combined type + height egg...")
    t0 = time.time()

    # Modulate type colors by height field: bumps stay bright, channels darken
    h_mod = (h_norm * 0.6 + 0.4).unsqueeze(-1)  # [0.4, 1.0] range
    combined_tex = color_tex * h_mod
    egg_combined = egg(combined_tex, normal_map)
    t1 = time.time()
    save_image(egg_combined, out / f"lattice_egg_combined_{ts}.png", timestamp=False)
    stats["phase7_combined"] = {"time_s": round(t1 - t0, 3)}
    print(f"  Time: {t1 - t0:.3f}s")
    print()

    # ================================================================
    # Summary
    # ================================================================
    total_time = sum(v.get("time_s", 0) for v in stats.values() if isinstance(v, dict))
    stats["total_time_s"] = round(total_time, 2)
    stats["images_written"] = 5

    stats_path = out / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 60)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Images: 5 written to {out}/")
    print(f"  Stats:  {stats_path}")
    print()

    # Checks
    checks = [
        ("nodes >= 1000", lattice["n"] >= 1000),
        ("edges >= 3000", lattice["n_edges"] >= 3000),
        ("distinct directions >= 17", lattice["n_distinct_directions"] >= 17),
        ("height field nonzero > 50%",
         (height_field.abs() > 1e-4).float().mean().item() > 0.5),
        ("height field has bumps AND channels",
         height_field.max().item() > 0.1 and height_field.min().item() < -0.1),
        ("egg render std > 0.05", egg_render.std().item() > 0.05),
        ("no NaN in renders",
         not (torch.isnan(egg_render).any() or torch.isnan(height_field).any())),
    ]

    print("Checks:")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    stats["checks"] = {name: passed for name, passed in checks}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
