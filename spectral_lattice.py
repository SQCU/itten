"""Lattice construction on curved surfaces with edge-based height field rendering.

Core idea: edges are lines (cover many pixels), nodes are points (one pixel).
Render lattice structure as:
  - Positive-height bumps at node positions
  - Negative-height channels along edge paths
  - Different lattice types produce visibly different channel patterns

Rendering pipeline:
  Graph on surface -> edge directions projected to tangent plane
  -> height field (bumps + channels) -> HeightToNormals -> EggSurfaceRenderer

Three lattice types wrapped on an egg surface, each extruded iteratively
to create high-dimensional structure (local dim >= 3 per patch, total
turning directions > 17).  UV jitter per extrusion layer creates distinct
angular channel patterns per lattice type.

All code is pure PyTorch -- no numpy.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Surface Parametrizations
# ============================================================================


def egg_surface(
    u: torch.Tensor, v: torch.Tensor,
    egg_factor: float = 0.25, scale: float = 1.0,
) -> torch.Tensor:
    """Map (u, v) in [0,1]^2 to an egg surface in R^3."""
    phi = v * math.pi
    theta = u * 2 * math.pi
    y = torch.cos(phi)
    r_base = torch.sin(phi)
    egg_mod = 1.0 - egg_factor * y
    x = r_base * torch.cos(theta) * egg_mod * scale
    z = r_base * torch.sin(theta) * egg_mod * scale
    y = y * scale
    return torch.stack([x, y, z], dim=-1)


def cylinder_surface(
    u: torch.Tensor, v: torch.Tensor,
    theta_range: float = 2.0 * math.pi / 3,
    theta_offset: float = 0.0,
    radius: float = 1.0, height: float = 2.0,
) -> torch.Tensor:
    """Map (u, v) in [0,1]^2 to a cylindrical patch in R^3."""
    theta = theta_offset + u * theta_range
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = v * height - height / 2
    return torch.stack([x, y, z], dim=-1)


# ============================================================================
# Lattice Constructors (return adj, uv, 3d coords, node count)
# ============================================================================


def triangular_lattice(
    rows: int, cols: int, device: torch.device,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Triangular lattice on a (u,v) patch. Returns (adj, uv, n)."""
    nodes = []
    u0, u1 = u_range
    v0, v1 = v_range
    for r in range(rows):
        for c in range(cols):
            u = u0 + (u1 - u0) * (c / max(cols - 1, 1) + (0.5 / max(cols - 1, 1) if r % 2 else 0.0))
            v = v0 + (v1 - v0) * r / max(rows - 1, 1)
            nodes.append((u, v))
    n = len(nodes)
    uv = torch.tensor(nodes, dtype=torch.float32, device=device)

    rows_list, cols_list = [], []
    node_grid = {}
    for idx, (r, c) in enumerate([(r, c) for r in range(rows) for c in range(cols)]):
        node_grid[(r, c)] = idx

    for r in range(rows):
        for c in range(cols):
            me = node_grid[(r, c)]
            if c + 1 < cols:
                nb = node_grid[(r, c + 1)]
                rows_list.extend([me, nb])
                cols_list.extend([nb, me])
            if r + 1 < rows:
                nb = node_grid[(r + 1, c)]
                rows_list.extend([me, nb])
                cols_list.extend([nb, me])
            if r % 2 == 0:
                if r + 1 < rows and c + 1 < cols:
                    nb = node_grid[(r + 1, c + 1)]
                    rows_list.extend([me, nb])
                    cols_list.extend([nb, me])
            else:
                if r + 1 < rows and c - 1 >= 0:
                    nb = node_grid[(r + 1, c - 1)]
                    rows_list.extend([me, nb])
                    cols_list.extend([nb, me])

    if rows_list:
        r_t = torch.tensor(rows_list, dtype=torch.long, device=device)
        c_t = torch.tensor(cols_list, dtype=torch.long, device=device)
        vals = torch.ones(len(rows_list), dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(torch.stack([r_t, c_t]), vals, (n, n)).coalesce()
    else:
        adj = _empty_adj(n, device)
    return adj, uv, n


def square_lattice(
    rows: int, cols: int, device: torch.device,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Square lattice (4-connected grid) on a (u,v) patch. Returns (adj, uv, n)."""
    n = rows * cols
    u0, u1 = u_range
    v0, v1 = v_range
    nodes = []
    for r in range(rows):
        for c in range(cols):
            u = u0 + (u1 - u0) * c / max(cols - 1, 1)
            v = v0 + (v1 - v0) * r / max(rows - 1, 1)
            nodes.append((u, v))
    uv = torch.tensor(nodes, dtype=torch.float32, device=device)

    rows_list, cols_list = [], []
    for r in range(rows):
        for c in range(cols):
            me = r * cols + c
            if c + 1 < cols:
                nb = r * cols + c + 1
                rows_list.extend([me, nb])
                cols_list.extend([nb, me])
            if r + 1 < rows:
                nb = (r + 1) * cols + c
                rows_list.extend([me, nb])
                cols_list.extend([nb, me])

    r_t = torch.tensor(rows_list, dtype=torch.long, device=device)
    c_t = torch.tensor(cols_list, dtype=torch.long, device=device)
    vals = torch.ones(len(rows_list), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(torch.stack([r_t, c_t]), vals, (n, n)).coalesce()
    return adj, uv, n


def hexagonal_lattice(
    rows: int, cols: int, device: torch.device,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Hexagonal (honeycomb) lattice on a (u,v) patch. Returns (adj, uv, n)."""
    n = rows * cols
    u0, u1 = u_range
    v0, v1 = v_range
    hex_h = math.sqrt(3) / 2
    nodes = []
    for r in range(rows):
        for c in range(cols):
            x_off = 0.5 if r % 2 else 0.0
            u = u0 + (u1 - u0) * (c + x_off) / max(cols, 1)
            v = v0 + (v1 - v0) * (r * hex_h) / max((rows - 1) * hex_h, 1)
            nodes.append((u, v))
    uv = torch.tensor(nodes, dtype=torch.float32, device=device)

    rows_list, cols_list = [], []
    for r in range(rows):
        for c in range(cols):
            me = r * cols + c
            if c + 1 < cols:
                nb = r * cols + c + 1
                rows_list.extend([me, nb])
                cols_list.extend([nb, me])
            if (r + c) % 2 == 0:
                if r + 1 < rows:
                    nb = (r + 1) * cols + c
                    rows_list.extend([me, nb])
                    cols_list.extend([nb, me])

    r_t = torch.tensor(rows_list, dtype=torch.long, device=device)
    c_t = torch.tensor(cols_list, dtype=torch.long, device=device)
    vals = torch.ones(len(rows_list), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(torch.stack([r_t, c_t]), vals, (n, n)).coalesce()
    return adj, uv, n


def _empty_adj(n: int, device: torch.device) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.long, device=device),
        torch.zeros(0, dtype=torch.float32, device=device), (n, n)
    ).coalesce()


# ============================================================================
# Graph Utilities
# ============================================================================


def adjacency_to_laplacian(adj: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    adj_c = adj.coalesce()
    rows = adj_c.indices()[0]
    cols = adj_c.indices()[1]
    vals = adj_c.values()
    off_diag = -vals.abs()
    degrees = torch.zeros(n, device=device, dtype=torch.float32)
    degrees.scatter_add_(0, rows, vals.abs())
    diag_idx = torch.arange(n, device=device, dtype=torch.long)
    all_rows = torch.cat([rows, diag_idx])
    all_cols = torch.cat([cols, diag_idx])
    all_vals = torch.cat([off_diag, degrees])
    return torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]), all_vals, (n, n)
    ).coalesce()


def compute_degrees(adj: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    adj_c = adj.coalesce()
    degrees = torch.zeros(n, device=device, dtype=torch.float32)
    degrees.scatter_add_(0, adj_c.indices()[0], torch.ones(adj_c._nnz(), device=device))
    return degrees


def merge_graphs(
    adj_a: torch.Tensor, uv_a: torch.Tensor, types_a: torch.Tensor, n_a: int,
    adj_b: torch.Tensor, uv_b: torch.Tensor, types_b: torch.Tensor, n_b: int,
    device: torch.device, n_bridges: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Merge two graphs, adding bridge edges between nearest boundary nodes."""
    n = n_a + n_b
    a_c = adj_a.coalesce()
    b_c = adj_b.coalesce()

    rows_list = [a_c.indices()[0], b_c.indices()[0] + n_a]
    cols_list = [a_c.indices()[1], b_c.indices()[1] + n_a]

    # Bridge edges: nearest pairs by UV distance
    dists = torch.cdist(uv_a, uv_b)
    for _ in range(min(n_bridges, min(n_a, n_b))):
        flat_idx = dists.argmin()
        i_a = flat_idx // n_b
        i_b = flat_idx % n_b
        rows_list.append(torch.tensor([i_a, i_b + n_a], device=device))
        cols_list.append(torch.tensor([i_b + n_a, i_a], device=device))
        dists[i_a, i_b] = float('inf')  # don't reuse

    all_rows = torch.cat(rows_list)
    all_cols = torch.cat(cols_list)
    vals = torch.ones(all_rows.shape[0], dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(torch.stack([all_rows, all_cols]), vals, (n, n)).coalesce()

    uv = torch.cat([uv_a, uv_b], dim=0)
    types = torch.cat([types_a, types_b], dim=0)
    return adj, uv, types, n


# ============================================================================
# Lattice Extrusion (adds layers -> high-dimensional structure)
# ============================================================================


# Jitter angles per lattice type per layer (radians in UV space).
# These create distinct angular channel patterns:
#   tri: 60-degree family (0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3)
#   sq:  45-degree family (pi/4, 3pi/4, 5pi/4, 7pi/4, ...)
#   hex: 30-degree family (pi/6, pi/2, 5pi/6, 7pi/6, ...)
_JITTER_ANGLES = {
    0: [0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3],
    1: [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4, math.pi / 8, 5 * math.pi / 8],
    2: [math.pi / 6, math.pi / 2, 5 * math.pi / 6, 7 * math.pi / 6, 3 * math.pi / 2, 11 * math.pi / 6],
}


def extrude_lattice(
    adj: torch.Tensor,
    uv: torch.Tensor,
    n: int,
    lattice_type: int,
    n_layers: int = 5,
    jitter_scale: float = 0.015,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Extrude a 2D surface lattice into higher dimension by adding offset layers.

    Each layer creates copies of base-layer nodes with UV coordinates jittered
    in a lattice-type-dependent angular direction.  Connections:
      - parent -> child (vertical/extrusion edges)
      - child_i -> child_j  if parent_i ~ parent_j (horizontal edges in new layer)

    The angular jitter per layer creates new edge directions in UV space,
    so different lattice types produce visually distinct channel textures.

    Returns: (new_adj, new_uv, new_n, layer_labels)
      layer_labels[i] = which layer node i belongs to (0 = base)
    """
    angles = _JITTER_ANGLES.get(lattice_type, _JITTER_ANGLES[0])
    base_n = n
    all_uv = [uv]
    all_layers = [torch.zeros(n, dtype=torch.long, device=device)]

    # Collect all edges (will add new ones per layer)
    adj_c = adj.coalesce()
    edge_rows = [adj_c.indices()[0]]
    edge_cols = [adj_c.indices()[1]]

    # Map from base node index -> latest copy index (for connecting consecutive layers)
    prev_copy = torch.arange(n, device=device, dtype=torch.long)

    current_n = n
    for layer in range(n_layers):
        angle = angles[layer % len(angles)]
        # Jitter direction in UV space
        du = jitter_scale * math.cos(angle)
        dv = jitter_scale * math.sin(angle)
        # Scale jitter per layer: each layer pushes further out
        # Different lattice types get different base scales for visual contrast
        type_scale = [1.0, 1.3, 0.8][lattice_type]
        scale = type_scale * (1.0 + 0.4 * layer)

        # New node UV coords: base positions + jitter
        new_uv = uv.clone()
        new_uv[:, 0] = (new_uv[:, 0] + du * scale) % 1.0
        new_uv[:, 1] = (new_uv[:, 1] + dv * scale).clamp(0.01, 0.99)
        all_uv.append(new_uv)
        all_layers.append(torch.full((base_n,), layer + 1, dtype=torch.long, device=device))

        new_start = current_n
        new_copy = torch.arange(new_start, new_start + base_n, device=device, dtype=torch.long)

        # Vertical edges: prev_copy[i] -> new_copy[i]
        edge_rows.append(prev_copy)
        edge_cols.append(new_copy)
        edge_rows.append(new_copy)
        edge_cols.append(prev_copy)

        # Horizontal edges: copy the base adjacency pattern into the new layer
        base_rows = adj_c.indices()[0]
        base_cols = adj_c.indices()[1]
        edge_rows.append(base_rows + new_start)
        edge_cols.append(base_cols + new_start)

        # Cross-layer diagonals: connect prev_copy[i] -> new_copy[j] where i~j
        # This adds "twisted" edges that create rich angular patterns
        edge_rows.append(prev_copy[base_rows])
        edge_cols.append(new_copy[base_cols])
        edge_rows.append(new_copy[base_cols])
        edge_cols.append(prev_copy[base_rows])

        prev_copy = new_copy
        current_n += base_n

    all_rows = torch.cat(edge_rows)
    all_cols = torch.cat(edge_cols)
    vals = torch.ones(all_rows.shape[0], dtype=torch.float32, device=device)
    new_adj = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]), vals, (current_n, current_n)
    ).coalesce()

    new_uv = torch.cat(all_uv, dim=0)
    layer_labels = torch.cat(all_layers)

    return new_adj, new_uv, current_n, layer_labels


# ============================================================================
# Edge-Based Height Field Rendering
# ============================================================================


def edges_to_height_field(
    adj: torch.Tensor,
    uv: torch.Tensor,
    n: int,
    tex_res: int = 256,
    bump_height: float = 1.0,
    bump_sigma: float = 1.8,
    channel_depth: float = -0.4,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert graph edges to a height field on the (u,v) texture space.

    Positive Gaussian bumps at each node's UV position.
    Negative channels along each edge (Bresenham rasterization).

    Returns: (tex_res, tex_res) height field.
    """
    height = torch.zeros(tex_res, tex_res, dtype=torch.float32, device=device)

    # --- Node bumps ---
    # Pixel positions for each node
    px = (uv[:, 0] * (tex_res - 1)).clamp(0, tex_res - 1)
    py = (uv[:, 1] * (tex_res - 1)).clamp(0, tex_res - 1)
    px_i = px.long()
    py_i = py.long()

    # Stamp Gaussian bumps (vectorized per node, local window)
    r = int(bump_sigma * 3)
    for i in range(n):
        cx = px[i].item()
        cy = py[i].item()
        x0 = max(0, int(cx) - r)
        x1 = min(tex_res, int(cx) + r + 1)
        y0 = max(0, int(cy) - r)
        y1 = min(tex_res, int(cy) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local_x = torch.arange(x0, x1, dtype=torch.float32, device=device)
        local_y = torch.arange(y0, y1, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(local_y, local_x, indexing='ij')
        dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
        bump = bump_height * torch.exp(-dist_sq / (2 * bump_sigma ** 2))
        height[y0:y1, x0:x1] = torch.max(height[y0:y1, x0:x1], bump)

    # --- Edge channels (batched rasterization with validity mask) ---
    adj_c = adj.coalesce()
    edge_src = adj_c.indices()[0]
    edge_dst = adj_c.indices()[1]
    # Only process each edge once (src < dst)
    keep = edge_src < edge_dst
    edge_src = edge_src[keep]
    edge_dst = edge_dst[keep]

    src_px = px[edge_src]
    src_py = py[edge_src]
    dst_px = px[edge_dst]
    dst_py = py[edge_dst]

    # Per-edge length in pixels
    dx = (dst_px - src_px).abs()
    dy = (dst_py - src_py).abs()
    edge_lengths = torch.maximum(dx, dy).long().clamp(min=1)
    max_steps = edge_lengths.max().clamp(max=tex_res).item()

    # t values: (max_steps+1,)
    t = torch.linspace(0, 1, max_steps + 1, device=device)

    # Interpolate all edges: (n_edges, max_steps+1)
    line_x = (src_px.unsqueeze(1) + t.unsqueeze(0) * (dst_px - src_px).unsqueeze(1))
    line_y = (src_py.unsqueeze(1) + t.unsqueeze(0) * (dst_py - src_py).unsqueeze(1))
    line_x = line_x.long().clamp(0, tex_res - 1)
    line_y = line_y.long().clamp(0, tex_res - 1)

    # Validity mask: sample index <= actual edge length
    sample_idx = torch.arange(max_steps + 1, device=device).unsqueeze(0)  # (1, max_steps+1)
    valid = sample_idx <= edge_lengths.unsqueeze(1)  # (n_edges, max_steps+1)

    flat_idx = (line_y * tex_res + line_x).reshape(-1)
    valid_flat = valid.reshape(-1)

    # Only accumulate valid samples (binary channel: each pixel counts once per edge)
    # Use a set-like approach: mark which pixels have ANY channel through them
    channel_mask = torch.zeros(tex_res * tex_res, dtype=torch.float32, device=device)
    valid_idx = flat_idx[valid_flat]
    channel_mask.scatter_add_(
        0, valid_idx,
        torch.ones_like(valid_idx, dtype=torch.float32),
    )
    # Convert hit count to depth: log-scale so high overlap doesn't crush everything
    # 1 hit = channel_depth, 10 hits = ~2.3 * channel_depth
    channel_field = channel_depth * torch.log1p(channel_mask)
    channel_field.clamp_(min=-bump_height * 1.5)
    height.view(-1).add_(channel_field)

    return height


def lattice_type_texture(
    uv: torch.Tensor,
    lattice_types: torch.Tensor,
    n: int,
    tex_res: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate (tex_res, tex_res, 3) color texture from lattice type via nearest-node Voronoi.

    Colors: tri=warm red, sq=blue, hex=green.
    """
    TYPE_COLORS = torch.tensor([
        [0.85, 0.25, 0.20],  # triangular
        [0.20, 0.40, 0.85],  # square
        [0.20, 0.80, 0.30],  # hexagonal
    ], dtype=torch.float32, device=device)

    texture = torch.zeros(tex_res, tex_res, 3, dtype=torch.float32, device=device)

    # Pixel grid in UV space
    uu = torch.linspace(0, 1, tex_res, device=device)
    vv = torch.linspace(0, 1, tex_res, device=device)
    grid_v, grid_u = torch.meshgrid(vv, uu, indexing='ij')  # (tex_res, tex_res)

    # Process in chunks to avoid OOM on large grids
    chunk = 32
    for y_start in range(0, tex_res, chunk):
        y_end = min(y_start + chunk, tex_res)
        chunk_u = grid_u[y_start:y_end].reshape(-1, 1)  # (chunk*tex_res, 1)
        chunk_v = grid_v[y_start:y_end].reshape(-1, 1)

        # Distance to each node
        du = chunk_u - uv[:, 0].unsqueeze(0)  # (pixels, n)
        dv = chunk_v - uv[:, 1].unsqueeze(0)
        dist = du ** 2 + dv ** 2

        nearest = dist.argmin(dim=1)  # (pixels,)
        pixel_types = lattice_types[nearest].clamp(0, 2)
        colors = TYPE_COLORS[pixel_types]  # (pixels, 3)

        texture[y_start:y_end] = colors.reshape(y_end - y_start, tex_res, 3)

    return texture


# ============================================================================
# Build Complete Lattice on Egg Surface
# ============================================================================


def build_egg_lattice(
    grid_size: int = 12,
    n_extrusion_layers: int = 5,
    jitter_scale: float = 0.015,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Build 3 lattice types on an egg surface, extrude each, merge.

    The egg surface is divided into 3 angular sectors:
      [0.0, 0.33] = triangular lattice
      [0.33, 0.66] = square lattice
      [0.66, 1.0]  = hexagonal lattice

    Each sector covers the full v range [0.05, 0.95] (avoiding poles).

    Returns dict with all graph data for rendering.
    """
    # Build seed lattices on separate UV sectors of the egg
    tri_adj, tri_uv, n_tri = triangular_lattice(
        grid_size, grid_size, device,
        u_range=(0.0, 0.32), v_range=(0.05, 0.95),
    )
    sq_adj, sq_uv, n_sq = square_lattice(
        grid_size, grid_size - 2, device,
        u_range=(0.34, 0.64), v_range=(0.05, 0.95),
    )
    hex_adj, hex_uv, n_hex = hexagonal_lattice(
        grid_size, grid_size, device,
        u_range=(0.66, 0.98), v_range=(0.05, 0.95),
    )

    # Extrude each lattice type
    tri_adj_e, tri_uv_e, n_tri_e, tri_layers = extrude_lattice(
        tri_adj, tri_uv, n_tri, lattice_type=0,
        n_layers=n_extrusion_layers, jitter_scale=jitter_scale, device=device,
    )
    tri_types = torch.zeros(n_tri_e, dtype=torch.long, device=device)

    sq_adj_e, sq_uv_e, n_sq_e, sq_layers = extrude_lattice(
        sq_adj, sq_uv, n_sq, lattice_type=1,
        n_layers=n_extrusion_layers, jitter_scale=jitter_scale, device=device,
    )
    sq_types = torch.ones(n_sq_e, dtype=torch.long, device=device)

    hex_adj_e, hex_uv_e, n_hex_e, hex_layers = extrude_lattice(
        hex_adj, hex_uv, n_hex, lattice_type=2,
        n_layers=n_extrusion_layers, jitter_scale=jitter_scale, device=device,
    )
    hex_types = torch.full((n_hex_e,), 2, dtype=torch.long, device=device)

    # Merge all three
    adj_01, uv_01, types_01, n_01 = merge_graphs(
        tri_adj_e, tri_uv_e, tri_types, n_tri_e,
        sq_adj_e, sq_uv_e, sq_types, n_sq_e,
        device=device, n_bridges=15,
    )
    adj_all, uv_all, types_all, n_all = merge_graphs(
        adj_01, uv_01, types_01, n_01,
        hex_adj_e, hex_uv_e, hex_types, n_hex_e,
        device=device, n_bridges=15,
    )

    # Compute graph properties
    degrees = compute_degrees(adj_all, n_all, device)
    n_edges = adj_all._nnz() // 2

    # Count distinct edge directions (quantized to 10-degree bins)
    adj_c = adj_all.coalesce()
    src = adj_c.indices()[0]
    dst = adj_c.indices()[1]
    keep = src < dst
    du = uv_all[dst[keep], 0] - uv_all[src[keep], 0]
    dv = uv_all[dst[keep], 1] - uv_all[src[keep], 1]
    angles = torch.atan2(dv, du)
    # Quantize to 10-degree bins
    angle_bins = ((angles / math.pi * 180 + 180) / 10).long() % 36
    n_distinct_directions = angle_bins.unique().shape[0]

    # Layer labels for the full graph
    layers = torch.cat([tri_layers, sq_layers + 0, hex_layers + 0])

    return {
        "adj": adj_all,
        "uv": uv_all,
        "types": types_all,
        "layers": layers,
        "degrees": degrees,
        "n": n_all,
        "n_edges": n_edges,
        "n_distinct_directions": n_distinct_directions,
        "seed_counts": {"tri": n_tri, "sq": n_sq, "hex": n_hex},
        "extruded_counts": {"tri": n_tri_e, "sq": n_sq_e, "hex": n_hex_e},
        "n_extrusion_layers": n_extrusion_layers,
    }
