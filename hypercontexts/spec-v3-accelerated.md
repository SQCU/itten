# Spec V3: Accelerated Spectral Ops

## Non-Negotiable Requirements

1. **Vectorized**: No Python for-loops over nodes/pixels
2. **CUDA-first**: torch.sparse, fallback to CPU torch, fail loudly if neither
3. **Structured concurrency**: trio, not threading/multiprocessing
4. **10-second timeout**: dump partial work and exit gracefully
5. **Saturate hardware**: >80% GPU utilization or >80% CPU cores

---

## Core Module: `spectral_ops_accel.py`

Replace the dict-based sparse ops with torch.sparse:

```python
import torch
import trio

class SparseLaplacian:
    """GPU-accelerated local Laplacian operations."""

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            print("WARNING: CUDA not available, falling back to CPU", file=sys.stderr)

    def from_graph_region(self, graph, seed_nodes, radius):
        """
        Materialize local Laplacian as sparse tensor.

        This is O(|local_nodes| * avg_degree), not O(n²).
        """
        # Expand neighborhood
        nodes = expand_neighborhood(graph, seed_nodes, radius)
        node_list = list(nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        # Build COO sparse tensor
        rows, cols, vals = [], [], []
        degrees = torch.zeros(n, device=self.device)

        for i, node in enumerate(node_list):
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor in node_to_idx:
                    j = node_to_idx[neighbor]
                    rows.append(i)
                    cols.append(j)
                    vals.append(-1.0)
                    degrees[i] += 1

        # Add diagonal
        for i in range(n):
            rows.append(i)
            cols.append(i)
            vals.append(float(degrees[i]))

        indices = torch.tensor([rows, cols], device=self.device)
        values = torch.tensor(vals, dtype=torch.float32, device=self.device)

        L = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        return L, node_list, node_to_idx

    def lanczos_fiedler(self, L, k=30):
        """
        Lanczos iteration entirely on GPU.
        Returns (fiedler_vector, lambda_2).
        """
        n = L.shape[0]

        # Random init, project out constant
        v = torch.randn(n, device=self.device)
        v = v - v.mean()
        v = v / v.norm()

        V = [v]
        alphas = []
        betas = [0.0]

        for i in range(min(k, n - 1)):
            # w = L @ v  (sparse matmul on GPU)
            w = torch.sparse.mm(L, V[-1].unsqueeze(1)).squeeze()

            alpha = V[-1].dot(w)
            alphas.append(alpha.item())

            w = w - alpha * V[-1]
            if i > 0:
                w = w - betas[-1] * V[-2]

            # Project out constant
            w = w - w.mean()

            beta = w.norm()
            if beta < 1e-10:
                break

            betas.append(beta.item())
            V.append(w / beta)

        # Solve tridiagonal (small, can do on CPU)
        T = torch.zeros(len(alphas), len(alphas))
        for i, a in enumerate(alphas):
            T[i, i] = a
            if i > 0:
                T[i, i-1] = betas[i]
                T[i-1, i] = betas[i]

        eigenvalues, eigenvectors = torch.linalg.eigh(T)

        # Find smallest non-trivial
        for idx in range(len(eigenvalues)):
            if eigenvalues[idx] > 1e-6:
                break

        lambda2 = eigenvalues[idx].item()

        # Reconstruct Fiedler on GPU
        y = eigenvectors[:, idx]
        fiedler = torch.zeros(n, device=self.device)
        for i, coeff in enumerate(y):
            if i < len(V):
                fiedler += coeff * V[i]

        return fiedler, lambda2
```

---

## Structured Concurrency: trio Pattern

```python
import trio

async def compute_expansion_map_parallel(graph, nodes, radius=2, lanczos_iters=15):
    """
    Compute expansion estimates for all nodes in parallel with timeout.
    """
    results = {}
    laplacian = SparseLaplacian()

    async def compute_one(node, task_status=trio.TASK_STATUS_IGNORED):
        task_status.started()
        L, node_list, _ = laplacian.from_graph_region(graph, [node], radius)
        _, lambda2 = laplacian.lanczos_fiedler(L, k=lanczos_iters)
        results[node] = lambda2

    partial_results = {}

    with trio.move_on_after(10) as cancel_scope:  # 10 second timeout
        async with trio.open_nursery() as nursery:
            for node in nodes:
                await nursery.start(compute_one, node)
                partial_results = dict(results)  # snapshot for partial dump
                await trio.sleep(0)  # yield point

    if cancel_scope.cancelled_caught:
        print(f"TIMEOUT: Computed {len(partial_results)}/{len(nodes)} nodes")
        save_partial_work(partial_results)
        return partial_results

    return results


async def main():
    # ... setup ...
    results = await compute_expansion_map_parallel(graph, nodes)
```

---

## Vectorized Rendering

```python
def render_egg_vectorized(height_field, size=512):
    """
    Fully vectorized egg rendering - no loops.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate all rays at once
    y_coords, x_coords = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij'
    )

    # Normalize to [-1, 1]
    x = (x_coords - size/2) / (size/2) * 1.2
    y = (y_coords - size/2) / (size/2) * 1.2

    # Ray-egg intersection (vectorized quadratic solve)
    # ... all operations are tensor ops, no loops

    # Sample texture at UV coordinates (vectorized gather)
    u_idx = (u * height_field.shape[1]).long().clamp(0, height_field.shape[1]-1)
    v_idx = (v * height_field.shape[0]).long().clamp(0, height_field.shape[0]-1)

    sampled = height_field[v_idx, u_idx]  # pure indexing, no loop

    return sampled
```

---

## Timeout with Partial Work Dump

```python
import json
from pathlib import Path

def save_partial_work(results, path="partial_work.json"):
    """Dump whatever we have so far."""
    Path(path).write_text(json.dumps({
        "completed_nodes": len(results),
        "results": {str(k): v for k, v in results.items()},
        "timestamp": time.time(),
        "status": "partial"
    }))
    print(f"Partial work saved to {path}")


def load_partial_work(path="partial_work.json"):
    """Resume from partial work if available."""
    if Path(path).exists():
        data = json.loads(Path(path).read_text())
        if data["status"] == "partial":
            return {eval(k): v for k, v in data["results"].items()}
    return {}
```

---

## Expected Structure

```
spectral_accel/
  __init__.py
  sparse_laplacian.py    # torch.sparse Laplacian ops
  lanczos_gpu.py         # GPU Lanczos iteration
  async_compute.py       # trio-based parallel computation
  timeout.py             # early termination + partial dump
  render_vectorized.py   # pure tensor rendering

lattice_extrusion_v3/    # uses spectral_accel
texture_synth_v3/        # uses spectral_accel
```

---

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.0",
    "trio>=0.22",
    "numpy>=1.24",
    "pillow>=10.0",
]
```

---

## Failure Modes

1. **No CUDA**: Warn loudly, fall back to CPU torch (still vectorized)
2. **Timeout (>10s)**: Dump partial results, exit 0 with warning
3. **OOM on GPU**: Catch, fall back to smaller batch size, retry
4. **trio cancelled**: Propagate cleanly, dump partial work

---

## Verification

After implementing, these must pass:

```bash
# Should complete in <10 seconds or dump partial
time uv run python -m texture_synth_v3.cli --demo-a

# Should show >80% GPU utilization
nvidia-smi  # while running

# Should be vectorized (no Python loops in hot path)
grep -r "for.*in.*range" spectral_accel/  # should return nothing
```
