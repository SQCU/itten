# MANDATE: Fast Code Only

## The Rule

**Slow code is not to spec. Slow code is to be deleted and replaced with fast code.**

There is no "preserve the slow version." There is no "fallback to Python loops."
There is only: vectorized, sparse, fast.

---

## Hardware Available

- Beefy CPU with many cores
- Assume CUDA available (fail loudly if not, but write for GPU first)

---

## What "Fast" Means

### Vectorized
```python
# WRONG (slow, not to spec, delete this)
for i in range(n):
    for j in range(n):
        result[i, j] = compute(data[i], data[j])

# RIGHT (fast, to spec)
result = compute_vectorized(data[:, None], data[None, :])
```

### Sparse Operations
```python
# WRONG (slow, not to spec, delete this)
L = np.zeros((n, n))
for i, node in enumerate(nodes):
    for j, neighbor in enumerate(graph.neighbors(node)):
        L[i, j] = -1
        L[i, i] += 1

# RIGHT (fast, to spec)
indices = torch.tensor([rows, cols], device='cuda')
values = torch.tensor(vals, device='cuda')
L = torch.sparse_coo_tensor(indices, values, (n, n))
```

### No Python Loops in Hot Paths
```python
# WRONG
for pixel in pixels:
    output[pixel] = kernel @ input[pixel]

# RIGHT
output = torch.sparse.mm(kernel, input)  # one call, all pixels
```

---

## Modules to Rewrite

### 1. spectral_ops.py → spectral_ops_fast.py
- `local_laplacian_matvec` → torch.sparse.mm
- `local_fiedler_vector` → GPU Lanczos with torch tensors
- `local_expansion_estimate` → batched, all nodes at once if possible
- `expand_neighborhood` → can stay Python (graph traversal, not hot path)

### 2. lattice_extrusion_v2/ → rewrite in place
- Delete all `for coord in nodes` loops
- Replace with tensor operations
- Rendering must be pure numpy/torch broadcasting

### 3. texture_synth_v2/ → rewrite in place
- `carrier_graph.py`: Build sparse adjacency as torch.sparse in one pass
- `spectral_modulate.py`: Heat diffusion as sparse matmul iterations
- `render_egg.py`: Already vectorized? Verify. If any loops, delete and vectorize.

---

## Concurrency

Use `trio` for structured concurrency where needed:
```python
import trio

async def compute_with_timeout(work_fn, timeout=10):
    result = None
    with trio.move_on_after(timeout):
        result = await trio.to_thread.run_sync(work_fn)
    if result is None:
        print("TIMEOUT - dumping partial work")
    return result
```

For CPU-parallel numerical work, prefer:
- numpy with MKL/OpenBLAS (auto-parallel)
- torch with multiple CPU threads
- numba with `parallel=True`

NOT threading.Thread, NOT multiprocessing.Pool.

---

## Verification

After rewriting, these must pass:

```bash
# No Python for-loops in hot paths
grep -rn "for.*in range" spectral_ops_fast.py  # must return NOTHING
grep -rn "for node in" lattice_extrusion_v2/   # must return NOTHING
grep -rn "for pixel in" texture_synth_v2/      # must return NOTHING

# Must complete fast
time uv run python -m texture_synth_v2.cli --demo-a  # <5 seconds

# Must use GPU if available
python -c "import torch; print(torch.cuda.is_available())"  # True
```

---

## Exit Hypercontext Requirements

When done, write exit hypercontext proving:
1. Zero Python for-loops in numerical code
2. All Laplacian ops use torch.sparse
3. Benchmark: time before vs time after
4. GPU utilization observed (nvidia-smi or equivalent)

---

## There Is No Alternative

If you find yourself writing `for i in range`, STOP.
Delete what you wrote. Think about the tensor operation.
Write the tensor operation.

Slow code is a bug. Fix the bug.
