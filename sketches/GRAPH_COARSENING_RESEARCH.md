# Graph Coarsening for Fast Fiedler Computation: Research Summary

## Executive Summary

The core insight: **Real image graphs aren't random graphs**. They have structure (homogeneous regions, coherent edges) that can be exploited to reduce computation from O(n) to O(coarse graph size).

## Most Promising Approaches (Ranked by Applicability)

### Tier 1: Ready to Implement

#### 1. GRACLUS - Avoid Eigenvectors Entirely
**Key insight**: Normalized cut objective is mathematically equivalent to weighted kernel k-means. Optimize the SAME objective without computing eigenvectors.

- **Paper**: Dhillon, Guan, Kulis. "Weighted Graph Cuts without Eigenvectors" (IEEE TPAMI 2007)
- **Complexity**: O(|E|) per iteration
- **Quality**: 67% of images achieved lower normalized cut than spectral methods
- **Speed**: 3.2x faster than spectral on average
- **Implementation**: https://www.cs.utexas.edu/~dml/Software/graclus.html (C++)

**Why promising**: Completely sidesteps eigenvector computation while optimizing the exact same objective. For image segmentation (our use case), this is likely the fastest path to production.

#### 2. Local Variation Coarsening (Loukas)
**Key insight**: Coarsen by contracting edges with smallest "local variation" - an upper bound on spectral approximation error.

- **Paper**: Loukas. "Graph Reduction with Spectral and Cut Guarantees" (JMLR 2019)
- **Paper**: Jin, Loukas, JaJa. "Graph Coarsening with Preserved Spectral Properties" (AISTATS 2020)
- **Complexity**: O(|E| log |E|)
- **Guarantees**: Eigenvalues of coarse graph approximate original within ε
- **Implementation**: https://github.com/loukasa/graph-coarsening (Python)

**Why promising**: Explicit theoretical guarantees on eigenvector preservation. 70% reduction achievable while maintaining ε < 1.

#### 3. GRASS (Graph Spectral Sparsifier)
**Key insight**: Build low-stretch spanning tree, then recover "spectrally critical" off-tree edges.

- **Authors**: Feng et al. (Michigan Tech)
- **Papers**: DAC 2016, 2018; IEEE TCAD 2020
- **Complexity**: Near-linear
- **Output**: Very sparse (0.02|V| off-tree edges to preserve first 30 eigenvalues)
- **Implementation**: https://sites.google.com/mtu.edu/zhuofeng-graphspar/home

**Why promising**: Explicitly designed to preserve first k eigenvalues/eigenvectors including Fiedler.

### Tier 2: GPU Acceleration (If Exact Eigenvectors Needed)

#### 4. GPU LOBPCG (cuGraph)
**Key insight**: LOBPCG is better than Lanczos for GPU parallelization due to blocked operations.

- **Implementation**: RAPIDS cuGraph `spectralBalancedCutClustering()`
- **Speedup**: 7x over CPU, 100-140 GFlop/s vs 20-25 for Lanczos
- **Availability**: `pip install cugraph` (requires NVIDIA GPU)

#### 5. Sphynx (Multi-GPU)
- **Paper**: Deveci et al. (Sandia, 2021)
- **Implementation**: Trilinos framework, C++
- **Advantage**: Multi-GPU scaling, better than cuGraph on irregular graphs

### Tier 3: Theoretical (Requires Implementation)

#### 6. Cascadic Multigrid for Fiedler
**Key insight**: Heavy-edge coarsening + cascade (one-way) refinement, specifically designed for Fiedler.

- **Paper**: Urschel, Xu, Hu, Zikatanov (J. Comp. Math 2015)
- **Complexity**: Near O(n)
- **Availability**: MATLAB code from authors

#### 7. Spielman-Srivastava Spectral Sparsification
**Key insight**: Sample edges by effective resistance. Preserves ALL eigenvalues.

- **Paper**: Spielman & Srivastava (STOC 2008)
- **Complexity**: O(m log n / ε²)
- **Implementation**: PyGSP `pygsp.reduction.graph_sparsify(G, epsilon)`

---

## Recommended Implementation Strategy

### For 100Hz Real-Time Goal:

```
Option A: GRACLUS Path (Avoid Eigenvectors)
1. Build image graph (O(n))
2. GRACLUS weighted kernel k-means (~O(n))
3. Use cluster assignments directly for gating/correspondence

Option B: Coarsening Path
1. Build image graph (O(n))
2. Local variation coarsening to ~500 nodes (O(n log n))
3. Dense eigensolve on 500x500 matrix (~1ms)
4. Map Fiedler back to pixels (O(n))

Option C: GPU Path
1. Build image graph, transfer to GPU
2. cuGraph LOBPCG (~3-5ms on GPU)
3. Full quality Fiedler
```

### Estimated Timings (256x256 = 65536 pixels):

| Method | Expected Time | Quality |
|--------|--------------|---------|
| GRACLUS | 2-5ms | Equivalent objective |
| Local Variation + dense solve | 5-10ms | ε-approximate |
| GPU LOBPCG | 3-5ms | Exact |
| Current Lanczos (CPU) | 15ms | Exact |

---

## Code to Try First

### 1. PyGSP Coarsening (Easiest Start)
```python
from pygsp import graphs, reduction

# Build image graph
G = graphs.Grid2d(H, W)
G.set_coordinates(...)  # Set edge weights from image

# Coarsen
G_coarse, Gc_indices, Gc_weights = reduction.graph_sparsify(G, epsilon=0.5)

# Or Kron reduction
G_kron = reduction.kron_reduction(G, boundary_indices)
```

### 2. Loukas Graph Coarsening
```python
# From https://github.com/loukasa/graph-coarsening
from graph_coarsening import coarsen

C, Gc, Call, Gall = coarsen(G, r=0.5, method='variation_neighborhoods')
# C is coarsening matrix, Gc is coarse graph
# Compute Fiedler on Gc, lift back via C
```

### 3. cuGraph (If GPU Available)
```python
import cugraph
import cudf

# Build edge list
edges_df = cudf.DataFrame({'src': src, 'dst': dst, 'weight': weights})
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight')

# Spectral clustering (computes eigenvectors internally)
labels = cugraph.spectralBalancedCutClustering(G, num_clusters=2)
```

---

## Citation List

### Spectral Sparsification
1. Spielman & Srivastava. "Graph Sparsification by Effective Resistances" SIAM J. Computing 2011. [arXiv:0803.0929](https://arxiv.org/abs/0803.0929)
2. Batson, Spielman, Srivastava. "Twice-Ramanujan Sparsifiers" SIAM J. Computing 2012. [arXiv:0808.0163](https://arxiv.org/abs/0808.0163)

### Graph Coarsening with Spectral Preservation
3. Loukas. "Graph Reduction with Spectral and Cut Guarantees" JMLR 2019. [arXiv:1808.10650](https://arxiv.org/abs/1808.10650)
4. Jin, Loukas, JaJa. "Graph Coarsening with Preserved Spectral Properties" AISTATS 2020. [PMLR](https://proceedings.mlr.press/v108/jin20a.html)
5. Feng et al. "GRASS: Graph Spectral Sparsification" DAC 2016-2020. [Project](https://sites.google.com/mtu.edu/zhuofeng-graphspar/home)

### Avoiding Eigenvectors
6. Dhillon, Guan, Kulis. "Weighted Graph Cuts without Eigenvectors" IEEE TPAMI 2007. [Software](https://www.cs.utexas.edu/~dml/Software/graclus.html)

### Multigrid Methods
7. Urschel, Xu, Hu, Zikatanov. "Cascadic Multigrid for Fiedler Vector" J. Comp. Math 2015. [arXiv:1412.0565](https://arxiv.org/abs/1412.0565)
8. Livne & Brandt. "LAMG: Lean Algebraic Multigrid" SIAM J. Sci. Comp. 2012. [arXiv:1108.1310](https://arxiv.org/abs/1108.1310)

### GNN Pooling with Spectral Properties
9. Bianchi, Grattarola, Alippi. "MinCutPool: Spectral Clustering with GNNs" ICML 2020. [arXiv:1907.00481](https://arxiv.org/abs/1907.00481)
10. Ying et al. "DiffPool: Hierarchical Graph Representation Learning" NeurIPS 2018. [arXiv:1806.08804](https://arxiv.org/abs/1806.08804)

### GPU Implementations
11. NVIDIA. "Fast Spectral Graph Partitioning on GPUs" 2016. [Blog](https://developer.nvidia.com/blog/fast-spectral-graph-partitioning-gpus/)
12. Deveci et al. "Sphynx: Multi-GPU Graph Partitioner" 2021. [arXiv:2105.00578](https://arxiv.org/abs/2105.00578)

### Randomized Methods
13. Halko, Martinsson, Tropp. "Finding Structure with Randomness" SIAM Review 2011. [arXiv:0909.4061](https://arxiv.org/abs/0909.4061)
14. Fowlkes et al. "Spectral Grouping Using Nystrom" IEEE TPAMI 2004. [Berkeley](https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf)

### Classic References
15. Shi & Malik. "Normalized Cuts and Image Segmentation" IEEE TPAMI 2000.
16. Karypis & Kumar. "METIS: Multilevel Graph Partitioning" SIAM J. Sci. Comp. 1998.

---

## Next Steps

1. **Immediate**: Try PyGSP coarsening or Loukas implementation on test images
2. **If GPU available**: Benchmark cuGraph vs current Lanczos
3. **For production**: Implement GRACLUS-style kernel k-means (avoids eigenvectors entirely)
4. **Research deep-dive**: Read Loukas 2019 for theoretical understanding of what coarsening can/cannot preserve
