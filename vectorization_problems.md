# Vectorization Programming Problems

A collection of problems that test fluency with vectorized tensor operations — broadcasting, reshaping, and avoiding Python-level loops. The goal in every problem is a fully broadcast NumPy (or PyTorch) solution; nested `for` loops are a red flag.

---

## 1. Pairwise IoU Matrix

### Why it matters

In detection post-processing — matching predictions to ground truth, computing eval metrics, running NMS — you repeatedly need pairwise overlaps between two sets of boxes. Writing this as a nested Python loop is a dead giveaway that someone isn't fluent with tensors yet. A fully broadcast solution is expected.

### Task

Implement `pairwise_iou(boxes_a, boxes_b)` that returns the IoU between every pair of boxes in two sets. Boxes are axis-aligned, in `(x1, y1, x2, y2)` format, with `x2 > x1` and `y2 > y1` for valid boxes.

### Shapes

| Argument   | Shape    | Per-row format         |
|------------|----------|------------------------|
| `boxes_a`  | `[N, 4]` | `(x1, y1, x2, y2)`     |
| `boxes_b`  | `[M, 4]` | `(x1, y1, x2, y2)`     |
| **return** | `[N, M]` | IoU for `(a_i, b_j)` ∈ `[0, 1]` |

### Examples

**EX 1 · identical**
```
a = [[0, 0, 10, 10]]
b = [[0, 0, 10, 10]]
→ [[1.0]]
```

**EX 2 · disjoint**
```
a = [[0, 0, 1, 1]]
b = [[5, 5, 6, 6]]
→ [[0.0]]
```

**EX 3 · half-overlap**
```
a = [[0,  0, 10, 10]]
b = [[5,  5, 15, 15]]
→ [[25 / 175 ≈ 0.1429]]
```

### Reference implementation

A basic NumPy version (returning the per-row best match instead of the full matrix) lives in [`vectorization_examples.py`](./vectorization_examples.py). It works fine for small to medium $N$ and $M$, but does not scale — see the next section.

### Scaling: what to do when $N$ and $M$ are large

The basic broadcast approach materializes intermediate arrays of shape `[N, M, 2]` and `[N, M]`, so memory grows as $\mathcal{O}(NM)$. With float64:

| $N = M$  | one `[N, M]` array | peak (≈3 such arrays) |
|----------|-------------------:|----------------------:|
| 1,000    | 8 MB               | ~50 MB                |
| 10,000   | 800 MB             | ~5 GB                 |
| 100,000  | 80 GB              | infeasible            |

Past a few thousand boxes per side the broadcast approach falls over. Mitigations, in roughly increasing effort:

1. **Chunk along `boxes_a`.** When the function only needs a per-row reduction (argmax / max / top-k), the full $[N, M]$ matrix is never required. Process `boxes_a` in chunks of size $K$ and reduce each $[K, M]$ block immediately. Peak memory drops to $\mathcal{O}(KM)$. Single biggest win for the existing API.
2. **Use float32 instead of float64.** Halves memory; precision is plenty for IoU. Free.
3. **1D pre-filter.** Most pairs don't overlap. A cheap separating-axis check on the $x$ (or both) ranges can skip non-overlapping pairs, but only helps if you can act on the sparsity (e.g. write to a sparse output).
4. **Spatial index.** Build an R-tree or uniform grid over `boxes_b` and query with each `boxes_a`. Complexity drops from $\mathcal{O}(NM)$ to roughly $\mathcal{O}((N+M)\log(N+M) + K)$, where $K$ is the count of actually-overlapping pairs. Libraries: `rtree`, `shapely`, or a hand-rolled grid hash.
5. **Move to GPU.** [`torchvision.ops.box_iou`](https://docs.pytorch.org/vision/main/generated/torchvision.ops.box_iou.html) is the standard. Same broadcast pattern but on GPU memory; handles tens of thousands easily. Combine with chunking if memory is still tight.
6. **Specialized libraries for downstream tasks.** If the goal is NMS or detection eval, [`torchvision.ops.nms`](https://docs.pytorch.org/vision/main/generated/torchvision.ops.nms.html) and `batched_nms` skip building the full IoU matrix entirely.

For most workloads, **(1) + (2)** is a small no-dependency change that buys an order of magnitude. Reach for **(4)** or **(5)** only when $N, M$ are genuinely in the tens of thousands or larger.
