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

---

## 2. Masked Mean of Agent Trajectories

### Why it matters

Agent tracks in real logs are ragged — occlusion, late entries into the scene, sensor dropouts. Every ML pipeline built on them relies on explicit validity masks. Computing masked statistics cleanly (without division-by-zero and without allocating garbage at masked positions) is bread-and-butter work.

### Task

Implement `masked_mean_xy(trajectories, validity)` that returns the mean `(x, y)` position per agent across time, considering only valid timesteps. For an agent with no valid timesteps, return `(0, 0)`.

### Shapes

| Argument        | Shape         | Meaning                                            |
|-----------------|---------------|----------------------------------------------------|
| `trajectories`  | `[B, A, T, 2]`| batch, agents, time, `(x, y)`                      |
| `validity`      | `[B, A, T]`   | bool — `True` if that timestep is observed         |
| **return**      | `[B, A, 2]`   | mean `(x, y)` per agent over valid timesteps       |

### Examples

**EX 1 · all valid**
```
traj     = [[[[1, 1], [3, 3]]]]   # shape [1, 1, 2, 2]
validity = [[[True, True]]]
→ [[[2.0, 2.0]]]
```

**EX 2 · partial masking**
```
traj     = [[[[0, 0], [10, 10], [999, 999]]]]
validity = [[[True, True, False]]]
→ [[[5.0, 5.0]]]                  # last step ignored
```

**EX 3 · fully masked agent**
```
validity[b, a, :] all False
→ [[[0.0, 0.0]]]
```

### Reference implementation

A basic NumPy version lives in [`vectorization_examples.py`](./vectorization_examples.py). It always materializes a dense `[B, A, T, 2]` masked tensor before reducing — fine for typical batches, but worth revisiting when most of the input is invalid (see below).

### Sparse input: when the basic solution is and isn't enough

"Sparse" can mean two very different things here, with different implications.

#### Scenario A — dense input, sparse mask

`trajectories` is still a fully-allocated `[B, A, T, 2]` array, but most entries of `validity` are `False` (e.g., $T = 1000$ and each agent has only ~10 valid frames).

**The basic solution is fine.** The input itself already costs $\mathcal{O}(B A T)$ memory; you can't beat that by being clever about the reduction. Vectorized sums over masked-out zeros are essentially free in modern memory-bandwidth terms.

A small tweak avoids the intermediate full-size `masked` array (saves one full-size copy):

```python
count  = validity.sum(axis=-1, keepdims=True).astype(float)                 # [B, A, 1]
summed = np.einsum('bati,bat->bai', trajectories, validity.astype(float))   # [B, A, 2]
return summed / np.maximum(count, 1.0)
```

Same compute cost, half the peak memory.

#### Scenario B — input itself is sparse / ragged

You don't actually have a dense `[B, A, T, 2]` tensor — you have a list of observations like `(batch, agent, t, x, y)`, or each agent has its own variable-length track. Padding everything up to $T_{\max}$ is wasteful when the typical track length is much smaller (e.g., median 20, max 1000).

The basic solution is **bad** here, and not for vectorization reasons — it forces you to materialize a dense tensor that dwarfs the actual data. Three better approaches:

**(1) Scatter from a flat observation array.** Store observations as `obs` of shape `[E, 4]`, columns `(batch, agent, x, y)`, where $E$ is total observation count across all agents (no time dim):

```python
bidx, aidx = obs[:, 0].astype(int), obs[:, 1].astype(int)
xy         = obs[:, 2:4]

summed = np.zeros((B, A, 2))
count  = np.zeros((B, A, 1))
np.add.at(summed, (bidx, aidx), xy)
np.add.at(count,  (bidx, aidx, 0), 1.0)

return summed / np.maximum(count, 1.0)
```

Memory and compute become $\mathcal{O}(E)$, not $\mathcal{O}(B \cdot A \cdot T_{\max})$. Note `np.add.at` is unbuffered (handles repeated indices correctly) but slower per-element than dense ops — break-even is around when sparsity gives you >10× fewer entries than the dense form.

**(2) Segment-sum from a CSR-style layout.** If observations are already grouped by agent with offsets:

```python
np.add.reduceat(xy, starts, axis=0)   # one sum per agent
counts = ends - starts
```

Cleaner than scatter when input is already sorted by agent.

**(3) Library support.** [`torch.nested`](https://docs.pytorch.org/docs/stable/nested.html) for ragged batches and [`torch_scatter.scatter_mean`](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) are the standard tools for trajectory / graph workloads on GPU — `scatter_mean` is essentially approach (1) done in one call.

#### Practical recommendation

- If your dataloader emits dense `[B, A, T, 2]` tensors with masks, keep the basic solution. Maybe switch to the `einsum` form to halve peak memory.
- If you have control over the input pipeline and the *typical* number of observations per agent is much smaller than $T_{\max}$, change the **representation** (flat obs + scatter) before optimizing the math. The biggest win is not allocating the dense tensor in the first place.
