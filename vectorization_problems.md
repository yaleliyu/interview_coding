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

---

## 3. Nearest Valid Agent to Ego

### Why it matters

Picking out the closest neighbor under a validity constraint is the building block for a lot of planner features — "closest lead vehicle", "nearest pedestrian", conflict detection. The trick is the **masked-argmin**: a naive `argmin` will happily pick a masked agent if its garbage coordinates happen to be close.

### Task

Given ego positions and other agents' positions with a validity mask, return the index of the nearest valid agent to ego for each batch element. If no valid agents exist for a batch element, return `-1` for that slot.

### Shapes

| Argument    | Shape       | Meaning                                              |
|-------------|-------------|------------------------------------------------------|
| `ego`       | `[B, 2]`    | `(x, y)` of ego per batch                            |
| `agents`    | `[B, A, 2]` | `(x, y)` of every agent, per batch                   |
| `validity`  | `[B, A]`    | bool — `True` if agent is present                    |
| **return**  | `[B]`       | index into `A` of nearest valid agent, or `-1`       |

### Examples

**EX 1 · one valid**
```
ego      = [[0, 0]]
agents   = [[[1, 0], [100, 100]]]
validity = [[True, False]]
→ [0]
```

**EX 2 · masked one would have won**
```
ego      = [[0, 0]]
agents   = [[[0.1, 0], [5, 5]]]
validity = [[False, True]]
→ [1]                                # not 0 — agent 0 is masked
```

**EX 3 · no valid**
```
validity = [[False, False]]
→ [-1]
```

### Reference implementation

Both the top-1 and top-k variants live in [`vectorization_examples.py`](./vectorization_examples.py) as `nearest_valid_agent` and `nearest_k_valid_agents`.

### Solution: top-1

Three steps, all elementwise / broadcast — no Python loops over agents:

```python
ego_b = ego[:, None, :]                                   # [B, 1, 2]
dist_sq = np.sum((ego_b - agents) ** 2, axis=-1)          # [B, A]
masked_dist = np.where(validity, dist_sq, np.inf)         # [B, A]

any_valid = validity.any(axis=-1)                         # [B]
return np.where(any_valid, np.argmin(masked_dist, axis=-1), -1)
```

Key ideas:

1. **Broadcast the ego against agents.** Inserting a singleton axis (`ego[:, None, :]` → `[B, 1, 2]`) makes the difference with `agents` (`[B, A, 2]`) broadcast cleanly to `[B, A, 2]`. Squaring and summing over the last axis gives the per-agent squared distance `[B, A]`.
2. **Masked-argmin via `+inf` substitution.** This is the central trick. Naively, `argmin` would happily pick a masked agent if its garbage coordinates happened to lie near the ego. Replacing masked entries with `np.inf` guarantees they cannot win the argmin. (Use `np.inf`, not a finite sentinel like `1e10` — finite sentinels can be beaten by real distances in large coordinate systems.)
3. **No-valid-agent guard.** When *every* agent in a batch row is masked, `argmin` still returns a (meaningless) `0`. Compute `validity.any(axis=-1)` separately and overwrite those rows with `-1` via `np.where`. Both branches of `np.where` always evaluate, so this costs an extra full reduction but no extra control flow.

Working with squared distance throughout is intentional — `argmin` of $d^2$ and of $d$ give the same answer, and skipping the `sqrt` saves work.

### Solution: top-k

The top-1 idea generalizes by replacing `argmin` with a partial sort. The masked-distance setup is identical:

```python
ego_b = ego[:, None, :]
dist_sq = np.sum((ego_b - agents) ** 2, axis=-1)          # [B, A]
masked_dist = np.where(validity, dist_sq, np.inf)         # [B, A]

k = min(k, agents.shape[1])                               # clamp if k > A
top_k_idx = np.argsort(masked_dist, axis=-1)[:, :k]       # [B, k]

top_k_dist = np.take_along_axis(masked_dist, top_k_idx, axis=-1)
return np.where(np.isinf(top_k_dist), -1, top_k_idx)
```

Notes:

1. **`-1` slots cover both edge cases for free.** If a batch row has fewer than `k` valid agents (or none at all), the trailing positions in `top_k_idx` correspond to masked agents whose distance is `np.inf`. Gathering those distances back via `take_along_axis` and checking with `np.isinf` lets a single `np.where` clear them all to `-1`. There's no separate "no valid agents" branch.
2. **Clamp `k`.** If the caller asks for more neighbors than there are agent slots, slice `[:, :k]` would just under-fill — explicit `min(k, A)` makes the output shape predictable.
3. **`argsort` vs. `argpartition`.** Full sort is `O(A \log A)` per row. For large `A` and small `k`, swap to `np.argpartition(masked_dist, kth=k, axis=-1)[:, :k]` (`O(A)`), then sort just the `k` selected entries. Worth it only when `A` is in the hundreds or more.
4. **Sort stability.** `np.argsort` defaults to stable sort, so ties (equal distances) are broken by original index — usually the desired behavior. `argpartition` is unstable, so the partition variant tie-breaks arbitrarily.
5. **Returning distances too.** If callers want both indices and distances, gather `top_k_dist` and apply `np.sqrt` *only at the end* — sorting on squared distances gives the same order, so the square root is dead work until the very last step.

---

## 4. minADE over Top-K Predictions

### Why it matters

Motion forecasting models emit multiple hypothesis trajectories per agent with confidence scores. A standard evaluation metric — **minADE@K** — takes the top-K most confident predictions and reports the minimum **average displacement error** against the ground truth. Getting the gather step right, without loops, trips up a lot of candidates.

### Task

Given `K` predicted trajectories per example with confidence scores, and a ground-truth trajectory, compute **minADE over the top-3 most confident predictions** for each example.

ADE = mean L2 distance between a predicted trajectory and the ground truth, averaged across timesteps.

### Shapes

| Argument    | Shape          | Meaning                                                |
|-------------|----------------|--------------------------------------------------------|
| `preds`     | `[B, K, T, 2]` | `K` predicted trajectories per example                 |
| `conf`      | `[B, K]`       | confidence scores; higher = more confident             |
| `gt`        | `[B, T, 2]`    | ground-truth trajectory                                |
| **return**  | `[B]`          | minADE over the top-3 predictions                      |

### Examples

**EX 1 · structure**
```
B = 2, K = 6, T = 12, 2-D positions
→ pick the top-3 by `conf`, compute ADE for each, take min per batch
→ shape [2], float
```

**EX 2 · edge: K = 3**
```
K equals 3 — use all predictions
→ reduces to min ADE over all K predictions per batch
```

### Reference implementation

Both variants live in [`vectorization_examples.py`](./vectorization_examples.py):

- `min_ade_top_k` — canonical minADE@K (`ade.min(axis=-1)`)
- `weighted_ade_top_k` — confidence-weighted ADE (`Σ softmax(top_conf) * ade`)

They share a private helper `_top_k_ade` that does the gather pipeline once; only the final reduction over `k` differs.

### Solution: key takeaways

#### Two related metrics

Once you have `ade` of shape `[B, k]` (mean L2 displacement per top-k prediction), there are two common reductions over the `k` axis:

| Metric              | Formula                                  | Interpretation                                                  |
|---------------------|------------------------------------------|-----------------------------------------------------------------|
| **minADE@K** (canonical) | `ade.min(axis=-1)`                  | optimistic — rewards the model if any of its top-K is close     |
| **Weighted ADE**    | `(softmax(conf_topk) * ade).sum(axis=-1)` | calibrated — penalizes confidence spent on bad predictions     |

Both share the same gather pipeline; only the last reduction differs. The reference implementation shows the weighted form.

#### 1. Gathering with `take_along_axis`

The index array must match the *rank* of the source array. `preds` has shape `[B, K, T, 2]`, but the top-k indices come back as `[B, k]`. To gather along axis 1, expand the index to broadcast over the trailing axes:

```python
conf_idx   = np.argpartition(-conf, kth=k-1, axis=-1)[:, :k]      # [B, k]
pred_top_k = np.take_along_axis(preds, conf_idx[..., None, None], axis=1)  # [B, k, T, 2]
```

This is where most bugs hide. The shape of the *index* must equal the shape of the *output*, with `1`s in the trailing dims that should broadcast.

#### 2. `argpartition` vs. `argsort`

`argpartition` is `O(K)` average via Quickselect (Introselect in NumPy gives `O(K)` worst case). `argsort` is `O(K log K)`. For minADE the top-k set doesn't need to be ordered internally, so `argpartition` is the right call. Only worth it at large `K` (e.g. diffusion models sampling 512+ hypotheses); below ~100, the difference is invisible.

#### 3. `k > K` guard

Always clamp or raise explicitly:

```python
actual_k = min(k, conf.shape[-1])
```

`np.argpartition` with `kth ≥ K` produces silent wrong behavior that's painful to debug. The clamp is a one-liner — never skip it.

#### 4. Weighted variant: shape and order discipline

For the weighted form, compute ADE first as `[B, k]`, then gather the matching top-k confidences (using the **same** index array), softmax those weights, multiply elementwise, sum over `k`:

```python
ade        = dist.mean(axis=-1)                               # [B, k]   temporal mean first
top_conf   = np.take_along_axis(conf, conf_idx, axis=-1)      # [B, k]
weights    = softmax(top_conf)                                # [B, k]
weighted   = (weights * ade).sum(axis=-1)                     # [B]
```

**Order matters.** Weighting before the temporal mean (e.g. multiplying weights into the per-timestep distances) gives a different and incorrect result — the weights are over predictions, not over timesteps.

#### 5. Numerically stable softmax

Always subtract the row max before exponentiating:

```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

Prevents overflow when confidence scores are large or unbounded (logits straight from a model).
