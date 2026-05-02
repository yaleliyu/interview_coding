# NumPy Vectorization Cheatsheet

Quick reference for replacing Python loops with array operations. Assumes `import numpy as np`.

### Shape conventions used in this cheatsheet

| Symbol | Meaning |
|--------|---------|
| `N`, `M` | number of items / rows in two collections (e.g. samples, queries) |
| `D` | feature dimension |
| `C` | number of classes / channels |
| `G` | number of groups |
| `K` | rank, intermediate dim, or top-k size |
| `B` | batch size |
| `H`, `W` | image / window height and width |

Each section starts with a `# Setup:` line declaring the shapes assumed in its examples.

---

## Broadcasting in 30 seconds

When operating on two arrays, NumPy compares shapes from the **right**. Two dimensions are compatible if they are equal, or one of them is 1.

```text
A      [N, 1, 4]
B      [   M, 4]   ← right-aligned: [1, M, 4]
A * B  [N, M, 4]
```

Add a singleton axis to make broadcasting do what you want:

```python
x[:, None]    # shape [N]   → [N, 1]
x[None, :]    # shape [N]   → [1, N]
x[..., None]  # add axis at the end
np.expand_dims(x, axis=1)  # equivalent, more explicit
```

---

## Pairwise patterns (the broadcast trick)

Most "for every i, for every j" computations collapse to a `[N, 1, ...]` × `[1, M, ...]` broadcast.

```python
# Setup: a shape [N, D], b shape [M, D]

# pairwise differences:           [N, M, D]
diff = a[:, None, :] - b[None, :, :]

# pairwise squared distance:      [N, M]
d2   = np.sum(diff ** 2, axis=-1)

# pairwise dot product:           [N, M]
dot  = a @ b.T

# pairwise cosine similarity:     [N, M]
cos  = (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None]
                    * np.linalg.norm(b, axis=1)[None, :])

# elementwise max/min over pairs:
hi   = np.maximum(a[:, None, :], b[None, :, :])  # [N, M, D]
```

For pairwise IoU, see [`vectorization_problems.md`](./vectorization_problems.md).

---

## Reductions along axes

A reduction collapses one or more axes of an array. The mental model:

> **`axis=k` means the kth axis disappears** (unless `keepdims=True`). Everything else stays.

### The `axis` parameter

An axis is just a numbered dimension of the array. For `arr` of shape `[A0, A1, A2]`:
- `axis=0` refers to the first dimension (`A0`),
- `axis=1` refers to the second (`A1`),
- `axis=-1` refers to the last (`A2`) — negative indices count from the end, same as Python lists.

When you pass `axis=k` to a reduction, NumPy iterates over every other axis, applies the reduction along axis `k`, and drops axis `k` from the output shape:

```python
# Setup: arr shape [4, 5, 6]

arr.sum(axis=0).shape   # [5, 6]   ← axis 0 dropped
arr.sum(axis=1).shape   # [4, 6]   ← axis 1 dropped
arr.sum(axis=2).shape   # [4, 5]   ← axis 2 dropped
arr.sum(axis=-1).shape  # [4, 5]   ← same as axis=2
```

Three special cases for `axis`:

| `axis=` value | Meaning                              | Output shape (for `[4, 5, 6]`)                |
|---------------|--------------------------------------|------------------------------------------------|
| `None` (default) | Collapse *every* axis             | `()` — a scalar                                |
| an `int`      | Collapse just that one axis           | original shape minus that dim                  |
| a `tuple` of ints | Collapse all listed axes at once  | original shape minus all listed dims           |

```python
arr.sum()                  # scalar  — same as axis=None
arr.sum(axis=(0, 2)).shape # [5]     — drop axis 0 and 2
```

The reason "axis k disappears" beats "operate on axis k" as a mental model: it tells you the output shape directly without having to re-derive what gets summed.

### 2-D example

```python
# Setup: x shape [N, D]    (N rows, D columns)

x.sum(axis=0).shape   # [D]   ← rows axis collapsed (sum down each column)
x.sum(axis=1).shape   # [N]   ← cols axis collapsed (sum across each row)
x.sum().shape         # ()    ← scalar; both axes collapsed
```

A common confusion: `axis=0` does *not* mean "sum the rows" — it means "the row axis disappears", which results in summing *down* each column.

### 3-D example to lock it in

```python
# Setup: cube shape [B, H, W]

cube.sum(axis=0).shape       # [H, W]
cube.sum(axis=(1, 2)).shape  # [B]      per-batch total
cube.mean(axis=-1).shape     # [B, H]   collapse last axis
```

### Common reductions

```python
# Setup: x shape [N, D]

x.sum(axis=1)         # [N] — sum per row
x.mean(axis=1)        # [N]
x.max(axis=1)         # [N] — max value per row
x.min(axis=1)         # [N]
x.std(axis=1)         # [N] — std (ddof=0 by default; pass ddof=1 for sample std)
x.var(axis=1)         # [N]
x.prod(axis=1)        # [N]
np.median(x, axis=1)  # [N] — only available as np function, not method
np.percentile(x, 95, axis=1)  # [N]
```

### Boolean reductions

```python
# Setup: mask shape [N, D], bool

mask.any(axis=1)              # [N] bool — any True per row
mask.all(axis=1)              # [N] bool — all True per row
mask.sum(axis=1)              # [N] int  — count of True per row
np.count_nonzero(mask, axis=1)  # alias for the count above
```

### Argmax / argmin

```python
# Setup: x shape [N, D]

x.argmax(axis=1)              # [N] int — column index of the row's max
x.argmin(axis=1)              # [N] int — column index of the row's min
```

Get index *and* value in one pass — see [Fancy indexing & gathering](#fancy-indexing--gathering) for the `take_along_axis` pattern.

### The `keepdims` parameter

Default is `False` — the reduced axis vanishes from the output. With `keepdims=True`, NumPy keeps a length-1 placeholder axis where the collapse happened, so the result has the same number of dimensions as the input.

```python
# Setup: x shape [N, D]

x.mean(axis=1).shape                 # [N]       — axis 1 gone
x.mean(axis=1, keepdims=True).shape  # [N, 1]    — axis 1 preserved as size 1
x.mean(axis=0, keepdims=True).shape  # [1, D]    — axis 0 preserved as size 1
x.mean(keepdims=True).shape          # [1, 1]    — both axes preserved
```

**Why it matters: broadcasting back against the original.** NumPy's broadcasting rules align shapes from the right and require either equal or size-1 dimensions. If the reduced axis is missing entirely, broadcasting fails or aligns the wrong axes. Compare:

```python
# Setup: x shape [N, D]

# WORKS — keepdims gives [N, 1], broadcasts cleanly with [N, D]:
x_centered = x - x.mean(axis=1, keepdims=True)

# FAILS — x.mean(axis=1) is [N], aligns to the column axis, not the row axis:
x_centered = x - x.mean(axis=1)        # ValueError unless N == D
```

You can do it without `keepdims` by re-inserting the axis manually, but it's verbose and easy to get wrong:

```python
x_centered = x - x.mean(axis=1)[:, None]   # [N] → [N, 1] then broadcast
```

**When NOT to use `keepdims`:**

- The reduction is the final answer (a per-row total, a per-class accuracy) and you don't intend to broadcast against the original shape.
- You want a scalar — `arr.sum()` returns `0.5`; `arr.sum(keepdims=True)` returns `array([[0.5]])`, a 2-D array with one element. The latter is rarely what you want for a scalar result.

**Rule of thumb:** if the next thing you do with the reduction is an arithmetic op against the original array, use `keepdims=True`. Otherwise leave it `False`.

### Multi-axis reductions

Pass a tuple of axes to collapse several at once:

```python
# Setup: img shape [H, W, C]

img.mean(axis=(0, 1))   # [C]  — per-channel mean over all pixels
img.sum(axis=(0, 1))    # [C]  — per-channel total
img.max(axis=(0, 1))    # [C]  — per-channel max
```

### Conditional / NaN-safe reductions

```python
# Setup: x shape [N, D] with possible NaNs

np.nansum(x, axis=1)    # ignore NaNs in the sum
np.nanmean(x, axis=1)   # ignore NaNs in the mean
np.nanmax(x, axis=1)    # ignore NaNs in the max

# Mask-based reduction (sum only where condition holds):
np.where(mask, x, 0).sum(axis=1)
```

### Cumulative & windowed reductions (no axis collapsed)

```python
# Setup: x shape [N, D]

np.cumsum(x, axis=0)        # [N, D] — running sum down rows
np.cumprod(x, axis=0)       # [N, D]
np.diff(x, axis=0)          # [N-1, D] — consecutive differences

# Reduce in groups defined by start indices:
np.add.reduceat(x, [0, 3, 7], axis=0)   # 3 partial sums over row ranges
```

### Common patterns

```python
# Row-wise normalize so each row sums to 1:
x / x.sum(axis=1, keepdims=True)

# Column-wise standardize (zero mean, unit variance per column):
(x - x.mean(axis=0)) / x.std(axis=0)

# Mean over a masked subset:
mean_per_row = (x * mask).sum(axis=1) / mask.sum(axis=1)
```

---

## Boolean masking & conditional logic

```python
# Setup: x, a, b shape [N, D]; mask shape [N, D] (bool, broadcastable to x)

mask = x > 0                      # bool array, same shape as x
x[mask]                           # 1-D array of selected elements (copy)
x[mask] = 0                       # in-place assign

np.where(mask, a, b)              # [N, D] — elementwise select between a / b
np.where(mask)                    # tuple of index arrays where True
np.select([c1, c2], [v1, v2], default=v0)  # multi-branch
np.clip(x, lo, hi)                # [N, D] — clamp values into [lo, hi]
```

---

## Fancy indexing & gathering

```python
# Setup: x shape [N, D]; rows shape [K] int; cols shape [K] int; idx shape [N] int

rows = np.array([0, 2, 5])         # K = 3
x[rows]                            # [3, D] — picks rows 0, 2, 5
x[rows, cols]                      # [K]    — 1-D pick of (row[i], col[i]) pairs

# gather one value per row using a column index per row:
np.take_along_axis(x, idx[:, None], axis=1).squeeze(1)   # [N]

# argmax + value in one go:
best_idx = x.argmax(axis=1)                                            # [N]
best_val = np.take_along_axis(x, best_idx[:, None], axis=1).squeeze(1) # [N]
# equivalent for the value alone:
best_val = x.max(axis=1)                                               # [N]
```

---

## Top-k per row

```python
# Setup: x shape [N, M] (e.g. similarity scores); k = number of top entries per row

k = 5
# unsorted top-k indices (faster than full sort):
top_k_idx = np.argpartition(-x, k, axis=1)[:, :k]   # [N, k]

# sort within the top-k if needed:
top_k_idx = np.take_along_axis(
    top_k_idx,
    np.argsort(-np.take_along_axis(x, top_k_idx, axis=1), axis=1),
    axis=1,
)                                                   # [N, k] sorted by score desc
```

---

## Group-by aggregations (no loops)

```python
# group_id ∈ [0, G), values shape [N]:

counts = np.bincount(group_id, minlength=G)                # [G]
sums   = np.bincount(group_id, weights=values, minlength=G)
means  = sums / np.maximum(counts, 1)

# scatter-add for non-integer or multi-D values:
out = np.zeros(G)
np.add.at(out, group_id, values)         # unbuffered, handles repeats correctly
```

---

## Numerically stable softmax

```python
# Setup: z shape [N, C] (logits over C classes for N items)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)        # [N, C] — shift for stability
    e = np.exp(z)                                   # [N, C]
    return e / e.sum(axis=axis, keepdims=True)      # [N, C]
```

Same trick for log-sum-exp (returns shape `[N]` if axis collapses C):

```python
m = z.max(axis=axis, keepdims=True)                # [N, 1]
lse = m.squeeze(axis) + np.log(np.exp(z - m).sum(axis=axis))   # [N]
```

---

## One-hot, indicator masks

```python
y = np.array([1, 0, 2, 1])          # labels, shape [N]
C = 3                                # num classes

one_hot = np.eye(C)[y]              # [N, C]
# or:
one_hot = (y[:, None] == np.arange(C)[None, :]).astype(np.float32)
```

---

## Sliding windows / patches

```python
# Setup: x shape [N] (1-D signal); img shape [H_img, W_img] (single-channel image);
#        W = window length; H, W = patch height/width

from numpy.lib.stride_tricks import sliding_window_view

# 1-D windows of size W:                                       → [N - W + 1, W]
sliding_window_view(x, W)

# 2-D patches of size H×W from an image:
sliding_window_view(img, (H, W))     # → [H_img - H + 1, W_img - W + 1, H, W]
```

Avoid hand-rolled `as_strided` — it's easy to corrupt memory.

---

## Linear algebra shortcuts

```python
# Setup: A shape [M, K]; B shape [K, P]; b shape [M]; x shape [N, D];
#        a shape [N, D], b_vec shape [M, D] for the einsum examples

A @ B                        # [M, P] — matmul (also np.matmul, np.dot for 2-D)
np.linalg.inv(A)             # [M, M] — only when A is square
np.linalg.solve(A, b)        # [M] — solves A @ y = b; prefer over inv(A) @ b
np.linalg.lstsq(A, b)        # least squares
np.linalg.norm(x, axis=-1)   # [N] — per-row L2 norm

# einsum for explicit contractions:
np.einsum('ij,jk->ik', A, B)             # [M, P] = A @ B
np.einsum('bij,bjk->bik', A, B)          # [B, M, P] — batched matmul
np.einsum('nd,md->nm', a, b_vec)         # [N, M] — pairwise dot
np.einsum('ij,ij->i', a, a)              # [N] — row-wise dot (squared norm)
```

`einsum` is verbose but unambiguous and often the only readable way to express batched/multi-axis contractions.

---

## Performance tips

- **In-place where possible:** `x += 1` allocates nothing; `x = x + 1` allocates a new array.
- **Use `out=` arg** for ufuncs to write into a preallocated buffer.
- **Drop precision:** `float32` halves memory and is usually enough.
- **Avoid `np.append` / `np.concatenate` in a loop** — preallocate once with the final shape.
- **Materialize lists into arrays once**, not per iteration.
- **Use views, not copies:** slicing returns a view; fancy indexing returns a copy.
- **Watch broadcasting memory:** `[N, 1, D] * [1, M, D]` allocates `[N, M, D]`. For large `N, M` chunk along one axis.
- **Profile, don't guess:** `%timeit` in IPython, `np.show_config()` to confirm BLAS backing.

---

## When *not* to vectorize

- The loop runs a tiny number of times (e.g. once per epoch).
- Each iteration mutates state in a way that is genuinely sequential (e.g. sampling with rejection that depends on prior draws).
- The intermediate broadcast would not fit in memory (chunk + Python loop over chunks is fine).
- The vectorized version is so cryptic the next reader will rewrite it. Readability wins for cold paths.

For hot inner loops where vectorization isn't a clean fit: **Numba** (`@njit`) or **Cython** are usually faster than NumPy and easier to write than `as_strided` gymnastics.

---

## Anti-patterns to avoid

```python
# BAD — Python loop over array elements
for i in range(len(x)):
    y[i] = x[i] ** 2 + 3
# GOOD
y = x ** 2 + 3

# BAD — np.append in loop is O(N²)
out = np.array([])
for v in source:
    out = np.append(out, v)
# GOOD
out = np.array(list(source))            # if source is a generator
# or preallocate:
out = np.empty(N); out[:] = source

# BAD — comparing arrays with `==` and `if`
if x == y: ...           # raises if shapes broadcast > 1 elt
# GOOD
if np.array_equal(x, y): ...
if (x == y).all(): ...

# BAD — boolean indexing then mutating original via copy
x[x > 0][:] = 0          # writes to a copy, not x
# GOOD
x[x > 0] = 0
```

---

## Quick lookup: "I want to..."

| Goal | Function |
|------|----------|
| Repeat a value `n` times | `np.full(n, val)`, `np.broadcast_to` |
| Insert a singleton axis | `arr[:, None]` or `np.expand_dims` |
| Drop a singleton axis | `arr.squeeze()` |
| Stack along new axis | `np.stack` |
| Concatenate along existing axis | `np.concatenate` |
| Find first index matching a condition | `np.argmax(cond)` (returns 0 if none — verify with `cond.any()`) |
| Binary search into a sorted array | `np.searchsorted` |
| Histogram / counts | `np.bincount`, `np.histogram` |
| Unique values + counts | `np.unique(x, return_counts=True)` |
| Sort indices, not values | `np.argsort` |
| Sort by multiple keys | `np.lexsort` |
| Cartesian product of two 1-D arrays | `np.ix_(a, b)` or broadcast `a[:, None], b[None, :]` |
