# NumPy Vectorization Cheatsheet

Quick reference for replacing Python loops with array operations. Assumes `import numpy as np`.

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

```python
x.sum(axis=0)       # collapse rows  → shape [D]
x.sum(axis=1)       # collapse cols  → shape [N]
x.mean(axis=-1)     # last axis
x.sum(axis=(0, 2))  # multiple axes

x.sum(axis=1, keepdims=True)  # keeps shape [N, 1] for broadcasting
```

Common: subtract a per-row mean.

```python
x_centered = x - x.mean(axis=1, keepdims=True)
```

---

## Boolean masking & conditional logic

```python
mask = x > 0                      # bool array, same shape as x
x[mask]                           # 1-D view of selected elements
x[mask] = 0                       # in-place assign

np.where(mask, a, b)              # elementwise select between a / b
np.where(mask)                    # tuple of index arrays where True
np.select([c1, c2], [v1, v2], default=v0)  # multi-branch
np.clip(x, lo, hi)
```

---

## Fancy indexing & gathering

```python
rows = np.array([0, 2, 5])
x[rows]                           # picks rows 0, 2, 5
x[rows, cols]                     # 1-D pick of (row[i], col[i])

# gather one value per row using a column index per row:
np.take_along_axis(x, idx[:, None], axis=1).squeeze(1)

# argmax + value in one go:
best_idx = x.argmax(axis=1)
best_val = np.take_along_axis(x, best_idx[:, None], axis=1).squeeze(1)
# equivalent:
best_val = x.max(axis=1)
```

---

## Top-k per row

```python
k = 5
# unsorted top-k indices (faster than full sort):
top_k_idx = np.argpartition(-x, k, axis=1)[:, :k]
# sort within the top-k if needed:
top_k_idx = np.take_along_axis(
    top_k_idx,
    np.argsort(-np.take_along_axis(x, top_k_idx, axis=1), axis=1),
    axis=1,
)
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
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # shift for stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
```

Same trick for log-sum-exp:

```python
m = z.max(axis=axis, keepdims=True)
lse = m.squeeze(axis) + np.log(np.exp(z - m).sum(axis=axis))
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
from numpy.lib.stride_tricks import sliding_window_view

# 1-D windows of size W:           [N - W + 1, W]
sliding_window_view(x, W)

# 2-D patches of size H×W from an image:
sliding_window_view(img, (H, W))   # last two axes contain each patch
```

Avoid hand-rolled `as_strided` — it's easy to corrupt memory.

---

## Linear algebra shortcuts

```python
A @ B               # matmul (also np.matmul, np.dot for 2-D)
np.linalg.inv(A)
np.linalg.solve(A, b)        # prefer over inv(A) @ b
np.linalg.lstsq(A, b)        # least squares
np.linalg.norm(x, axis=-1)   # vector norms

# einsum for explicit contractions:
np.einsum('ij,jk->ik', A, B)             # = A @ B
np.einsum('bij,bjk->bik', A, B)          # batched matmul
np.einsum('nd,md->nm', a, b)             # pairwise dot
np.einsum('ij,ij->i', a, b)              # row-wise dot product
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
