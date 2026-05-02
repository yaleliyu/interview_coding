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
