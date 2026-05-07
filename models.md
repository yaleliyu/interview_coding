# Models from Scratch

Minimal NumPy implementations of common ML models, used as references for the math and implementation patterns.

- **Linear regression** and **logistic regression** — gradient-trained, parametric. See *Optimization*, *Linear Regression*, *Logistic Regression*, *Regularization*.
- **K-nearest neighbours (KNN)** — non-parametric, lazy classifier. See *K-Nearest Neighbors*.

Files:

- `generate_data.py` — synthetic data generators for regression and classification.
- `model.py` — training loops and the `KNNClassifier` class.

## Optimization: Mini-batch SGD

Both models are trained by mini-batch stochastic gradient descent. The general recipe: compute the gradient of the loss on a small random subset of the data, then take a step in the direction that decreases the loss.

### The update rule

Given a differentiable loss $\mathcal{L}(w)$, gradient descent updates the parameters by

$$
w \leftarrow w - \eta \, \nabla_w \mathcal{L}(w),
$$

where $\eta > 0$ is the **learning rate**. Intuition: $\nabla_w \mathcal{L}$ points in the direction of fastest *increase* of the loss, so subtracting it moves toward a lower-loss region. A small enough $\eta$ guarantees the loss decreases on convex problems; too large and the iterates can oscillate or diverge.

### Full-batch vs. stochastic vs. mini-batch

- **Full-batch GD** computes $\nabla_w \mathcal{L}$ on all $n$ samples each step. Accurate gradient, but expensive per step.
- **Stochastic GD (SGD)** uses one sample per step. Cheap and noisy; the noise can help escape shallow local minima (less relevant for convex losses, but useful in deep nets).
- **Mini-batch SGD** is the compromise used here: each step uses a random subset of $B$ samples (we use $B = 32$). This keeps gradient estimates reasonably accurate while remaining cheap and providing useful stochasticity.

For a mini-batch indexed by $\mathcal{B}$, the gradient estimate is

$$
\hat{g} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_w \ell_i(w),
$$

and the update becomes $w \leftarrow w - \eta \, \hat{g}$.

### What this looks like in code

```python
idx = np.random.choice(n_samples, batch_size, replace=False)  # sample mini-batch
x   = data[idx, :-1]                                          # inputs
y   = data[idx,  -1]                                          # targets
y_hat   = forward(x, weights)                                 # prediction
gradient = x.T @ (y_hat - y) / batch_size                     # ĝ
weights -= lr * gradient                                      # w ← w − η ĝ
```

The exact form of `forward` is the only thing that differs between the two models:

- **Linear regression:** `forward(x, w) = x @ w`.
- **Logistic regression:** `forward(x, w) = sigmoid(x @ w)`.

Why the gradient takes the same shape $X^\top (\hat{y} - y) / B$ in both cases is explained in each model's theory section below.

## Linear Regression

### Theory

Given a feature matrix $X \in \mathbb{R}^{n \times d}$ and targets $y \in \mathbb{R}^{n}$, we model

$$
\hat{y} = X w
$$

where $X$ is augmented with a leading column of ones so the bias term is absorbed into $w \in \mathbb{R}^{d+1}$. The objective is the mean squared error

$$
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2.
$$

The gradient with respect to $w$ is

$$
\nabla_w \mathcal{L} = \frac{2}{n} X^\top (X w - y).
$$

The factor of 2 is folded into the learning rate, giving the gradient used in code:

$$
g = \frac{1}{B} X^\top (\hat{y} - y).
$$

### Training process

1. Generate `(features, labels, true_weights)` from `generate_regression_data`. Labels are produced as $y = X_{\text{aug}} w_{\text{true}} + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 0.3)$.
2. Augment features with a bias column and stack labels for batched sampling.
3. Initialize weights to ones, set `lr = 0.02`, `batch_size = 32`.
4. Each step: sample a random mini-batch (without replacement), compute predictions and gradient, update `weights -= lr * gradient`.
5. Every 25 steps, print the per-batch MSE.
6. After 1000 steps, print learned vs. true weights.

## Logistic Regression

### Theory

For binary classification with labels $y \in \{0, 1\}^n$, we model the class probability via the sigmoid

$$
\hat{y} = \sigma(z), \qquad z = X w, \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The loss is the average binary cross-entropy

$$
\mathcal{L}(w) = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \big].
$$

A useful identity: $\partial \mathcal{L} / \partial z_i = \hat{y}_i - y_i$. So the gradient w.r.t. $w$ has the same form as linear regression:

$$
\nabla_w \mathcal{L} = \frac{1}{n} X^\top (\hat{y} - y).
$$

### Numerical stability

- **Sigmoid:** `np.exp(-z)` overflows for very negative `z`. We use the branched form `where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))`, which selects whichever branch keeps the exponent argument $\le 0$.
- **Log:** `log(0)` is $-\infty$. Before computing the loss we clip $\hat{y}$ to $[10^{-12}, 1 - 10^{-12}]$.

### Training process

1. Generate `(features, labels, true_weights)` from `generate_classification_data`.
2. Augment features with a bias column and stack labels.
3. Initialize weights to ones, set `lr = 0.01`, `batch_size = 32`.
4. Each step: sample a mini-batch, compute logits $z$, sigmoid $\hat{y}$, gradient $X^\top(\hat{y} - y)/B$, and update.
5. Every 25 steps, print the per-batch cross-entropy (with clipping inside the log).
6. After 1000 steps, print learned vs. true weights.

## Regularization

Both losses above are *unregularized* — the optimizer is free to grow weights as large as the data allows. With limited samples, correlated features, or perfectly separable classes (logistic regression), this can cause overfitting or even divergence. Regularization adds a penalty on the weight magnitude to bias the solution toward simpler models.

Both training functions accept `l1` and `l2` keyword arguments (defaults `0.0`):

```python
train_linear_regression(l2=0.1)
train_logistic_regression(l1=0.05, l2=0.05)  # elastic net
```

### L2 (Ridge)

Add a squared-norm penalty to the loss:

$$
\mathcal{L}_{\text{reg}}(w) = \mathcal{L}(w) + \lambda \, \lVert w \rVert_2^2.
$$

Gradient picks up an extra $2\lambda w$ term, so the update becomes

$$
w \leftarrow w - \eta \big( \hat{g} + 2\lambda w \big) = (1 - 2\eta\lambda)\, w - \eta\, \hat{g}.
$$

This is sometimes called **weight decay**: each step shrinks $w$ toward zero by a constant factor before applying the data gradient. L2 keeps all weights small but rarely makes any exactly zero. For linear regression it has a closed form ($w = (X^\top X + \lambda I)^{-1} X^\top y$). It is equivalent to MAP estimation with a Gaussian prior on $w$.

### L1 (Lasso)

Add an absolute-value penalty:

$$
\mathcal{L}_{\text{reg}}(w) = \mathcal{L}(w) + \lambda \, \lVert w \rVert_1.
$$

The (sub)gradient adds $\lambda \, \mathrm{sign}(w)$. L1 induces *sparsity*: many components of $w$ are driven to exactly zero, so it doubles as feature selection. It corresponds to MAP estimation with a Laplace prior. There is no closed form, and naive subgradient SGD converges slowly near zero — proximal methods (soft-thresholding) are the standard fix.

**Elastic net** combines both: $\lambda_1 \lVert w \rVert_1 + \lambda_2 \lVert w \rVert_2^2$.

### Penalty in the gradient

Both penalties combine into a single `penalty` term and are folded into the gradient before the SGD step:

```python
penalty = 2 * l2 * weights + l1 * np.sign(weights)
penalty[0] = 0  # don't regularize the bias
gradient = x.T @ (y_hat - y) / batch_size + penalty
weights -= lr * gradient
```

Setting `l1=0` gives pure L2, `l2=0` gives pure L1, both nonzero gives elastic net. The line `penalty[0] = 0` implements the "don't penalize the bias" practice noted below.

To make the regularizer's effect visible at log time, report the data loss and the penalty value separately:

```python
reg = l2 * np.sum(weights[1:]**2) + l1 * np.sum(np.abs(weights[1:]))
print(f"step {epoch} data_loss: {data_loss:.4f} reg: {reg:.4f}")
```

### Practical notes

- **Don't penalize the bias.** The bias term shifts the prediction globally; shrinking it toward zero biases predictions toward zero with no benefit. The training loops handle this by zeroing `penalty[0]` before adding to the gradient (see snippet above).
- **Standardize features first.** L1/L2 penalize each weight equally, so feature scale directly affects how strongly each feature is regularized. Standardizing (zero mean, unit variance) puts all features on the same footing.
- **Choose $\lambda$ by cross-validation.** A common sweep is a log-scale grid (e.g. $10^{-4}, 10^{-3}, \dots, 10^{1}$) using held-out validation loss.
- **Same recipe for both models.** The penalty is added to whichever loss you use — MSE for linear regression, cross-entropy for logistic — so the gradient modification above is identical in both training loops.

## K-Nearest Neighbors (KNN)

### Theory

KNN is a **non-parametric, instance-based** classifier — there's no model in the parametric sense. Training is just "store the data"; all the work happens at query time.

To predict the label of a query point $x$:

1. Compute the distance from $x$ to every training point.
2. Take the $k$ closest training points.
3. Output the most frequent label among them (majority vote).

Properties:

- No "training" cost beyond memorising the data: $\mathcal{O}(N \cdot D)$ storage.
- All cost is at query time: $\mathcal{O}(N \cdot D)$ per query.
- No assumptions about data distribution — a *lazy* learner.
- Decision boundaries are arbitrarily complex (driven by data, not parameters).

### Reference implementation

`KNNClassifier` in `model.py`:

```python
clf = KNNClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test, k=5)
```

`predict` is fully vectorised — no Python loops over points or neighbours.

### Implementation walkthrough

```python
# 1. Pairwise squared distances:                            [M, N]
dist_sqr = ((X_test[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=-1)

# 2. Indices of the k nearest training points per query:   [M, k]
top_k = np.argpartition(dist_sqr, kth=k - 1, axis=-1)[:, :k]

# 3. Majority vote via one-hot + sum:
top_k_y = self.y[top_k]                                       # [M, k]
C = int(self.y.max()) + 1
votes = np.eye(C, dtype=np.int64)[top_k_y].sum(axis=1)        # [M, C]
return votes.argmax(axis=-1).astype(self.y.dtype)
```

Three patterns to highlight:

- **Squared distance, not Euclidean.** `argmin` of $\lVert \cdot \rVert^2$ and $\lVert \cdot \rVert$ give the same answer, so `np.sqrt` is dead work. Only re-introduce it when you need actual distance values (thresholds, RBF kernels, returning distances to the caller).
- **`argpartition`, not `argsort`.** $\mathcal{O}(N)$ average via Quickselect vs $\mathcal{O}(N \log N)$. We need the *set* of $k$ nearest, not their order — partial sort is the right primitive.
- **`np.eye(C)[labels]` for voting.** Fancy-indexing the identity matrix one-hot encodes; summing across the $k$ axis gives per-class vote counts. See [`np.eye`, one-hot, and voting](./numpy_vectorization_cheatsheet.md#npeye-one-hot-and-voting) in the cheatsheet.

### Caveats / bugs to fix

- **`k > N` not handled.** `np.argpartition(..., kth=k - 1)` raises `IndexError` for an out-of-range `kth`. Either clip `k = min(k, N)` (silent fallback) or raise a descriptive `ValueError`. The current default of "implicit IndexError" is hard to debug.
- **`y` dtype not validated.** Both `np.bincount` and `np.eye(C)[y]` indexing require **non-negative integer** labels. Negative labels (e.g., `-1`) and float labels will fail loudly or silently corrupt the votes. Add a check in `fit`, or remap labels via `np.unique(y, return_inverse=True)` and store the reverse map for output.
- **`n_classes` not stored.** When the test set's top-$k$ neighbours don't span all training classes, a per-row `np.bincount(row)` (without `minlength`) returns variable-length vote arrays, breaking vectorisation. The fix is to store `self.n_classes = int(y.max()) + 1` in `fit` and pass it as `minlength` (or as `C` in the `np.eye` form). The current code recomputes `C` per call, which is fine but wasteful.
- **Tie-breaking is by smallest class index.** `votes.argmax(axis=-1)` returns the first maximum on ties. Common alternatives: weight votes by inverse distance, or break ties by the closest neighbour's label.
- **`predict` before `fit`.** Calling `predict` before `fit` raises `AttributeError: 'KNNClassifier' object has no attribute 'X'`. A one-line guard or an `__init__` that initialises `self.X = self.y = None` makes the failure mode explicit.

### Complexity

- **Time:** $\mathcal{O}(M \cdot N \cdot D)$, dominated by the pairwise distance computation. `argpartition` is $\mathcal{O}(M \cdot N)$. The vote step is $\mathcal{O}(M \cdot k \cdot C + M \cdot C)$ — typically negligible.
- **Space:** $\mathcal{O}(M \cdot N \cdot D)$ — *not* the distance matrix. The real memory cliff is the broadcast intermediate `X_test[:, None, :] - self.X[None, :, :]`, which materialises a `[M, N, D]` tensor before the squared-sum reduction.

### Scaling

Mitigations to the naive memory and time cost, in order of effort:

1. **Inner-product trick to avoid the `[M, N, D]` intermediate.** Use the identity $\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2 \, a \cdot b$ to compute distances via BLAS:
   ```python
   sq_a  = (X_test ** 2).sum(axis=-1)        # [M]
   sq_b  = (self.X  ** 2).sum(axis=-1)       # [N]
   cross = X_test @ self.X.T                 # [M, N]   ← BLAS
   dist_sqr = sq_a[:, None] + sq_b[None, :] - 2 * cross
   ```
   Same result; only `[M, N]` is allocated, and `@` dispatches to BLAS — typically much faster.

2. **Batch `X_test` (large $M$).** Process queries in chunks to cap peak memory:
   ```python
   batch_size = memory_budget_bytes // (N * D * 8)   # 8 bytes per float64
   ```
   A default of 512–1000 is reasonable. Same time complexity, but bounded memory.

3. **Tree methods (large $N$, low-to-moderate $D$).** Build a spatial index over training data once, query in sublinear time:
   - `sklearn.neighbors.BallTree` and `KDTree` — build $\mathcal{O}(N \log N)$, query $\mathcal{O}(D \log N)$ per point.
   - Worthwhile when $N$ is in the hundreds of thousands or more.

4. **High-dimensional input (curse of dimensionality).** Tree methods degrade to linear scan above roughly 20–30 dimensions because nearly every cell of the tree gets visited. The production answer at scale is **approximate nearest neighbours**:
   - **FAISS** — IVF + product quantisation, GPU support.
   - **HNSW** — hierarchical small-world graphs.
   - **ScaNN** — anisotropic quantisation.
   
   These give good-enough neighbours at orders-of-magnitude lower cost, at the price of recall < 100%.

> **Coarse-to-fine heuristic:** find candidate classes via centroid distance, then run KNN within those. Creative but not exact — breaks down with overlapping or non-convex class geometry. Don't reach for it unless you've measured that approximate neighbours are acceptable for your task.

### Discussion questions

Worth being able to answer cleanly:

- **Why `argpartition` over `argsort`?** $\mathcal{O}(N)$ average vs $\mathcal{O}(N \log N)$. We only need the *set* of $k$ nearest, not their order.
- **When does `sqrt` matter for distance?** Not for `argmin`/`argmax` — square root is monotonic, so ordering is preserved. It matters when you're computing actual distance values (thresholds, RBF kernels, exposed to the caller).
- **What is the full time and space complexity?** Don't forget $D$. Both are $\mathcal{O}(M \cdot N \cdot D)$ with the broadcast form. The inner-product trick keeps time at $\mathcal{O}(M \cdot N \cdot D)$ but drops memory to $\mathcal{O}(M \cdot N)$.
- **When do tree methods stop helping?** Above roughly 20–30 dimensions.
- **Production path for exact KNN at scale?** `sklearn.neighbors.BallTree` (or `KDTree`) for moderate $N$ and $D$; otherwise shard the data.
- **Production path when approximate is acceptable?** FAISS or HNSW.

## Caveats: Linear / Logistic Regression

Issues identified during code review that are not yet fixed:

- **`n_epochs` is misnamed.** Each iteration is a single mini-batch update, so what the code calls 1000 "epochs" is really 1000 *steps*. With 1000 samples and `batch_size = 32`, one true epoch ≈ 31 steps; the loop runs for ≈ 32 epochs. Rename to `n_steps` or restructure as nested epoch/batch loops.
- **Per-batch loss is noisy.** Logging MSE/CE on the current 32-sample batch produces a jittery curve. Computing it on the full dataset at log time would give a smoother and more meaningful signal.
- **Pre-update loss reporting.** The printed loss uses `y_hat` from before the weight update, so it lags the current `weights` by one step. Minor.
- **Wildcard import.** `from generate_data import *` should be replaced with explicit imports.
- **No held-out evaluation.** Only training-set loss is reported. Adequate for a synthetic-data sanity check, not for a real model.
- **Logistic regression: weights are not directly comparable.** Decision boundaries are invariant to positive scaling of `w`, so direction matters but magnitude does not. A cosine similarity between learned and true weights is more informative than printing both.
- **Logistic regression: no accuracy reported.** Cross-entropy on a 32-sample batch is hard to interpret. Adding `acc = np.mean((y_hat > 0.5) == y)` at log time would help.
- **Classification labels are not from a clean logistic model.** `generate_classification_data` adds Gaussian noise in *probability* space and then thresholds at 0.5. Noise can push values outside $[0, 1]$, and the implied generative process is not Bernoulli given a logistic link. Even with infinite data, recovered weights will not perfectly align with `true_weights`.
- **`np.where` sigmoid still evaluates both branches.** The output is correct, but `RuntimeWarning: overflow in exp` may appear. Wrap in `np.errstate(over='ignore')` or use `scipy.special.expit` to silence.
- **Initialization to ones.** Works for these convex problems but is unconventional; zeros or small random values are more typical.
