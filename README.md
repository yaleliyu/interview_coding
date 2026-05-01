# Linear & Logistic Regression from Scratch

Minimal NumPy implementations of linear and logistic regression trained with mini-batch SGD on synthetic data. Files:

- `generate_data.py` — synthetic data generators for regression and classification.
- `model.py` — training loops for both models.

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

## Caveats

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
