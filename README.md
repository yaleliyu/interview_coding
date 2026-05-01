# Linear & Logistic Regression from Scratch

Minimal NumPy implementations of linear and logistic regression trained with mini-batch SGD on synthetic data. Files:

- `generate_data.py` — synthetic data generators for regression and classification.
- `model.py` — training loops for both models.

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
