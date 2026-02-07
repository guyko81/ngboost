# SymPy-Based Auto-Generated Distributions

## Why SymPy?

Hand-writing `d_score()` and `metric()` methods for NGBoost distributions is
tedious and error-prone. Each distribution requires:

- Correct symbolic derivatives (often involving digamma/polygamma functions)
- Chain-rule adjustments for log-transformed parameters
- Fisher Information computation (expected Hessian)

Two factories are provided:

- `make_distribution` — creates a **complete distribution class** ready for
  `NGBRegressor(Dist=...)`. This is the recommended entry point.
- `make_sympy_log_score` — creates just the **LogScore subclass** (for advanced
  use when you need a custom `Distn` wrapper, e.g. classification or extra
  parameters).

## Built-in Distributions

All SymPy-powered distributions are importable directly:

```python
from ngboost.distns import Beta, BetaBernoulli, BetaBinomial, LogitNormal
```

| Distribution   | Type           | Factory used          | Use case                                  |
|---------------|----------------|-----------------------|-------------------------------------------|
| Beta          | Regression     | `make_distribution`   | Bounded (0,1) outcomes                    |
| LogitNormal   | Regression     | `make_distribution`   | Bounded (0,1) with logistic-normal model  |
| BetaBernoulli | Classification | `make_sympy_log_score` + wrapper | Binary with calibrated uncertainty |
| BetaBinomial  | Regression     | `make_sympy_log_score` + wrapper | Overdispersed count data          |

BetaBernoulli and BetaBinomial use `make_sympy_log_score` (the lower-level
factory) because they require custom `Distn` wrappers — see
[Writing a custom wrapper](#writing-a-custom-distn-wrapper) below.

## Quickstart: One Function Call

```python
import sympy as sp
import sympy.stats as symstats
import scipy.stats
from ngboost.distns.sympy_utils import make_distribution

alpha, beta, y = sp.symbols("alpha beta y", positive=True)

Beta = make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Beta("Y", alpha, beta),
    scipy_dist_cls=scipy.stats.beta,
    scipy_kwarg_map={"a": alpha, "b": beta},
    name="Beta",
)

# Ready to use with NGBoost
from ngboost import NGBRegressor
ngb = NGBRegressor(Dist=Beta)
ngb.fit(X_train, Y_train)
ngb.predict(X_test)
```

That's it. The factory auto-derives score, gradients, and Fisher Information
from the SymPy distribution, and auto-generates fit/sample/mean from scipy.

## When You Need a Manual Score Expression

For distributions without a `sympy.stats` equivalent (e.g., LogitNormal), or
where the auto-derived density is too complex, you can provide the score
expression explicitly:

```python
mu, sigma, y = sp.symbols("mu sigma y", positive=True)
logit_y = sp.log(y / (1 - y))
score = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi) + sp.log(sigma)
    + (logit_y - mu)**2 / (2 * sigma**2)
    + sp.log(y) + sp.log(1 - y)
)

LogitNormalLogScore = make_sympy_log_score(
    params=[(mu, False), (sigma, True)],
    y=y,
    score_expr=score,
    name="LogitNormalLogScore",
)
```

## Parameter Handling

### Log-transformed parameters

Most NGBoost parameters are log-transformed to ensure positivity (e.g.,
`alpha = exp(log_alpha)`). When `log_transformed=True`, the factory
automatically applies the chain rule:

```
d/d(log theta) = theta * d/d(theta)
```

### Identity parameters

Some parameters use an identity link (e.g., the mean `mu` of a Normal).
Set `log_transformed=False` and derivatives are computed directly.

### Extra (non-optimized) parameters

Parameters like `n` in BetaBinomial appear in the score but are not
optimized by NGBoost. Pass them via `extra_params`:

```python
make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score_expr,
    extra_params=[n],  # not differentiated
)
```

At runtime, the score class reads `self.n` from the distribution instance.

## Fisher Information Strategy

The factory computes the Fisher Information (metric) analytically when
possible, using a three-tier strategy:

1. **y-free Hessian**: If the second derivatives of the score don't depend
   on `y`, then `FI = Hessian` directly (no expectation needed). This is
   the case for exponential family distributions like Beta and Gamma.

2. **SymPy E[]**: If the Hessian depends on `y` and `sympy_dist` is
   provided, the factory substitutes `y` with the random variable and
   computes `E[Hessian]` symbolically. This works for Normal, Bernoulli,
   and other distributions where SymPy can evaluate the expectation.

3. **Monte Carlo fallback**: If neither analytical approach succeeds, the
   class inherits `LogScore.metric()` which estimates the FI via Monte
   Carlo sampling. This is used for LogitNormal and BetaBinomial.

## Discrete Distribution Support

For discrete distributions like Bernoulli, `sympy.stats.density()` returns
a `Piecewise` expression. The factory automatically converts Bernoulli-like
distributions (support {0, 1}) to the smooth form `p^y * (1-p)^(1-y)` for
differentiation:

```python
alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

# Just pass the Bernoulli — Piecewise is handled automatically
BetaBernoulliLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
)
```

## Writing a Custom Distn Wrapper

`make_distribution` produces a complete `RegressionDistn`. Use the lower-level
`make_sympy_log_score` when you need:

- **Classification** (`ClassificationDistn` with `class_probs()`)
- **Extra parameters** (like `n` in BetaBinomial) that require custom
  `__init__`, `fit`, and `sample`

The factory generates the score/gradient/FI; you write the thin wrapper:

### Classification wrapper (BetaBernoulli)

```python
from ngboost.distns.distn import ClassificationDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

BetaBernoulliLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
)

class BetaBernoulli(ClassificationDistn):
    n_params = 2
    scores = [BetaBernoulliLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.alpha = np.exp(np.clip(params[0], -150, 700))
        self.beta = np.exp(np.clip(params[1], -150, 700))

    def class_probs(self):
        p = self.alpha / (self.alpha + self.beta)
        return np.column_stack([1 - p, p])

    def fit(Y):  # initial params before boosting
        ...

    def sample(self, m):
        ...
```

### Extra-parameter wrapper (BetaBinomial)

```python
from ngboost.distns.distn import RegressionDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

alpha, beta = sp.symbols("alpha beta", positive=True)
y, n = sp.symbols("y n", positive=True, integer=True)

BetaBinomialLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score,       # manual NLL expression
    extra_params=[n],       # not optimized — read from self.n at runtime
)

class BetaBinomial(RegressionDistn):
    n_params = 2
    scores = [BetaBinomialLogScore]

    def __init__(self, params, n=1):
        super().__init__(params)
        self.alpha = np.exp(np.clip(params[0], -150, 700))
        self.beta = np.exp(np.clip(params[1], -150, 700))
        self.n = n            # extra param — set here, read by score class

    def fit(Y, n=1):  # initial params before boosting
        ...

    def sample(self, m):
        ...
```

To use a specific `n`, subclass with the value baked in:

```python
class BetaBinomial20(BetaBinomial):
    def __init__(self, params):
        super().__init__(params, n=20)
    def fit(Y):
        return BetaBinomial.fit(Y, n=20)

ngb = NGBRegressor(Dist=BetaBinomial20)
```

## Worked Examples

### Continuous — auto-derived from sympy.stats (Beta)

```python
alpha, beta, y = sp.symbols("alpha beta y", positive=True)

BetaLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Beta("Y", alpha, beta),
)
```

FI path: y-free Hessian (tier 1).

### Continuous with manual score (LogitNormal)

```python
mu, sigma, y = sp.symbols("mu sigma y", positive=True)
logit_y = sp.log(y / (1 - y))
score = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi) + sp.log(sigma)
    + (logit_y - mu)**2 / (2 * sigma**2)
    + sp.log(y) + sp.log(1 - y)
)

LogitNormalLogScore = make_sympy_log_score(
    params=[(mu, False), (sigma, True)],
    y=y,
    score_expr=score,
)
```

FI path: Monte Carlo fallback (tier 3) — no `sympy_dist` available.

### Discrete classification (BetaBernoulli)

```python
alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

BetaBernoulliLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
)
```

FI path: SymPy E[] (tier 2) — Hessian depends on y, but E[] is tractable.

### Extra non-optimized params (BetaBinomial)

```python
alpha, beta = sp.symbols("alpha beta", positive=True)
y, n = sp.symbols("y n", positive=True, integer=True)

score = -(
    sp.loggamma(n + 1) - sp.loggamma(y + 1) - sp.loggamma(n - y + 1)
    + sp.loggamma(y + alpha) + sp.loggamma(n - y + beta)
    - sp.loggamma(n + alpha + beta)
    + sp.loggamma(alpha + beta)
    - sp.loggamma(alpha) - sp.loggamma(beta)
)

BetaBinomialLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score,
    extra_params=[n],
)
```

FI path: Monte Carlo fallback (tier 3) — E[] over BetaBinomial produces
unevaluated sums.

## Testing

### Gradient correctness

`tests/test_score.py` uses `scipy.optimize.approx_fprime` to compare
`d_score()` against finite differences of `score()`. All four SymPy
distributions (Beta, LogitNormal, BetaBernoulli, BetaBinomial) are included.

### Metric correctness

`tests/test_score.py` compares analytical `metric()` against a Monte Carlo
estimate. Distributions with analytical FI (Beta, BetaBernoulli) are
included in the metric test.

### Existing distribution parity

`tests/test_sympy_existing_distns.py` verifies that SymPy-generated score
classes for Normal, Gamma, and Poisson match their hand-written
implementations numerically (score, d_score, and metric) — using only
`sympy_dist` (no manual score expressions).

## Example Notebooks

| Notebook                          | Pattern demonstrated                          |
|----------------------------------|-----------------------------------------------|
| `notebooks/example_normal.ipynb`         | `make_distribution` from scratch, verified against built-in Normal |
| `notebooks/example_beta.ipynb`           | `make_distribution` with `sympy.stats` + scipy |
| `notebooks/example_logitnormal.ipynb`    | `make_distribution` with manual `score_expr` + custom `sample_fn` |
| `notebooks/example_betabernoulli.ipynb`  | `make_sympy_log_score` + `ClassificationDistn` wrapper |
| `notebooks/example_betabinomial.ipynb`   | `make_sympy_log_score` + extra params + custom `fit` |
| `notebooks/example_mixture_lognormal.ipynb` | Advanced: 8-param mixture of 3 log-normals with logsumexp path |
| `notebooks/sympy_normal_demo.ipynb`      | Deep dive: symbolic expressions, verification against hand-written code |

## Reference: SymPy-Powered Distributions

| Distribution    | Params (link)                    | Score source   | FI method      |
|----------------|----------------------------------|----------------|----------------|
| Beta           | alpha (log), beta (log)          | auto from dist | y-free Hessian |
| BetaBernoulli  | alpha (log), beta (log)          | auto from dist | SymPy E[]      |
| BetaBinomial   | alpha (log), beta (log) + n      | manual         | Monte Carlo    |
| LogitNormal    | mu (identity), sigma (log)       | manual         | Monte Carlo    |
