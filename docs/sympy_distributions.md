# SymPy-Based Auto-Generated Distributions

## Why SymPy?

Hand-writing `d_score()` and `metric()` methods for NGBoost distributions is
tedious and error-prone. Each distribution requires:

- Correct symbolic derivatives (often involving digamma/polygamma functions)
- Chain-rule adjustments for log-transformed parameters
- Fisher Information computation (expected Hessian)

The `make_sympy_log_score` factory automates all of this. You define the
distribution, and the factory produces a complete `LogScore` subclass with
correct `score()`, `d_score()`, and `metric()` methods — no calculus needed.

## How to Add a New Distribution (2 Steps)

### Step 1: Define the distribution and call the factory

```python
import sympy as sp
import sympy.stats as symstats
from ngboost.distns.sympy_utils import make_sympy_log_score

alpha, beta, y = sp.symbols("alpha beta y", positive=True)

BetaLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],  # True = log-transformed
    y=y,
    sympy_dist=symstats.Beta("Y", alpha, beta),
    name="BetaLogScore",
)
```

That's it. The factory auto-derives `score()` from the distribution's density,
computes `d_score()` via symbolic differentiation, and tries to compute
`metric()` analytically (falling back to Monte Carlo if needed).

### Step 2: Wire up the distribution class

```python
class Beta(RegressionDistn):
    n_params = 2
    scores = [BetaLogScore]
    # ... __init__, fit, sample, params as usual
```

No hand-written derivatives needed.

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

## Worked Examples

### Continuous — just define the distribution (Beta)

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
`d_score()` against finite differences of `score()`. All four new
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

## Reference: SymPy-Powered Distributions

| Distribution    | Params (link)                    | Score source   | FI method      |
|----------------|----------------------------------|----------------|----------------|
| Beta           | alpha (log), beta (log)          | auto from dist | y-free Hessian |
| BetaBernoulli  | alpha (log), beta (log)          | auto from dist | SymPy E[]      |
| BetaBinomial   | alpha (log), beta (log) + n      | manual         | Monte Carlo    |
| LogitNormal    | mu (identity), sigma (log)       | manual         | Monte Carlo    |
