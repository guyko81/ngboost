"""The NGBoost Logit-Normal distribution — built via the SymPy factory.

Use for regression on bounded (0, 1) outcomes where the distribution
of logit(Y) is approximately Normal.  Compared to Beta, LogitNormal can
capture heavier tails and is more natural when the generative process
applies a logistic transform to a latent Gaussian.

Examples: bioassay dose-response curves, compositional data proportions,
species prevalence modeling.

This demonstrates the manual-expression pattern: when no ``sympy.stats``
equivalent exists, you provide ``score_expr`` (the negative log-likelihood)
directly, plus a custom ``sample_fn``.  The factory auto-derives score,
gradient, Fisher Information, and fit (via numerical MLE).
"""

import numpy as np
import sympy as sp

from ngboost.distns.sympy_utils import make_distribution

_mu, _sigma, _y = sp.symbols("mu sigma y", positive=True)
_logit_y = sp.log(_y / (1 - _y))

_score_expr = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi)
    + sp.log(_sigma)
    + (_logit_y - _mu) ** 2 / (2 * _sigma**2)
    + sp.log(_y)
    + sp.log(1 - _y)
)


def _sample(self, m):
    mu, sigma = np.squeeze(self.mu), np.squeeze(self.sigma)
    return np.array([
        1 / (1 + np.exp(-np.random.normal(mu, sigma)))
        for _ in range(m)
    ])


LogitNormal = make_distribution(
    params=[(_mu, False), (_sigma, True)],
    y=_y,
    score_expr=_score_expr,
    sample_fn=_sample,
    name="LogitNormal",
)
