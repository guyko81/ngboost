"""The NGBoost Beta-Binomial distribution and scores"""

import numpy as np
import sympy as sp
from scipy.special import digamma

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

# --- SymPy-generated LogScore ---
_alpha, _beta = sp.symbols("alpha beta", positive=True)
_y, _n = sp.symbols("y n", positive=True, integer=True)

_score_expr = -(
    sp.loggamma(_n + 1)
    - sp.loggamma(_y + 1)
    - sp.loggamma(_n - _y + 1)
    + sp.loggamma(_y + _alpha)
    + sp.loggamma(_n - _y + _beta)
    - sp.loggamma(_n + _alpha + _beta)
    + sp.loggamma(_alpha + _beta)
    - sp.loggamma(_alpha)
    - sp.loggamma(_beta)
)

BetaBinomialLogScore = make_sympy_log_score(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    score_expr=_score_expr,
    extra_params=[_n],
    name="BetaBinomialLogScore",
)


class BetaBinomial(RegressionDistn):
    """
    Implements the Beta-Binomial distribution for NGBoost.

    The Beta-Binomial distribution has parameters n, alpha, and beta.
    The score and gradient are auto-derived via SymPy.
    """

    n_params = 2
    scores = [BetaBinomialLogScore]

    def __init__(self, params, n=1):
        super().__init__(params)
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])
        self.n = n

    def fit(Y, n=1):
        """Fit alpha, beta using fixed-point iteration on digamma equations."""
        p = np.clip(np.mean(Y) / n, 0.01, 0.99)
        alpha = p * 2
        beta = (1 - p) * 2

        for _ in range(100):
            ab = alpha + beta
            psi_ab = digamma(ab)
            alpha_new = alpha * (np.mean(digamma(Y + alpha)) - psi_ab) / (
                digamma(alpha) - psi_ab + 1e-10
            )
            beta_new = beta * (np.mean(digamma(n - Y + beta)) - psi_ab) / (
                digamma(beta) - psi_ab + 1e-10
            )
            alpha = np.clip(alpha_new, 1e-4, 1e4)
            beta = np.clip(beta_new, 1e-4, 1e4)

        return np.array([np.log(alpha), np.log(beta)])

    def sample(self, m):
        p = np.random.beta(self.alpha, self.beta, size=m)
        return np.random.binomial(self.n, p)

    def mean(self):
        return self.n * self.alpha / (self.alpha + self.beta)

    @property
    def params(self):
        return {"n": self.n, "alpha": self.alpha, "beta": self.beta}
