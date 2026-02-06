"""The NGBoost Beta-Bernoulli distribution and scores"""

import numpy as np
import sympy as sp
import sympy.stats as symstats
from scipy.special import digamma

from ngboost.distns.distn import ClassificationDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

# --- SymPy-generated LogScore ---
_alpha, _beta, _y = sp.symbols("alpha beta y")
_p = _alpha / (_alpha + _beta)

BetaBernoulliLogScore = make_sympy_log_score(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Bernoulli("Y", _p),
    name="BetaBernoulliLogScore",
)


class BetaBernoulli(ClassificationDistn):
    """
    Implements the Beta-Bernoulli distribution for NGBoost.

    Uses a Beta prior with parameters alpha and beta.
    The predictive probability for class 1 is alpha / (alpha + beta).
    """

    n_params = 2
    scores = [BetaBernoulliLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])

    def class_probs(self):
        p = self.alpha / (self.alpha + self.beta)
        return np.column_stack([1 - p, p])

    def fit(Y):
        """Fit alpha, beta using fixed-point iteration on digamma equations."""
        p = np.clip(np.mean(Y), 0.01, 0.99)
        # Start from method of moments
        alpha = p * 2
        beta = (1 - p) * 2

        for _ in range(100):
            ab = alpha + beta
            psi_ab = digamma(ab)
            alpha_new = alpha * (np.mean(digamma(Y + alpha)) - psi_ab) / (
                digamma(alpha) - psi_ab + 1e-10
            )
            beta_new = beta * (np.mean(digamma(1 - Y + beta)) - psi_ab) / (
                digamma(beta) - psi_ab + 1e-10
            )
            alpha = np.clip(alpha_new, 1e-4, 1e4)
            beta = np.clip(beta_new, 1e-4, 1e4)

        return np.array([np.log(alpha), np.log(beta)])

    def sample(self, m):
        p = np.squeeze(self.alpha / (self.alpha + self.beta))
        return np.random.binomial(1, p, size=m).astype(float)

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
