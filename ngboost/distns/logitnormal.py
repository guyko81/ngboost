"""The NGBoost Logit-Normal distribution and scores"""

import numpy as np
import sympy as sp
from scipy.stats import norm

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

# --- SymPy-generated LogScore ---
_mu, _sigma, _y = sp.symbols("mu sigma y", positive=True)
_logit_y = sp.log(_y / (1 - _y))
_score_expr = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi)
    + sp.log(_sigma)
    + (_logit_y - _mu) ** 2 / (2 * _sigma**2)
    + sp.log(_y)
    + sp.log(1 - _y)
)

LogitNormalLogScore = make_sympy_log_score(
    params=[(_mu, False), (_sigma, True)],
    y=_y,
    score_expr=_score_expr,
    name="LogitNormalLogScore",
)


class LogitNormal(RegressionDistn):
    """
    Implements the Logit-Normal distribution for NGBoost.

    The Logit-Normal distribution has two parameters, mu and sigma.
    If X ~ Normal(mu, sigma), then Y = 1/(1+exp(-X)) ~ LogitNormal(mu, sigma).
    """

    n_params = 2
    scores = [LogitNormalLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.mu = params[0]
        self.logsigma = params[1]
        self.sigma = np.exp(params[1])
        self.var = self.sigma**2

    def fit(Y):
        logit_Y = np.log(Y / (1 - Y))
        m = np.mean(logit_Y)
        s = np.std(logit_Y)
        return np.array([m, np.log(s)])

    def sample(self, m):
        mu = np.squeeze(self.mu)
        sigma = np.squeeze(self.sigma)
        logit_samples = np.random.normal(mu, sigma, size=m)
        return 1 / (1 + np.exp(-logit_samples))

    def mean(self):
        # No closed-form mean; approximate via Monte Carlo
        samples = self.sample(1000)
        return np.mean(samples, axis=0)

    def predict(self):
        return self.mean()

    @property
    def params(self):
        return {"mu": self.mu, "sigma": self.sigma}
