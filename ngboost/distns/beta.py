"""The NGBoost Beta distribution and scores"""

import numpy as np
import sympy as sp
import sympy.stats as symstats
from scipy.stats import beta as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

# --- SymPy-generated LogScore ---
_alpha, _beta, _y = sp.symbols("alpha beta y", positive=True)

BetaLogScore = make_sympy_log_score(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Beta("Y", _alpha, _beta),
    name="BetaLogScore",
)


class Beta(RegressionDistn):
    """
    Implements the Beta distribution for NGBoost.

    The Beta distribution has two parameters, alpha and beta, both positive.
    Internally parameterized as log(alpha) and log(beta).
    """

    n_params = 2
    scores = [BetaLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])
        self.dist = dist(a=self.alpha, b=self.beta)

    def fit(Y):
        a, b, _, _ = dist.fit(Y, floc=0, fscale=1)
        return np.array([np.log(a), np.log(b)])

    def sample(self, m):
        return np.array([self.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
