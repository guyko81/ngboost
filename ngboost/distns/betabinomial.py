"""The NGBoost Beta-Binomial distribution — built via the SymPy factory.

Count regression with overdispersion.  The Beta-Binomial models the number
of successes in ``n`` trials where the success probability itself follows
a Beta(alpha, beta) prior, producing greater variance than a plain Binomial.

Uses a **manual score expression** with an ``extra_params=[n]`` non-optimized
parameter.  Fisher Information is computed via Monte Carlo (SymPy E[] over
BetaBinomial produces unevaluated sums).
"""

import numpy as np
import sympy as sp

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

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
    """Beta-Binomial distribution for NGBoost.

    Parameters n, alpha, beta — where n is fixed (non-optimized) and
    alpha, beta are learned via natural gradient boosting.
    """

    n_params = 2
    scores = [BetaBinomialLogScore]

    def __init__(self, params, n=1):
        super().__init__(params)
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(np.clip(params[0], -150, 700))
        self.beta = np.exp(np.clip(params[1], -150, 700))
        self.n = n

    def fit(Y, n=1):
        """Estimate initial alpha, beta from data via method of moments.

        NGBoost calls ``fit(Y)`` once before boosting to get the starting
        (intercept) values for log(alpha) and log(beta).  The boosting
        iterations then refine these per-sample using the score gradient.

        We use method of moments rather than MLE because BetaBinomial MLE
        requires iterative digamma solvers that are numerically fragile
        for count data (Y in {0..n}).  Method of moments is closed-form:

            p_hat = mean(Y) / n
            rho   = (Var(Y) / (n * p * (1-p)) - 1) / (n - 1)

        where rho is the overdispersion parameter = 1/(alpha+beta+1).
        From rho we recover the concentration alpha+beta, then split
        by p_hat to get individual alpha, beta.
        """
        Y = np.asarray(Y, dtype=float)
        p_hat = np.clip(np.mean(Y) / n, 0.01, 0.99)
        var_hat = np.var(Y)
        binom_var = n * p_hat * (1 - p_hat)
        if binom_var > 0 and n > 1:
            rho = np.clip((var_hat / binom_var - 1) / (n - 1), 0.01, 0.99)
        else:
            rho = 0.5
        concentration = 1.0 / rho - 1  # alpha + beta
        alpha = np.clip(p_hat * concentration, 1e-4, 1e4)
        beta = np.clip((1 - p_hat) * concentration, 1e-4, 1e4)
        return np.array([np.log(alpha), np.log(beta)])

    def sample(self, m):
        alpha = np.squeeze(self.alpha)
        beta = np.squeeze(self.beta)
        return np.array([
            np.random.binomial(self.n, np.random.beta(alpha, beta))
            for _ in range(m)
        ])

    def mean(self):
        return self.n * self.alpha / (self.alpha + self.beta)

    @property
    def params(self):
        return {"n": self.n, "alpha": self.alpha, "beta": self.beta}
