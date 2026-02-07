"""The NGBoost Beta-Bernoulli distribution — built via the SymPy factory.

Binary classification with calibrated uncertainty.  Unlike standard classifiers
that output a single probability, the Beta-Bernoulli gives a full *distribution
over the probability itself* via its Beta(alpha, beta) prior.

The predictive probability for class 1 is ``alpha / (alpha + beta)``.
When alpha and beta are both large, the model is confident.  When they're
small, there's high uncertainty.

This is a **classification** distribution, so it uses ``make_sympy_log_score``
(the lower-level factory) with a thin ``ClassificationDistn`` wrapper.
"""

import numpy as np
import sympy as sp
import sympy.stats as symstats
from scipy.special import digamma

from ngboost.distns.distn import ClassificationDistn
from ngboost.distns.sympy_utils import make_sympy_log_score

_alpha, _beta, _y = sp.symbols("alpha beta y")
_p = _alpha / (_alpha + _beta)

BetaBernoulliLogScore = make_sympy_log_score(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Bernoulli("Y", _p),
    name="BetaBernoulliLogScore",
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

    def fit(Y):
        p = np.clip(np.mean(Y), 0.01, 0.99)
        a, b = p * 2, (1 - p) * 2
        for _ in range(100):
            ab = a + b
            psi_ab = digamma(ab)
            a = np.clip(
                a * (np.mean(digamma(Y + a)) - psi_ab) / (digamma(a) - psi_ab + 1e-10),
                1e-4,
                1e4,
            )
            b = np.clip(
                b * (np.mean(digamma(1 - Y + b)) - psi_ab) / (digamma(b) - psi_ab + 1e-10),
                1e-4,
                1e4,
            )
        return np.array([np.log(a), np.log(b)])

    def sample(self, m):
        p = np.squeeze(self.alpha / (self.alpha + self.beta))
        return np.random.binomial(1, p, size=m).astype(float)

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
