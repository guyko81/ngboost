"""SymPy-based auto-generation of NGBoost LogScore subclasses.

Provides ``make_sympy_log_score``, a factory that takes a SymPy
distribution (or a symbolic negative-log-likelihood) and produces a
complete ``LogScore`` subclass with auto-derived ``score``, ``d_score``,
and (optionally analytical) ``metric`` methods.
"""

import numpy as np
import sympy as sym
import sympy.stats as symstats
from sympy.utilities.lambdify import lambdify

from ngboost.scores import LogScore


def _density_from_piecewise(pw, y):
    """Convert a Piecewise density to a smooth expression for differentiation.

    SymPy represents discrete PMFs as ``Piecewise`` expressions (e.g.
    ``Piecewise((p, Eq(y, 1)), (1-p, Eq(y, 0)), (0, True))``).  These
    cannot be differentiated w.r.t. continuous parameters.

    For Bernoulli-like distributions with support {0, 1} this function
    returns the equivalent smooth form ``p1**y * p0**(1-y)``.
    """
    if not isinstance(pw, sym.Piecewise):
        return pw

    prob_map = {}
    for value, cond in pw.args:
        if isinstance(cond, sym.Eq):
            lhs, rhs = cond.args
            if lhs == y and rhs.is_number:
                prob_map[int(rhs)] = value
            elif rhs == y and lhs.is_number:
                prob_map[int(lhs)] = value

    # Bernoulli-like: support {0, 1} → p1^y * p0^(1-y)
    if set(prob_map.keys()) == {0, 1}:
        return prob_map[1] ** y * prob_map[0] ** (1 - y)

    return None


def _build_lambdified(expr, ordered_symbols, extra_symbols=None):
    """Lambdify a SymPy expression into a NumPy/SciPy-callable function.

    Uses ``sympy.Dummy`` symbols internally so that user-chosen symbol names
    (e.g. ``beta``) never collide with SciPy function names during lambdify.
    """
    all_symbols = list(ordered_symbols)
    if extra_symbols:
        all_symbols.extend(extra_symbols)

    # Replace every user symbol with a Dummy to avoid name collisions
    dummies = [sym.Dummy(s.name) for s in all_symbols]
    subs = list(zip(all_symbols, dummies))
    expr_safe = expr.subs(subs)

    return lambdify(dummies, expr_safe, modules=["scipy", "numpy"])


def _try_analytical_fi(score_expr, params, y, grad_exprs, extra_params, sympy_dist):
    """Attempt to compute analytical Fisher Information.

    Strategy
    --------
    1. Compute the Hessian of the score w.r.t. the *raw* (natural) parameters.
    2. If all Hessian elements are free of ``y``, the FI in raw-parameter space
       equals the Hessian (since E[Hessian] = Hessian when it's constant in y).
    3. Transform to internal-parameter space via Jacobian:
       ``FI_internal[i,j] = J[i] * J[j] * FI_raw[i,j]``
       where ``J[k] = theta_k`` if log-transformed, else ``J[k] = 1``.
    4. If any Hessian element depends on ``y`` and ``sympy_dist`` is given,
       try ``symstats.E()``; fall back to None on failure.

    Returns
    -------
    list[list[sympy.Expr]] or None
    """
    n_params = len(params)
    param_symbols = [p for p, _ in params]

    try:
        # Compute raw-parameter Hessian: d²score / d(theta_i) d(theta_j)
        raw_grads = [sym.diff(score_expr, p) for p in param_symbols]
        raw_hessian = [[None] * n_params for _ in range(n_params)]

        all_y_free = True
        for i in range(n_params):
            for j in range(i, n_params):
                hij = sym.simplify(sym.diff(raw_grads[i], param_symbols[j]))
                raw_hessian[i][j] = hij
                raw_hessian[j][i] = hij
                if y in hij.free_symbols:
                    all_y_free = False

        if all_y_free:
            # FI_raw = Hessian (no expectation needed)
            # Transform: FI_internal[i,j] = J_i * J_j * FI_raw[i,j]
            fi_exprs = [[None] * n_params for _ in range(n_params)]
            for i in range(n_params):
                for j in range(n_params):
                    ji = params[i][0] if params[i][1] else sym.Integer(1)
                    jj = params[j][0] if params[j][1] else sym.Integer(1)
                    fi_exprs[i][j] = sym.simplify(ji * jj * raw_hessian[i][j])
            return fi_exprs

        # Some elements depend on y: try E[] with sympy_dist
        if sympy_dist is not None:
            fi_exprs = [[None] * n_params for _ in range(n_params)]
            for i in range(n_params):
                for j in range(i, n_params):
                    hij = raw_hessian[i][j]
                    if y not in hij.free_symbols:
                        eij = hij
                    else:
                        # Substitute the y symbol with the random variable,
                        # then take the unconditional expectation.
                        hij_rv = hij.subs(y, sympy_dist)
                        eij = symstats.E(hij_rv)
                        eij = sym.simplify(eij)
                        if eij.has(symstats.Expectation) or y in eij.free_symbols:
                            return None

                    ji = params[i][0] if params[i][1] else sym.Integer(1)
                    jj = params[j][0] if params[j][1] else sym.Integer(1)
                    fi_ij = sym.simplify(ji * jj * eij)
                    fi_exprs[i][j] = fi_ij
                    fi_exprs[j][i] = fi_ij
            return fi_exprs

        return None
    except Exception:
        return None


def make_sympy_log_score(
    params,
    y,
    score_expr=None,
    sympy_dist=None,
    extra_params=None,
    name=None,
):
    """Create a ``LogScore`` subclass from a symbolic score expression.

    Either ``score_expr`` or ``sympy_dist`` (or both) must be provided.
    When only ``sympy_dist`` is given the score is derived automatically
    as ``-log(density(sympy_dist)(y))``.

    Parameters
    ----------
    params : list[tuple[sympy.Symbol, bool]]
        Each entry is ``(symbol, log_transformed)``.  If ``log_transformed``
        is True the internal NGBoost parameter is ``log(symbol)`` and the
        chain-rule factor ``symbol * d/d(symbol)`` is applied automatically.
        If False the parameter is used with an identity link.
    y : sympy.Symbol
        The symbol representing the observed data in *score_expr*.
    score_expr : sympy.Expr or None
        The *negative log-likelihood* as a SymPy expression
        (i.e. ``-log p(y|theta)``).  If ``None``, it is derived
        automatically from ``sympy_dist``.
    sympy_dist : sympy.stats random variable or None
        A ``sympy.stats`` random variable.  Used to (a) derive
        ``score_expr`` when it is not given, and (b) compute the Fisher
        Information analytically when the Hessian depends on ``y``.
        Falls back to Monte-Carlo metric (inherited from ``LogScore``) on
        failure.
    extra_params : list[sympy.Symbol] or None
        Non-optimised parameters (e.g. ``n`` in BetaBinomial) that appear in
        the score expression but are *not* differentiated.
    name : str or None
        Class name for the generated ``LogScore`` subclass.

    Returns
    -------
    type
        A ``LogScore`` subclass with ``score``, ``d_score``, and ``metric``
        methods.
    """
    if score_expr is None and sympy_dist is None:
        raise ValueError("At least one of score_expr or sympy_dist must be provided")

    if score_expr is None:
        density = symstats.density(sympy_dist)(y)
        if isinstance(density, sym.Piecewise):
            density = _density_from_piecewise(density, y)
            if density is None:
                raise ValueError(
                    "Cannot auto-derive score from Piecewise density. "
                    "Provide score_expr explicitly."
                )
        score_expr = -sym.log(density)

    extra_params = extra_params or []
    param_symbols = [p for p, _ in params]

    # ---- score function ----
    _score_fn = _build_lambdified(score_expr, [y], param_symbols + extra_params)

    # ---- d_score (gradient w.r.t. internal parameters) ----
    grad_exprs = []
    for psym, is_log in params:
        raw_deriv = sym.diff(score_expr, psym)
        if is_log:
            grad_exprs.append(sym.simplify(psym * raw_deriv))
        else:
            grad_exprs.append(sym.simplify(raw_deriv))

    _grad_fns = [
        _build_lambdified(g, [y], param_symbols + extra_params) for g in grad_exprs
    ]

    # ---- metric (Fisher Information) ----
    n_params = len(params)
    fi_fns = None

    fi_exprs = _try_analytical_fi(
        score_expr, params, y, grad_exprs, extra_params, sympy_dist
    )
    if fi_exprs is not None:
        fi_fns = []
        for i in range(n_params):
            row = []
            for j in range(n_params):
                row.append(
                    _build_lambdified(
                        fi_exprs[i][j], param_symbols, extra_params or None
                    )
                )
            fi_fns.append(row)

    # ---- Build the class ----
    class_name = name or "SympyLogScore"

    _param_names = [str(p) for p, _ in params]
    _extra_names = [str(e) for e in extra_params]

    def _get_param_arrays(self_obj):
        param_arrays = [getattr(self_obj, n) for n in _param_names]
        extra_arrays = [getattr(self_obj, n) for n in _extra_names]
        return param_arrays, extra_arrays

    def score_method(self, Y):
        param_arrays, extra_arrays = _get_param_arrays(self)
        return _score_fn(Y, *param_arrays, *extra_arrays)

    def d_score_method(self, Y):
        param_arrays, extra_arrays = _get_param_arrays(self)
        D = np.zeros((len(Y), n_params))
        for k, gfn in enumerate(_grad_fns):
            D[:, k] = gfn(Y, *param_arrays, *extra_arrays)
        return D

    if fi_fns is not None:

        def metric_method(self):
            param_arrays, extra_arrays = _get_param_arrays(self)
            n = param_arrays[0].shape[0]
            FI = np.zeros((n, n_params, n_params))
            for i in range(n_params):
                for j in range(n_params):
                    FI[:, i, j] = fi_fns[i][j](*param_arrays, *extra_arrays)
            return FI

    else:
        metric_method = None

    attrs = {
        "score": score_method,
        "d_score": d_score_method,
    }
    if metric_method is not None:
        attrs["metric"] = metric_method

    cls = type(class_name, (LogScore,), attrs)
    return cls
