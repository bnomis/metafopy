"""Regression test for funnel plot asymmetry (metafor's ``regtest``).

Sterne & Egger (2005), doi:10.1016/j.jclinepi.2004.01.022 -- the test adds the
standard error (or another measure of precision) as a predictor to the
meta-analytic model. See https://wviechtb.github.io/metafor/reference/regtest.html.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy._data import MetaData
from metafopy.bias._inputs import data_of, fp, fx, sign_flag, vector_inputs
from metafopy.methods._common import refit_arrays
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult, Test, fit_rma_uni, level_to_alpha

__all__ = ['RegtestResult', 'regtest']

type FloatArray = NDArray[np.float64]
type Predictor = Literal['sei', 'vi', 'ni', 'ninv', 'sqrtni', 'sqrtninv']

PREDICTOR_LABEL = {
    'sei': 'standard error',
    'vi': 'sampling variance',
    'ni': 'sample size',
    'ninv': 'inverse of the sample size',
    'sqrtni': 'square root sample size',
    'sqrtninv': 'inverse of the square root sample size',
}
LIMIT_LABEL = {
    'sei': 'Limit Estimate (as sei -> 0):   ',
    'vi': 'Limit Estimate (as vi -> 0):    ',
    'ninv': 'Limit Estimate (as ni -> inf):  ',
    'sqrtninv': 'Limit Estimate (as ni -> inf):  ',
}


@dataclass(frozen=True, slots=True)
class RegtestResult:
    """Result of :func:`regtest`; ``fit`` is the meta-regression fitted (an :class:`RMAResult` for ``model='rma'``)."""

    model: str
    predictor: str
    zval: float
    pval: float
    ddf: int | None
    method: str
    digits: int
    ret_fit: bool
    fit: RMAResult | None
    est: float | None
    ci_lb: float | None
    ci_ub: float | None

    @property
    def dfs(self) -> int | None:
        return self.ddf

    def summary(self, digits: int | None = None, ret_fit: bool | None = None) -> str:
        d = self.digits if digits is None else digits
        show_fit = self.ret_fit if ret_fit is None else ret_fit
        if self.model == 'lm':
            model_line = 'Model:     weighted regression with multiplicative dispersion'
        else:
            kind = 'fixed-effects' if self.method in FIXED_EFFECTS else 'mixed-effects'
            model_line = f'Model:     {kind} meta-regression model'
        out = f'\nRegression Test for Funnel Plot Asymmetry\n\n{model_line}\nPredictor: {PREDICTOR_LABEL[self.predictor]}\n'
        if show_fit and self.fit is not None:
            out += str(self.fit) + '\n'
        else:
            out += '\n'
        flag = sign_flag(self.zval, self.est) if self.est is not None else ''
        if self.ddf is None:
            out += f'Test for Funnel Plot Asymmetry: z = {fx(self.zval, d, flag)}, {fp(self.pval, d, equal=True, sep=True, pname="p ")}\n'
        else:
            out += f'Test for Funnel Plot Asymmetry: t = {fx(self.zval, d, flag)}, df = {self.ddf}, {fp(self.pval, d, equal=True, sep=True, pname="p ")}\n'
        if self.est is not None and self.ci_lb is not None and self.ci_ub is not None:
            out += f'{LIMIT_LABEL[self.predictor]}b = {fx(self.est, d, sign_flag(self.zval, self.est))} (CI: {fx(self.ci_lb, d)}, {fx(self.ci_ub, d)})\n'
        return out

    def __str__(self) -> str:
        return self.summary()


def regtest(
    x: RMAResult | ArrayLike | str,
    vi: ArrayLike | str | None = None,
    *,
    sei: ArrayLike | str | None = None,
    ni: ArrayLike | str | None = None,
    subset: ArrayLike | None = None,
    data: pd.DataFrame | None = None,
    model: Literal['lm', 'rma'] = 'rma',
    predictor: Predictor = 'sei',
    ret_fit: bool = False,
    digits: int | None = None,
    level: float | None = None,
    method: str | None = None,
    test: str | None = None,
) -> RegtestResult:
    """Regression test for funnel plot asymmetry.

    ``x`` is a fitted model or a vector of estimates (with ``vi`` or ``sei``). ``level`` is a
    confidence level in percent (default 95). With ``model='lm'`` (weighted regression with
    multiplicative dispersion, the classical Egger test) ``ret_fit`` prints nothing extra.
    """
    if model not in ('lm', 'rma'):
        raise ValueError(f'unknown model {model!r}')
    if predictor not in PREDICTOR_LABEL:
        raise ValueError(f'unknown predictor {predictor!r}')

    if isinstance(x, RMAResult):
        if vi is not None or sei is not None or subset is not None:
            warnings.warn("Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object.", stacklevel=2)
        yi, v, n = data_of(x)
        if ni is not None:
            n_full = np.asarray(ni, dtype=np.float64)
            if n_full.shape[0] != x.data.k_all:
                raise ValueError(
                    f"Length of the variable specified via 'ni' ({n_full.shape[0]}) does not correspond to the size of the original dataset ({x.data.k_all})."
                )
            n = n_full[x.data.keep]
        X = x.X
        p = x.p
        alpha = x.level if level is None else level_to_alpha(level)
        meth = x.method if method is None else method
        tst = x.test if test is None else test
        weighted = x.weighted
        tau2_fix = x.tau2_fix
        tau2 = x.tau2
        control = x.control
        weights = x.weights
        coef_names = tuple(x.coef_names)
        slab = x.slab
    else:
        yi, v_opt, n = vector_inputs(x, vi, sei, ni=ni, subset=subset, data=data, what='the test')
        if v_opt is None:
            raise ValueError("Must specify the 'vi' or 'sei' argument.")
        v = v_opt
        p = 1
        X = np.ones((yi.shape[0], 1))
        alpha = 0.05 if level is None else level_to_alpha(level)
        meth = 'REML' if method is None else method
        tst = 'z' if test is None else test
        weighted = True
        tau2_fix = False
        tau2 = math.nan
        control = None
        weights = None
        coef_names = ('intrcpt',)
        slab = tuple(str(i + 1) for i in range(yi.shape[0]))
    d = 4 if digits is None else digits

    if predictor == 'sei':
        extra = np.sqrt(v)
    elif predictor == 'vi':
        extra = v
    else:
        if n is None:
            raise ValueError('No sample size information available to use this predictor.')
        extra = {'ni': n, 'ninv': 1 / n, 'sqrtni': np.sqrt(n), 'sqrtninv': 1 / np.sqrt(n)}[predictor]
    Xn = np.column_stack([X, extra])
    names = (*coef_names, predictor if predictor in ('sei', 'vi') else 'ni')

    if np.linalg.matrix_rank(Xn) < Xn.shape[1]:
        raise ValueError('Model matrix no longer of full rank after addition of predictor. Cannot fit model.')

    fit: RMAResult | None = None
    if model == 'rma':
        if isinstance(x, RMAResult):
            fit = refit_arrays(x, X=Xn, names=names, intercept=True, tau2='same', method=meth, skip_r2=False)
            if fit is not None and (tst != x.test or alpha != x.level):
                fit = None
            if fit is None:
                fit = _fit_vectors(yi, v, Xn, names, n, slab, meth, tst, alpha, weighted, weights, tau2 if tau2_fix else None, control)
        else:
            fit = _fit_vectors(yi, v, Xn, names, n, slab, meth, tst, alpha, weighted, weights, None, control)
        zval = float(fit.zval[p])
        pval = float(fit.pval[p])
        ddf = fit.ddf
        meth_out = fit.method
    else:
        w = 1 / v
        k = yi.shape[0]
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Xn * sw[:, None], yi * sw, rcond=None)
        resid = yi - Xn @ beta
        df = k - Xn.shape[1]
        sigma2 = float(np.sum(w * resid**2) / df)
        cov = sigma2 * np.linalg.inv((Xn * w[:, None]).T @ Xn)
        se = np.sqrt(np.diag(cov))
        tv = beta / se
        zval = float(tv[p])
        pval = float(2 * stats.t.sf(abs(zval), df))
        ddf = df
        meth_out = meth

    est = ci_lb = ci_ub = None
    if predictor in ('sei', 'vi', 'ninv', 'sqrtninv') and p == 1 and np.all(Xn[:, 0] == 1):
        if model == 'lm':
            est = float(beta[0])
            crit = float(stats.t.isf(alpha / 2, k - Xn.shape[1]))
            ci_lb, ci_ub = est - crit * float(se[0]), est + crit * float(se[0])
        elif fit is not None:
            est, ci_lb, ci_ub = float(fit.beta[0]), float(fit.ci_lb[0]), float(fit.ci_ub[0])
    return RegtestResult(
        model=model,
        predictor=predictor,
        zval=zval,
        pval=pval,
        ddf=ddf if model == 'lm' else (fit.ddf if fit is not None else None),
        method=meth_out,
        digits=d,
        ret_fit=ret_fit,
        fit=fit,
        est=est,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
    )


def cast_test(test: str) -> Test:
    if test not in ('z', 't', 'knha', 'adhoc'):
        raise ValueError(f'unknown test {test!r}')
    return cast(Test, test)


def _fit_vectors(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    names: Sequence[str],
    ni: FloatArray | None,
    slab: Sequence[str],
    method: str,
    test: str,
    alpha: float,
    weighted: bool,
    weights: FloatArray | None,
    tau2: float | None,
    control: Any,
) -> RMAResult:
    md = MetaData(
        yi=yi,
        vi=vi,
        X=X,
        X_names=tuple(names),
        slab=tuple(slab),
        keep=np.ones(yi.shape[0], dtype=np.bool_),
        weights=weights,
        ni=ni,
        intercept=True,
    )
    from dataclasses import asdict, is_dataclass

    ctrl = asdict(control) if is_dataclass(control) and not isinstance(control, type) else control
    return fit_rma_uni(md, method=method, weighted=weighted, test=cast_test(test), level=1 - alpha, tau2=tau2, control=ctrl, _skip_r2=False)
