"""Regression test for funnel plot asymmetry (metafor's ``regtest``).

Sterne & Egger (2005), doi:10.1016/j.jclinepi.2004.01.022 -- the test adds the
standard error (or another measure of precision) as a predictor to the
meta-analytic model. See https://wviechtb.github.io/metafor/reference/regtest.html.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.bias.inputs
import metafopy.errors
import metafopy.metadata
import metafopy.methods.common
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.models.tau2

__all__ = ['RegtestResult', 'regtest']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Predictor = typing.Literal['sei', 'vi', 'ni', 'ninv', 'sqrtni', 'sqrtninv']

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PREDICTOR_LABEL = {
    'sei': 'standard error',
    'vi': 'sampling variance',
    'ni': 'sample size',
    'ninv': 'inverse of the sample size',
    'sqrtni': 'square root sample size',
    'sqrtninv': 'inverse of the square root sample size',
}
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
LIMIT_LABEL = {
    'sei': 'Limit Estimate (as sei -> 0):   ',
    'vi': 'Limit Estimate (as vi -> 0):    ',
    'ninv': 'Limit Estimate (as ni -> inf):  ',
    'sqrtninv': 'Limit Estimate (as ni -> inf):  ',
}


@dataclasses.dataclass(frozen=True, slots=True)
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
    fit: metafopy.models.rma_uni.RMAResult | None
    est: float | None
    ci_lb: float | None
    ci_ub: float | None

    @property
    def dfs(self) -> int | None:
        return self.ddf

    def summary(self, digits: int | None = None, ret_fit: bool | None = None) -> str:
        d = metafopy.mfopt.option_digits(digits, self.digits)
        show_fit = self.ret_fit if ret_fit is None else ret_fit
        if self.model == 'lm':
            model_line = 'Model:     weighted regression with multiplicative dispersion'
        else:
            kind = 'fixed-effects' if self.method in metafopy.models.tau2.FIXED_EFFECTS else 'mixed-effects'
            model_line = f'Model:     {kind} meta-regression model'
        out = f'\nRegression Test for Funnel Plot Asymmetry\n\n{model_line}\nPredictor: {PREDICTOR_LABEL[self.predictor]}\n'
        if show_fit and self.fit is not None:
            out += str(self.fit) + '\n'
        else:
            out += '\n'
        flag = metafopy.bias.inputs.sign_flag(self.zval, self.est) if self.est is not None else ''
        stat = metafopy.bias.inputs.fx(self.zval, d, flag)
        pval = metafopy.bias.inputs.fp(self.pval, d, equal=True, sep=True, pname='p ')
        if self.ddf is None:
            out += f'Test for Funnel Plot Asymmetry: z = {stat}, {pval}\n'
        else:
            out += f'Test for Funnel Plot Asymmetry: t = {stat}, df = {self.ddf}, {pval}\n'
        if self.est is not None and self.ci_lb is not None and self.ci_ub is not None:
            est = metafopy.bias.inputs.fx(self.est, d, metafopy.bias.inputs.sign_flag(self.zval, self.est))
            ci = f'{metafopy.bias.inputs.fx(self.ci_lb, d)}, {metafopy.bias.inputs.fx(self.ci_ub, d)}'
            out += f'{LIMIT_LABEL[self.predictor]}b = {est} (CI: {ci})\n'
        return out

    def __str__(self) -> str:
        return self.summary()


def regtest(
    x: metafopy.models.rma_uni.RMAResult | numpy.typing.ArrayLike | str,
    vi: numpy.typing.ArrayLike | str | None = None,
    *,
    sei: numpy.typing.ArrayLike | str | None = None,
    ni: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    data: pandas.DataFrame | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    model: typing.Literal['lm', 'rma'] = 'rma',
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

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if model not in ('lm', 'rma'):
        raise metafopy.errors.InputError(f'unknown model {model!r}')
    if predictor not in PREDICTOR_LABEL:
        raise metafopy.errors.InputError(f'unknown predictor {predictor!r}')

    if isinstance(x, metafopy.models.rma_uni.RMAResult):
        if vi is not None or sei is not None or subset is not None:
            warnings.warn("Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object.", stacklevel=2)
        yi, v, n = metafopy.bias.inputs.data_of(x)
        if ni is not None:
            n_full = numpy.asarray(ni, dtype=numpy.float64)
            if n_full.shape[0] != x.data.k_all:
                raise metafopy.errors.InputError(
                    f"Length of the variable specified via 'ni' ({n_full.shape[0]}) does not correspond to the size of the original dataset ({x.data.k_all})."
                )
            n = n_full[x.data.keep]
        X = x.X
        p = x.p
        alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
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
        yi, v_opt, n = metafopy.bias.inputs.vector_inputs(x, vi, sei, ni=ni, subset=subset, data=data, what='the test')
        if v_opt is None:
            raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
        v = v_opt
        p = 1
        X = numpy.ones((yi.shape[0], 1))
        alpha = 0.05 if level is None else metafopy.models.rma_uni.level_to_alpha(level)
        meth = 'REML' if method is None else method
        tst = 'z' if test is None else test
        weighted = True
        tau2_fix = False
        tau2 = math.nan
        control = None
        weights = None
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        coef_names = ('intrcpt',)
        slab = tuple(str(i + 1) for i in range(yi.shape[0]))
    d = 4 if digits is None else digits

    if predictor == 'sei':
        extra = numpy.sqrt(v)
    elif predictor == 'vi':
        extra = v
    else:
        if n is None:
            raise metafopy.errors.InputError('No sample size information available to use this predictor.')
        extra = {'ni': n, 'ninv': 1 / n, 'sqrtni': numpy.sqrt(n), 'sqrtninv': 1 / numpy.sqrt(n)}[predictor]
    Xn = numpy.column_stack([X, extra])
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    names = (*coef_names, predictor if predictor in ('sei', 'vi') else 'ni')

    if numpy.linalg.matrix_rank(Xn) < Xn.shape[1]:
        raise metafopy.errors.InputError('Model matrix no longer of full rank after addition of predictor. Cannot fit model.')

    fit: metafopy.models.rma_uni.RMAResult | None = None
    if model == 'rma':
        if isinstance(x, metafopy.models.rma_uni.RMAResult):
            fit = metafopy.methods.common.refit_arrays(x, X=Xn, names=names, intercept=True, tau2='same', method=meth, skip_r2=False)
            if fit is not None and (tst != x.test or alpha != x.level):
                fit = None
            if fit is None:
                fit = fit_vectors(yi, v, Xn, names, n, slab, meth, tst, alpha, weighted, weights, tau2 if tau2_fix else None, control)
        else:
            fit = fit_vectors(yi, v, Xn, names, n, slab, meth, tst, alpha, weighted, weights, None, control)
        zval = float(fit.zval[p])
        pval = float(fit.pval[p])
        ddf = fit.ddf
        meth_out = fit.method
    else:
        w = 1 / v
        k = yi.shape[0]
        sw = numpy.sqrt(w)
        beta, *_ = numpy.linalg.lstsq(Xn * sw[:, None], yi * sw, rcond=None)
        resid = yi - Xn @ beta
        df = k - Xn.shape[1]
        sigma2 = float(numpy.sum(w * resid**2) / df)
        cov = sigma2 * numpy.linalg.inv((Xn * w[:, None]).T @ Xn)
        se = numpy.sqrt(numpy.diag(cov))
        tv = beta / se
        zval = float(tv[p])
        pval = float(2 * scipy.stats.t.sf(abs(zval), df))
        ddf = df
        meth_out = meth

    est = ci_lb = ci_ub = None
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if predictor in ('sei', 'vi', 'ninv', 'sqrtninv') and p == 1 and numpy.all(Xn[:, 0] == 1):
        if model == 'lm':
            est = float(beta[0])
            crit = float(scipy.stats.t.isf(alpha / 2, k - Xn.shape[1]))
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


def cast_test(test: str) -> metafopy.models.rma_uni.Test:
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test not in ('z', 't', 'knha', 'adhoc'):
        raise metafopy.errors.InputError(f'unknown test {test!r}')
    return typing.cast(metafopy.models.rma_uni.Test, test)


def fit_vectors(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    names: collections.abc.Sequence[str],
    ni: FloatArray | None,
    slab: collections.abc.Sequence[str],
    method: str,
    test: str,
    alpha: float,
    weighted: bool,
    weights: FloatArray | None,
    tau2: float | None,
    control: typing.Any,
) -> metafopy.models.rma_uni.RMAResult:
    md = metafopy.metadata.MetaData(
        yi=yi,
        vi=vi,
        X=X,
        X_names=tuple(names),
        slab=tuple(slab),
        keep=numpy.ones(yi.shape[0], dtype=numpy.bool_),
        weights=weights,
        ni=ni,
        intercept=True,
    )
    ctrl = dataclasses.asdict(control) if dataclasses.is_dataclass(control) and not isinstance(control, type) else control
    return metafopy.models.rma_uni.fit_rma_uni(
        md, method=method, weighted=weighted, test=cast_test(test), level=1 - alpha, tau2=tau2, control=ctrl, skip_r2=False
    )
