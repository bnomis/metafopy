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
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.binary
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

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
class RegtestResult(metafopy.printing.HtmlRepr):
    """Result of [regtest][metafopy.regtest].

    Attributes:
        model: 'rma' or 'lm', the regression model used.
        predictor: the predictor that was used ('sei', 'vi', 'ni', 'ninv', 'sqrtni' or 'sqrtninv').
        zval: test statistic for the coefficient of the predictor (a t-value when the test is based on the t-distribution).
        pval: p-value of that test.
        ddf: degrees of freedom of the test, or None for a z-test.
        method: tau^2 estimator of the regression model.
        digits: number of decimals used when printing.
        ret_fit: whether the printed summary shows the fitted model.
        fit: the fitted regression model ([RMAResult][metafopy.RMAResult] for `model='rma'`), or None.
        est: estimate of the model at zero of the predictor (the effect at infinite precision), or None if it is not defined.
        ci_lb: lower bound of the confidence interval of `est`, or None.
        ci_ub: upper bound of the confidence interval of `est`, or None.
    """

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
        """Degrees of freedom of the test (alias of `ddf`); None for a z-test."""
        return self.ddf

    def summary(self, digits: int | None = None, ret_fit: bool | None = None) -> str:
        """Format the result as text, as metafor prints it.

        Args:
            digits: number of decimals. Default None: the digits stored in the result.
            ret_fit: whether to show the fitted regression model. Default None: the value stored in the result.

        Returns:
            The printed output as a string.
        """
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
    x: metafopy.models.rma_uni.RMAResult | metafopy.models.binary.FixedBinaryResult | numpy.typing.ArrayLike | str,
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
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> RegtestResult:
    """Regression test for funnel plot asymmetry (Egger's test and extensions).

    Mirrors metafor's `regtest`, see https://wviechtb.github.io/metafor/reference/regtest.html. The effect sizes are regressed on a measure of
    their precision (the standard error by default) and the test for the coefficient of that predictor is the test for asymmetry
    (Egger et al., 1997, doi:10.1136/bmj.315.7109.629). With a fitted model that has moderators, the predictor is added to the model.
    The intercept of the model with `sei` (or `vi`) as the predictor estimates the effect at infinite precision.

    Args:
        x: a fitted model ([rma_uni][metafopy.rma_uni] result, or a fixed-effects `rma_mh` / `rma_peto` result with `model='lm'`), or the
            observed effect sizes (array-like, or the name of a column of `data`).
        vi: sampling variances of a vector `x` (array-like or column name of `data`); ignored with a warning when `x` is a model.
        sei: standard errors, used when `vi` is not given.
        ni: sample sizes (array-like or column name of `data`), needed for the predictors based on `ni`. For a model it must have the length
            of the original data set. For a vector `x` it defaults to the sample sizes stored by [escalc][metafopy.escalc].
        subset: 0-based integer indices or boolean mask of the studies to use (vector input only).
        data: data frame in which string values of `x`, `vi`, `sei` and `ni` are looked up as column names.
        model: 'rma' | 'lm'. 'rma' (default) fits a meta-regression model (a mixed-effects model for a random-effects `x`), 'lm' the weighted
            regression with multiplicative dispersion (the classical Egger test).
        predictor: 'sei' | 'vi' | 'ni' | 'ninv' | 'sqrtni' | 'sqrtninv': the standard error (default), the sampling variance, the sample size,
            its inverse, its square root or the inverse of its square root. The last four need `ni`.
        ret_fit: whether the printed summary also shows the fitted regression model (the model is always in the result).
            It prints nothing extra for `model='lm'`.
        digits: number of decimals used when the result is printed. Default None: the digits of the model, or 4.
        level: confidence level in percent for the limit estimate (the intercept) and the model fit. Default None: the level of the model, or 95.
        method: tau^2 estimator of the meta-regression model, e.g. 'REML', 'DL' or 'FE'. Default None: the method of `x`, or 'REML' for a vector.
        test: test for the model coefficients: 'z' | 't' | 'knha' | 'adhoc'. Default None: the test of `x`, or 'z' for a vector.
        na_action: what to do with studies with missing values in a vector `x`: 'omit' | 'exclude' | 'pass' omit them with a warning,
            'fail' raises. It has no effect on a fitted model.

    Returns:
        A [RegtestResult][metafopy.RegtestResult] with the test statistic `zval` (a t-value for `test='t'`), its p-value `pval`, the
        degrees of freedom `ddf` (None for a z-test), the estimate `est` and confidence interval `ci_lb`, `ci_ub` of the model at zero of the
        predictor (when `predictor` is 'sei', 'vi', 'ninv' or 'sqrtninv' and the model has no moderators), and the regression model `fit`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or studies are missing and `na_action='fail'`.
    """
    metafopy.methods.common.check_na_action(na_action)
    vec_measure: str | None = None
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if model not in ('lm', 'rma'):
        raise metafopy.errors.InputError(f'unknown model {model!r}')
    if predictor not in PREDICTOR_LABEL:
        raise metafopy.errors.InputError(f'unknown predictor {predictor!r}')

    if metafopy.methods.fitview.is_rma(x):
        metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
        if model == 'rma' and not isinstance(x, metafopy.models.rma_uni.RMAResult):
            # rma.mh and rma.peto objects get as far as refitting the model with their tau2 (NULL), which metafor rejects
            raise metafopy.errors.InputError("Argument 'tau2' must be a scalar.")
        if vi is not None or sei is not None or subset is not None:
            warnings.warn("Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object.", stacklevel=2)
    if isinstance(x, metafopy.models.binary.FixedBinaryResult):
        # model='lm' of an rma.mh or rma.peto fit: the weighted regression of the estimates on the predictor, with no model to refit
        view = metafopy.methods.fitview.fit_view(x)
        yi, v, n = view.yi, view.vi, view.ni
        if ni is not None:
            n = metafopy.bias.inputs.as_vector(ni, 'ni', data)
            if n is None or n.shape[0] != view.k_all:
                size = 0 if n is None else n.shape[0]
                raise metafopy.errors.InputError(
                    f"Length of the variable specified via 'ni' ({size}) does not correspond to the size of the original dataset ({view.k_all})."
                )
            n = n[view.keep]
        X = view.X
        p = X.shape[1]
        alpha = view.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
        meth, tst, weighted, tau2_fix, tau2, control, weights = view.method, 'z', True, False, math.nan, None, None
        coef_names: tuple[str, ...] = ('intrcpt',)
        slab = view.slab
    elif isinstance(x, metafopy.models.rma_uni.RMAResult):
        yi, v, n = metafopy.bias.inputs.data_of(x)
        if ni is not None:
            n_full = metafopy.bias.inputs.as_vector(ni, 'ni', data)
            if n_full is None:
                raise metafopy.errors.InternalError('ni must not be None')
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
        yi, v_opt, n = metafopy.bias.inputs.vector_inputs(x, vi, sei, ni=ni, subset=subset, data=data, what='the test', attr_ni=True, na_action=na_action)
        if isinstance(x, str) and data is not None and subset is None and data.attrs.get('measure'):
            # metafor takes the measure from the attribute of the yi column of the escalc object
            vec_measure = str(data.attrs['measure'])
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
    d = metafopy.mfopt.stored_digits(digits, metafopy.mfopt.model_digits(x))

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
    names = (*coef_names, predictor if predictor in ('sei', 'vi', 'ninv') else 'ni')

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
            if vec_measure is not None:
                fit = dataclasses.replace(fit, measure=vec_measure)
        if isinstance(x, metafopy.models.rma_uni.RMAResult):
            # the yi of a trimfill() object carries the labels 1, 2, ... whatever labels were given
            dropped = x.data.na_rows is not None and bool(x.data.na_rows.any())
            fit = metafopy.models.rma_uni.as_refit_on_used(
                fit, dropped=dropped, measure=x.measure, numbered=hasattr(x, 'k0'), row_names=x.data.rownames if x.p > 1 or not x.intercept else None
            )
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
