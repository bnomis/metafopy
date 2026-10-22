"""Test of excess significance (metafor's ``tes``).

Ioannidis & Trikalinos (2007), doi:10.1177/1740774507079441. The power of each study is evaluated
at an assumed true effect (by default the weighted average); the observed number of significant
results is compared with the expected one. See https://wviechtb.github.io/metafor/reference/tes.html.
"""

import dataclasses
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.integrate
import scipy.stats

import metafopy.bias.inputs
import metafopy.core.parallel
import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['TesResult', 'tes']

type FloatArray = numpy.typing.NDArray[numpy.float64]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type Alternative = typing.Literal['two.sided', 'greater', 'less']

EPS = sys.float_info.epsilon
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
SCIPY_ALT: dict[str, typing.Literal['two-sided', 'greater', 'less']] = {'two.sided': 'two-sided', 'greater': 'greater', 'less': 'less'}


@dataclasses.dataclass(frozen=True, slots=True)
class TesResult(metafopy.printing.HtmlRepr):
    """Result of [tes][metafopy.tes].

    Attributes:
        k: number of studies.
        O: observed number of significant studies.
        E: expected number of significant studies (the sum of the powers).
        OEratio: ratio of the observed to the expected number.
        test: the test used ('chi2', 'binom' or 'exact').
        X2: chi-square statistic (NaN unless `test` is 'chi2').
        pval: p-value of the test of excess significance.
        power: estimated power of each study.
        sig: whether each study is significant.
        theta: assumed true effect (a single value or one per study).
        theta_lim: true effect(s) at which the p-value equals `tes_alpha` (one or two values, NaN if not estimable), or None.
        tes_alternative: alternative of the test of excess significance.
        tes_alpha: significance level of the test of excess significance.
        digits: number of decimals used when printing.
    """

    k: int
    O: int  # noqa: E741
    E: float
    OEratio: float
    test: str
    X2: float
    pval: float
    power: FloatArray
    sig: numpy.typing.NDArray[numpy.bool_]
    theta: float | FloatArray
    theta_lim: tuple[float, ...] | None
    tes_alternative: str
    tes_alpha: float
    digits: int = 4

    def summary(self, digits: int | None = None) -> str:
        """Format the result as text, as metafor prints it.

        Args:
            digits: number of decimals. Default None: the digits stored in the result.

        Returns:
            The printed output as a string.
        """
        d = metafopy.mfopt.option_digits(digits, self.digits)
        out = ['', 'Test of Excess Significance', '']
        out += [
            f'Observed Number of Significant Findings: {self.O} (out of {self.k})',
            f'Expected Number of Significant Findings: {metafopy.bias.inputs.fx(self.E, d)}',
            f'Observed Number / Expected Number:       {metafopy.bias.inputs.fx(self.OEratio, d)}',
            '',
        ]
        if numpy.ndim(self.theta) == 0:
            out.append(f'Estimated Power of Tests (based on theta = {metafopy.bias.inputs.fx(float(numpy.asarray(self.theta)), d)})')
        else:
            out.append('Estimated Power of Tests: ')
        out.append('')
        if self.k > 5:
            q = numpy.quantile(self.power, [0, 0.25, 0.5, 0.75, 1])
            names = ['min', 'q1', 'median', 'q3', 'max']
            vals = [metafopy.bias.inputs.fx(float(v), d) for v in q]
        else:
            names = [str(i + 1) for i in range(self.k)]
            vals = [metafopy.bias.inputs.fx(float(v), d) for v in self.power]
        out += [metafopy.bias.inputs.print_vector(names, vals), '']
        line = f'Test of Excess Significance: {metafopy.bias.inputs.fp(self.pval, d, equal=True, sep=True, pname="p ")}'
        if self.test == 'chi2':
            line += f' (X^2 = {metafopy.bias.inputs.fx(self.X2, d)}, df = 1)'
        elif self.test == 'binom':
            line += ' (binomial test)'
        else:
            line += ' (exact test)'
        out.append(line)
        if self.theta_lim is not None:
            lim = 'Limit Estimate (theta_lim):  ' + ('not estimable' if math.isnan(self.theta_lim[0]) else metafopy.bias.inputs.fx(self.theta_lim[0], d))
            if len(self.theta_lim) == 2:
                lim += ', ' + ('not estimable' if math.isnan(self.theta_lim[1]) else metafopy.bias.inputs.fx(self.theta_lim[1], d))
            if any(not math.isnan(t) for t in self.theta_lim):
                p_at = self.tes_alpha / 2 if self.tes_alternative == 'two.sided' else self.tes_alpha
                lim += f' (where p = {p_at:.7g})'
            out.append(lim)
        return '\n'.join([*out, ''])

    def __str__(self) -> str:
        return self.summary()


def tes_power(theta: FloatArray, sei: FloatArray, h0: float, alternative: str, crit: float) -> FloatArray:
    mu = (theta - h0) / sei
    if alternative == 'two.sided':
        pw = scipy.stats.norm.sf(crit, loc=mu) + scipy.stats.norm.cdf(-crit, loc=mu)
    elif alternative == 'greater':
        pw = scipy.stats.norm.sf(crit, loc=mu)
    else:
        pw = scipy.stats.norm.cdf(crit, loc=mu)
    return numpy.asarray(pw, dtype=numpy.float64)


def poisson_binomial(p: FloatArray, progbar: bool = False) -> FloatArray:
    pmf = numpy.array([1.0])
    for pi in metafopy.core.parallel.progress(p, p.shape[0], progbar):
        pmf = numpy.concatenate([pmf * (1 - pi), [0.0]]) + numpy.concatenate([[0.0], pmf * pi])
    return pmf


def tes(
    x: metafopy.methods.fitview.Fit | numpy.typing.ArrayLike | str,
    vi: numpy.typing.ArrayLike | str | None = None,
    *,
    sei: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    data: pandas.DataFrame | None = None,
    H0: float = 0.0,
    alternative: Alternative = 'two.sided',
    alpha: float = 0.05,
    theta: numpy.typing.ArrayLike | None = None,
    tau2: float | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    test: typing.Literal['chi2', 'binom', 'exact'] | None = None,
    tes_alternative: Alternative = 'greater',
    tes_alpha: float = 0.1,
    progbar: bool = False,
    digits: int | None = None,
    correct: bool = False,
    rel_tol: float = EPS**0.25,
    subdivisions: int = 100,
    tau2_lb: float = 0.0,
    find_lim: bool = True,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> TesResult:
    """Test of excess significance.

    Mirrors metafor's `tes`, see https://wviechtb.github.io/metafor/reference/tes.html. The power of each study is computed for an assumed
    true effect and the observed number of significant studies is compared with the expected number (Ioannidis & Trikalinos, 2007,
    doi:10.1177/1740774507079441). Also estimates the true effect `theta_lim` at which the test would be just significant.

    Args:
        x: a fitted model (the power is based on its estimate, or its fitted values with moderators, and its tau^2), or the observed effect
            sizes (array-like, or the name of a column of `data`).
        vi: sampling variances of a vector `x` (array-like or column name of `data`).
        sei: standard errors, used when `vi` is not given. One of `vi` and `sei` is required for a vector `x`.
        subset: 0-based integer indices or boolean mask of the studies to use (vector input only).
        data: data frame in which string values of `x`, `vi` and `sei` are looked up as column names.
        H0: value of the effect under the null hypothesis of the significance tests of the studies (default 0).
        alternative: 'two.sided' | 'greater' | 'less': the alternative of the significance tests of the studies (default 'two.sided').
        alpha: significance level of those tests (default 0.05).
        theta: assumed true effect, a single value or one per study. Default None: the weighted average of the effects.
        tau2: amount of heterogeneity; the power is then averaged over the distribution of the true effects. Default None: no heterogeneity.
        test: 'chi2' | 'binom' | 'exact': the test of excess significance. Default None: 'exact' unless it needs more than 10^6
            terms, then 'chi2'.
        tes_alternative: 'two.sided' | 'greater' | 'less': the alternative of the test of excess significance (default 'greater').
        tes_alpha: significance level of the test of excess significance, used for `theta_lim` (default 0.1).
        progbar: whether to show a progress bar on stderr during the exact test (metafor's default is True, here it is False).
        digits: number of decimals used when the result is printed. Default None: the digits of the model, or 4.
        correct: whether the continuity correction of the chi-square test is used (default False).
        rel_tol: relative tolerance of the numerical integration over tau^2 (default `eps**0.25`).
        subdivisions: maximum number of subdivisions of that integration (default 100).
        tau2_lb: `tau2` values not above this lower bound are treated as zero (default 0).
        find_lim: whether `theta_lim` is estimated (default True; only when `theta` is a single value or estimated).
        na_action: what to do with studies with missing values in a vector `x`: 'omit' | 'exclude' | 'pass' omit them with a warning,
            'fail' raises. It has no effect on a fitted model.

    Returns:
        A [TesResult][metafopy.TesResult] with the number of studies `k`, the observed number `O` and expected number `E` of significant
        studies, their ratio `OEratio`, the test used `test` with its statistic `X2` (chi-square test only) and p-value `pval`, the power
        of each study `power`, the significance flags `sig`, the assumed effect `theta` and `theta_lim`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or studies are missing and `na_action='fail'`.
    """
    metafopy.methods.common.check_na_action(na_action)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if alternative not in ('two.sided', 'greater', 'less') or tes_alternative not in ('two.sided', 'greater', 'less'):
        raise metafopy.errors.InputError("'alternative' and 'tes_alternative' must be 'two.sided', 'greater' or 'less'")
    if metafopy.methods.fitview.is_rma(x):
        metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
        view = metafopy.methods.fitview.fit_view(x)
        th: numpy.typing.ArrayLike = numpy.array([view.beta[0]]) if view.int_only else view.fitted
        return tes(
            view.yi,
            view.vi,
            H0=H0,
            alternative=alternative,
            alpha=alpha,
            theta=th,
            tau2=view.tau2,
            test=test,
            tes_alternative=tes_alternative,
            tes_alpha=tes_alpha,
            progbar=progbar,
            digits=metafopy.mfopt.stored_digits(digits, metafopy.mfopt.model_digits(x)),
            correct=correct,
            rel_tol=rel_tol,
            subdivisions=subdivisions,
            tau2_lb=tau2_lb,
            find_lim=find_lim,
            na_action=na_action,
        )
    yi = metafopy.bias.inputs.as_vector(x, 'x', data)
    v = metafopy.bias.inputs.as_vector(vi, 'vi', data)
    s = metafopy.bias.inputs.as_vector(sei, 'sei', data)
    if yi is None:
        raise metafopy.errors.InputError("Argument 'x' must be a vector with estimates or an 'rma' model object.")
    if v is None and s is not None:
        v = s**2
    if v is None:
        raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
    if yi.shape != v.shape:
        raise metafopy.errors.InputError("Length of 'yi' and 'vi' (or 'sei') are not the same.")
    if not 0 < alpha < 1:
        raise metafopy.errors.InputError("Value of 'alpha' needs to be > 0 and < 1.")
    if not 0 < tes_alpha < 1:
        raise metafopy.errors.InputError("Value of 'tes_alpha' needs to be > 0 and < 1.")
    if alternative == 'two.sided':
        crit = float(scipy.stats.norm.isf(alpha / 2))
    elif alternative == 'greater':
        crit = float(scipy.stats.norm.isf(alpha))
    else:
        crit = float(scipy.stats.norm.ppf(alpha))
    k_f = yi.shape[0]
    if theta is None:
        single, est_theta = True, True
        th_arr = numpy.zeros(k_f)
        theta_1 = math.nan
    else:
        arr = numpy.atleast_1d(numpy.asarray(theta, dtype=numpy.float64))
        est_theta = False
        if arr.shape[0] == 1:
            single = True
            theta_1 = float(arr[0])
            th_arr = numpy.full(k_f, theta_1)
        else:
            single = False
            theta_1 = math.nan
            th_arr = arr
            if th_arr.shape[0] != k_f:
                raise metafopy.errors.InputError("Length of 'theta' and 'yi' are not the same.")
    if subset is not None:
        sub = numpy.asarray(subset)
        sel = sub if sub.dtype == numpy.bool_ else sub.astype(int)
        yi, v, th_arr = yi[sel], v[sel], th_arr[sel]
    bad = numpy.isnan(yi) | numpy.isnan(v) | numpy.isnan(th_arr)
    if bad.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('Missing values in data.')
        warnings.warn(f'{int(bad.sum())} {"studies" if bad.sum() > 1 else "study"} with NAs omitted from the test.', stacklevel=2)
        yi, v, th_arr = yi[~bad], v[~bad], th_arr[~bad]
    k = yi.shape[0]
    if k == 0:
        raise metafopy.errors.InputError('Stopped because k = 0.')
    sei_ = numpy.sqrt(v)
    zi = (yi - H0) / sei_
    use_tau2 = tau2 is not None and tau2 > tau2_lb
    wi = 1 / (v + tau2) if use_tau2 and tau2 is not None else 1 / v
    if est_theta:
        theta_1 = float(numpy.sum(yi * wi) / numpy.sum(wi))
        th_arr = numpy.full(k, theta_1)
    if not use_tau2 or tau2 is None:
        pow_ = tes_power(th_arr, sei_, H0, alternative, crit)
    else:
        tau = math.sqrt(tau2)
        pow_ = numpy.empty(k)
        for i in range(k):

            def intfun(u: float, i: int = i) -> float:
                return float(tes_power(numpy.array([u]), sei_[i : i + 1], H0, alternative, crit)[0] * scipy.stats.norm.pdf(u, th_arr[i], tau))

            val, _ = scipy.integrate.quad(intfun, th_arr[i] - 5 * tau, th_arr[i] + 5 * tau, epsabs=rel_tol, epsrel=rel_tol, limit=subdivisions)
            pow_[i] = val
    if alternative == 'two.sided':
        sig = numpy.abs(zi) >= crit
    elif alternative == 'greater':
        sig = zi >= crit
    else:
        sig = zi <= crit
    e = float(numpy.sum(pow_))
    o = int(numpy.sum(sig))
    if tes_alternative == 'two.sided':
        js = list(range(k + 1))
    elif tes_alternative == 'greater':
        js = list(range(o, k + 1))
    else:
        js = list(range(o + 1))
    if test is None:
        tot = sum(math.comb(k, j) for j in js)
        test = 'exact' if tot <= 10**6 else 'chi2'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif test not in ('chi2', 'binom', 'exact'):
        raise metafopy.errors.InputError(f'unknown test {test!r}')
    x2 = math.nan
    p0 = e / k
    if test in ('chi2', 'binom') and not 0 < p0 < 1:
        raise metafopy.errors.InputError('The expected proportion of significant findings must be in (0, 1).')
    if test == 'chi2':
        yates = min(0.5, abs(o - k * p0)) if correct else 0.0
        x2 = (abs(o - k * p0) - yates) ** 2 / (k * p0 * (1 - p0))
        z = math.copysign(math.sqrt(x2), o / k - p0)
        if tes_alternative == 'two.sided':
            pval = float(scipy.stats.chi2.sf(x2, 1))
        elif tes_alternative == 'greater':
            pval = float(scipy.stats.norm.sf(z))
        else:
            pval = float(scipy.stats.norm.cdf(z))
    elif test == 'binom':
        pval = float(scipy.stats.binomtest(o, k, p0, alternative=SCIPY_ALT[tes_alternative]).pvalue)
    else:
        prj = poisson_binomial(pow_, progbar)[js]
        if tes_alternative == 'two.sided':
            pval = float(numpy.sum(prj[prj <= prj[o] + EPS**0.5]))
        else:
            pval = float(numpy.sum(prj))
        pval = min(pval, 1.0)

    theta_lim: tuple[float, ...] | None = None
    if find_lim and single:

        def lim_fn(t: float) -> float:
            return (
                tes(
                    yi,
                    v,
                    H0=H0,
                    alternative=alternative,
                    alpha=alpha,
                    theta=t,
                    tau2=tau2,
                    test=test,
                    tes_alternative=tes_alternative,
                    tes_alpha=tes_alpha,
                    correct=correct,
                    rel_tol=rel_tol,
                    subdivisions=subdivisions,
                    tau2_lb=tau2_lb,
                    find_lim=False,
                ).pval
                - tes_alpha
            )

        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        def solve(direction: typing.Literal['upX', 'downX'], lo_first: bool) -> float:
            lo, hi = (H0, theta_1) if lo_first else (theta_1, H0)
            try:
                return metafopy.core.zeroin.uniroot(lim_fn, lo, hi, extend_int=direction).root
            except ValueError, ArithmeticError:
                return math.nan

        if tes_alternative == 'greater':
            if lim_fn(H0) >= 0:
                theta_lim = (math.nan,)
            else:
                theta_lim = (solve('upX', True) if theta_1 >= H0 else solve('downX', False),)
        elif tes_alternative == 'less':
            if lim_fn(H0) <= 0:
                theta_lim = (math.nan,)
            else:
                theta_lim = (solve('downX', True) if theta_1 >= H0 else solve('upX', False),)
        else:
            lbs, ubs = (
                tes(
                    yi,
                    v,
                    H0=H0,
                    alternative=alternative,
                    alpha=alpha,
                    theta=theta_1,
                    tau2=tau2,
                    test=test,
                    tes_alternative=alt,
                    tes_alpha=tes_alpha / 2,
                    correct=correct,
                    rel_tol=rel_tol,
                    subdivisions=subdivisions,
                    tau2_lb=tau2_lb,
                    find_lim=True,
                ).theta_lim
                for alt in ('greater', 'less')
            )
            theta_lim = (lbs[0] if lbs else math.nan, ubs[0] if ubs else math.nan)
    return TesResult(
        k=k,
        O=o,
        E=e,
        OEratio=o / e,
        test=test,
        X2=x2,
        pval=pval,
        power=pow_,
        sig=sig,
        theta=theta_1 if single else th_arr,
        theta_lim=theta_lim,
        tes_alternative=tes_alternative,
        tes_alpha=tes_alpha,
        digits=metafopy.mfopt.stored_digits(digits, 4),
    )
