"""Mantel-Haenszel meta-analysis of 2x2 tables and incidence rates (metafor's ``rma.mh``).

The pooled odds ratio follows Mantel and Haenszel (1959),
doi:10.1093/jnci/22.4.719, with the variance estimator of Robins, Breslow and
Greenland (1986), doi:10.2307/2531783; the pooled risk ratio and incidence rate
ratio follow Greenland and Robins (1985), doi:10.2307/2530923; the risk
difference variance is that of Sato, Greenland and Robins (1989),
doi:10.2307/2531593. See
https://wviechtb.github.io/metafor/reference/rma.mh.html and the source at
https://github.com/wviechtb/metafor/blob/master/R/rma.mh.r.
"""

import collections.abc
import dataclasses
import math

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.verbose
import metafopy.errors
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.binary
import metafopy.models.call
import metafopy.models.tables
import metafopy.printing

__all__ = ['MHResult', 'rma_mh']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
MEASURES = ('OR', 'RR', 'RD', 'IRR', 'IRD')
NAN = math.nan


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class MHResult(metafopy.models.binary.FixedBinaryResult):
    """A fitted ``rma.mh`` model; ``CO``/``MH`` are the Cochran and Mantel-Haenszel tests, ``BD``/``TA`` the Breslow-Day and Tarone tests (odds ratios only)."""

    CO: float = NAN
    COp: float = NAN
    MH: float = NAN
    MHp: float = NAN
    BD: float = NAN
    BDp: float = NAN
    TA: float = NAN
    TAp: float = NAN
    correct: bool = True

    def weight_values(self) -> FloatArray:
        c = self.tables.cells
        if self.measure in ('RR', 'OR', 'RD'):
            ni = c['ai'] + c['bi'] + c['ci'] + c['di']
            if self.measure == 'OR':
                return (c['bi'] / ni) * c['ci']
            if self.measure == 'RR':
                return (c['ci'] / ni) * (c['ai'] + c['bi'])
            return ((c['ai'] + c['bi']) / ni) * (c['ci'] + c['di'])
        ti = c['t1i'] + c['t2i']
        if self.measure == 'IRR':
            return (c['x2i'] / ti) * c['t1i']
        return (c['t1i'] / ti) * c['t2i']

    def render(self, digits: int, showfit: bool) -> str:
        out = metafopy.models.binary.format_header(self, digits, showfit, max(self.k_yi - 1, 0))
        scale = self.measure if self.measure in ('OR', 'RR', 'IRR') else None
        out += metafopy.models.binary.result_rows(self, digits, scale)
        if self.measure == 'OR':
            out.append('')
            mh = metafopy.printing.fmt_num(self.MH, digits)
            ta = metafopy.printing.fmt_num(self.TA, digits)
            width = 1 if (mh == 'NA' and ta == 'NA') else max(len(mh) if mh != 'NA' else 0, len(ta) if ta != 'NA' else 0)
            if mh == 'NA':
                out.append('Cochran-Mantel-Haenszel Test:    test value not computable for these data')
            else:
                k_pos = self.k_pos if self.k_pos is not None else 0
                pad = ' ' * (len(str(k_pos)) - 1)
                out.append(f'Cochran-Mantel-Haenszel Test:    CMH = {mh.rjust(width)}, df = 1,{pad} p-val {add0_pval(self.MHp, digits)}')
            if ta == 'NA':
                out.append("Tarone's Test for Heterogeneity: test value not computable for these data")
            else:
                k_pos = self.k_pos if self.k_pos is not None else 0
                out.append(f"Tarone's Test for Heterogeneity: X^2 = {ta.rjust(width)}, df = {k_pos - 1}, p-val {add0_pval(self.TAp, digits)}")
        if self.measure == 'IRR':
            out.append('')
            if math.isnan(self.MH):
                out.append('Mantel-Haenszel Test: test value not computable for these data')
            else:
                out.append(
                    f'Mantel-Haenszel Test: MH = {metafopy.printing.fmt_num(self.MH, digits)}, df = 1, p-val {metafopy.models.rma_uni.pfmt(self.MHp, digits)}'
                )
        out.append('')
        return '\n'.join(out)


def add0_pval(p: float, digits: int) -> str:
    """R's ``fmtp(p, digits, equal=TRUE, sep=TRUE, add0=TRUE)``."""
    s = metafopy.models.rma_uni.pfmt(p, digits)
    return s.replace('< .', '< 0.') if s.startswith('<') else s


def sum_is_zero(x: FloatArray) -> bool:
    return bool(numpy.sum(x) == 0)


def fdiv(a: float, b: float) -> float:
    """``a / b`` as R computes it: a zero divisor gives ``Inf`` or ``NaN`` instead of an error."""
    with numpy.errstate(all='ignore'):
        return float(numpy.float64(a) / numpy.float64(b))


def fsqrt(x: float) -> float:
    """Square root that is ``NaN`` for negative values (R's ``sqrt``, which warns)."""
    return math.sqrt(x) if x >= 0 else NAN


def mh_estimate(measure: str, c: dict[str, FloatArray], correct: bool) -> tuple[float, float, dict[str, float], int | None]:
    """Pooled estimate (``NaN`` when not defined), its standard error and the extra test statistics."""
    extra: dict[str, float] = {'CO': NAN, 'COp': NAN, 'MH': NAN, 'MHp': NAN, 'BD': NAN, 'BDp': NAN, 'TA': NAN, 'TAp': NAN}
    k_pos: int | None = None
    cc = 0.5 if correct else 0.0
    with numpy.errstate(all='ignore'):
        if measure == 'OR':
            ai, bi, ci, di = c['ai'], c['bi'], c['ci'], c['di']
            n1i, n2i = ai + bi, ci + di
            ni = ai + bi + ci + di
            pi = ai / ni + di / ni
            qi = bi / ni + ci / ni
            ri = (ai / ni) * di
            si = (bi / ni) * ci
            r, s = float(ri.sum()), float(si.sum())
            if r == 0 or s == 0:
                beta, se, beta_exp = NAN, NAN, NAN
            else:
                beta_exp = r / s
                beta = math.log(beta_exp)
                se = fsqrt(0.5 * (float((pi * ri).sum()) / r**2 + float((pi * si + qi * ri).sum()) / (r * s) + float((qi * si).sum()) / s**2))
            xt = ai + ci
            yt = bi + di
            if sum_is_zero(xt) or sum_is_zero(yt):
                co = cop = mh = mhp = NAN
            else:
                num = (abs(float((ai - (n1i / ni) * xt).sum())) - cc) ** 2
                co = num / float(((n1i / ni) * (n2i / ni) * (xt * (yt / ni))).sum())
                cop = float(scipy.stats.chi2.sf(co, 1))
                mh = num / float(((n1i / ni) * (n2i / ni) * (xt * (yt / (ni - 1)))).sum())
                mhp = float(scipy.stats.chi2.sf(mh, 1))
            extra.update(CO=co, COp=cop, MH=mh, MHp=mhp)
            if math.isnan(beta):
                extra.update(BD=NAN, TA=NAN, BDp=NAN, TAp=NAN)
                k_pos = 0
            else:
                if beta_exp == 1:
                    n11 = (n1i / ni) * xt
                else:
                    a = beta_exp * (n1i + xt) + (n2i - xt)
                    b = numpy.sqrt(a**2 - 4 * n1i * xt * beta_exp * (beta_exp - 1))
                    n11 = (a - b) / (2 * (beta_exp - 1))
                pos = (n11 > 0) & (xt > 0) & (yt > 0)
                k_pos = int(pos.sum())
                n11 = n11[pos]
                n12 = n1i[pos] - n11
                n21 = xt[pos] - n11
                n22 = n11 - n1i[pos] - xt[pos] + ni[pos]
                v = 1 / (1 / n11 + 1 / n12 + 1 / n21 + 1 / n22)
                bd = max(0.0, float((((ai[pos] - n11) ** 2) / v).sum()))
                ta = max(0.0, bd - float((ai[pos] - n11).sum()) ** 2 / float(v.sum()))
                extra.update(BD=bd, TA=ta)
                if k_pos > 1:
                    extra.update(BDp=float(scipy.stats.chi2.sf(bd, k_pos - 1)), TAp=float(scipy.stats.chi2.sf(ta, k_pos - 1)))
            return beta, se, extra, k_pos

        if measure == 'RR':
            ai, bi, ci, di = c['ai'], c['bi'], c['ci'], c['di']
            n1i, n2i = ai + bi, ci + di
            ni = ai + bi + ci + di
            r = float((ai * (n2i / ni)).sum())
            s = float((ci * (n1i / ni)).sum())
            if sum_is_zero(ai) or sum_is_zero(ci):
                return NAN, NAN, extra, k_pos
            beta = math.log(r / s)
            se = fsqrt(fdiv(float(((n1i / ni) * (n2i / ni) * (ai + ci) - (ai / ni) * ci).sum()), r * s))
            return beta, se, extra, k_pos

        if measure == 'RD':
            ai, bi, ci, di = c['ai'], c['bi'], c['ci'], c['di']
            n1i, n2i = ai + bi, ci + di
            ni = ai + bi + ci + di
            denom = float((n1i * (n2i / ni)).sum())
            beta = fdiv(float((ai * (n2i / ni) - ci * (n1i / ni)).sum()), denom)
            num = beta * float((ci * (n1i / ni) ** 2 - ai * (n2i / ni) ** 2 + (n1i / ni) * (n2i / ni) * (n2i - n1i) / 2).sum()) + float(
                (ai * (n2i - ci) / ni + ci * (n1i - ai) / ni).sum() / 2
            )
            return beta, fsqrt(fdiv(num, denom**2)), extra, k_pos

        x1i, x2i, t1i, t2i = c['x1i'], c['x2i'], c['t1i'], c['t2i']
        ti = t1i + t2i
        if measure == 'IRR':
            r = float((x1i * (t2i / ti)).sum())
            s = float((x2i * (t1i / ti)).sum())
            if sum_is_zero(x1i) or sum_is_zero(x2i):
                beta, se = NAN, NAN
            else:
                beta = math.log(r / s)
                se = fsqrt(fdiv(float(((t1i / ti) * (t2i / ti) * (x1i + x2i)).sum()), r * s))
            xt = x1i + x2i
            if not sum_is_zero(xt):
                mh = (abs(float((x1i - xt * (t1i / ti)).sum())) - cc) ** 2 / float((xt * (t1i / ti) * (t2i / ti)).sum())
                extra.update(MH=mh, MHp=float(scipy.stats.chi2.sf(mh, 1)))
            return beta, se, extra, k_pos

        denom = float(((t1i / ti) * t2i).sum())
        beta = fdiv(float(((x1i * t2i - x2i * t1i) / ti).sum()), denom)
        se = fdiv(fsqrt(float((((t1i / ti) * t2i) ** 2 * (x1i / t1i**2 + x2i / t2i**2)).sum())), denom)
        return beta, se, extra, k_pos


@metafopy.models.call.records_call
def rma_mh(
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    measure: str = 'OR',
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | collections.abc.Sequence[float] = 0.5,
    to: str | collections.abc.Sequence[str] = 'only0',
    drop00: bool | collections.abc.Sequence[bool] = True,
    correct: bool = True,
    level: float = 95,
    verbose: bool | int = False,
    digits: int = 4,
    na_action: metafopy.metadata.NAAction = 'omit',
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> MHResult:
    """Fit a Mantel-Haenszel meta-analysis of 2x2 tables (``measure`` ``'OR'``, ``'RR'``, ``'RD'``) or rate data (``'IRR'``, ``'IRD'``).

    Mirrors metafor's ``rma.mh()``, see <https://wviechtb.github.io/metafor/reference/rma.mh.html>. The pooled effect is the Mantel-Haenszel
    estimate (Mantel & Haenszel, 1959; Greenland & Robins, 1985; Robins, Breslow & Greenland, 1986; Nurminen, 1981). The individual
    effect sizes and sampling variances are computed as by [escalc][metafopy.escalc]. For odds ratios the result also holds the
    Cochran-Mantel-Haenszel test and Tarone's test of heterogeneity. The first element of ``add``, ``to`` and ``drop00`` is used when computing
    the individual ``yi``/``vi`` values, the second one for the counts entering the Mantel-Haenszel estimator (by default nothing is added to them).

    Args:
        ai: Frequencies of the 2x2 tables, upper left cell (event in group 1), as an array or the name of a column of ``data``.
        bi: Frequencies of the 2x2 tables, upper right cell (no event in group 1).
        ci: Frequencies of the 2x2 tables, lower left cell (event in group 2).
        di: Frequencies of the 2x2 tables, lower right cell (no event in group 2).
        n1i: Size of group 1 (row total); can be given instead of ``bi``.
        n2i: Size of group 2 (row total); can be given instead of ``di``.
        x1i: Number of events in group 1 (rate data, ``'IRR'`` and ``'IRD'``).
        x2i: Number of events in group 2 (rate data).
        t1i: Total person-time of group 1 (rate data).
        t2i: Total person-time of group 2 (rate data).
        measure: Effect measure: ``'OR'`` (log odds ratio, the default), ``'RR'`` (log risk ratio), ``'RD'`` (risk difference), ``'IRR'`` (log
            incidence rate ratio) or ``'IRD'`` (incidence rate difference). Other measures raise an error.
        data: Data frame holding the variables named in the other arguments.
        slab: Study labels, as a sequence or a column name. Defaults to ``1, 2, ...``.
        subset: Which studies to use: a boolean array, an array of 0-based indices, or a string expression evaluated on ``data``.
        add: Constant added to the cell counts of tables with zero cells. A single value, or a pair: the first for the ``yi``/``vi`` values and the
            second for the Mantel-Haenszel estimator. Default ``0.5``, so that nothing is added to the counts of the estimator.
        to: Which tables get ``add``: ``'all'``, ``'only0'`` (those with a zero cell, the default), ``'if0all'`` (all, if any table has a zero cell)
            or ``'none'``. A single value, or a pair as for ``add``.
        drop00: Drop studies with no events (or only events) in both groups. A single value, or a pair as for ``add``. Default ``True``.
        correct: Use the continuity correction in the Cochran-Mantel-Haenszel test (odds ratios only).
        level: Confidence interval level in percent (``95`` is 95%).
        verbose: Print metafor's progress messages to stderr.
        digits: Number of decimals used when printing, stored in the result.
        na_action: What to do with missing values: ``'omit'`` (drop those studies, the default), ``'fail'`` (raise) or ``'pass'`` (keep them).
        onlyo1: Passed to [escalc][metafopy.escalc]: only the first outcome of the 2x2 tables is used.
        addyi: Passed to [escalc][metafopy.escalc]: apply ``add`` to the effect size ``yi``.
        addvi: Passed to [escalc][metafopy.escalc]: apply ``add`` to the sampling variance ``vi``.

    Returns:
        The fitted model. The pooled estimate is in ``beta`` with ``se``, ``zval``, ``pval``, ``ci_lb`` and ``ci_ub`` (on the log scale for
        ``'OR'``, ``'RR'`` and ``'IRR'``); ``QE``, ``QEp``, ``I2`` and ``H2`` describe the heterogeneity; ``CO``/``COp``, ``MH``/``MHp``,
        ``BD``/``BDp`` and ``TA``/``TAp`` are the Cochran, Mantel-Haenszel, Breslow-Day and Tarone tests (odds ratios only); ``summary()`` prints
        the table.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or ``measure`` is not one of the supported measures.

    Examples:
        >>> from metafopy import datasets, rma_mh
        >>> dat = datasets.load("dat.bcg")  # doctest: +SKIP
        >>> rma_mh(ai="tpos", bi="tneg", ci="cpos", di="cneg", data=dat)  # doctest: +SKIP
    """
    metafopy.core.verbose.binary_start(int(verbose), subset)
    if measure not in MEASURES:
        raise metafopy.errors.InputError('Mantel-Haenszel method can only be used with measures OR, RR, RD, IRR, and IRD.')
    add2, to2, drop2 = metafopy.models.binary.argument_pairs(add, to, drop00)
    tab = metafopy.models.tables.prepare_tables(
        measure,
        ai=ai,
        bi=bi,
        ci=ci,
        di=di,
        n1i=n1i,
        n2i=n2i,
        x1i=x1i,
        x2i=x2i,
        t1i=t1i,
        t2i=t2i,
        data=data,
        slab=slab,
        subset=subset,
        add=add2,
        to=to2,
        drop00=drop2,
        onlyo1=onlyo1,
        addyi=addyi,
        addvi=addvi,
        na_action=na_action,
    )
    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    beta, se, extra, k_pos = mh_estimate(measure, tab.cells, correct)
    crit = float(scipy.stats.norm.isf(alpha / 2))
    with numpy.errstate(all='ignore'):
        zval = float(numpy.float64(beta) / numpy.float64(se))
    pval = 2 * float(scipy.stats.norm.sf(abs(zval))) if not math.isnan(zval) else NAN

    yi, vi, k_yi = tab.yi, tab.vi, tab.k_yi
    with numpy.errstate(all='ignore'):
        wi = 1.0 / vi
        if k_yi > 1:
            qe = metafopy.models.binary.positive_max(float(numpy.sum(wi * (yi - beta) ** 2)))
            qep = float(scipy.stats.chi2.sf(qe, k_yi - 1)) if not math.isnan(qe) else NAN
            i2 = metafopy.models.binary.positive_max(100 * (qe - (k_yi - 1)) / qe) if qe != 0 else NAN
            h2 = qe / (k_yi - 1)
        else:
            qe, qep, i2, h2 = 0.0, 1.0, 0.0, 1.0
    ml, reml = metafopy.models.binary.fit_statistics(yi, vi, qe, k_yi)

    metafopy.core.verbose.binary_end(int(verbose), not bool(tab.not_na.all()), not bool(tab.not_na_yivi.all()))
    return MHResult(
        beta=numpy.array([beta]),
        se=numpy.array([se]),
        zval=numpy.array([zval]),
        pval=numpy.array([pval]),
        ci_lb=numpy.array([beta - crit * se]),
        ci_ub=numpy.array([beta + crit * se]),
        vb=numpy.array([[se**2]]),
        tau2=0.0,
        I2=i2,
        H2=h2,
        QE=qe,
        QEp=qep,
        k=tab.k,
        k_f=tab.k_f,
        k_yi=k_yi,
        k_all=tab.k_all,
        k_pos=k_pos,
        p=1,
        parms=1,
        measure=measure,
        level=alpha,
        add=add2,
        to=to2,
        drop00=drop2,
        fit_stats_ml=ml,
        fit_stats_reml=reml,
        tables=tab,
        data=data,
        digits=metafopy.mfopt.reduce_digits(digits),
        CO=extra['CO'],
        COp=extra['COp'],
        MH=extra['MH'],
        MHp=extra['MHp'],
        BD=extra['BD'],
        BDp=extra['BDp'],
        TA=extra['TA'],
        TAp=extra['TAp'],
        correct=correct,
    )
