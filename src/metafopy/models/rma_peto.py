"""Peto's one-step method for pooling 2x2 tables (metafor's ``rma.peto``).

Yusuf et al. (1985), doi:10.1016/S0033-0620(85)80003-7, popularised the
method; the heterogeneity test is Peto's. See
https://wviechtb.github.io/metafor/reference/rma.peto.html and the source at
https://github.com/wviechtb/metafor/blob/master/R/rma.peto.r.
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
import metafopy.models.binary
import metafopy.models.call
import metafopy.models.rma_uni
import metafopy.models.tables

__all__ = ['PetoResult', 'rma_peto']

type FloatArray = numpy.typing.NDArray[numpy.float64]


def peto_terms(cells: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Observed minus expected counts ``ai - Ei``, the hypergeometric variances ``Vi`` and the expected counts ``Ei``."""
    ai, bi, ci, di = cells['ai'], cells['bi'], cells['ci'], cells['di']
    n1i, n2i = ai + bi, ci + di
    ni = ai + bi + ci + di
    xt, yt = ai + ci, bi + di
    ei = xt * n1i / ni
    with numpy.errstate(all='ignore'):
        vi = xt * yt * (n1i / ni) * (n2i / ni) / (ni - 1)
    return ai - ei, vi, ei


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PetoResult(metafopy.models.binary.FixedBinaryResult):
    """A fitted ``rma.peto`` model (the pooled effect is a log odds ratio)."""

    def weight_values(self) -> FloatArray:
        return peto_terms(self.tables.cells)[1]

    def render(self, digits: int, showfit: bool) -> str:
        k_pos = self.k_pos if self.k_pos is not None else 0
        out = metafopy.models.binary.format_header(self, digits, showfit, k_pos - 1)
        out += metafopy.models.binary.result_rows(self, digits, 'OR')
        out.append('')
        return '\n'.join(out)


@metafopy.models.call.records_call
def rma_peto(
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | collections.abc.Sequence[float] = 0.5,
    to: str | collections.abc.Sequence[str] = 'only0',
    drop00: bool | collections.abc.Sequence[bool] = True,
    level: float = 95,
    verbose: bool | int = False,
    digits: int = 4,
    na_action: metafopy.metadata.NAAction = 'omit',
) -> PetoResult:
    """Fit Peto's one-step method to a set of 2x2 tables.

    Mirrors metafor's ``rma.peto()``; see
    https://wviechtb.github.io/metafor/reference/rma.peto.html. The first
    element of ``add``, ``to`` and ``drop00`` is used for the individual
    ``yi``/``vi`` values, the second for the counts used in the pooled
    estimate (by default nothing is added to them).

    Examples
    --------
    >>> from metafopy import datasets, rma_peto
    >>> dat = datasets.load("dat.bcg")  # doctest: +SKIP
    >>> rma_peto(ai="tpos", bi="tneg", ci="cpos", di="cneg", data=dat)  # doctest: +SKIP

    ``verbose`` prints metafor's progress messages (to stderr).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or one of the two outcomes never occurred.
    """
    metafopy.core.verbose.binary_start(int(verbose), subset)
    add2, to2, drop2 = metafopy.models.binary.argument_pairs(add, to, drop00)
    tab = metafopy.models.tables.prepare_tables(
        'PETO',
        ai=ai,
        bi=bi,
        ci=ci,
        di=di,
        n1i=n1i,
        n2i=n2i,
        data=data,
        slab=slab,
        subset=subset,
        add=add2,
        to=to2,
        drop00=drop2,
        na_action=na_action,
    )
    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    o_minus_e, vi_peto, _ = peto_terms(tab.cells)
    sum_vi = float(vi_peto.sum())
    if sum_vi == 0:
        raise metafopy.errors.InputError("One of the two outcomes never occurred in any of the tables. Peto's method cannot be used.")
    beta = float(o_minus_e.sum()) / sum_vi
    se = math.sqrt(1 / sum_vi)
    zval = beta / se
    pval = 2 * float(scipy.stats.norm.sf(abs(zval)))
    crit = float(scipy.stats.norm.isf(alpha / 2))

    k_pos = int((vi_peto > 0).sum())
    vi_na = numpy.where(vi_peto == 0, numpy.nan, vi_peto)
    qe = max(0.0, float(numpy.nansum(o_minus_e**2 / vi_na)) - float(o_minus_e.sum()) ** 2 / float(numpy.nansum(vi_na)))
    k_yi = tab.k_yi
    if k_pos > 1:
        qep = float(scipy.stats.chi2.sf(qe, k_yi - 1))
        i2 = max(0.0, 100 * (qe - (k_yi - 1)) / qe) if qe != 0 else math.nan
        h2 = qe / (k_yi - 1)
    else:
        qep, i2, h2 = 1.0, 0.0, 1.0

    with numpy.errstate(all='ignore'):
        rss = float(numpy.sum((tab.yi - beta) ** 2 / tab.vi))
    ml, reml = metafopy.models.binary.fit_statistics(tab.yi, tab.vi, rss, k_yi)

    metafopy.core.verbose.binary_end(int(verbose), not bool(tab.not_na.all()), not bool(tab.not_na_yivi.all()))
    return PetoResult(
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
        measure='PETO',
        level=alpha,
        add=add2,
        to=to2,
        drop00=drop2,
        fit_stats_ml=ml,
        fit_stats_reml=reml,
        tables=tab,
        data=data,
        digits=digits,
    )
