"""Multiplicity adjustments of emmeans summaries: critical values for the intervals and adjusted p-values.

R source: https://github.com/rvlenth/emmeans/blob/master/R/summary.R (``.adj.critval``, ``.adj.p.value``) and
``R/emm-contr.R`` (``.pdunnx``, ``.qdunnx``); reference: https://rvlenth.github.io/emmeans/reference/summary.emmGrid.html.
Tukey and Dunnett adjustments use the studentized range of :mod:`metafopy.core.tukey` as R does. The ``mvt`` adjustment integrates
the multivariate t distribution with SciPy's quasi-Monte Carlo routine (R uses ``mvtnorm``), so it agrees with R to about 1e-3.
"""

import collections.abc
import dataclasses
import math
import warnings

import numpy
import numpy.typing
import scipy.optimize
import scipy.stats

import metafopy.core.padjust
import metafopy.core.tukey
import metafopy.core.zeroin
import metafopy.errors

__all__ = ['Adjustment', 'Family', 'critical_values', 'p_values', 'parse_side']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] the order in which emmeans matches abbreviations
TWO_SIDED_ONLY = ('tukey', 'scheffe', 'dunnettx')
P_METHODS = ('sidak', 'tukey', 'scheffe', 'dunnettx', 'mvt', *metafopy.core.padjust.P_ADJUST_METHODS)
MVT_SEED = 20260101
MVT_MAXPTS = 20000


@dataclasses.dataclass(frozen=True, slots=True)
class Family:
    """What the adjustments need to know about the estimates being summarised (emmeans' ``fam.info``, ``by.rows``, ``sch.rank``, ``corrmat``).

    ``ragged`` is set when the ``by`` groups differ in size (or in Scheffe rank), when emmeans shows ``(varies)`` for the family size.
    """

    fam_size: float
    n_contr: int
    est_type: int
    by_rows: tuple[tuple[int, ...], ...]
    sch_rank: tuple[int, ...]
    corr: FloatArray | None
    ragged: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Adjustment:
    """Critical values (or p-values) with the message emmeans shows and the method used."""

    values: FloatArray
    message: str | None
    adjust: str
    notes: tuple[str, ...] = ()


def match_method(adjust: str, methods: collections.abc.Sequence[str]) -> str | None:
    """R's ``pmatch``: an exact match, else a unique partial match."""
    if adjust in methods:
        return adjust
    hits = [m for m in methods if m.startswith(adjust)] if adjust else []
    return hits[0] if len(hits) == 1 else None


# auditme: accept R's pmatch is positional; the order maps each name to its code in SIDE_CODES
SIDE_OPTIONS = ('left', 'both', 'right', 'two-sided', 'noninferiority', 'nonsuperiority', 'equivalence', 'superiority', 'inferiority')
# auditme: accept positional, paired with SIDE_CODES
SIDE_SYMBOLS = ('0', '2', '-1', '1', '+1', '<', '>', '!=', '=')
SIDE_CODES = (1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3, 3, 1, 3, 2, 2)


def parse_side(side: int | float | str) -> int:
    """The tail of the tests and intervals, -1 (left), 0 (both) or 1 (right), from emmeans' ``side`` argument.

    ``side`` is a number (``-1``, ``0``, ``1``), a symbol (``'<'``, ``'>'``, ``'!='``), or a possibly abbreviated word
    (``'left'``, ``'right'``, ``'both'``, ``'two-sided'``, ``'noninferiority'``, ``'nonsuperiority'``, ``'equivalence'``,
    ``'superiority'``, ``'inferiority'``); anything else means both sides, as in emmeans.
    """
    options = (*SIDE_OPTIONS, *SIDE_SYMBOLS)
    text = str(int(side)) if isinstance(side, int | float) and float(side).is_integer() else str(side)
    found = match_method(text, options)
    return 0 if found is None else SIDE_CODES[options.index(found)] - 2


def pf_sf(x: float, df1: float, df2: float) -> float:
    return float(scipy.stats.chi2.sf(x * df1, df1)) if math.isinf(df2) else float(scipy.stats.f.sf(x, df1, df2))


def pf_cdf(x: float, df1: float, df2: float) -> float:
    return float(scipy.stats.chi2.cdf(x * df1, df1)) if math.isinf(df2) else float(scipy.stats.f.cdf(x, df1, df2))


def qf(p: float, df1: float, df2: float) -> float:
    return float(scipy.stats.chi2.ppf(p, df1) / df1) if math.isinf(df2) else float(scipy.stats.f.ppf(p, df1, df2))


def qt(p: float, df: float) -> float:
    return float(scipy.stats.norm.ppf(p)) if math.isinf(df) else float(scipy.stats.t.ppf(p, df))


def pt_sf(x: float, df: float) -> float:
    return float(scipy.stats.norm.sf(x)) if math.isinf(df) else float(scipy.stats.t.sf(x, df))


def pdunnx(x: float, k: float, df: float) -> float:
    """Lund and Lund's approximation to the Dunnett distribution used by emmeans' ``dunnettx`` (``.pdunnx``)."""
    twt = (k - 1) / k
    tukey = metafopy.core.tukey.ptukey(math.sqrt(2) * x, (1 + math.sqrt(1 + 8 * k)) / 2, df)
    sidak = pf_cdf(x * x, 1, df) ** k
    return float(twt * tukey + (1 - twt) * sidak)


def qdunnx(p: float, k: float, df: float) -> float:
    """Critical value of ``dunnettx`` (``.qdunnx``); the root is found with R's ``uniroot`` at its tolerance of 5e-4."""
    if k < 1.005:
        return qt(1 - 0.5 * (1 - p), df)
    fam = (1 + math.sqrt(1 + 8 * k)) / 2
    xtuk = metafopy.core.tukey.qtukey(p, fam, df) / math.sqrt(2)
    xsid = math.sqrt(qf(p ** (1 / k), 1, df))
    if abs(xtuk - xsid) < 5e-04:
        return xtuk
    try:
        return metafopy.core.zeroin.uniroot(lambda x: pdunnx(x, k, df) - p, xtuk, xsid, tol=5e-04).root
    except ValueError, metafopy.errors.InputError:
        warnings.warn('Root-finding failed; using qtukey approximation for Dunnett quantile', stacklevel=2)
        return xtuk


def pmvt_both(x: float, df: float, corr: FloatArray) -> float:
    """P(|T_i| <= x for all i) for a multivariate t with correlation ``corr``."""
    k = corr.shape[0]
    if k == 1:
        return 1 - 2 * pt_sf(x, df)
    lower = numpy.full(k, -x)
    upper = numpy.full(k, x)
    if math.isinf(df):
        return float(
            scipy.stats.multivariate_normal.cdf(
                upper, cov=corr, allow_singular=True, maxpts=MVT_MAXPTS, lower_limit=lower, rng=numpy.random.default_rng(MVT_SEED)
            )
        )
    dist = scipy.stats.multivariate_t(shape=corr, df=int(df), allow_singular=True)
    return float(dist.cdf(upper, lower_limit=lower, random_state=numpy.random.default_rng(MVT_SEED), maxpts=MVT_MAXPTS))


def pmvt_lower(x: float, df: float, corr: FloatArray) -> float:
    """P(T_i <= x for all i) for a multivariate t with correlation ``corr``."""
    k = corr.shape[0]
    if k == 1:
        return 1 - pt_sf(x, df)
    upper = numpy.full(k, x)
    if math.isinf(df):
        return float(scipy.stats.multivariate_normal.cdf(upper, cov=corr, allow_singular=True, maxpts=MVT_MAXPTS, rng=numpy.random.default_rng(MVT_SEED)))
    dist = scipy.stats.multivariate_t(shape=corr, df=int(df), allow_singular=True)
    return float(dist.cdf(upper, random_state=numpy.random.default_rng(MVT_SEED), maxpts=MVT_MAXPTS))


def qmvt_lower(p: float, df: float, corr: FloatArray) -> float:
    """The ``c`` with P(T_i <= c for all i) = ``p`` for a multivariate t with correlation ``corr``."""
    return float(scipy.optimize.brentq(lambda c: pmvt_lower(c, df, corr) - p, -100.0, 100.0, xtol=1e-6))


def qmvt_both(p: float, df: float, corr: FloatArray) -> float:
    """The ``c`` with P(|T_i| <= c for all i) = ``p`` for a multivariate t with correlation ``corr``."""
    return float(scipy.optimize.brentq(lambda c: pmvt_both(c, df, corr) - p, 1e-6, 100.0, xtol=1e-6))


def tidy_method(adjust: str, methods: collections.abc.Sequence[str], default: str | None) -> str:
    adjust = adjust.lower()
    if adjust in ('bh', 'by'):
        adjust = adjust.upper()
    found = match_method(adjust, methods)
    if found is None:
        if default is None:
            raise metafopy.errors.InputError(f"Adjust method '{adjust}' is not recognized or not valid")
        return default
    return found


def change(adjust: str, new: str, reason: str) -> tuple[str, str]:
    return new, f'Note: adjust = "{adjust}" was changed to "{new}" because "{adjust}" is {reason}'


def critical_values(est: FloatArray, level: float, df: FloatArray, adjust: str, fam: Family, tail: int = 0) -> Adjustment:
    """Multipliers of the standard errors for the intervals (emmeans' ``.adj.critval``); ``tail`` is -1, 0 or 1 for one-sided intervals.

    Raises:
        metafopy.errors.InputError: the adjustment is not known, or ``mvt`` is asked for without the correlations of the estimates.
    """
    methods = ('sidak', 'tukey', 'scheffe', 'dunnettx', 'mvt', 'bonferroni', 'none')
    n_contr = fam.n_contr
    notes: list[str] = []
    scheffe = adjust.lower().startswith('sch')
    method = 'none' if (n_contr == 1 and not fam.ragged and not scheffe) else tidy_method(adjust, methods, 'bonferroni')
    adaptive = bool(numpy.any(numpy.isnan(est)))
    ragged = fam.ragged or adaptive or method == 'mvt'
    if tail != 0 and method in TWO_SIDED_ONLY:
        method, note = change(method, 'sidak', 'not appropriate for one-sided inferences')
        notes.append(note)
    if fam.est_type != 3 and method == 'tukey':
        method, note = change(method, 'sidak', 'only appropriate for one set of pairwise comparisons')
        notes.append(note)
    df = numpy.where(numpy.isnan(df), numpy.inf, df)
    by_rows = fam.by_rows
    divisor = 1 if tail != 0 else 2
    cv = numpy.zeros(est.shape[0])
    fam_size = fam.fam_size
    for jj, rows in enumerate(by_rows):
        idx = list(rows)
        d = df[idx]
        if ragged:
            n_contr = max(1, int(numpy.sum(~numpy.isnan(est[idx]))))
            fam_size = (1 + math.sqrt(1 + 8 * n_contr)) / 2
        for pos, row in enumerate(rows):
            dfi = float(d[pos])
            match method:
                case 'none':
                    value = -qt((1 - level) / divisor, dfi)
                case 'sidak':
                    value = -qt((1 - level ** (1 / n_contr)) / divisor, dfi)
                case 'bonferroni':
                    value = -qt((1 - level) / n_contr / divisor, dfi)
                case 'tukey':
                    value = metafopy.core.tukey.qtukey(level, fam_size, dfi) / math.sqrt(2)
                case 'scheffe':
                    value = math.sqrt(fam.sch_rank[jj] * qf(level, fam.sch_rank[jj], dfi))
                case 'dunnettx':
                    value = qdunnx(level, n_contr, dfi)
                case _:
                    if fam.corr is None:
                        raise metafopy.errors.InputError("The 'mvt' adjustment needs the correlations of the estimates.")
                    sub = fam.corr[numpy.ix_(idx, idx)]
                    value = qmvt_both(level, dfi, sub) if tail == 0 else qmvt_lower(level, dfi, sub)
            cv[row] = value
    if ragged:
        size = max(len(r) for r in by_rows)
        shown_fam = (1 + math.sqrt(1 + 8 * size)) / 2
        scheffe_dim = '(varies)'
    else:
        shown_fam = fam.fam_size
        scheffe_dim = str(fam.sch_rank[0])
    chk = {'none': 1, 'tukey': 2, 'scheffe': 3}.get(method, 99)
    message = None
    if chk > 1 and not (shown_fam < 3 and chk < 10):
        if adaptive and len(by_rows) > 1:
            xtra = {'tukey': 'for varying family sizes', 'scheffe': 'with varying rank'}.get(method, 'for varying numbers of estimates')
        elif method == 'tukey':
            xtra = f'for comparing a family of {fmt_num(fam_size)} estimates'
        elif method == 'scheffe':
            xtra = f'with rank {scheffe_dim}'
        else:
            xtra = f'for {n_contr} estimates'
        message = f'Conf-level adjustment: {method} method {xtra}'
    return Adjustment(cv, message, method, tuple(notes))


def fmt_num(x: float) -> str:
    return f'{x:.15g}'


def p_values(t: FloatArray, df: FloatArray, adjust: str, fam: Family, tail: int = 0) -> Adjustment:
    """The p-values for the t (or z) ratios (emmeans' ``.adj.p.value``); ``tail`` is -1 (left-sided), 0 (two-sided) or 1 (right-sided).

    Raises:
        metafopy.errors.InputError: the adjustment is not known, or ``mvt`` is asked for without the correlations of the estimates.
    """
    methods = ('sidak', 'tukey', 'scheffe', 'dunnettx', 'mvt', *metafopy.core.padjust.P_ADJUST_METHODS)
    n_contr = fam.n_contr
    notes: list[str] = []
    scheffe = adjust.lower().startswith('sch')
    method = 'none' if (n_contr == 1 and not fam.ragged and not scheffe) else tidy_method(adjust, methods, None)
    adaptive = method != 'none' and bool(numpy.any(numpy.isnan(t)))
    if tail != 0 and method in TWO_SIDED_ONLY:
        method, note = change(method, 'sidak', 'not appropriate for one-sided inferences')
        notes.append(note)
    if fam.est_type != 3 and method == 'tukey':
        method, note = change(method, 'sidak', 'only appropriate for one set of pairwise comparisons')
        notes.append(note)
    df = numpy.where(numpy.isnan(df), numpy.inf, df)
    ragged = fam.ragged or method == 'mvt' or method in metafopy.core.padjust.P_ADJUST_METHODS or adaptive
    if tail == 0:
        p_unadj = numpy.array([2 * pt_sf(abs(ti), float(di)) for ti, di in zip(t, df, strict=True)])
    else:
        p_unadj = numpy.array([pt_sf(-ti if tail < 0 else ti, float(di)) for ti, di in zip(t, df, strict=True)])
    pval = numpy.zeros(t.shape[0])
    fam_size = fam.fam_size
    by_rows = fam.by_rows if ragged else (tuple(range(t.shape[0])),)
    for jj, rows in enumerate(by_rows):
        idx = list(rows)
        unadj = p_unadj[idx]
        abst = numpy.abs(t[idx])
        d = df[idx]
        if ragged:
            n_contr = max(int(numpy.sum(~numpy.isnan(unadj))), 1)
            fam_size = (1 + math.sqrt(1 + 8 * n_contr)) / 2
        if method in metafopy.core.padjust.P_ADJUST_METHODS:
            pval[idx] = metafopy.core.padjust.p_adjust(unadj, method)
            continue
        for pos, row in enumerate(idx):
            dfi = float(d[pos])
            match method:
                case 'none':
                    pval[row] = unadj[pos]
                case 'sidak':
                    pval[row] = 1 - (1 - unadj[pos]) ** n_contr
                case 'tukey':
                    pval[row] = metafopy.core.tukey.ptukey(math.sqrt(2) * abst[pos], fam_size, dfi, lower_tail=False)
                case 'scheffe':
                    rank = fam.sch_rank[jj]
                    pval[row] = pf_sf(t[row] ** 2 / rank, rank, dfi)
                case 'dunnettx':
                    pval[row] = 1 - pdunnx(abst[pos], n_contr, dfi)
                case _:
                    if fam.corr is None:
                        raise metafopy.errors.InputError("The 'mvt' adjustment needs the correlations of the estimates.")
                    sub = fam.corr[numpy.ix_(idx, idx)]
                    if tail == 0:
                        pval[row] = 1 - pmvt_both(abst[pos], dfi, sub)
                    else:
                        pval[row] = 1 - pmvt_lower(t[row] if tail > 0 else -t[row], dfi, sub)
    if ragged:
        nc = max(len(r) for r in fam.by_rows)
        fs = (1 + math.sqrt(1 + 8 * nc)) / 2
        scheffe_dim = '(varies)'
    else:
        nc, fs = n_contr, fam.fam_size
        scheffe_dim = str(fam.sch_rank[0])
    chk = {'none': 1, 'tukey': 2, 'scheffe': 3}.get(method, 99)
    message = None
    if chk > 1 and not (fs < 3 and chk < 10):
        if adaptive and len(fam.by_rows) > 1:
            xtra = {'tukey': 'for varying family sizes', 'scheffe': 'with varying rank'}.get(method, 'for varying numbers of tests')
        elif method == 'tukey':
            xtra = f'for comparing a family of {fmt_num(fam_size)} estimates'
        elif method == 'scheffe':
            xtra = f'with rank {scheffe_dim}'
        else:
            xtra = f'for {n_contr} tests'
        message = f'P value adjustment: {method} method {xtra}'
    return Adjustment(pval, message, method, tuple(notes))
