"""Independent re-derivations of the quantities metafopy computes, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as the value of an ``expect`` key. The function is called with the arguments of the case that it declares
(see ``cases.formula_value``).
"""

import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]
type FsnType = typing.Literal['Rosenthal', 'Orwin', 'Rosenberg', 'General']

KNAPP_HARTUNG_TESTS = ('adhoc', 'knha')


def risk_difference_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Risk difference ``p1 - p2`` of two independent proportions, ``p1 = ai / (ai + bi)`` and ``p2 = ci / (ci + di)`` (the definition)."""
    return ai / (ai + bi) - ci / (ci + di)


def risk_difference_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Variance of the difference of two independent binomial proportions, ``p1 (1 - p1) / n1 + p2 (1 - p2) / n2``.

    Derivation: the variance of a difference of independent estimates is the sum of their variances, and that of a binomial proportion is ``p (1 - p) / n``.
    """
    n1, n2 = ai + bi, ci + di
    p1, p2 = ai / n1, ci / n2
    return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2


def adjusted_table(
    ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0'
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Continuity correction of 2x2 tables: ``add`` goes to all four cells of a table with a zero cell (``to='only0'``) or of every table (``to='all'``).

    The one-half correction is Haldane (1956, Annals of Human Genetics) and Anscombe (1956, Biometrika).
    """
    if to == 'all':
        rows = numpy.ones(ai.shape, dtype=bool)
    elif to == 'only0':
        rows = (ai == 0) | (bi == 0) | (ci == 0) | (di == 0)
    else:
        raise ValueError(f'unsupported "to": {to}')
    bump = numpy.where(rows, add, 0.0)
    return ai + bump, bi + bump, ci + bump, di + bump


def adjusted_rates(x1i: FloatArray, x2i: FloatArray, add: float = 0.5, to: str = 'only0') -> tuple[FloatArray, FloatArray]:
    """The same correction for two event counts: ``add`` goes to both counts of a study with a zero count (``to='only0'``) or of every study."""
    rows = numpy.ones(x1i.shape, dtype=bool) if to == 'all' else (x1i == 0) | (x2i == 0)
    bump = numpy.where(rows, add, 0.0)
    return x1i + bump, x2i + bump


def group_proportions(
    ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str, add: float, to: str
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Group sizes and the proportions at which a variance is evaluated.

    ``LS`` and ``UB`` use each study's own proportions; ``AV`` the size-weighted mean over the studies (Sanchez-Meca & Marin-Martinez 2001, Psychological
    Methods); ``H0`` the pooled proportion of both groups, the estimate under the null hypothesis of equal proportions.
    """
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    if vtype == 'AV':
        p1 = numpy.full(a.shape, float(numpy.sum(n1 * p1) / numpy.sum(n1)))
        p2 = numpy.full(a.shape, float(numpy.sum(n2 * p2) / numpy.sum(n2)))
    elif vtype == 'H0':
        pooled = (a + c) / (n1 + n2)
        p1, p2 = pooled, pooled
    return n1, n2, p1, p2


def log_risk_ratio_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Log risk ratio ``log(p1 / p2)`` (the definition) on the corrected table."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    return numpy.log(a / (a + b)) - numpy.log(c / (c + d))


def log_risk_ratio_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str = 'LS', add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Delta-method variance of a log proportion, ``(1 - p) / (n p)``, summed over the two groups (Rothman, Greenland & Lash 2008, Modern Epidemiology)."""
    n1, n2, p1, p2 = group_proportions(ai, bi, ci, di, vtype, add, to)
    return (1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)


def log_odds_ratio_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Log odds ratio ``log(a d / (b c))`` (the definition) on the corrected table."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    return numpy.log(a * d / (b * c))


def log_odds_ratio_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str = 'LS', add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance of the log odds ratio, Woolf (1955, Annals of Human Genetics): ``1/a + 1/b + 1/c + 1/d`` (``LS``).

    ``AV`` and ``H0`` evaluate the delta-method variance of a log odds, ``1 / (n p (1 - p))``, per group at the average or the pooled proportion.
    """
    if vtype == 'LS':
        a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
        return 1 / a + 1 / b + 1 / c + 1 / d
    n1, n2, p1, p2 = group_proportions(ai, bi, ci, di, vtype, add, to)
    return 1 / (n1 * p1 * (1 - p1)) + 1 / (n2 * p2 * (1 - p2))


def risk_difference_yi_adjusted(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Risk difference ``p1 - p2`` (the definition) on the corrected table."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    return a / (a + b) - c / (c + d)


def risk_difference_vi_by_type(
    ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str = 'LS', add: float = 0.5, to: str = 'only0'
) -> FloatArray:
    """Variance of a difference of two independent proportions, ``p (1 - p) / n`` per group.

    ``LS`` evaluates at the study's own proportions, ``UB`` uses the unbiased ``p (1 - p) / (n - 1)``, ``AV`` and ``H0`` as in ``group_proportions``.
    """
    n1, n2, p1, p2 = group_proportions(ai, bi, ci, di, vtype, add, to)
    if vtype == 'UB':
        return p1 * (1 - p1) / (n1 - 1) + p2 * (1 - p2) / (n2 - 1)
    return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2


def peto_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Peto's one-step log odds ratio ``(a - E) / V`` (Yusuf et al. 1985, Progress in Cardiovascular Diseases 27:335-371).

    ``E = n1 m1 / N`` is the expected count of the first cell, and ``V = n1 n2 m1 m2 / (N^2 (N - 1))`` its hypergeometric variance.
    """
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    n1, n2, m1, m2, n = a + b, c + d, a + c, b + d, a + b + c + d
    return (a - n1 * m1 / n) / (n1 * n2 * m1 * m2 / (n**2 * (n - 1)))


def peto_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance ``1 / V`` of Peto's log odds ratio (Yusuf et al. 1985), with ``V`` the hypergeometric variance (see ``peto_yi``)."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    n1, n2, m1, m2, n = a + b, c + d, a + c, b + d, a + b + c + d
    return n**2 * (n - 1) / (n1 * n2 * m1 * m2)


def arcsine_difference_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Arcsine risk difference ``asin(sqrt(p1)) - asin(sqrt(p2))`` (Rucker et al. 2009, BMC Medical Research Methodology 9:79)."""
    return numpy.arcsin(numpy.sqrt(ai / (ai + bi))) - numpy.arcsin(numpy.sqrt(ci / (ci + di)))


def arcsine_difference_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Variance ``1 / (4 n1) + 1 / (4 n2)``: the arcsine-square-root transform stabilises a binomial proportion's variance at ``1 / (4 n)`` (Rucker 2009)."""
    return 1 / (4 * (ai + bi)) + 1 / (4 * (ci + di))


def multinomial_delta_variance(statistic: collections.abc.Callable[[FloatArray], FloatArray], counts: FloatArray) -> FloatArray:
    """Delta-method variance of a smooth function of the cell proportions of a table under multinomial sampling.

    ``Var = grad' (diag(p) - p p') grad / N`` (Agresti 2013, Categorical Data Analysis, section 16.1). The gradient is taken by complex-step
    differentiation, exact to rounding for an analytic function. ``counts`` has one row per table and four columns.
    """
    out = numpy.empty(counts.shape[0])
    for row, cells in enumerate(counts):
        n = cells.sum()
        p = cells / n
        grad = numpy.empty(4)
        for j in range(4):
            shifted = p.astype(numpy.complex128)
            shifted[j] += 1e-30j
            grad[j] = numpy.imag(statistic(shifted)) / 1e-30
        cov = numpy.diag(p) - numpy.outer(p, p)
        out[row] = grad @ cov @ grad / n
    return out


def cells_matrix(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> FloatArray:
    """The four cells of each table as one row."""
    return numpy.column_stack([a, b, c, d])


def phi_of(p: FloatArray) -> FloatArray:
    """Phi of cell proportions ``(a, b, c, d)``: ``(a d - b c) / sqrt((a + b)(c + d)(a + c)(b + d))``."""
    a, b, c, d = p
    return typing.cast(FloatArray, (a * d - b * c) / numpy.sqrt((a + b) * (c + d) * (a + c) * (b + d)))


def phi_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Phi coefficient, the product-moment correlation of two binary variables (Pearson 1904; Yule 1912, Journal of the Royal Statistical Society)."""
    return numpy.array([phi_of(numpy.array([a, b, c, d])) for a, b, c, d in zip(ai, bi, ci, di, strict=True)], dtype=numpy.float64)


def phi_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Large-sample variance of phi by the delta method.

    ``LS`` and ``CS``: the multinomial delta method, with the whole table sampled (see ``multinomial_delta_variance``). ``ST``: the two group sizes held
    fixed, each group's proportion binomial, so ``Var = (dphi/dp1)^2 p1 (1 - p1) / n1 + (dphi/dp2)^2 p2 (1 - p2) / n2`` (Agresti 2013, section 16.1),
    the gradient again by complex-step differentiation.
    """
    if vtype != 'ST':
        return multinomial_delta_variance(phi_of, cells_matrix(ai, bi, ci, di))
    out = numpy.empty(ai.shape[0])
    step = 1e-30
    for row, (a, b, c, d) in enumerate(zip(ai, bi, ci, di, strict=True)):
        n1, n2 = a + b, c + d
        p1, p2 = a / n1, c / n2

        def phi_of_proportions(q1: complex, q2: complex, n1: float = n1, n2: float = n2) -> FloatArray:
            return phi_of(numpy.array([n1 * q1, n1 * (1 - q1), n2 * q2, n2 * (1 - q2)]))

        d1 = numpy.imag(phi_of_proportions(p1 + 1j * step, p2)) / step
        d2 = numpy.imag(phi_of_proportions(p1, p2 + 1j * step)) / step
        out[row] = d1**2 * p1 * (1 - p1) / n1 + d2**2 * p2 * (1 - p2) / n2
    return out


def fisher_z_phi_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Fisher's z transform ``atanh(phi)`` of the phi coefficient (Fisher 1921, Metron 1:3-32)."""
    return numpy.arctanh(phi_yi(ai, bi, ci, di))


def fisher_z_phi_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``atanh(phi)`` by the delta method: ``Var(phi) / (1 - phi^2)^2``."""
    return phi_vi(ai, bi, ci, di, vtype) / (1 - phi_yi(ai, bi, ci, di) ** 2) ** 2


def yule_q_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Yule's Q ``(OR - 1) / (OR + 1)`` (Yule 1900, Philosophical Transactions of the Royal Society A) on the corrected table."""
    odds = numpy.exp(log_odds_ratio_yi(ai, bi, ci, di, add, to))
    return typing.cast(FloatArray, (odds - 1) / (odds + 1))


def yule_q_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Delta-method variance of Q: ``Q = tanh(log(OR) / 2)``, so ``dQ/dlog(OR) = (1 - Q^2) / 2`` and ``Var = (1 - Q^2)^2 / 4 * Var(log OR)``."""
    q = yule_q_yi(ai, bi, ci, di, add, to)
    return (1 - q**2) ** 2 / 4 * log_odds_ratio_vi(ai, bi, ci, di, 'LS', add, to)


def yule_y_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Yule's Y, the coefficient of colligation ``(sqrt(OR) - 1) / (sqrt(OR) + 1)`` (Yule 1912) on the corrected table."""
    root = numpy.exp(log_odds_ratio_yi(ai, bi, ci, di, add, to) / 2)
    return typing.cast(FloatArray, (root - 1) / (root + 1))


def yule_y_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Delta-method variance of Y: ``Y = tanh(log(OR) / 4)``, so ``Var = (1 - Y^2)^2 / 16 * Var(log OR)``."""
    y = yule_y_yi(ai, bi, ci, di, add, to)
    return (1 - y**2) ** 2 / 16 * log_odds_ratio_vi(ai, bi, ci, di, 'LS', add, to)


def probit_difference_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Probit difference ``z(p1) - z(p2)`` (the definition, ``z`` the standard normal quantile) on the corrected table."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    return scipy.stats.norm.ppf(a / (a + b)) - scipy.stats.norm.ppf(c / (c + d))


def probit_difference_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Delta-method variance of a probit, ``p (1 - p) / (n phi(z)^2)`` with ``phi`` the standard normal density, summed over the two groups."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    n1, n2, p1, p2 = a + b, c + d, a / (a + b), c / (c + d)
    var1 = p1 * (1 - p1) / (n1 * scipy.stats.norm.pdf(scipy.stats.norm.ppf(p1)) ** 2)
    var2 = p2 * (1 - p2) / (n2 * scipy.stats.norm.pdf(scipy.stats.norm.ppf(p2)) ** 2)
    return var1 + var2


def odds_ratio_scale(measure: str) -> float:
    """The factor that turns a log odds ratio into a standardised mean difference: ``sqrt(3) / pi`` for ``OR2D`` and ``OR2DL``, ``1 / 1.65`` for ``OR2DN``."""
    return 1 / 1.65 if measure == 'OR2DN' else math.sqrt(3) / math.pi


def odds_ratio_to_d_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, measure: str, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """A log odds ratio rescaled to a standardised mean difference.

    ``d = log(OR) * sqrt(3) / pi`` (Hasselblad & Hedges 1995, Psychological Bulletin), or ``log(OR) / 1.65``
    (Sanchez-Meca, Marin-Martinez & Chacon-Moscoso 2003, Psychological Methods).
    """
    return log_odds_ratio_yi(ai, bi, ci, di, add, to) * odds_ratio_scale(measure)


def odds_ratio_to_d_vi(
    ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, measure: str, vtype: str = 'LS', add: float = 0.5, to: str = 'only0'
) -> FloatArray:
    """Variance of a rescaled log odds ratio: ``scale^2 * Var(log OR)`` (variance of a constant multiple)."""
    return odds_ratio_scale(measure) ** 2 * log_odds_ratio_vi(ai, bi, ci, di, vtype, add, to)


def paired_cells(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float, to: str) -> FloatArray:
    """The corrected cells of paired-outcome tables, one row per table."""
    a, b, c, d = adjusted_table(ai, bi, ci, di, add, to)
    return cells_matrix(a, b, c, d)


def paired_risk_difference_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Difference of the paired marginal proportions ``(a + b) / N - (a + c) / N`` (the definition; the cells are the 2x2 table of paired outcomes)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    return (cells[:, 1] - cells[:, 2]) / cells.sum(axis=1)


def paired_risk_difference_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Multinomial delta-method variance of the difference of marginal proportions."""
    return multinomial_delta_variance(lambda p: (p[0] + p[1]) - (p[0] + p[2]), paired_cells(ai, bi, ci, di, add, to))


def paired_log_risk_ratio_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Log of the ratio of the paired marginal proportions, ``log((a + b) / (a + c))`` (the definition)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    return numpy.log(cells[:, 0] + cells[:, 1]) - numpy.log(cells[:, 0] + cells[:, 2])


def paired_log_risk_ratio_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Multinomial delta-method variance of the log ratio of marginal proportions."""
    return multinomial_delta_variance(lambda p: numpy.log(p[0] + p[1]) - numpy.log(p[0] + p[2]), paired_cells(ai, bi, ci, di, add, to))


def paired_log_odds_ratio_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Log of the ratio of the odds of the paired marginal proportions (the definition)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    p1, p2 = (cells[:, 0] + cells[:, 1]) / cells.sum(axis=1), (cells[:, 0] + cells[:, 2]) / cells.sum(axis=1)
    return numpy.log(p1 / (1 - p1)) - numpy.log(p2 / (1 - p2))


def paired_log_odds_ratio_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Multinomial delta-method variance of the log odds ratio of the marginal proportions."""

    def statistic(p: FloatArray) -> FloatArray:
        p1, p2 = p[0] + p[1], p[0] + p[2]
        return typing.cast(FloatArray, numpy.log(p1 / (1 - p1)) - numpy.log(p2 / (1 - p2)))

    return multinomial_delta_variance(statistic, paired_cells(ai, bi, ci, di, add, to))


def conditional_log_odds_ratio_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Conditional (matched-pairs) log odds ratio ``log(b / c)`` of the discordant pairs (Agresti 2013, Categorical Data Analysis, section 10.1)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    return numpy.log(cells[:, 1]) - numpy.log(cells[:, 2])


def conditional_log_odds_ratio_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance ``1 / b + 1 / c``: given the discordant pairs, ``b`` is binomial, and the delta method for ``log(b / c)`` gives it (Agresti 2013)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    return 1 / cells[:, 1] + 1 / cells[:, 2]


def paired_peto_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Peto's one-step estimate on the discordant pairs.

    Under no effect ``b`` is binomial with ``b + c`` trials and probability one half, so ``E = (b + c) / 2``, ``V = (b + c) / 4`` and ``yi = (b - E) / V``
    (Yusuf et al. 1985, applied to matched pairs).
    """
    cells = paired_cells(ai, bi, ci, di, add, to)
    b, c = cells[:, 1], cells[:, 2]
    return (b - (b + c) / 2) / ((b + c) / 4)


def paired_peto_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance ``1 / V = 4 / (b + c)`` of the paired Peto estimate (see ``paired_peto_yi``)."""
    cells = paired_cells(ai, bi, ci, di, add, to)
    return 4 / (cells[:, 1] + cells[:, 2])


def log_rate_ratio_yi(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Log incidence rate ratio ``log((x1 / t1) / (x2 / t2))`` (the definition) on the corrected counts."""
    x1, x2 = adjusted_rates(x1i, x2i, add, to)
    return numpy.log(x1 / t1i) - numpy.log(x2 / t2i)


def log_rate_ratio_vi(x1i: FloatArray, x2i: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance ``1 / x1 + 1 / x2``: a Poisson count has variance ``mu``, so the delta method gives ``1 / mu`` for its log (Rothman et al. 2008)."""
    x1, x2 = adjusted_rates(x1i, x2i, add, to)
    return 1 / x1 + 1 / x2


def rate_difference_yi(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Incidence rate difference ``x1 / t1 - x2 / t2`` (the definition) on the corrected counts."""
    x1, x2 = adjusted_rates(x1i, x2i, add, to)
    return x1 / t1i - x2 / t2i


def rate_difference_vi(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray, add: float = 0.5, to: str = 'only0') -> FloatArray:
    """Variance ``x1 / t1^2 + x2 / t2^2``: a Poisson count ``x`` has variance ``x``, so the rate ``x / t`` has ``x / t^2``."""
    x1, x2 = adjusted_rates(x1i, x2i, add, to)
    return x1 / t1i**2 + x2 / t2i**2


def sqrt_rate_difference_yi(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> FloatArray:
    """Difference of square-root rates ``sqrt(x1 / t1) - sqrt(x2 / t2)`` (the definition; the Poisson variance-stabilising transform)."""
    return numpy.sqrt(x1i / t1i) - numpy.sqrt(x2i / t2i)


def sqrt_rate_difference_vi(t1i: FloatArray, t2i: FloatArray) -> FloatArray:
    """Variance ``1 / (4 t1) + 1 / (4 t2)``: the square root of a Poisson count has variance one quarter (Anscombe 1948, Biometrika)."""
    return 1 / (4 * t1i) + 1 / (4 * t2i)


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """``P(X < h, Y < k)`` for standard bivariate normal variates of correlation ``rho``.

    The integral over ``x < h`` of ``phi(x) Phi((k - rho x) / sqrt(1 - rho^2))`` (Owen 1956, Annals of Mathematical Statistics), by adaptive quadrature.
    """
    root = math.sqrt(1 - rho**2)
    value, _ = scipy.integrate.quad(lambda x: scipy.stats.norm.pdf(x) * scipy.stats.norm.cdf((k - rho * x) / root), -math.inf, h, epsabs=1e-13, epsrel=1e-13)
    return float(value)


def tetrachoric_negative_log_likelihood(par: FloatArray, cells: tuple[float, float, float, float]) -> float:
    """Negative multinomial log likelihood of a 2x2 table under a latent bivariate normal (Pearson 1900, Philosophical Transactions of the Royal Society A).

    ``par`` is the correlation and the row and column thresholds; the cells are a = both below, b = column above, c = row above, d = both above.
    """
    rho, row, col = par
    p_a = bivariate_normal_cdf(col, row, rho)
    p_b = scipy.stats.norm.cdf(row) - p_a
    p_c = scipy.stats.norm.cdf(col) - p_a
    p_d = 1 - p_a - p_b - p_c
    probs = numpy.array([p_a, p_b, p_c, p_d])
    if numpy.any(probs <= 0):
        return math.inf
    return -float(numpy.dot(numpy.array(cells), numpy.log(probs)))


@functools.cache
def tetrachoric_fit(ai: float, bi: float, ci: float, di: float) -> tuple[float, float]:
    """Maximum-likelihood tetrachoric correlation and its variance: the inverse of the observed information, by central differences of the log likelihood."""
    cells = (ai, bi, ci, di)
    n = ai + bi + ci + di
    start = numpy.array([0.0, scipy.stats.norm.ppf((ai + bi) / n), scipy.stats.norm.ppf((ai + ci) / n)])
    minimize = typing.cast(collections.abc.Callable[..., scipy.optimize.OptimizeResult], scipy.optimize.minimize)
    fit = minimize(
        lambda par: tetrachoric_negative_log_likelihood(par, cells),
        start,
        method='Nelder-Mead',
        options={'xatol': 1e-11, 'fatol': 1e-14, 'maxiter': 20000, 'maxfev': 20000},
    )
    best = fit.x
    step = 1e-3
    hessian = numpy.empty((3, 3))
    f0 = tetrachoric_negative_log_likelihood(best, cells)
    for i in range(3):
        for j in range(3):
            ei, ej = numpy.zeros(3), numpy.zeros(3)
            ei[i], ej[j] = step, step
            if i == j:
                hessian[i, j] = (
                    -tetrachoric_negative_log_likelihood(best + 2 * ei, cells)
                    + 16 * tetrachoric_negative_log_likelihood(best + ei, cells)
                    - 30 * f0
                    + 16 * tetrachoric_negative_log_likelihood(best - ei, cells)
                    - tetrachoric_negative_log_likelihood(best - 2 * ei, cells)
                ) / (12 * step**2)
            else:
                hessian[i, j] = (
                    tetrachoric_negative_log_likelihood(best + ei + ej, cells)
                    - tetrachoric_negative_log_likelihood(best + ei - ej, cells)
                    - tetrachoric_negative_log_likelihood(best - ei + ej, cells)
                    + tetrachoric_negative_log_likelihood(best - ei - ej, cells)
                ) / (4 * step**2)
    return float(best[0]), float(numpy.linalg.inv(hessian)[0, 0])


def tetrachoric_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Maximum-likelihood tetrachoric correlation of each table (see ``tetrachoric_fit``)."""
    return numpy.array([tetrachoric_fit(a, b, c, d)[0] for a, b, c, d in zip(ai, bi, ci, di, strict=True)])


def tetrachoric_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Sampling variance of the tetrachoric correlation (see ``tetrachoric_fit``)."""
    return numpy.array([tetrachoric_fit(a, b, c, d)[1] for a, b, c, d in zip(ai, bi, ci, di, strict=True)])


def fisher_z_tetrachoric_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Fisher's z transform of the tetrachoric correlation."""
    return numpy.arctanh(tetrachoric_yi(ai, bi, ci, di))


def fisher_z_tetrachoric_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Variance of ``atanh(r)`` by the delta method: ``Var(r) / (1 - r^2)^2``."""
    return tetrachoric_vi(ai, bi, ci, di) / (1 - tetrachoric_yi(ai, bi, ci, di) ** 2) ** 2


def hedges_correction(m: FloatArray) -> FloatArray:
    """Bias-correction factor ``c(m) = Gamma(m / 2) / (sqrt(m / 2) Gamma((m - 1) / 2))`` (Hedges 1981, Journal of Educational Statistics 6:107-128, eq. 6a).

    ``g = c(m) d`` is then unbiased for the standardised mean difference when ``m`` is the degrees of freedom of the standardiser.
    """
    return numpy.exp(scipy.special.gammaln(m / 2) - scipy.special.gammaln((m - 1) / 2)) / numpy.sqrt(m / 2)


def standardised_variance(g: FloatArray, a: FloatArray, m: FloatArray, n: FloatArray, vtype: str, weights: FloatArray | None = None) -> FloatArray:
    """Sampling variance of ``g = c(m) d`` where ``d = (difference of means) / s``, the difference has variance ``a sigma^2`` and ``s^2 ~ sigma^2 chi2_m / m``.

    ``d`` is then a scaled non-central t, so ``Var(d) = m / (m - 2) (a + delta^2) - delta^2 / c^2`` exactly (Hedges 1981; Hedges & Olkin 1985, ch. 5), and
    ``Var(g) = c^2 Var(d)``.

    - ``LS``: the large-sample form ``a + g^2 / (2 n)`` (Hedges & Olkin 1985, eq. 8, with ``n`` the total sample size);
    - ``LS2``: the same on ``d = g / c``, multiplied by ``c^2``;
    - ``UB``: the exact form with ``delta^2`` replaced by its unbiased estimate ``(m - 2) d^2 / m - a`` (Viechtbauer 2007, Journal of Educational and
      Behavioral Statistics 32:39-60), which reduces to ``a + g^2 (1 - (m - 2) / (m c^2))``;
    - ``AV``: ``LS`` with ``g`` replaced by the ``weights``-weighted mean of ``g`` over the studies (Sanchez-Meca & Marin-Martinez 2001);
    - ``H0``: the exact form at ``delta = 0``, ``c^2 a m / (m - 2)``.
    """
    c = hedges_correction(m)
    if vtype == 'LS':
        return a + g**2 / (2 * n)
    if vtype == 'LS2':
        return c**2 * (a + (g / c) ** 2 / (2 * n))
    if vtype == 'UB':
        return a + g**2 * (1 - (m - 2) / (m * c**2))
    if vtype == 'AV' and weights is not None:
        mean_g = numpy.sum(weights * g) / numpy.sum(weights)
        return a + mean_g**2 / (2 * n)
    if vtype == 'H0':
        return numpy.where(m > 2, c**2 * a * m / (m - 2), numpy.nan)
    raise ValueError(f'unsupported vtype: {vtype}')


def pooled_sd(sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Pooled standard deviation ``sqrt(((n1 - 1) s1^2 + (n2 - 1) s2^2) / (n1 + n2 - 2))``."""
    return numpy.sqrt(((n1i - 1) * sd1i**2 + (n2i - 1) * sd2i**2) / (n1i + n2i - 2))


def mean_difference_yi(m1i: FloatArray, m2i: FloatArray) -> FloatArray:
    """Raw mean difference ``m1 - m2`` (the definition)."""
    return m1i - m2i


def mean_difference_vi(sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of a difference of independent means: ``s1^2 / n1 + s2^2 / n2`` (``LS``, ``UB``), or ``s_p^2 (1 / n1 + 1 / n2)`` (``HO``)."""
    if vtype == 'HO':
        return pooled_sd(sd1i, sd2i, n1i, n2i) ** 2 * (1 / n1i + 1 / n2i)
    return sd1i**2 / n1i + sd2i**2 / n2i


def pomp_mean_difference_yi(m1i: FloatArray, m2i: FloatArray, mini: FloatArray, maxi: FloatArray) -> FloatArray:
    """Mean difference on the percent-of-maximum-possible scale, ``100 (m1 - m2) / (max - min)`` (Cohen, Cohen, Aiken & West 1999, Multivariate Behavioral
    Research 34:315-346)."""
    return 100 * (m1i - m2i) / (maxi - mini)


def pomp_mean_difference_vi(
    sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, mini: FloatArray, maxi: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """Variance of the POMP-scaled mean difference: the raw variance times ``(100 / (max - min))^2``, since the rescaling is linear."""
    return (100 / (maxi - mini)) ** 2 * mean_difference_vi(sd1i, sd2i, n1i, n2i, vtype)


def smd_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Hedges' g, ``c(m) (m1 - m2) / s_p`` with ``m = n1 + n2 - 2`` (Hedges 1981, eq. 6a)."""
    return hedges_correction(n1i + n2i - 2) * (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)


def smd_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of Hedges' g; see ``standardised_variance`` (``a = 1 / n1 + 1 / n2``, ``m = n1 + n2 - 2``, ``n = n1 + n2``)."""
    g = smd_yi(m1i, m2i, sd1i, sd2i, n1i, n2i)
    return standardised_variance(g, 1 / n1i + 1 / n2i, n1i + n2i - 2, n1i + n2i, vtype, n1i + n2i)


def glass_yi(m1i: FloatArray, m2i: FloatArray, sd2i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Glass' delta with the second group's standard deviation, corrected by ``c(n2 - 1)`` (Hedges 1981, section on Glass' estimator)."""
    return hedges_correction(n2i - 1) * (m1i - m2i) / sd2i


def glass_vi(m1i: FloatArray, m2i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'UB') -> FloatArray:
    """Variance of the corrected Glass delta from the exact non-central t moments (``a = 1 / n1 + 1 / n2``, ``m = n2 - 1``); ``UB`` and ``H0`` only."""
    g = glass_yi(m1i, m2i, sd2i, n2i)
    return standardised_variance(g, 1 / n1i + 1 / n2i, n2i - 1, n2i, vtype)


def smd1h_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Variance of the Glass delta without equal variances (Bonett 2008, Psychological Methods 13:99-109, delta method with ``n - 1``).

    ``Var(D) / s2^2 + d^2 Var(s2^2) / (4 s2^4)`` with ``Var(D) = s1^2 / (n1 - 1) + s2^2 / (n2 - 1)`` and ``Var(s2^2) = 2 s2^4 / (n2 - 1)``.
    """
    g = glass_yi(m1i, m2i, sd2i, n2i)
    return sd1i**2 / (sd2i**2 * (n1i - 1)) + 1 / (n2i - 1) + g**2 / (2 * (n2i - 1))


def smdh_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Standardised mean difference with the root mean square of the two standard deviations (Bonett 2008, Psychological Methods 13:99-109).

    The bias correction is ``c(n1 + n2 - 2)``.
    """
    return hedges_correction(n1i + n2i - 2) * (m1i - m2i) / numpy.sqrt((sd1i**2 + sd2i**2) / 2)


def bonett_variance(d: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str) -> FloatArray:
    """Delta-method variance of ``d = D / s`` with ``s^2 = (s1^2 + s2^2) / 2`` (Bonett 2008, Psychological Methods 13:99-109).

    ``Var(D) / s^2 + d^2 (Var(s1^2) + Var(s2^2)) / (16 s^4)``, with ``Var(s_i^2) = 2 s_i^4 / df_i``. ``LS`` takes ``df_i = n_i - 1`` in both terms, ``LS2``
    ``df_i = n_i`` in both, and ``LS3`` ``n_i`` in the first and ``n_i - 1`` in the second.
    """
    s2 = (sd1i**2 + sd2i**2) / 2
    first = (n1i, n2i) if vtype in ('LS2', 'LS3') else (n1i - 1, n2i - 1)
    second = (n1i, n2i) if vtype == 'LS2' else (n1i - 1, n2i - 1)
    return (sd1i**2 / first[0] + sd2i**2 / first[1]) / s2 + d**2 * (sd1i**4 / second[0] + sd2i**4 / second[1]) / (8 * s2**2)


def smdh_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``smdh_yi``: ``bonett_variance`` evaluated at the corrected estimate."""
    return bonett_variance(smdh_yi(m1i, m2i, sd1i, sd2i, n1i, n2i), sd1i, sd2i, n1i, n2i, vtype)


def log_response_ratio_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Log ratio of means with the second-order bias correction (Hedges, Gurevitch & Curtis 1999, Ecology 80:1150-1156; Lajeunesse 2015, Ecology 96:2056-2063).

    ``log(m1 / m2) + (s1^2 / (n1 m1^2) - s2^2 / (n2 m2^2)) / 2``.
    """
    return numpy.log(m1i / m2i) + (sd1i**2 / (n1i * m1i**2) - sd2i**2 / (n2i * m2i**2)) / 2


def log_response_ratio_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """Delta-method variance of the log ratio of means (Hedges, Gurevitch & Curtis 1999, eq. 2; Lajeunesse 2015 for the second-order ``LS2``).

    ``LS``: ``s1^2 / (n1 m1^2) + s2^2 / (n2 m2^2)``; ``HO``: the pooled variance in both terms; ``AV``: the coefficient of variation replaced by its mean over
    studies, weighted by group size, per group; ``AVHO``: by the mean over both groups and all studies (Nakagawa et al. 2015, Methods in Ecology and
    Evolution 6:143-152); ``LS2`` adds ``s^4 / (2 n^2 m^4)`` per group.
    """
    cv1, cv2 = sd1i / m1i, sd2i / m2i
    if vtype == 'LS':
        return cv1**2 / n1i + cv2**2 / n2i
    if vtype == 'HO':
        sp2 = pooled_sd(sd1i, sd2i, n1i, n2i) ** 2
        return sp2 / (n1i * m1i**2) + sp2 / (n2i * m2i**2)
    if vtype == 'AV':
        return (numpy.sum(n1i * cv1) / numpy.sum(n1i)) ** 2 / n1i + (numpy.sum(n2i * cv2) / numpy.sum(n2i)) ** 2 / n2i
    if vtype == 'AVHO':
        return ((numpy.sum(n1i * cv1) + numpy.sum(n2i * cv2)) / (numpy.sum(n1i) + numpy.sum(n2i))) ** 2 * (1 / n1i + 1 / n2i)
    if vtype == 'LS2':
        return cv1**2 / n1i + cv2**2 / n2i + cv1**4 / (2 * n1i**2) + cv2**4 / (2 * n2i**2)
    raise ValueError(f'unsupported vtype: {vtype}')


def log_variability_ratio_yi(sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Log ratio of standard deviations with the small-sample correction ``1 / (2 (n - 1))`` per group (Nakagawa et al. 2015, lnVR)."""
    return numpy.log(sd1i / sd2i) + 1 / (2 * (n1i - 1)) - 1 / (2 * (n2i - 1))


def log_variability_ratio_vi(sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``log s``, ``1 / (2 (n - 1))`` (``LS``) or its second-order form ``1 / (2 (n - 1)) + 1 / (2 (n - 1)^2)`` (``LS2``), summed over the
    groups (Nakagawa et al. 2015, Methods in Ecology and Evolution 6:143-152)."""
    if vtype == 'LS2':
        return (n1i / (n1i - 1) ** 2 + n2i / (n2i - 1) ** 2) / 2
    return 1 / (2 * (n1i - 1)) + 1 / (2 * (n2i - 1))


def log_cv_ratio_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Log ratio of coefficients of variation (Nakagawa et al. 2015, lnCVR) with both small-sample corrections.

    ``E[log s] = log sigma - 1 / (2 (n - 1))`` and ``E[log m] = log mu - s^2 / (2 n m^2)`` to second order, so the bias of ``log s - log m`` is removed by
    adding ``1 / (2 (n - 1))`` and subtracting ``s^2 / (2 n m^2)``, per group.
    """
    return numpy.log(sd1i / m1i) - numpy.log(sd2i / m2i) + 1 / (2 * (n1i - 1)) - 1 / (2 * (n2i - 1)) + (sd2i**2 / (n2i * m2i**2) - sd1i**2 / (n1i * m1i**2)) / 2


def log_cv_ratio_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of the log CV ratio: that of ``log s`` plus that of ``log m`` per group, as the mean and variance of a normal sample are independent
    (Nakagawa et al. 2015)."""
    return log_variability_ratio_vi(sd1i, sd2i, n1i, n2i, vtype) + log_response_ratio_vi(m1i, m2i, sd1i, sd2i, n1i, n2i, vtype)


def point_biserial_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Point-biserial correlation from ``d`` (Tate 1954, Annals of Mathematical Statistics 25:603-607; Jacobs & Viechtbauer 2017, Research Synthesis Methods
    8:161-180): ``r = d / sqrt(d^2 + h)`` with ``d`` not bias-corrected and ``h = m / n1 + m / n2``, ``m = n1 + n2 - 2``."""
    m = n1i + n2i - 2
    d = (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)
    return d / numpy.sqrt(d**2 + m / n1i + m / n2i)


def point_biserial_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, fisher_z: bool = False
) -> FloatArray:
    """Delta-method variance of ``r = d / sqrt(d^2 + h)`` from ``Var(d) = 1 / n1 + 1 / n2 + d^2 / (2 (n1 + n2))``: ``dr/dd = h / (d^2 + h)^(3/2)``.

    With ``fisher_z`` the variance of ``atanh(r)``, ``Var(r) / (1 - r^2)^2``.
    """
    m = n1i + n2i - 2
    d = (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)
    h = m / n1i + m / n2i
    var_r = h**2 / (h + d**2) ** 3 * (1 / n1i + 1 / n2i + d**2 / (2 * (n1i + n2i)))
    if fisher_z:
        return var_r / (1 - d**2 / (d**2 + h)) ** 2
    return var_r


def fisher_z_point_biserial_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Fisher's z of the point-biserial correlation, ``atanh(r)`` (Fisher 1921, Metron 1:3-32)."""
    return numpy.arctanh(point_biserial_yi(m1i, m2i, sd1i, sd2i, n1i, n2i))


def biserial_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Biserial correlation ``r_b = sqrt(p q) r_pb / phi(z)``, ``z`` the normal quantile with upper tail ``p`` (Soper 1914, Biometrika 10:384-390)."""
    n = n1i + n2i
    p, q = n1i / n, n2i / n
    z = scipy.stats.norm.isf(p)
    return numpy.sqrt(p * q) / scipy.stats.norm.pdf(z) * point_biserial_yi(m1i, m2i, sd1i, sd2i, n1i, n2i)


def fisher_z_biserial_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Fisher's z of the biserial correlation, ``atanh(r_b)``."""
    return numpy.arctanh(biserial_yi(m1i, m2i, sd1i, sd2i, n1i, n2i))


def d_to_log_odds_ratio_yi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, scale: float = math.pi / math.sqrt(3)
) -> FloatArray:
    """Log odds ratio from ``d``: ``scale * d``, with ``scale = pi / sqrt(3)`` for a logistic latent variable (Hasselblad & Hedges 1995, Psychological
    Bulletin 117:167-178; Sanchez-Meca, Marin-Martinez & Chacon-Moscoso 2003, Psychological Methods 8:448-467) and ``1.65`` after Cox (1970)."""
    return scale * (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)


def d_to_log_odds_ratio_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, scale: float = math.pi / math.sqrt(3)
) -> FloatArray:
    """Variance of ``scale * d``: ``scale^2 (1 / n1 + 1 / n2 + d^2 / (2 (n1 + n2)))`` (the variance of ``d``, Hedges & Olkin 1985, eq. 8)."""
    d = (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)
    return scale**2 * (1 / n1i + 1 / n2i + d**2 / (2 * (n1i + n2i)))


def change_sd(sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray) -> FloatArray:
    """Standard deviation of the change score, ``sqrt(s1^2 + s2^2 - 2 r s1 s2)`` (the variance of a difference of correlated variables)."""
    return numpy.sqrt(sd1i**2 + sd2i**2 - 2 * ri * sd1i * sd2i)


def mean_change_yi(m1i: FloatArray, m2i: FloatArray) -> FloatArray:
    """Raw mean change ``m1 - m2`` (the definition)."""
    return m1i - m2i


def mean_change_vi(sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance of a mean change, ``Var(change score) / n``."""
    return change_sd(sd1i, sd2i, ri) ** 2 / ni


def pomp_mean_change_yi(m1i: FloatArray, m2i: FloatArray, mini: FloatArray, maxi: FloatArray) -> FloatArray:
    """Mean change on the POMP scale (Cohen et al. 1999): ``100 (m1 - m2) / (max - min)``."""
    return 100 * (m1i - m2i) / (maxi - mini)


def pomp_mean_change_vi(sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, mini: FloatArray, maxi: FloatArray) -> FloatArray:
    """Variance of the POMP-scaled mean change: the raw variance times ``(100 / (max - min))^2``."""
    return (100 / (maxi - mini)) ** 2 * mean_change_vi(sd1i, sd2i, ri, ni)


def smcc_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray) -> FloatArray:
    """Standardised mean change with change-score standardisation, ``c(n - 1) (m1 - m2) / sd_change`` (Gibbons, Hedeker & Davis 1993, Archives of General
    Psychiatry 50:739-750; Becker 1988, British Journal of Mathematical and Statistical Psychology 41:257-278)."""
    return hedges_correction(ni - 1) * (m1i - m2i) / change_sd(sd1i, sd2i, ri)


def smcc_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``smcc_yi``; the change scores are one sample of size ``n``, so ``standardised_variance`` applies with ``a = 1 / n`` and ``m = n - 1``."""
    g = smcc_yi(m1i, m2i, sd1i, sd2i, ri, ni)
    return standardised_variance(g, 1 / ni, ni - 1, ni, vtype)


def smcr_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, ni: FloatArray) -> FloatArray:
    """Standardised mean change with raw-score standardisation, ``c(n - 1) (m1 - m2) / s1`` (Becker 1988; Morris 2008, Organizational Research Methods
    11:364-386)."""
    return hedges_correction(ni - 1) * (m1i - m2i) / sd1i


def unbiased_correlation(ri: FloatArray, ni: FloatArray) -> FloatArray:
    """Olkin & Pratt (1958, Annals of Mathematical Statistics 29:201-211) unbiased correlation, ``r 2F1(1/2, 1/2; (n - 2) / 2; 1 - r^2)``."""
    return ri * scipy.special.hyp2f1(0.5, 0.5, (ni - 2) / 2, 1 - ri**2)


def smcr_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``smcr_yi`` (Morris 2008): ``standardised_variance`` with ``a = 2 (1 - r) / n`` and ``m = n - 1``; ``UB`` and ``H0`` use the unbiased
    correlation of Olkin & Pratt (1958)."""
    g = smcr_yi(m1i, m2i, sd1i, ni)
    r = unbiased_correlation(ri, ni) if vtype in ('UB', 'H0') else ri
    return standardised_variance(g, 2 * (1 - r) / ni, ni - 1, ni, vtype)


def smcrh_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Delta-method variance of ``d = D / s1`` for paired data without equal variances (Bonett 2008, Psychological Methods 13:99-109).

    ``Var(D) / s1^2 + d^2 Var(s1^2) / (4 s1^4)`` with ``Var(D) = sd_change^2 / df`` and ``Var(s1^2) = 2 s1^4 / df``; ``df = n - 1`` (``LS``) or ``n`` (``LS2``).
    """
    g = smcr_yi(m1i, m2i, sd1i, ni)
    df = ni if vtype == 'LS2' else ni - 1
    return change_sd(sd1i, sd2i, ri) ** 2 / (sd1i**2 * df) + g**2 / (2 * df)


def paired_bonett_variance(d: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str) -> FloatArray:
    """Delta-method variance of ``d = D / s`` with ``s^2 = (s1^2 + s2^2) / 2`` for paired data.

    ``Var(s^2) = (Var(s1^2) + Var(s2^2) + 2 Cov(s1^2, s2^2)) / 4`` with ``Var(s_i^2) = 2 s_i^4 / df`` and ``Cov(s1^2, s2^2) = 2 r^2 s1^2 s2^2 / df`` (normal
    theory), so ``Var(d) = sd_change^2 / (s^2 df) + d^2 (s1^4 + s2^4 + 2 r^2 s1^2 s2^2) / (8 s^4 df)``; ``df = n - 1`` (``LS``) or ``n`` (``LS2``).
    """
    s2 = (sd1i**2 + sd2i**2) / 2
    df = ni if vtype == 'LS2' else ni - 1
    return change_sd(sd1i, sd2i, ri) ** 2 / (s2 * df) + d**2 * (sd1i**4 + sd2i**4 + 2 * ri**2 * sd1i**2 * sd2i**2) / (8 * s2**2 * df)


def common_language_paired_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray) -> FloatArray:
    """Probability of superiority ``Phi(d / sqrt(2))`` for normal data (McGraw & Wong 1992, Psychological Bulletin 111:361-365), ``d`` standardised by the
    root mean square of the two standard deviations."""
    return scipy.stats.norm.cdf((m1i - m2i) / numpy.sqrt((sd1i**2 + sd2i**2) / 2) / math.sqrt(2))


def common_language_paired_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray) -> FloatArray:
    """Delta-method variance of ``Phi(d / sqrt(2))``: ``phi(d / sqrt(2))^2 Var(d) / 2 = exp(-d^2 / 2) Var(d) / (4 pi)`` with ``Var(d)`` from
    ``paired_bonett_variance``."""
    d = (m1i - m2i) / numpy.sqrt((sd1i**2 + sd2i**2) / 2)
    return numpy.exp(-(d**2) / 2) / (4 * math.pi) * paired_bonett_variance(d, sd1i, sd2i, ri, ni, 'LS')


def log_response_ratio_change_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ni: FloatArray) -> FloatArray:
    """Log ratio of means for dependent samples with the second-order bias correction (Lajeunesse 2015, Ecology 96:2056-2063; Nakagawa et al. 2015)."""
    return numpy.log(m1i / m2i) + (sd1i**2 / (ni * m1i**2) - sd2i**2 / (ni * m2i**2)) / 2


def log_response_ratio_change_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """Delta-method variance of ``log m1 - log m2`` for correlated means: ``cv1^2 / n + cv2^2 / n - 2 r cv1 cv2 / n``. ``LS2`` adds the second-order terms
    ``cv1^4 / (2 n^2) + cv2^4 / (2 n^2) - r^2 cv1^2 cv2^2 / n^2`` (Lajeunesse 2015)."""
    cv1, cv2 = sd1i / m1i, sd2i / m2i
    first = cv1**2 / ni + cv2**2 / ni - 2 * ri * cv1 * cv2 / ni
    if vtype == 'LS2':
        return first + cv1**4 / (2 * ni**2) + cv2**4 / (2 * ni**2) - ri**2 * cv1**2 * cv2**2 / ni**2
    return first


def log_variability_ratio_change_yi(sd1i: FloatArray, sd2i: FloatArray) -> FloatArray:
    """Log ratio of standard deviations for dependent samples, ``log(s1 / s2)`` (Raudenbush & Bryk 1987, Journal of Educational Statistics 12:241-269)."""
    return numpy.log(sd1i / sd2i)


def log_variability_ratio_change_vi(ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of ``log s1 - log s2``: ``Var(log s_i) = 1 / (2 (n - 1))`` and ``Cov(log s1, log s2) = r^2 / (2 (n - 1))`` (normal theory), so
    ``(1 - r^2) / (n - 1)``; ``LS2`` adds the second-order term ``(1 - r^4) / (n - 1)^2``."""
    first = (1 - ri**2) / (ni - 1)
    return first + (1 - ri**4) / (ni - 1) ** 2 if vtype == 'LS2' else first


def log_cv_ratio_change_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ni: FloatArray) -> FloatArray:
    """Log ratio of coefficients of variation for dependent samples with the mean-bias correction of Lajeunesse (2015); the ``1 / (2 (n - 1))`` terms of
    the two occasions cancel."""
    return numpy.log(sd1i / m1i) - numpy.log(sd2i / m2i) + (sd2i**2 / (ni * m2i**2) - sd1i**2 / (ni * m1i**2)) / 2


def log_cv_ratio_change_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, ri: FloatArray, ni: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """Variance of the log CV ratio for dependent samples: that of the log ratio of the means plus that of the log ratio of the standard deviations, which
    are independent under normality."""
    return log_response_ratio_change_vi(m1i, m2i, sd1i, sd2i, ri, ni, vtype) + log_variability_ratio_change_vi(ri, ni, vtype)


def auc_yi(ai: FloatArray) -> FloatArray:
    """The area under the curve, or common-language effect size, is the effect size itself (McGraw & Wong 1992; Ruscio 2008, Psychological Methods 13:19-30)."""
    return ai


def auc_vi(ai: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'H0') -> FloatArray:
    """Sampling variance of the Mann-Whitney probability ``A``.

    - ``H0``: ``(n1 + n2 + 1) / (12 n1 n2)``, the exact null variance of ``U / (n1 n2)`` (Mann & Whitney 1947, Annals of Mathematical Statistics 18:50-60);
    - ``LS2`` (no ties): Hanley & McNeil (1982, Radiology 143:29-36) ``(A (1 - A) + (n1 - 1) (Q1 - A^2) + (n2 - 1) (Q2 - A^2)) / (n1 n2)`` with
      ``Q1 = A / (2 - A)`` and ``Q2 = 2 A^2 / (1 + A)``, with the denominator ``(n1 - 1) (n2 - 1)`` of Newcombe (2006, Statistics in Medicine 25:559-573).
    """
    if vtype == 'H0':
        return (n1i + n2i + 1) / (12 * n1i * n2i)
    q0, q1, q2 = ai * (1 - ai), ai / (2 - ai), 2 * ai**2 / (1 + ai)
    if vtype == 'LS2':
        return (q0 + (n1i - 1) * (q1 - ai**2) + (n2i - 1) * (q2 - ai**2)) / ((n1i - 1) * (n2i - 1))
    raise ValueError(f'unsupported vtype: {vtype}')


def common_language_normal_yi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """``Phi(d / sqrt(2))`` (McGraw & Wong 1992), ``d`` standardised by the pooled standard deviation (``HO``) or the root mean square of the two (others)."""
    sp = pooled_sd(sd1i, sd2i, n1i, n2i) if vtype == 'HO' else numpy.sqrt((sd1i**2 + sd2i**2) / 2)
    return scipy.stats.norm.cdf((m1i - m2i) / sp / math.sqrt(2))


def common_language_normal_vi(
    m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray, vtype: str = 'LS'
) -> FloatArray:
    """Delta-method variance of ``Phi(d / sqrt(2))``, ``exp(-d^2 / 2) Var(d) / (4 pi)``, with ``Var(d)`` as follows.

    ``HO`` (pooled standard deviation): ``1 / n1 + 1 / n2 + d^2 / (2 (n1 + n2))`` (Hedges & Olkin 1985, eq. 8). Otherwise ``bonett_variance`` for ``LS``,
    ``LS2`` and ``LS3``; and for ``H0`` (``d = 0``) its first term only.
    """
    if vtype == 'HO':
        d = (m1i - m2i) / pooled_sd(sd1i, sd2i, n1i, n2i)
        return numpy.exp(-(d**2) / 2) / (4 * math.pi) * (1 / n1i + 1 / n2i + d**2 / (2 * (n1i + n2i)))
    d = (m1i - m2i) / numpy.sqrt((sd1i**2 + sd2i**2) / 2)
    if vtype == 'H0':
        return bonett_variance(numpy.zeros_like(d), sd1i, sd2i, n1i, n2i, 'LS') / (4 * math.pi)
    return numpy.exp(-(d**2) / 2) / (4 * math.pi) * bonett_variance(d, sd1i, sd2i, n1i, n2i, vtype)


def fisher_z_point_biserial_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Variance of the Fisher z of the point-biserial correlation: ``point_biserial_vi`` divided by ``(1 - r^2)^2`` (delta method for ``atanh``)."""
    return point_biserial_vi(m1i, m2i, sd1i, sd2i, n1i, n2i, True)


def normal_d_to_log_odds_ratio_yi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Log odds ratio from ``d`` as ``1.65 d`` (Cox 1970, The Analysis of Binary Data, Chapman & Hall; Sanchez-Meca et al. 2003)."""
    return d_to_log_odds_ratio_yi(m1i, m2i, sd1i, sd2i, n1i, n2i, 1.65)


def normal_d_to_log_odds_ratio_vi(m1i: FloatArray, m2i: FloatArray, sd1i: FloatArray, sd2i: FloatArray, n1i: FloatArray, n2i: FloatArray) -> FloatArray:
    """Variance of ``1.65 d``: ``1.65^2 Var(d)`` (Sanchez-Meca et al. 2003)."""
    return d_to_log_odds_ratio_vi(m1i, m2i, sd1i, sd2i, n1i, n2i, 1.65)


def weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    return float(numpy.sum(weights * values) / numpy.sum(weights))


def correlation_yi(ri: FloatArray) -> FloatArray:
    """The Pearson correlation is its own effect size."""
    return ri


def correlation_vi(ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Sampling variance of a Pearson correlation.

    - ``LS``: ``(1 - r^2)^2 / (n - 1)``, the large-sample variance (Olkin & Pratt 1958, Annals of Mathematical Statistics 29:201-211; Hedges & Olkin 1985);
    - ``AV``: ``LS`` at the sample-size weighted mean correlation (Hunter & Schmidt 2004);
    - ``H0``: ``(1 - r^2) / (n - 2)``, the squared standard error of the ``t`` test of ``rho = 0`` with ``n - 2`` degrees of freedom;
    - ``H0b``: ``1 / n``, the large-sample variance at ``rho = 0``.
    """
    if vtype == 'LS':
        return (1 - ri**2) ** 2 / (ni - 1)
    if vtype == 'AV':
        return (1 - weighted_mean(ri, ni) ** 2) ** 2 / (ni - 1)
    if vtype in ('H0', 'H0a'):
        return (1 - ri**2) / (ni - 2)
    if vtype == 'H0b':
        return 1 / ni
    raise ValueError(f'unsupported vtype: {vtype}')


def olkin_pratt_yi(ri: FloatArray, ni: FloatArray) -> FloatArray:
    """The unbiased estimator of rho, ``r 2F1(1/2, 1/2; (n - 2) / 2; 1 - r^2)`` (Olkin & Pratt 1958, eq. 2.3)."""
    return ri * scipy.special.hyp2f1(0.5, 0.5, (ni - 2) / 2, 1 - ri**2)


def olkin_pratt_vi(ri: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of the Olkin-Pratt estimator.

    ``UB``: the unbiased estimate of its variance, ``r_u^2 - (1 - (n - 3) / (n - 2) (1 - r^2) 2F1(1, 1; n / 2; 1 - r^2))`` (Olkin & Pratt 1958, eq. 2.6);
    ``LS`` and ``AV`` as ``correlation_vi`` evaluated at ``r_u``.
    """
    ru = olkin_pratt_yi(ri, ni)
    if vtype == 'UB':
        return numpy.asarray(ru**2 - (1 - (ni - 3) / (ni - 2) * (1 - ri**2) * scipy.special.hyp2f1(1, 1, ni / 2, 1 - ri**2)))
    return correlation_vi(ru, ni, vtype)


# auditme: accept[passthrough-shim] case files name a formula by function, so each measure's yi has its own named formula
def fisher_z_yi(ri: FloatArray) -> FloatArray:
    """Fisher's variance-stabilising transform ``atanh(r)`` (Fisher 1915, Biometrika 10:507-521)."""
    return numpy.arctanh(ri)


def fisher_z_vi(ni: FloatArray) -> FloatArray:
    """Variance of Fisher's z, ``1 / (n - 3)`` (Fisher 1921, Metron 1:3-32)."""
    return 1 / (ni - 3)


def partial_correlation_vi(ri: FloatArray, mi: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of a partial correlation with ``m`` predictors: ``(1 - r^2)^2 / (n - m)``, a correlation's variance with ``n - m - 1`` residual degrees of
    freedom (Aloe & Becker 2012, Journal of Educational and Behavioral Statistics 37:340-364). ``AV`` uses the weighted mean correlation."""
    base = weighted_mean(ri, ni) if vtype == 'AV' else ri
    return (1 - base**2) ** 2 / (ni - mi)


def fisher_z_partial_vi(mi: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance of Fisher's z of a partial correlation, ``1 / (n - m - 2)`` (Aloe & Becker 2012)."""
    return 1 / (ni - mi - 2)


def r2_vi(r2i: FloatArray, mi: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of a squared multiple correlation (Olkin & Finn 1995, Psychological Bulletin 118:155-164).

    ``LS``: ``4 R^2 (1 - R^2)^2 / n``; ``LS2``: ``4 R^2 (1 - R^2)^2 (n - m - 1)^2 / ((n^2 - 1) (n + 3))``, which replaces the factor ``1 / n`` by the exact
    one; ``AV`` and ``AV2`` evaluate these at the sample-size weighted mean of ``R^2``.
    """
    base = weighted_mean(r2i, ni) if vtype in ('AV', 'AV2') else r2i
    if vtype in ('LS2', 'AV2'):
        return 4 * base * (1 - base) ** 2 * (ni - mi - 1) ** 2 / ((ni**2 - 1) * (ni + 3))
    return 4 * base * (1 - base) ** 2 / ni


def fisher_z_multiple_yi(r2i: FloatArray) -> FloatArray:
    """Fisher's z of the multiple correlation ``sqrt(R^2)``."""
    return numpy.arctanh(numpy.sqrt(r2i))


def proportion_vi(xi: FloatArray, mi: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Sampling variance of a binomial proportion: ``p (1 - p) / n`` (``LS``), ``p (1 - p) / (n - 1)`` (``UB``, unbiased), or ``LS`` at the weighted mean
    proportion (``AV``)."""
    ni = xi + mi
    p = xi / ni
    if vtype == 'UB':
        return p * (1 - p) / (ni - 1)
    if vtype == 'AV':
        mean_p = weighted_mean(p, ni)
        return mean_p * (1 - mean_p) / ni
    return p * (1 - p) / ni


def proportion_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    return xi / (xi + mi)


def log_proportion_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    return numpy.log(xi / (xi + mi))


def log_proportion_vi(xi: FloatArray, mi: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Delta-method variance of ``log p``: ``(1 - p) / (n p) = 1 / x - 1 / n``; ``AV`` at the weighted mean proportion."""
    ni = xi + mi
    p = weighted_mean(xi / ni, ni) if vtype == 'AV' else xi / ni
    return (1 - p) / (ni * p)


def logit_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    return numpy.log(xi / mi)


def logit_vi(xi: FloatArray, mi: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Delta-method variance of ``logit p``: ``1 / (n p (1 - p)) = 1 / x + 1 / m``; ``AV`` at the weighted mean proportion."""
    ni = xi + mi
    p = weighted_mean(xi / ni, ni) if vtype == 'AV' else xi / ni
    return 1 / (ni * p * (1 - p))


def probit_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    return scipy.stats.norm.ppf(xi / (xi + mi))


def probit_vi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    """Delta-method variance of ``Phi^-1(p)``: ``p (1 - p) / (n phi(z)^2)``."""
    p = xi / (xi + mi)
    return p * (1 - p) / ((xi + mi) * scipy.stats.norm.pdf(scipy.stats.norm.ppf(p)) ** 2)


def arcsine_proportion_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    return numpy.arcsin(numpy.sqrt(xi / (xi + mi)))


def arcsine_proportion_vi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    """Variance ``1 / (4 n)`` of the arcsine square-root transform (Anscombe 1948, Biometrika 35:246-254)."""
    return 1 / (4 * (xi + mi))


def freeman_tukey_proportion_yi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    """Double arcsine transform ``(asin sqrt(x / (n + 1)) + asin sqrt((x + 1) / (n + 1))) / 2`` (Freeman & Tukey 1950; Miller 1978, Am Stat 32:138)."""
    ni = xi + mi
    return (numpy.arcsin(numpy.sqrt(xi / (ni + 1))) + numpy.arcsin(numpy.sqrt((xi + 1) / (ni + 1)))) / 2


def freeman_tukey_proportion_vi(xi: FloatArray, mi: FloatArray) -> FloatArray:
    """Variance ``1 / (4 n + 2)`` (Miller 1978)."""
    return 1 / (4 * (xi + mi) + 2)


def incidence_rate_yi(xi: FloatArray, ti: FloatArray) -> FloatArray:
    return xi / ti


def incidence_rate_vi(xi: FloatArray, ti: FloatArray) -> FloatArray:
    """Poisson variance of a rate: ``x / t^2``."""
    return xi / ti**2


def log_incidence_rate_yi(xi: FloatArray, ti: FloatArray) -> FloatArray:
    return numpy.log(xi / ti)


def log_incidence_rate_vi(xi: FloatArray) -> FloatArray:
    """Delta-method variance of the log of a Poisson rate, ``1 / x``."""
    return 1 / xi


def sqrt_incidence_rate_yi(xi: FloatArray, ti: FloatArray) -> FloatArray:
    return numpy.sqrt(xi / ti)


def sqrt_incidence_rate_vi(ti: FloatArray) -> FloatArray:
    """Variance-stabilised square-root rate, ``1 / (4 t)`` (Anscombe 1948)."""
    return 1 / (4 * ti)


def freeman_tukey_rate_yi(xi: FloatArray, ti: FloatArray) -> FloatArray:
    """Freeman-Tukey transform of a rate, ``(sqrt(x / t) + sqrt((x + 1) / t)) / 2`` (Freeman & Tukey 1950, Annals of Mathematical Statistics 21:607-611)."""
    return (numpy.sqrt(xi / ti) + numpy.sqrt((xi + 1) / ti)) / 2


def freeman_tukey_rate_vi(ti: FloatArray) -> FloatArray:
    return 1 / (4 * ti)


def mean_yi(mi: FloatArray) -> FloatArray:
    return mi


def mean_vi(sdi: FloatArray, ni: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of a sample mean, ``s^2 / n``; ``HO`` uses the pooled variance ``sum((n - 1) s^2) / sum(n - 1)`` over studies."""
    if vtype == 'HO':
        return numpy.sum((ni - 1) * sdi**2) / numpy.sum(ni - 1) / ni
    return sdi**2 / ni


def pomp_mean_yi(mi: FloatArray, mini: FloatArray, maxi: FloatArray) -> FloatArray:
    """Percent of maximal possible score, ``100 (m - min) / (max - min)`` (Cohen, Cohen, Aiken & West 1999, Multivariate Behavioral Research 34:315-346)."""
    return 100 * (mi - mini) / (maxi - mini)


def pomp_mean_vi(sdi: FloatArray, ni: FloatArray, mini: FloatArray, maxi: FloatArray, vtype: str = 'LS') -> FloatArray:
    """Variance of the POMP mean: that of the mean after rescaling the standard deviations by ``100 / (max - min)``."""
    return mean_vi(100 * sdi / (maxi - mini), ni, vtype)


def standardised_mean_yi(mi: FloatArray, sdi: FloatArray, ni: FloatArray) -> FloatArray:
    """Single-group standardised mean ``c(n - 1) m / s`` with Hedges' small-sample correction (Hedges 1981)."""
    return hedges_correction(ni - 1) * mi / sdi


def standardised_mean_vi(mi: FloatArray, sdi: FloatArray, ni: FloatArray) -> FloatArray:
    """Large-sample variance ``1 / n + g^2 / (2 n)`` of a one-sample standardised mean (Hedges & Olkin 1985, ch. 5)."""
    return 1 / ni + standardised_mean_yi(mi, sdi, ni) ** 2 / (2 * ni)


# auditme: accept[passthrough-shim] case files name a formula by function, so each measure's yi has its own named formula
def log_mean_yi(mi: FloatArray) -> FloatArray:
    return numpy.log(mi)


def log_mean_vi(mi: FloatArray, sdi: FloatArray, ni: FloatArray) -> FloatArray:
    """Delta-method variance of ``log m``, ``s^2 / (n m^2)`` (Raudenbush & Bryk 1987, Journal of Educational Statistics 12:241-269)."""
    return sdi**2 / (ni * mi**2)


def log_sd_yi(sdi: FloatArray, ni: FloatArray) -> FloatArray:
    """``log s`` plus its bias correction ``1 / (2 (n - 1))`` (Nakagawa et al. 2015, Methods in Ecology and Evolution 6:143-152)."""
    return numpy.log(sdi) + 1 / (2 * (ni - 1))


def log_sd_vi(ni: FloatArray) -> FloatArray:
    """Variance of ``log s``, ``1 / (2 (n - 1))`` (Nakagawa et al. 2015)."""
    return 1 / (2 * (ni - 1))


def log_cv_yi(mi: FloatArray, sdi: FloatArray, ni: FloatArray) -> FloatArray:
    return numpy.log(sdi / mi) + 1 / (2 * (ni - 1))


def log_cv_vi(mi: FloatArray, sdi: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance of ``log(s / m)``: that of ``log s`` plus that of ``log m`` (Nakagawa et al. 2015)."""
    return 1 / (2 * (ni - 1)) + sdi**2 / (ni * mi**2)


def cronbach_alpha_yi(ai: FloatArray) -> FloatArray:
    """The reliability coefficient itself (Cronbach 1951, Psychometrika 16:297-334)."""
    return ai


def cronbach_alpha_vi(ai: FloatArray, mi: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance ``2 m (1 - a)^2 / ((m - 1) (n - 2))`` of the raw coefficient (Bonett 2002, Journal of Educational and Behavioral Statistics 27:335-340)."""
    return 2 * mi * (1 - ai) ** 2 / ((mi - 1) * (ni - 2))


def hakstian_whalen_yi(ai: FloatArray) -> FloatArray:
    """Cube-root transform ``1 - (1 - a)^(1/3)`` (Hakstian & Whalen 1976, Journal of the American Statistical Association 71:460-462)."""
    return 1 - (1 - ai) ** (1 / 3)


def hakstian_whalen_vi(ai: FloatArray, mi: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance ``18 m (n - 1) (1 - a)^(2/3) / ((m - 1) (9 n - 11)^2)`` (Hakstian & Whalen 1976)."""
    return 18 * mi * (ni - 1) * (1 - ai) ** (2 / 3) / ((mi - 1) * (9 * ni - 11) ** 2)


def bonett_alpha_yi(ai: FloatArray) -> FloatArray:
    """Log transform ``-log(1 - a)`` (Bonett 2002)."""
    return -numpy.log(1 - ai)


def bonett_alpha_vi(mi: FloatArray, ni: FloatArray) -> FloatArray:
    """Variance ``2 m / ((m - 1) (n - 2))`` of the transformed coefficient (Bonett 2002)."""
    return 2 * mi / ((mi - 1) * (ni - 2))


def heterozygosity_statistic(p: FloatArray) -> FloatArray:
    """Log ratio of the heterozygote proportion to the geometric mean of the homozygote proportions, ``log(p1 / (2 sqrt(p0 p2)))``."""
    return numpy.asarray(numpy.log(p[1]) - numpy.log(2 * numpy.sqrt(p[0] * p[2])))


def excess_heterozygosity_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray) -> FloatArray:
    p = numpy.vstack([ai, bi, ci]) / (ai + bi + ci)
    return heterozygosity_statistic(p)


def excess_heterozygosity_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray) -> FloatArray:
    """Delta-method variance of ``excess_heterozygosity_yi`` under multinomial sampling of the three genotypes (Agresti 2013, section 16.1)."""
    counts = numpy.column_stack([ai, bi, ci, numpy.zeros_like(ai)])
    return multinomial_delta_variance(lambda p: heterozygosity_statistic(p[:3]), counts)


def squared_correlation_yi(r2i: FloatArray) -> FloatArray:
    """The squared multiple correlation is its own effect size."""
    return r2i


# --- rma_uni: the tau2 estimators and the weighted fit that follows them ------------------------------------------------------------------------------
#
# Notation (Viechtbauer 2005, JEBS 30:261-293; Raudenbush 2009, in the Handbook of Research Synthesis): y is the vector of k effects, V = diag(v) their
# sampling variances, X the k x p model matrix (a column of ones, then any moderators), tau2 the residual heterogeneity. For a weight matrix A the
# projection is P(A) = A - A X (X' A X)^-1 X' A, and the weighted residual sum of squares is Q(A) = y' P(A) y. The random-effects weights are
# W(tau2) = diag(1 / (v + tau2)).


def model_matrix(count: int, mods: str | None, data: object | None, intercept: bool = True) -> FloatArray:
    """``X``: a column of ones (unless ``intercept`` is false or the formula ends in ``- 1``), then the columns of the one-sided formula ``mods``
    (``'~ ablat + year'``) read from ``data``."""
    terms = (mods or '').replace('~', '')
    if terms.rstrip().endswith('- 1'):
        terms = terms.rstrip()[:-3]
        intercept = False
    columns = [numpy.ones(count)] if intercept else []
    if mods:
        for name in terms.split('+'):
            columns.append(numpy.asarray(data[name.strip()], dtype=numpy.float64))  # type: ignore[index]
    return numpy.column_stack(columns)


def projection(weights: FloatArray, design: FloatArray) -> FloatArray:
    """``P(A) = A - A X (X' A X)^-1 X' A`` for the diagonal weight matrix ``A = diag(weights)``."""
    a = numpy.diag(weights)
    return a - a @ design @ numpy.linalg.inv(design.T @ a @ design) @ design.T @ a


def random_effects_q(yi: FloatArray, vi: FloatArray, design: FloatArray, tau2: float) -> float:
    """``Q(tau2) = y' P(W(tau2)) y``, the weighted residual sum of squares of the random-effects model."""
    return float(yi @ projection(1 / (vi + tau2), design) @ yi)


def moment_tau2(yi: FloatArray, vi: FloatArray, design: FloatArray, weights: FloatArray) -> float:
    """Method of moments for fixed weights ``a``: ``tau2 = (y' P y - tr(P V)) / tr(P)``, from ``E[y' P y] = tau2 tr(P) + tr(P V)``.

    Equal weights give Hedges (1983), inverse-variance weights DerSimonian & Laird (1986) generalised to moderators by Raudenbush (2009), and any other
    weights the generalised Q estimator of DerSimonian & Kacker (2007). Truncation at zero is not applied here.
    """
    p = projection(weights, design)
    return float((yi @ p @ yi - numpy.trace(p @ numpy.diag(vi))) / numpy.trace(p))


def solve_decreasing(func: collections.abc.Callable[[float], float]) -> float:
    """The root of ``func`` on ``tau2 >= 0``, where ``func`` is positive below it and negative above; zero if ``func(0) <= 0``."""
    if func(0.0) <= 0:
        return 0.0
    upper = 1.0
    while func(upper) > 0:
        upper *= 4
    return float(scipy.optimize.brentq(func, 0.0, upper, xtol=1e-14, rtol=1e-14, maxiter=1000))


def quadratic_form_cdf(x: float, eigenvalues: FloatArray) -> float:
    """``P(Q <= x)`` for ``Q = sum(lambda_j * chi2_1)``, positive eigenvalues, by Imhof's (1961, Biometrika 48:419-426) inversion integral."""

    def integrand(u: float) -> float:
        theta = 0.5 * numpy.sum(numpy.arctan(eigenvalues * u)) - 0.5 * x * u
        rho = numpy.prod((1 + (eigenvalues * u) ** 2) ** 0.25)
        return float(numpy.sin(theta) / (u * rho))

    with warnings.catch_warnings():
        # the oscillating tail of the integrand cannot reach the requested accuracy; 1e-9 is reached well before it, and the cases compare at 1e-6
        warnings.simplefilter('ignore', scipy.integrate.IntegrationWarning)
        integral = scipy.integrate.quad(integrand, 0, numpy.inf, limit=200, epsabs=1e-9, epsrel=1e-9)[0]
    return 0.5 - integral / math.pi


def median_unbiased_general_q(yi: FloatArray, vi: FloatArray, design: FloatArray, weights: FloatArray) -> float:
    """Median-unbiased generalised Q estimator (Viechtbauer 2021, Behavior Research Methods): ``tau2`` at which the median of ``y' P y`` is the observed.

    Under normality ``y' P y`` is ``sum(lambda_j chi2_1)`` with ``lambda`` the non-zero eigenvalues of ``P (V + tau2 I)``; the median is found by
    solving ``P(Q <= m) = 1/2``. Lower bound zero.
    """
    p = projection(weights, design)
    observed = float(yi @ p @ yi)

    def median(tau2: float) -> float:
        eig = numpy.linalg.eigvals(p @ numpy.diag(vi + tau2)).real
        eig = eig[eig > 1e-10 * eig.max()]
        return float(scipy.optimize.brentq(lambda m: quadratic_form_cdf(m, eig) - 0.5, 1e-12, 50 * eig.sum() + 50, xtol=1e-10, rtol=1e-10))

    return solve_decreasing(lambda tau2: observed - median(tau2))


def rma_tau2(
    yi: FloatArray,
    vi: FloatArray,
    method: str,
    weights: FloatArray | None = None,
    mods: str | None = None,
    data: object | None = None,
    intercept: bool = True,
) -> float:
    """The estimate of ``tau2`` of ``method``, written from the estimating equation or moment equation of its publication.

    ``DL``, ``HE`` (and ``CO``, ``VC``, which the metafor documentation lists as Hedges' variance component estimator), ``HS``, ``HSk``, ``SJ`` closed
    forms; ``DLIT``, ``SJIT``, ``ML``, ``REML``, ``EB``, ``PM`` (``MP``), ``PMM`` solved by root finding on their equations; ``GENQ`` and ``GENQM``
    with the given ``weights``; the equal-effects methods ``EE``, ``FE`` and ``CE`` fix ``tau2 = 0``.
    """
    k = len(yi)
    design = model_matrix(k, mods, data, intercept)
    df: int = k - int(design.shape[1])
    w0 = 1 / vi
    if method in ('EE', 'FE', 'CE'):
        return 0.0
    if method == 'DL':
        return max(0.0, moment_tau2(yi, vi, design, w0))
    if method in ('HE', 'CO', 'VC'):
        return max(0.0, moment_tau2(yi, vi, design, numpy.ones(k)))
    if method == 'GENQ':
        assert weights is not None
        return max(0.0, moment_tau2(yi, vi, design, weights))
    if method == 'GENQM':
        assert weights is not None
        return median_unbiased_general_q(yi, vi, design, weights)
    if method == 'HS':
        # Hunter & Schmidt (1990): (Q - k) / sum(w), with the sampling variances' average k / sum(w) as the correction; Viechtbauer et al. (2015)
        return max(0.0, (random_effects_q(yi, vi, design, 0.0) - k) / float(w0.sum()))
    if method == 'HSk':
        # Brannick et al. (2019): the weighted variance Q / sum(w) is divided by k - p, not k, before the average sampling variance is subtracted
        return max(0.0, (random_effects_q(yi, vi, design, 0.0) * k / df - k) / float(w0.sum()))
    if method == 'SJ':
        # Sidik & Jonkman (2005, JRSS C 54:367-384): crude estimate sum((y - ybar)^2) / k, then sum(w (y - ybar_w)^2) / (k - 1) with w = 1 / (v / crude + 1)
        crude = float(numpy.sum((yi - yi.mean()) ** 2) / k)
        return random_effects_q(yi, vi, design, crude) * crude / df
    if method == 'DLIT':
        # the fixed point of the DL moment equation with the weights re-computed at its own estimate (DerSimonian & Laird 1986, iterated)
        return solve_decreasing(lambda t: moment_tau2(yi, vi, design, 1 / (vi + t)) - t)
    if method in ('SJIT', 'EB', 'PM', 'MP'):
        # Paule & Mandel (1982): Q(tau2) = k - p; Morris (1983) EB and the fixed point of the Sidik-Jonkman update
        # tau2 Q(tau2) / (k - p) reach the same equation
        return solve_decreasing(lambda t: random_effects_q(yi, vi, design, t) - df)
    if method == 'PMM':
        target = float(scipy.stats.chi2.median(df))
        return solve_decreasing(lambda t: random_effects_q(yi, vi, design, t) - target)
    if method in ('REML', 'ML'):
        # Hardy & Thompson (1996) and Viechtbauer (2005): the score equations y' P^2 y = tr(W) (ML) and y' P^2 y = tr(P) (REML)
        def score(t: float) -> float:
            w = 1 / (vi + t)
            p = projection(w, design)
            return float(yi @ p @ p @ yi - (numpy.trace(p) if method == 'REML' else w.sum()))

        return solve_decreasing(score)
    raise ValueError(f'unsupported method: {method}')


def fixed_effects_fit(
    yi: FloatArray, vi: FloatArray, method: str, weights: FloatArray | None = None, mods: str | None = None, data: object | None = None
) -> tuple[float, FloatArray, FloatArray]:
    """``(tau2, beta, se)``: the estimate of ``tau2`` and the weighted least squares coefficients with weights ``1 / (v + tau2)`` and their standard errors."""
    tau2 = rma_tau2(yi, vi, method, weights, mods, data)
    design = model_matrix(len(yi), mods, data)
    # auditme: accept[literal-duplication] this module imports numpy and scipy only, so it cannot share the tuple with test_properties.py
    if method in ('GENQ', 'GENQM'):
        assert weights is not None
        # the supplied weights also weight the coefficients; the standard errors are then the sandwich of a misspecified weight matrix (Jackson et al. 2014)
        a = numpy.diag(weights)
        bread = numpy.linalg.inv(design.T @ a @ design)
        cov = bread @ design.T @ a @ numpy.diag(vi + tau2) @ a @ design @ bread
        return tau2, bread @ design.T @ a @ yi, numpy.sqrt(numpy.diag(cov))
    w = numpy.diag(1 / (vi + tau2))
    cov = numpy.linalg.inv(design.T @ w @ design)
    return tau2, cov @ design.T @ w @ yi, numpy.sqrt(numpy.diag(cov))


def rma_beta(yi: FloatArray, vi: FloatArray, method: str, weights: FloatArray | None = None, mods: str | None = None, data: object | None = None) -> FloatArray:
    """The coefficients of ``rma_uni`` with ``method``."""
    return fixed_effects_fit(yi, vi, method, weights, mods, data)[1]


def rma_se(yi: FloatArray, vi: FloatArray, method: str, weights: FloatArray | None = None, mods: str | None = None, data: object | None = None) -> FloatArray:
    """The standard errors of the coefficients of ``rma_uni`` with ``method``."""
    return fixed_effects_fit(yi, vi, method, weights, mods, data)[2]


# Inference for the coefficients. With the random-effects weights W = diag(1 / (v + tau2)) the covariance matrix of the coefficients is (X' W X)^-1.
# The Wald tests use the normal (z) or the t distribution with k - p degrees of freedom (t). Knapp & Hartung (2003, Stat Med 22:2693-2710) multiply the
# covariance matrix by s2 = y' P(W) y / (k - p) and refer the statistics to the t distribution with k - p degrees of freedom; the ad hoc variant
# restricts s2 to at least one (the truncation that metafor's documentation attributes to Knapp and Hartung).


def alpha_of(level: float) -> float:
    """The significance level of a confidence level given in percent."""
    return 1 - level / 100


def coefficient_covariance(
    yi: FloatArray, vi: FloatArray, method: str, test: str, mods: str | None, data: object | None
) -> tuple[FloatArray, FloatArray, float]:
    """``(beta, covariance, degrees of freedom)`` for the random-effects fit of ``method`` under ``test``; the degrees of freedom are infinite for ``z``."""
    k = len(yi)
    design = model_matrix(k, mods, data)
    df = k - int(design.shape[1])
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    w = numpy.diag(1 / (vi + tau2))
    cov = numpy.linalg.inv(design.T @ w @ design)
    beta = cov @ design.T @ w @ yi
    if test in KNAPP_HARTUNG_TESTS:
        s2 = random_effects_q(yi, vi, design, tau2) / df
        cov = cov * (max(1.0, s2) if test == 'adhoc' else s2)
    return beta, cov, math.inf if test == 'z' else float(df)


def critical_value(test: str, df: float, level: float) -> float:
    """The two-sided critical value of the test: normal for ``z``, t for the others."""
    p = 1 - alpha_of(level) / 2
    return float(scipy.stats.norm.ppf(p) if test == 'z' else scipy.stats.t.ppf(p, df))


def rma_zval(yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None) -> FloatArray:
    """The test statistics ``beta / se`` of the coefficients."""
    beta, cov, _ = coefficient_covariance(yi, vi, method, test, mods, data)
    return beta / numpy.sqrt(numpy.diag(cov))


def rma_pval(yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None) -> FloatArray:
    """The two-sided p-values of the coefficients."""
    _, _, df = coefficient_covariance(yi, vi, method, test, mods, data)
    stat = numpy.abs(rma_zval(yi, vi, method, test, mods, data))
    return 2 * (scipy.stats.norm.sf(stat) if test == 'z' else scipy.stats.t.sf(stat, df))


def rma_ci_lb(
    yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None, level: float = 95
) -> FloatArray:
    """The lower confidence bounds of the coefficients."""
    beta, cov, df = coefficient_covariance(yi, vi, method, test, mods, data)
    return beta - critical_value(test, df, level) * numpy.sqrt(numpy.diag(cov))


def rma_ci_ub(
    yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None, level: float = 95
) -> FloatArray:
    """The upper confidence bounds of the coefficients."""
    beta, cov, df = coefficient_covariance(yi, vi, method, test, mods, data)
    return beta + critical_value(test, df, level) * numpy.sqrt(numpy.diag(cov))


def prediction_bounds(
    yi: FloatArray, vi: FloatArray, method: str, test: str, mods: str | None, data: object | None, level: float
) -> tuple[FloatArray, FloatArray]:
    """Prediction interval at each row of the model matrix (Riley, Higgins & Deeks 2011, JRSS A 174:369-382, equation 12):
    ``x'beta +- crit * sqrt(tau2 + x' Cov x)``, with the critical value of the test."""
    beta, cov, df = coefficient_covariance(yi, vi, method, test, mods, data)
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    design = model_matrix(len(yi), mods, data)
    centre = design @ beta
    half = critical_value(test, df, level) * numpy.sqrt(tau2 + numpy.einsum('ij,jk,ik->i', design, cov, design))
    return centre - half, centre + half


def prediction_lb(
    yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None, level: float = 95
) -> FloatArray:
    """The lower prediction bounds at the rows of the model matrix."""
    return prediction_bounds(yi, vi, method, test, mods, data, level)[0]


def prediction_ub(
    yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None, level: float = 95
) -> FloatArray:
    """The upper prediction bounds at the rows of the model matrix."""
    return prediction_bounds(yi, vi, method, test, mods, data, level)[1]


# Heterogeneity. QE is Cochran's Q about the equal-effects fit, QE = y' P(W(0)) y. The test of the moderators is the Wald statistic b' Cov^-1 b of the
# coefficients that are not the intercept (the intercept for a model with no moderators), divided by their number when the t distribution is used, which
# makes it an F statistic. I2 and H2 follow Higgins & Thompson (2002, Stat Med 21:1539-1558, equations 7 and 9) with the typical within-study variance
# vt = (k - p) / tr(P(W(0))): H2 = (vt + tau2) / vt and I2 = 100 tau2 / (vt + tau2).


def rma_QE(yi: FloatArray, vi: FloatArray, mods: str | None = None, data: object | None = None, intercept: bool = True) -> float:
    """Cochran's Q: the weighted residual sum of squares with the inverse-variance weights."""
    return random_effects_q(yi, vi, model_matrix(len(yi), mods, data, intercept), 0.0)


def rma_QM(yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None) -> float:
    """The omnibus test statistic of the moderators (``chi2`` for ``z``, ``F`` for the t-based tests)."""
    beta, cov, _ = coefficient_covariance(yi, vi, method, test, mods, data)
    which = slice(1, None) if mods else slice(0, 1)
    b = beta[which]
    stat = float(b @ numpy.linalg.inv(cov[which, which]) @ b)
    return stat if test == 'z' else stat / len(b)


def design_typical_variance(vi: FloatArray, design: FloatArray) -> float:
    """``vt = (k - p) / tr(P(W(0)))`` for the model matrix ``design``: the typical within-study variance of Higgins and Thompson (2002), equation 9."""
    return float(design.shape[0] - design.shape[1]) / float(numpy.trace(projection(1 / vi, design)))


def typical_variance(yi: FloatArray, vi: FloatArray, mods: str | None, data: object | None) -> float:
    """``vt = (k - p) / tr(P(W(0)))``, the typical sampling variance."""
    return design_typical_variance(vi, model_matrix(len(yi), mods, data))


def rma_I2(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None) -> float:
    """The share of the total variability that is heterogeneity, in percent."""
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    return 100 * tau2 / (typical_variance(yi, vi, mods, data) + tau2)


def rma_H2(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None) -> float:
    """The ratio of the total variability to the sampling variability."""
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    vt = typical_variance(yi, vi, mods, data)
    return (vt + tau2) / vt


# Confidence intervals for tau2. Each returns the array (lower, upper); `level` is a percentage.


def qprofile_bounds(yi: FloatArray, vi: FloatArray, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Q-profile interval (Viechtbauer 2007, Stat Med 26:37-52): the values of tau2 at which Q(tau2) = y' P(W(tau2)) y, which is chi-square distributed
    with k - p degrees of freedom at the true tau2, equals the upper and the lower ``alpha / 2`` quantile. Q falls as tau2 grows; a bound is zero where
    Q(0) is already below its quantile."""
    design = model_matrix(len(yi), mods, data)
    df = len(yi) - design.shape[1]
    alpha = alpha_of(level)
    lower = solve_decreasing(lambda t: random_effects_q(yi, vi, design, t) - float(scipy.stats.chi2.ppf(1 - alpha / 2, df)))
    upper = solve_decreasing(lambda t: random_effects_q(yi, vi, design, t) - float(scipy.stats.chi2.ppf(alpha / 2, df)))
    return numpy.array([lower, upper])


def general_q_cdf(yi: FloatArray, vi: FloatArray, design: FloatArray, weights: FloatArray, tau2: float) -> float:
    """``P(Q <= observed)`` for the generalised Q statistic y' P(A) y at the given tau2, as a sum of ``lambda_j chi2_1`` with ``lambda`` the non-zero
    eigenvalues of ``P(A) (V + tau2 I)`` (Jackson 2013, Stat Med 32:2837-2850; DerSimonian & Kacker 2007)."""
    p = projection(weights, design)
    eig = numpy.linalg.eigvals(p @ numpy.diag(vi + tau2)).real
    eig = eig[eig > 1e-10 * eig.max()]
    return quadratic_form_cdf(float(yi @ p @ yi), eig)


def general_q_bounds(yi: FloatArray, vi: FloatArray, weights: FloatArray, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Generalised Q interval (Jackson 2013): the tau2 at which the observed statistic is the ``1 - alpha / 2`` and the ``alpha / 2`` quantile of its
    distribution. The distribution function falls as tau2 grows."""
    design = model_matrix(len(yi), mods, data)
    alpha = alpha_of(level)
    lower = solve_decreasing(lambda t: general_q_cdf(yi, vi, design, weights, t) - (1 - alpha / 2))
    upper = solve_decreasing(lambda t: general_q_cdf(yi, vi, design, weights, t) - alpha / 2)
    return numpy.array([lower, upper])


def profile_log_likelihood(yi: FloatArray, vi: FloatArray, design: FloatArray, tau2: float, restricted: bool) -> float:
    """The part of the profile (restricted) log-likelihood that depends on tau2 (Hardy & Thompson 1996; Viechtbauer 2005):
    ``-1/2 (sum log(v + tau2) + Q(tau2) [+ log |X' W X|])``, the bracketed term for the restricted likelihood."""
    w = 1 / (vi + tau2)
    value = numpy.sum(numpy.log(vi + tau2)) + random_effects_q(yi, vi, design, tau2)
    if restricted:
        value += numpy.linalg.slogdet(design.T @ (w[:, None] * design))[1]
    return float(-0.5 * value)


def profile_likelihood_bounds(
    yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None, level: float = 95
) -> FloatArray:
    """Profile-likelihood interval (Hardy & Thompson 1996, Stat Med 15:619-629): the tau2 either side of the estimate where twice the drop of the
    profile log-likelihood from its maximum equals the ``level`` quantile of chi-square with one degree of freedom. The estimate is that of
    ``method`` (``REML`` or ``ML``); a lower bound is zero where the drop at zero is below the quantile."""
    design = model_matrix(len(yi), mods, data)
    restricted = method == 'REML'
    estimate = rma_tau2(yi, vi, method, None, mods, data)
    crit = float(scipy.stats.chi2.ppf(level / 100, 1))

    def log_likelihood(t: float) -> float:
        return profile_log_likelihood(yi, vi, design, t, restricted)

    top = log_likelihood(estimate)

    def excess(t: float) -> float:
        # auditme: accept[redundant-sink-caller-plus-callee] the maximum is evaluated once at the estimate; the root finder evaluates other values of tau2
        return 2 * (top - log_likelihood(t)) - crit

    lower = 0.0 if excess(0.0) < 0 else float(scipy.optimize.brentq(excess, 0.0, estimate, xtol=1e-14, rtol=1e-14))
    bound = max(1.0, 2 * estimate)
    while excess(bound) < 0:
        bound *= 2
    upper = float(scipy.optimize.brentq(excess, estimate, bound, xtol=1e-14, rtol=1e-14))
    return numpy.array([lower, upper])


def higgins_thompson_bounds(yi: FloatArray, vi: FloatArray, level: float = 95) -> FloatArray:
    """Test-based interval of Higgins & Thompson (2002, equations 24 and 25) for a model with no moderators: ``ln H +- z se`` with
    ``H = sqrt(Q / (k - 1))`` and ``se(ln H) = (ln Q - ln(k - 1)) / (2 (sqrt(2Q) - sqrt(2k - 3)))`` when ``Q > k`` and
    ``sqrt(1 / (2 (k - 2)) (1 - 1 / (3 (k - 2)^2)))`` otherwise; the bounds of H2 give those of tau2 as ``vt (H2 - 1)``, with H2 at least one."""
    k = len(yi)
    q = rma_QE(yi, vi)
    if q > k:
        se = 0.5 * (math.log(q) - math.log(k - 1)) / (math.sqrt(2 * q) - math.sqrt(2 * k - 3))
    else:
        se = math.sqrt(1 / (2 * (k - 2)) * (1 - 1 / (3 * (k - 2) ** 2)))
    log_h = 0.5 * math.log(q / (k - 1))
    z = float(scipy.stats.norm.ppf(1 - alpha_of(level) / 2))
    vt = typical_variance(yi, vi, None, None)
    h2 = numpy.maximum(1.0, numpy.exp(2 * numpy.array([log_h - z * se, log_h + z * se])))
    return vt * (h2 - 1)


def tau2_standard_error(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None) -> float:
    """Standard error of the ML or REML estimate of tau2 from the expected information (Viechtbauer 2005): ``sqrt(2 / tr(W^2))`` for ML and
    ``sqrt(2 / tr(P^2))`` for REML."""
    design = model_matrix(len(yi), mods, data)
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    w = 1 / (vi + tau2)
    if method == 'REML':
        p = projection(w, design)
        return math.sqrt(2 / float(numpy.trace(p @ p)))
    return math.sqrt(2 / float(numpy.sum(w**2)))


def wald_bounds(yi: FloatArray, vi: FloatArray, method: str, form: str, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Wald intervals for tau2 from its standard error: on the scale of tau2 (``wald``), of log tau2 (``wald.log``, delta method ``se / tau2``) and of
    tau (``wald.sqrt``, ``se / (2 tau)``), back-transformed. Lower bounds are truncated at zero where the scale allows negative values."""
    tau2 = rma_tau2(yi, vi, method, None, mods, data)
    se = tau2_standard_error(yi, vi, method, mods, data)
    z = float(scipy.stats.norm.ppf(1 - alpha_of(level) / 2))
    if form == 'wald':
        return numpy.array([max(0.0, tau2 - z * se), tau2 + z * se])
    if form == 'wald.log':
        half = z * se / tau2
        return numpy.array([tau2 * math.exp(-half), tau2 * math.exp(half)])
    tau = math.sqrt(tau2)
    half = z * se / (2 * tau)
    return numpy.array([max(0.0, tau - half) ** 2, (tau + half) ** 2])


def wald_tau2_bounds(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Wald interval on the scale of tau2."""
    return wald_bounds(yi, vi, method, 'wald', mods, data, level)


def wald_log_bounds(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Wald interval on the log scale of tau2."""
    return wald_bounds(yi, vi, method, 'wald.log', mods, data, level)


def wald_sqrt_bounds(yi: FloatArray, vi: FloatArray, method: str, mods: str | None = None, data: object | None = None, level: float = 95) -> FloatArray:
    """Wald interval on the scale of tau."""
    return wald_bounds(yi, vi, method, 'wald.sqrt', mods, data, level)


# Mantel-Haenszel and Peto pooling of 2x2 tables and incidence rates. Each estimator takes the tables as they are (no continuity correction), the way
# `rma_mh` and `rma_peto` pool them by default; a study with a zero cell contributes its zero terms.


def mh_or_beta(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Log of the Mantel-Haenszel odds ratio ``sum(a d / n) / sum(b c / n)`` (Mantel and Haenszel, 1959, J Natl Cancer Inst 22:719-748)."""
    n = ai + bi + ci + di
    return math.log(float(numpy.sum(ai * di / n)) / float(numpy.sum(bi * ci / n)))


def mh_or_se(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Standard error of the log Mantel-Haenszel odds ratio (Robins, Breslow and Greenland, 1986, Biometrics 42:311-323).

    With ``P = (a + d) / n``, ``Q = (b + c) / n``, ``R = a d / n`` and ``S = b c / n`` summed over the tables, the variance is
    ``sum(P R) / (2 R^2) + sum(P S + Q R) / (2 R S) + sum(Q S) / (2 S^2)``.
    """
    n = ai + bi + ci + di
    p, q, r, s = (ai + di) / n, (bi + ci) / n, ai * di / n, bi * ci / n
    big_r, big_s = float(numpy.sum(r)), float(numpy.sum(s))
    variance = float(numpy.sum(p * r)) / (2 * big_r**2) + float(numpy.sum(p * s + q * r)) / (2 * big_r * big_s) + float(numpy.sum(q * s)) / (2 * big_s**2)
    return math.sqrt(variance)


def mh_rr_terms(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> tuple[float, float]:
    """``R = sum(a n2 / n)`` and ``S = sum(c n1 / n)``, the numerator and denominator of the Mantel-Haenszel risk ratio."""
    n1, n2 = ai + bi, ci + di
    n = n1 + n2
    return float(numpy.sum(ai * n2 / n)), float(numpy.sum(ci * n1 / n))


def mh_rr_beta(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Log of the Mantel-Haenszel risk ratio ``sum(a n2 / n) / sum(c n1 / n)`` (Greenland and Robins, 1985, Biometrics 41:55-68)."""
    big_r, big_s = mh_rr_terms(ai, bi, ci, di)
    return math.log(big_r / big_s)


def mh_rr_se(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Standard error of the log Mantel-Haenszel risk ratio (Greenland and Robins, 1985).

    The variance is ``sum((n1 n2 (a + c) - a c n) / n^2) / (R S)`` with ``R`` and ``S`` as in ``mh_rr_terms``.
    """
    n1, n2 = ai + bi, ci + di
    n = n1 + n2
    big_r, big_s = mh_rr_terms(ai, bi, ci, di)
    return math.sqrt(float(numpy.sum((n1 * n2 * (ai + ci) - ai * ci * n) / n**2)) / (big_r * big_s))


def mh_rd_beta(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Mantel-Haenszel risk difference ``sum((a n2 - c n1) / n) / sum(n1 n2 / n)`` (Greenland and Robins, 1985)."""
    n1, n2 = ai + bi, ci + di
    n = n1 + n2
    return float(numpy.sum((ai * n2 - ci * n1) / n)) / float(numpy.sum(n1 * n2 / n))


def mh_rd_se(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Standard error of the Mantel-Haenszel risk difference, the variance consistent for sparse tables (Sato, Greenland and Robins, 1989,
    Biometrics 45:1323-1324).

    With ``P = (n1^2 c - n2^2 a + n1 n2 (n2 - n1) / 2) / n^2``, ``Q = (a (n2 - c) + c (n1 - a)) / (2 n)`` and ``R = n1 n2 / n``, the variance is
    ``(RD sum(P) + sum(Q)) / sum(R)^2``.
    """
    n1, n2 = ai + bi, ci + di
    n = n1 + n2
    rd = mh_rd_beta(ai, bi, ci, di)
    p = (n1**2 * ci - n2**2 * ai + n1 * n2 * (n2 - n1) / 2) / n**2
    q = (ai * (n2 - ci) + ci * (n1 - ai)) / (2 * n)
    r = n1 * n2 / n
    return math.sqrt((rd * float(numpy.sum(p)) + float(numpy.sum(q))) / float(numpy.sum(r)) ** 2)


def mh_irr_terms(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> tuple[float, float]:
    """``R = sum(x1 t2 / T)`` and ``S = sum(x2 t1 / T)`` with ``T = t1 + t2``, the numerator and denominator of the Mantel-Haenszel rate ratio."""
    total = t1i + t2i
    return float(numpy.sum(x1i * t2i / total)), float(numpy.sum(x2i * t1i / total))


def mh_irr_beta(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> float:
    """Log of the Mantel-Haenszel incidence rate ratio ``sum(x1 t2 / T) / sum(x2 t1 / T)`` (Greenland and Robins, 1985)."""
    big_r, big_s = mh_irr_terms(x1i, x2i, t1i, t2i)
    return math.log(big_r / big_s)


def mh_irr_se(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> float:
    """Standard error of the log Mantel-Haenszel rate ratio (Greenland and Robins, 1985): variance ``sum(t1 t2 (x1 + x2) / T^2) / (R S)``."""
    total = t1i + t2i
    big_r, big_s = mh_irr_terms(x1i, x2i, t1i, t2i)
    return math.sqrt(float(numpy.sum(t1i * t2i * (x1i + x2i) / total**2)) / (big_r * big_s))


def mh_ird_beta(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> float:
    """Mantel-Haenszel incidence rate difference ``sum((x1 t2 - x2 t1) / T) / sum(t1 t2 / T)`` (Greenland and Robins, 1985)."""
    total = t1i + t2i
    return float(numpy.sum((x1i * t2i - x2i * t1i) / total)) / float(numpy.sum(t1i * t2i / total))


def mh_ird_se(x1i: FloatArray, x2i: FloatArray, t1i: FloatArray, t2i: FloatArray) -> float:
    """Standard error of the Mantel-Haenszel rate difference: variance ``sum((t1 t2 / T)^2 (x1 / t1^2 + x2 / t2^2)) / sum(t1 t2 / T)^2``.

    It is the variance of a weighted sum of two independent Poisson rates, ``Var(x / t) = x / t^2``, with the weights ``t1 t2 / T`` over their sum.
    """
    total = t1i + t2i
    numerator = float(numpy.sum((t1i * t2i / total) ** 2 * (x1i / t1i**2 + x2i / t2i**2)))
    return math.sqrt(numerator / float(numpy.sum(t1i * t2i / total)) ** 2)


def peto_terms(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Observed minus expected ``a - E`` and the hypergeometric variance ``V`` of each table, ``E = n1 m1 / n``, ``V = n1 n2 m1 m2 / (n^2 (n - 1))``."""
    n1, n2 = ai + bi, ci + di
    n = n1 + n2
    m1, m2 = ai + ci, bi + di
    return ai - n1 * m1 / n, n1 * n2 * m1 * m2 / (n**2 * (n - 1))


def peto_beta(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Peto's pooled log odds ratio ``sum(a - E) / sum(V)`` (Yusuf et al. 1985, Progress in Cardiovascular Diseases 27:335-371)."""
    observed_minus_expected, variance = peto_terms(ai, bi, ci, di)
    return float(numpy.sum(observed_minus_expected)) / float(numpy.sum(variance))


def peto_se(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> float:
    """Standard error ``1 / sqrt(sum(V))`` of Peto's pooled log odds ratio (Yusuf et al. 1985)."""
    return 1 / math.sqrt(float(numpy.sum(peto_terms(ai, bi, ci, di)[1])))


# Multivariate and multilevel models (rma_mv): the marginal model y ~ N(X b, M), M = V + the sum of the random-effects matrices of the terms.

SIGMA2 = 'SIGMA2'
MV_GRADIENT_LIMIT = 1e-6


UNSTRUCTURED = ('GEN', 'UN')
PER_LEVEL_VARIANCES = ('DIAG', 'GDIAG', 'GEN', 'HAR', 'HCS', 'UN')
SPATIAL_STRUCTS = ('SPEXP', 'SPGAU', 'SPLIN', 'SPRAT', 'SPSPH')
COEFFICIENT_STRUCTS = ('GDIAG', 'GEN')


@dataclasses.dataclass(frozen=True)
class RandomTerm:
    """One random-effects term of ``rma_mv``: a ``sigma2`` term (``~ 1 | outer``, ``struct`` is ``SIGMA2``) or a structured ``~ inner | outer`` term.

    ``outer`` and ``inner`` hold one level code per row; rows share the random effect when their ``outer`` codes agree. ``n_inner`` is the number of
    inner levels (1 for a ``sigma2`` term, the number of random coefficients for ``GEN`` and ``GDIAG``). ``levels`` holds the value of each inner level
    (the time points of ``CAR``), ``design`` the ``k x p`` matrix of random coefficients (``GEN``, ``GDIAG``), ``dist`` the ``k x k`` distances between
    rows (the spatial structures) and ``R`` the known correlation matrix of the outer levels (a ``sigma2`` term). ``fixed_variances`` and
    ``fixed_correlations`` hold values held fixed (``nan`` where estimated); a ``UN`` or ``GEN`` matrix cannot have fixed values.
    """

    struct: str
    outer: IntArray
    inner: IntArray
    n_inner: int
    levels: FloatArray | None = None
    design: FloatArray | None = None
    dist: FloatArray | None = None
    R: FloatArray | None = None
    fixed_variances: FloatArray | None = None
    fixed_correlations: FloatArray | None = None

    def n_variances(self) -> int:
        return self.n_inner if self.struct in PER_LEVEL_VARIANCES else 1

    def n_correlations(self) -> int:
        if self.struct in UNSTRUCTURED:
            return self.n_inner * (self.n_inner - 1) // 2
        return 1 if self.struct in ('AR', 'CAR', 'CS', 'HAR', 'HCS', *SPATIAL_STRUCTS) else 0


@dataclasses.dataclass(frozen=True)
class MVFit:
    """The optimum of ``mv_fit``: the variances and correlations of each term, ``b``, ``vb = (X' M^-1 X)^-1``, the log-likelihood and convergence."""

    variances: tuple[FloatArray, ...]
    correlations: tuple[FloatArray, ...]
    b: FloatArray
    vb: FloatArray
    loglik: float
    success: bool
    gradient_norm: float
    message: str


def spatial_correlation(struct: str, dist: FloatArray, scale: float) -> FloatArray:
    """The correlation of two points at distance ``dist`` (the equations of the table in the ``rma.mv`` help page, after Pinheiro and Bates 2000).

    Exponential ``exp(-d/r)``, Gaussian ``exp(-d^2/r^2)``, linear ``(1 - d/r) I(d < r)``, rational quadratic ``1 - (d/r)^2 / (1 + (d/r)^2)``,
    spherical ``(1 - 1.5 d/r + 0.5 (d/r)^3) I(d < r)``, with the range ``r``. Cressie (1993, section 2.3.1) gives the same shapes with the range
    scaled by a constant for some of them (the Gaussian and exponential reach 95 per cent of the sill at a multiple of ``r``); the parametrisation here is
    the software's convention, not a difference of the models.
    """
    ratio = dist / scale
    if struct == 'SPEXP':
        return numpy.asarray(numpy.exp(-ratio), dtype=numpy.float64)
    if struct == 'SPGAU':
        return numpy.asarray(numpy.exp(-(ratio**2)), dtype=numpy.float64)
    if struct == 'SPLIN':
        return numpy.asarray((1 - ratio) * (ratio < 1), dtype=numpy.float64)
    if struct == 'SPRAT':
        return numpy.asarray(1 - ratio**2 / (1 + ratio**2), dtype=numpy.float64)
    if struct == 'SPSPH':
        return numpy.asarray((1 - 1.5 * ratio + 0.5 * ratio**3) * (ratio < 1), dtype=numpy.float64)
    raise ValueError(f'spatial_correlation: unknown structure {struct!r}')


def struct_matrix(struct: str, inner: FloatArray, tau2: FloatArray, rho: FloatArray, dist: FloatArray | None = None) -> FloatArray:
    """``G`` for one outer level, the covariance matrix of the effects at the ``len(inner)`` inner levels (Wolfinger 1993, table 1).

    ``ID``: ``tau2 I``. ``DIAG``: ``diag(tau2_1, ...)``. ``CS``: ``tau2`` on the diagonal and ``rho tau2`` off it. ``HCS``: ``tau2_j`` on the
    diagonal and ``rho sqrt(tau2_i tau2_j)`` off it. ``UN``: ``tau2_j`` on the diagonal and ``rho_il sqrt(tau2_i tau2_l)`` off it. The ``rho`` of
    ``UN`` runs down the columns of the lower triangle (the order is the software's convention, not Wolfinger's). ``AR``: ``tau2 rho^|i - l|`` for
    the ordinal positions ``i`` and ``l`` of the inner levels (the values of ``inner`` do not matter, as the ``rma.mv`` help page says), ``HAR``: the
    same correlations with ``tau2_j`` on the diagonal (Wolfinger 1993). ``CAR``: ``tau2 rho^|t_i - t_l|`` for the time points ``t`` in ``inner``.
    ``GEN`` and ``GDIAG``: the covariance matrix ``D`` of the random coefficients of the model of Laird and Ware (1982, equation 2.1), unstructured
    (as ``UN``) or diagonal. With ``dist``, the spatial structures: ``tau2`` times the correlation of the distances (``spatial_correlation``).
    """
    n = len(inner)
    if struct in SPATIAL_STRUCTS:
        if dist is None:
            raise ValueError(f'struct_matrix: {struct} needs distances')
        return float(tau2[0]) * spatial_correlation(struct, dist, float(rho[0]))
    if dist is not None:
        raise ValueError(f'struct_matrix: {struct} takes no distances')
    if struct == 'ID':
        return float(tau2[0]) * numpy.eye(n)
    if struct in ('DIAG', 'GDIAG'):
        return numpy.diag(tau2)
    if struct == 'CS':
        return float(tau2[0]) * (float(rho[0]) * numpy.ones((n, n)) + (1.0 - float(rho[0])) * numpy.eye(n))
    if struct in ('AR', 'CAR'):
        gaps = numpy.abs(numpy.subtract.outer(inner, inner)) if struct == 'CAR' else numpy.abs(numpy.subtract.outer(numpy.arange(n), numpy.arange(n)))
        return float(tau2[0]) * numpy.power(float(rho[0]), gaps)
    scale = numpy.sqrt(tau2)
    if struct == 'HAR':
        corr = numpy.power(float(rho[0]), numpy.abs(numpy.subtract.outer(numpy.arange(n), numpy.arange(n))))
    elif struct == 'HCS':
        corr = float(rho[0]) * numpy.ones((n, n)) + (1.0 - float(rho[0])) * numpy.eye(n)
    elif struct in UNSTRUCTURED:
        corr = numpy.eye(n)
        position = 0
        for column in range(n):
            for row in range(column + 1, n):
                corr[row, column] = corr[column, row] = rho[position]
                position += 1
    else:
        raise ValueError(f'struct_matrix: unknown or unsupported structure {struct!r}')
    return scale[:, None] * corr * scale[None, :]


def term_matrix(term: RandomTerm, variances: FloatArray, correlations: FloatArray) -> FloatArray:
    """The ``k x k`` matrix a term adds to ``M``.

    ``sigma2``: ``sigma2`` times ``1`` for two rows of an outer level (times the known correlation of their outer levels when ``R`` is given). The
    factor structures: ``G[inner_i, inner_l]`` when rows ``i`` and ``l`` share an outer level, else 0. ``GEN`` and ``GDIAG``: ``Z G Z'`` of the random
    coefficients ``Z`` (Laird and Ware 1982) within an outer level. The spatial structures: the correlation of the distance times ``tau2`` within one.
    """
    same = (term.outer[:, None] == term.outer[None, :]).astype(numpy.float64)
    if term.struct == SIGMA2:
        known = same if term.R is None else term.R[term.outer[:, None], term.outer[None, :]]
        return numpy.asarray(float(variances[0]) * known, dtype=numpy.float64)
    if term.struct in SPATIAL_STRUCTS:
        return numpy.asarray(struct_matrix(term.struct, numpy.zeros(0), variances, correlations, term.dist) * same, dtype=numpy.float64)
    if term.struct in COEFFICIENT_STRUCTS:
        assert term.design is not None
        d = struct_matrix(term.struct, numpy.arange(term.n_inner, dtype=numpy.float64), variances, correlations)
        return numpy.asarray((term.design @ d @ term.design.T) * same, dtype=numpy.float64)
    levels = numpy.arange(term.n_inner, dtype=numpy.float64) if term.levels is None else term.levels
    g = struct_matrix(term.struct, levels, variances, correlations)
    return numpy.asarray(g[term.inner[:, None], term.inner[None, :]] * same, dtype=numpy.float64)


def rescale_r(R: FloatArray, rscale: str) -> FloatArray:
    """A known matrix ``R`` rescaled as the ``Rscale`` text of the ``rma.mv`` help page says (the convention of that software).

    ``none``: no scaling. ``cor``: ``cov2cor``, ``R_il / sqrt(R_ii R_ll)``. ``cor0``: ``cov2cor`` and then ``(R - min(R)) / (1 - min(R))``, which puts
    a correlation of zero at the split at the root of the tree of the species analysed. ``cov0``: ``R - min(R)``, which roots a covariance matrix at
    its lowest split.
    """
    if rscale not in ('cor', 'cor0', 'cov0', 'none'):
        raise ValueError(f'rescale_r: unknown Rscale {rscale!r}')
    if rscale == 'none':
        return R
    if rscale == 'cov0':
        return R - float(R.min())
    scale = numpy.sqrt(numpy.diag(R))
    corr = R / scale[:, None] / scale[None, :]
    if rscale == 'cor':
        return numpy.asarray(corr, dtype=numpy.float64)
    low = float(corr.min())
    return numpy.asarray((corr - low) / (1 - low), dtype=numpy.float64)


def gls(yi: FloatArray, M: FloatArray, X: FloatArray, W: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
    """The generalised least squares ``b = (X' W X)^-1 X' W y`` with ``W = M^-1`` unless given, and its variance.

    For ``W = M^-1`` the variance is ``(X' M^-1 X)^-1``; for another ``W`` it is the sandwich ``(X' W X)^-1 X' W M W X (X' W X)^-1``.
    """
    if W is None:
        weight = numpy.linalg.inv(M)
        bread = numpy.linalg.inv(X.T @ weight @ X)
        return bread @ X.T @ weight @ yi, bread
    bread = numpy.linalg.inv(X.T @ W @ X)
    return bread @ X.T @ W @ yi, bread @ X.T @ W @ M @ W @ X @ bread


def mv_loglik(yi: FloatArray, V: FloatArray, X: FloatArray, M_extra: FloatArray, method: str, W: FloatArray | None = None) -> float:
    """The log-likelihood of ``y ~ N(X b, M)``, ``M = V + M_extra``, at the generalised least squares ``b`` (Harville 1977, section 3).

    ``ML``: ``-(k log 2 pi + log|M| + r' M^-1 r) / 2`` with ``r = y - X b``. ``REML``: ``-((k - p) log 2 pi - log|X'X| + log|M| + log|X' M^-1 X| +
    r' M^-1 r) / 2``, the restricted likelihood of the ``k - p`` error contrasts, which carries ``log|X'X|`` so that it does not depend on the choice
    of contrasts (that constant is the software's convention). The fixed-effects methods are scored as ``ML`` with ``M = V``.
    """
    k, p = X.shape
    M = V + M_extra
    b, _ = gls(yi, M, X, W)
    resid = yi - X @ b
    sign, logdet = numpy.linalg.slogdet(M)
    if sign <= 0:
        return float('-inf')
    quad = float(resid @ numpy.linalg.solve(M, resid))
    if method == 'ML':
        return float(-0.5 * (k * math.log(2 * math.pi) + logdet + quad))
    if method == 'REML':
        info = X.T @ numpy.linalg.solve(M, X)
        return float(-0.5 * ((k - p) * math.log(2 * math.pi) - numpy.linalg.slogdet(X.T @ X)[1] + logdet + numpy.linalg.slogdet(info)[1] + quad))
    raise ValueError(f'mv_loglik: method must be ML or REML, not {method!r}')


def n_free(term: RandomTerm) -> int:
    """The number of unconstrained parameters of a term in ``mv_fit`` (a ``UN`` or ``GEN`` matrix uses its Cholesky factor)."""
    n = term.n_inner
    return n * (n + 1) // 2 if term.struct in UNSTRUCTURED else term.n_variances() + term.n_correlations()


def correlation_transform(struct: str, values: FloatArray) -> FloatArray:
    """The correlation parameter from its unconstrained value: ``tanh`` in (-1, 1), the logistic function in (0, 1) for ``CAR``, ``exp`` for a range."""
    if struct == 'CAR':
        return numpy.asarray(scipy.special.expit(values), dtype=numpy.float64)
    if struct in SPATIAL_STRUCTS:
        return numpy.exp(values)
    return numpy.tanh(values)


def fixed_mask(terms: list[RandomTerm]) -> FloatArray:
    """Whether each unconstrained parameter of ``mv_fit`` is estimated (1) or held at a fixed value (0)."""
    parts: list[FloatArray] = []
    for term in terms:
        if term.struct in UNSTRUCTURED:
            parts.append(numpy.ones(n_free(term)))
            continue
        variances = numpy.ones(term.n_variances()) if term.fixed_variances is None else numpy.isnan(term.fixed_variances).astype(numpy.float64)
        correlations = numpy.ones(term.n_correlations()) if term.fixed_correlations is None else numpy.isnan(term.fixed_correlations).astype(numpy.float64)
        parts.append(numpy.concatenate([variances, correlations[: term.n_correlations()]]))
    return numpy.concatenate(parts) if parts else numpy.zeros(0)


def unpack_terms(terms: list[RandomTerm], theta: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
    """The variances and correlations of each term from the unconstrained vector of ``mv_fit``.

    A variance is ``exp`` of its entry, a correlation is ``correlation_transform`` of its entry, and a ``UN`` or ``GEN`` matrix is ``L L'`` for a lower
    triangular ``L`` whose diagonal is ``exp`` of its entries (the Cholesky parametrisation, which keeps the matrix positive definite). A fixed value
    replaces the entry's value.
    """
    variances: list[FloatArray] = []
    correlations: list[FloatArray] = []
    position = 0
    for term in terms:
        part = theta[position : position + n_free(term)]
        position += n_free(term)
        if term.struct in UNSTRUCTURED:
            n = term.n_inner
            low = numpy.zeros((n, n))
            numpy.fill_diagonal(low, numpy.exp(part[:n]))
            entries = iter(part[n:])
            for column in range(n):
                for row in range(column + 1, n):
                    low[row, column] = next(entries)
            cov = low @ low.T
            scale = numpy.sqrt(numpy.diag(cov))
            corr = cov / scale[:, None] / scale[None, :]
            variances.append(numpy.diag(cov).copy())
            correlations.append(numpy.array([corr[row, column] for column in range(n) for row in range(column + 1, n)]))
            continue
        var = numpy.exp(part[: term.n_variances()])
        cor = correlation_transform(term.struct, part[term.n_variances() :])
        if term.fixed_variances is not None:
            var = numpy.where(numpy.isnan(term.fixed_variances), var, term.fixed_variances)
        if term.fixed_correlations is not None:
            cor = numpy.where(numpy.isnan(term.fixed_correlations[: len(cor)]), cor, term.fixed_correlations[: len(cor)])
        variances.append(var)
        correlations.append(cor)
    return variances, correlations


def mv_fit(yi: FloatArray, V: FloatArray, X: FloatArray, terms: list[RandomTerm], method: str, W: FloatArray | None = None) -> MVFit:
    """Maximise ``mv_loglik`` over the components of ``terms`` with ``scipy.optimize.minimize`` (BFGS on log variances and atanh correlations).

    It starts from variances of ``var(y) / (2 n_terms)`` and zero correlations (a range starts at 1), restarts BFGS from its own optimum until the
    objective stops improving, polishes with Newton steps, and reports the norm of the central-difference gradient of the negative log-likelihood at
    the optimum, over the parameters that are not fixed.
    """
    count = sum(n_free(term) for term in terms)
    theta = numpy.zeros(count)
    position = 0
    start = math.log(float(numpy.var(yi)) / (2 * max(len(terms), 1)))
    for term in terms:
        theta[position : position + term.n_variances()] = start
        position += n_free(term)
    free = numpy.flatnonzero(fixed_mask(terms) > 0)

    def extra_of(values: FloatArray) -> FloatArray:
        variances, correlations = unpack_terms(terms, values)
        extra = numpy.zeros_like(V)
        for term, var, cor in zip(terms, variances, correlations, strict=True):
            extra = extra + term_matrix(term, var, cor)
        return extra

    def embed(part: FloatArray) -> FloatArray:
        full = theta.copy()
        full[free] = part
        return full

    def negative(part: FloatArray) -> float:
        value = mv_loglik(yi, V, X, extra_of(embed(part)), method, W)
        return -value if math.isfinite(value) else 1e300

    def gradient_of(part: FloatArray) -> FloatArray:
        step = 1e-4
        return numpy.array([(negative(part + step * unit) - negative(part - step * unit)) / (2 * step) for unit in numpy.eye(len(free))])

    x = theta[free]
    message = 'no random effects'
    if len(free):
        outcome = scipy.optimize.minimize(negative, x, method='BFGS', jac=gradient_of, options={'gtol': 1e-9, 'maxiter': 10000})
        previous = math.inf
        for _ in range(20):
            if abs(previous - outcome.fun) < 1e-13:
                break
            previous = outcome.fun
            outcome = scipy.optimize.minimize(negative, outcome.x, method='BFGS', jac=gradient_of, options={'gtol': 1e-9, 'maxiter': 10000})
        x, message = outcome.x, str(outcome.message)
        for _ in range(10):
            gradient = gradient_of(x)
            if numpy.linalg.norm(gradient) < 1e-9:
                break
            hessian = numpy.array([(gradient_of(x + 1e-3 * unit) - gradient_of(x - 1e-3 * unit)) / 2e-3 for unit in numpy.eye(len(free))])
            candidate = x - numpy.linalg.solve((hessian + hessian.T) / 2, gradient)
            if negative(candidate) > negative(x) + 1e-10:
                break
            x = candidate
    norm = float(numpy.linalg.norm(gradient_of(x))) if len(free) else 0.0
    optimum = embed(x)
    variances, correlations = unpack_terms(terms, optimum)
    extra = extra_of(optimum)
    b, vb = gls(yi, V + extra, X, W)
    return MVFit(tuple(variances), tuple(correlations), b, vb, mv_loglik(yi, V, X, extra, method, W), norm <= MV_GRADIENT_LIMIT, norm, message)


# rma_mv: inference on the coefficients, QE and QM, containment degrees of freedom, and profile-likelihood intervals of the components.


@dataclasses.dataclass(frozen=True)
class MVTests:
    """The tests of the coefficients of an ``rma_mv`` fit: standard errors, test statistics, p-values, interval bounds, ``QM`` with its p-value, and the
    degrees of freedom of each coefficient (``nan`` for a z test)."""

    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    QM: float
    QMp: float
    ddf: FloatArray


def knapp_hartung_factor(yi: FloatArray, M: FloatArray, X: FloatArray, ad_hoc: bool) -> float:
    """The Knapp-Hartung variance factor ``r' M^-1 r / (k - p)`` (Knapp and Hartung 2003, equation 2, generalised from ``M = diag(vi + tau2)`` to a full
    marginal covariance matrix, as in the ``rma.mv`` help page), with ``r = y - X b`` at the generalised least squares ``b``. The ad hoc version
    (``adhoc``) floors the factor at 1, so that the standard errors are never below the Wald ones."""
    k, p = X.shape
    b, _ = gls(yi, M, X)
    resid = yi - X @ b
    factor = float(resid @ numpy.linalg.solve(M, resid)) / (k - p)
    return float(max(factor, 1.0) if ad_hoc else factor)


def contain_ddf(X: FloatArray, groupings: list[IntArray]) -> FloatArray:
    """Containment degrees of freedom (the SAS/STAT ``PROC MIXED`` option ``DDFM=CONTAIN``): a coefficient's column is contained in a random effect when
    it is constant within each level of that effect's grouping; its degrees of freedom are the number of levels of the smallest such grouping minus
    the number of coefficients, at least 1. A column contained in no grouping has ``k - p``.

    ``groupings`` holds one level code per row for each grouping variable: the outer variable of each term and, for a structured term, the
    inner-by-outer combination. Making the numbers of levels the degrees of freedom is the software's reading of SAS's rank condition.
    """
    k, p = X.shape
    ddf = numpy.full(p, float(k - p))
    for j in range(p):
        for codes in groupings:
            levels = numpy.unique(codes)
            if all(numpy.unique(X[codes == level, j]).shape[0] == 1 for level in levels):
                ddf[j] = min(ddf[j], len(levels) - p)
        ddf[j] = max(ddf[j], 1.0)
    return ddf


def mv_tests(yi: FloatArray, M: FloatArray, X: FloatArray, test: str, ddf: FloatArray, btt: list[int], level: float = 95) -> MVTests:
    """Wald (``z``), ``t``, Knapp-Hartung (``knha``) and ad hoc (``adhoc``) tests of the coefficients at the generalised least squares estimate.

    ``z`` uses the normal distribution and ``QM`` against chi-square(m); the others scale the variance by the Knapp-Hartung factor (``t`` does not) and
    use the t distribution with ``ddf`` degrees of freedom for each coefficient, and ``QM / m`` against F(m, min ``ddf`` over ``btt``).
    """
    b, vb = gls(yi, M, X)
    if test in KNAPP_HARTUNG_TESTS:
        vb = knapp_hartung_factor(yi, M, X, test == 'adhoc') * vb
    se = numpy.sqrt(numpy.diag(vb))
    zval = b / se
    qm = float(b[btt] @ numpy.linalg.solve(vb[numpy.ix_(btt, btt)], b[btt]))
    half = alpha_of(level) / 2
    if test == 'z':
        crit = numpy.full(len(b), float(scipy.stats.norm.ppf(1 - half)))
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        return MVTests(se, zval, pval, b - crit * se, b + crit * se, qm, float(scipy.stats.chi2.sf(qm, len(btt))), numpy.full(len(b), math.nan))
    crit = numpy.asarray(scipy.stats.t.ppf(1 - half, ddf), dtype=numpy.float64)
    pval = numpy.asarray(2 * scipy.stats.t.sf(numpy.abs(zval), ddf), dtype=numpy.float64)
    m = len(btt)
    qm = qm / m
    return MVTests(se, zval, pval, b - crit * se, b + crit * se, qm, float(scipy.stats.f.sf(qm, m, float(numpy.min(ddf[btt])))), ddf)


def mv_qe(yi: FloatArray, V: FloatArray, X: FloatArray) -> tuple[float, float]:
    """The test for residual heterogeneity: ``QE = r' V^-1 r`` with ``r`` the residuals of the generalised least squares fit with weights ``V^-1`` (no
    random effects), and its p-value from chi-square(k - p) (Cochran 1954; Hedges and Olkin 1985, chapter 7)."""
    k, p = X.shape
    b, _ = gls(yi, V, X)
    resid = yi - X @ b
    q = float(resid @ numpy.linalg.solve(V, resid))
    return q, float(scipy.stats.chi2.sf(q, k - p))


def fix_component(terms: list[RandomTerm], index: int, kind: str, position: int, value: float) -> list[RandomTerm]:
    """``terms`` with the variance (``kind`` ``variance``) or correlation (``correlation``) at ``position`` of term ``index`` held at ``value``."""
    term = terms[index]
    if kind == 'variance':
        held = numpy.full(term.n_variances(), math.nan) if term.fixed_variances is None else term.fixed_variances.copy()
        held[position] = value
        return [*terms[:index], dataclasses.replace(term, fixed_variances=held), *terms[index + 1 :]]
    held = numpy.full(term.n_correlations(), math.nan) if term.fixed_correlations is None else term.fixed_correlations.copy()
    held[position] = value
    return [*terms[:index], dataclasses.replace(term, fixed_correlations=held), *terms[index + 1 :]]


def mv_profile_bounds(
    yi: FloatArray,
    V: FloatArray,
    X: FloatArray,
    terms: list[RandomTerm],
    method: str,
    index: int,
    kind: str,
    position: int,
    level: float = 95,
) -> FloatArray:
    """Profile-likelihood interval of one variance or correlation component (Hardy and Thompson 1996): the values either side of the estimate at which
    twice the drop of the log-likelihood, maximised over the other components, equals the ``level`` quantile of chi-square(1). The lower end of a
    variance is 0 and that of a correlation -1 where the drop there is below the quantile; the upper end of a correlation is 1 likewise.
    """
    full = mv_fit(yi, V, X, terms, method)
    estimate = float((full.variances if kind == 'variance' else full.correlations)[index][position])
    crit = float(scipy.stats.chi2.ppf(level / 100, 1))

    def excess(value: float) -> float:
        held = fix_component(terms, index, kind, position, value)
        return 2 * (full.loglik - mv_fit(yi, V, X, held, method).loglik) - crit

    low_end, high_end = (0.0, math.inf) if kind == 'variance' else (-1 + 1e-9, 1 - 1e-9)
    lower = low_end if excess(low_end) < 0 else float(scipy.optimize.brentq(excess, low_end, estimate, xtol=1e-13, rtol=1e-13))
    if kind == 'variance':
        high_end = max(1.0, 2 * estimate)
        while excess(high_end) < 0:
            high_end *= 2
    upper = high_end if excess(high_end) < 0 else float(scipy.optimize.brentq(excess, estimate, high_end, xtol=1e-13, rtol=1e-13))
    return numpy.array([lower, upper])


def mv_prediction_interval(
    yi: FloatArray, M: FloatArray, X: FloatArray, test: str, ddf: FloatArray, rows: FloatArray, component_variance: float, level: float = 95
) -> tuple[FloatArray, FloatArray]:
    """The prediction interval at the model-matrix ``rows`` of an ``rma_mv`` fit with ``sigma2`` components only (Riley, Higgins and Deeks 2011,
    equation 4, generalised): ``x'b +- crit sqrt(x' vb x + sum of the sigma2)``, with ``crit`` from the normal distribution for ``z`` and from the t
    distribution with the smallest ``ddf`` of the coefficients the row involves otherwise."""
    b, vb = gls(yi, M, X)
    if test in KNAPP_HARTUNG_TESTS:
        vb = knapp_hartung_factor(yi, M, X, test == 'adhoc') * vb
    spread = numpy.sqrt(numpy.einsum('ij,jk,ik->i', rows, vb, rows) + component_variance)
    half = alpha_of(level) / 2
    if test == 'z':
        crit = numpy.full(len(rows), float(scipy.stats.norm.ppf(1 - half)))
    else:
        crit = numpy.asarray(scipy.stats.t.ppf(1 - half, numpy.array([numpy.min(ddf[row != 0]) for row in rows])), dtype=numpy.float64)
    centre = rows @ b
    return centre - crit * spread, centre + crit * spread


# auditme: accept[non-alphabetical-literal] the documented option order, as in the code under test
CR_TYPES = ('CR0', 'CR1', 'CR1p', 'CR1S', 'CR2', 'CR3')  # auditme: accept[literal-duplication] the same tuple as the code under test
EDF_MIN_DF = 4.1


def pseudo_inverse_root(a: FloatArray, power: float) -> FloatArray:
    """``a ** power`` of a symmetric positive semi-definite matrix through its spectral decomposition, with the null space left at zero (the
    Moore-Penrose form of Pustejovsky and Tipton 2018, equation 7)."""
    values, vectors = numpy.linalg.eigh((a + a.T) / 2)
    keep = values > 1e-12 * max(1.0, float(values.max()))
    powered = numpy.zeros_like(values)
    powered[keep] = values[keep] ** power
    return numpy.asarray((vectors * powered) @ vectors.T, dtype=numpy.float64)


def adjustment_matrices(X: FloatArray, W: FloatArray, theta: FloatArray, cluster: IntArray, vcov: str) -> list[FloatArray]:
    """The adjustment matrix ``A_j`` of each cluster for the estimators of Pustejovsky and Tipton (2018, section 2), with ``X``, the weights ``W`` and the
    working covariance ``theta`` (``Phi``) as full matrices, block-diagonal over the clusters.

    ``CR0`` the identity (Liang and Zeger 1986); ``CR1`` ``sqrt(m / (m - 1))``; ``CR1S``
    ``sqrt(m (N - 1) / ((m - 1) (N - p)))`` (Stata's factor, Cameron and Miller 2015, equation 12; the paper's ``c_S`` prints ``N`` for ``N - 1``);
    ``CR1p`` ``sqrt(m / (m - p))`` (the ``clubSandwich`` documentation's variant that uses ``p`` in place of one); ``CR2`` ``D' B^(+1/2) D`` with ``D``
    the upper Cholesky factor of ``Phi_j`` and ``B = D (I - H) Phi (I - H)' D'`` (equations 7 and 8); ``CR3`` ``(I - X_j M X_j' W_j)^-1``
    (MacKinnon and White 1985, the jackknife form of the paper's section 2).
    """
    N, p = X.shape
    m = int(cluster.max()) + 1
    bread = numpy.linalg.inv(X.T @ W @ X)
    ih = numpy.eye(N) - X @ bread @ X.T @ W
    scale = {'CR0': 1.0, 'CR1': math.sqrt(m / (m - 1)), 'CR1p': math.sqrt(m / (m - p)), 'CR1S': math.sqrt(m * (N - 1) / ((m - 1) * (N - p)))}
    out: list[FloatArray] = []
    for j in range(m):
        rows = numpy.flatnonzero(cluster == j)
        if vcov in scale:
            out.append(scale[vcov] * numpy.eye(len(rows)))
        elif vcov == 'CR3':
            out.append(numpy.linalg.inv(numpy.eye(len(rows)) - X[rows] @ bread @ X[rows].T @ W[numpy.ix_(rows, rows)]))
        elif vcov == 'CR2':
            D = numpy.linalg.cholesky(theta[numpy.ix_(rows, rows)]).T
            B = D @ ih[rows] @ theta @ ih[rows].T @ D.T
            out.append(D.T @ pseudo_inverse_root(B, -0.5) @ D)
        else:
            raise ValueError(f'unknown estimator {vcov!r}')
    return out


def cluster_robust_vcov(X: FloatArray, W: FloatArray, resid: FloatArray, cluster: IntArray, vcov: str, working: FloatArray | None = None) -> FloatArray:
    """The cluster-robust variance ``(X'WX)^-1 [sum_j X_j' W_j A_j e_j e_j' A_j' W_j X_j] (X'WX)^-1`` of the weighted least squares ``b`` (Hedges, Tipton
    and Johnson 2010, equation 6; Pustejovsky and Tipton 2018, equation 4 with adjustment matrices ``A_j``).

    ``vcov`` is a factor of the meta-regression papers or one of the ``CR_TYPES``. The factors multiply the variance: ``CR0`` none; ``Hedges``
    ``n / (n - p)`` (Hedges et al., the degrees-of-freedom correction after equation 6); ``Stata1`` ``n / (n - 1) * (k - 1) / (k - p)`` (Cameron and
    Miller 2015, equation 12) and ``Stata2`` ``n / (n - 1)`` (the same section's ``G / (G - 1)``); both are Stata's convention, not the meta-analysis
    papers'. The ``CR_TYPES`` use the adjustment matrices of :func:`adjustment_matrices`, which need ``working`` (the working covariance, the inverse of
    ``W`` for an inverse-variance working model).

    ``W`` is the full weight matrix, block-diagonal over the clusters, and ``W_j`` its block of cluster ``j``; ``n`` is the number of clusters, ``k`` the
    number of rows and ``p`` the number of coefficients.
    """
    k, p = X.shape
    n = int(cluster.max()) + 1
    bread = numpy.linalg.inv(X.T @ W @ X)
    meat = numpy.zeros((p, p))
    covered = numpy.zeros(W.shape, dtype=bool)
    if vcov in CR_TYPES and vcov != 'CR0':
        if working is None:
            raise ValueError('the CR estimators need the working covariance')
        adjustments = adjustment_matrices(X, W, working, cluster, vcov)
    else:
        adjustments = [numpy.eye(int((cluster == j).sum())) for j in range(n)]
    for j in range(n):
        rows = numpy.flatnonzero(cluster == j)
        covered[numpy.ix_(rows, rows)] = True
        score = X[rows].T @ W[numpy.ix_(rows, rows)] @ adjustments[j] @ resid[rows]
        meat += numpy.outer(score, score)
    if numpy.any(W[~covered] != 0):
        raise ValueError('the weight matrix is not block-diagonal over the clusters')
    factors = {'CR0': 1.0, 'Hedges': n / (n - p), 'Stata1': n / (n - 1) * (k - 1) / (k - p), 'Stata2': n / (n - 1)}
    return numpy.asarray(factors.get(vcov, 1.0) * bread @ meat @ bread, dtype=numpy.float64)


def robust_tests(b: FloatArray, vb: FloatArray, n: int, btt: list[int], level: float = 95) -> MVTests:
    """The tests of the coefficients ``b`` with the cluster-robust variance ``vb`` over ``n`` clusters: t with ``n - p`` degrees of freedom for each
    coefficient (Hedges, Tipton and Johnson 2010, section 2) and ``QM / m`` against F(m, n - p) for the ``btt`` coefficients."""
    p = len(b)
    ddf = numpy.full(p, float(n - p))
    se = numpy.sqrt(numpy.diag(vb))
    zval = b / se
    half = alpha_of(level) / 2
    crit = float(scipy.stats.t.ppf(1 - half, n - p))
    m = len(btt)
    qm = float(b[btt] @ numpy.linalg.solve(vb[numpy.ix_(btt, btt)], b[btt])) / m
    pval = numpy.asarray(2 * scipy.stats.t.sf(numpy.abs(zval), n - p), dtype=numpy.float64)
    return MVTests(se, zval, pval, b - crit * se, b + crit * se, qm, float(scipy.stats.f.sf(qm, m, n - p)), ddf)


def club_components(
    X: FloatArray, W: FloatArray, theta: FloatArray, cluster: IntArray, vcov: str, C: FloatArray
) -> tuple[list[FloatArray], FloatArray, FloatArray]:
    """The matrices ``B_j = C M X_j' W_j A_j (I - X M X' W)_j`` (``q x N``, the rows of cluster ``j`` of the last factor; Tipton and Pustejovsky 2015,
    equation 9, before the standardisation by ``Omega^(-1/2)``) with ``Omega = sum_j B_j Theta B_j'``, the expected value of ``C V^R C'`` under the working
    model ``Theta`` (equation 7), and the whitening matrix ``Omega^(-1/2)``. Returns ``(B, Omega, Omega^(-1/2))``."""
    N = X.shape[0]
    M = numpy.linalg.inv(X.T @ W @ X)
    ih = numpy.eye(N) - X @ M @ X.T @ W
    adjustments = adjustment_matrices(X, W, theta, cluster, vcov)
    B = []
    for j, A in enumerate(adjustments):
        rows = numpy.flatnonzero(cluster == j)
        B.append(C @ M @ X[rows].T @ W[numpy.ix_(rows, rows)] @ A @ ih[rows])
    omega = numpy.sum([b @ theta @ b.T for b in B], axis=0)
    return B, omega, pseudo_inverse_root(omega, -0.5)


def covariance_of_d(B: list[FloatArray], theta: FloatArray, whiten: FloatArray) -> FloatArray:
    """``Cov(d_st, d_uv)`` as the array ``[s, t, u, v]`` for ``D = Omega^(-1/2) C V^R C' Omega^(-1/2)`` under normal errors with covariance ``theta``
    (Tipton and Pustejovsky 2015, equation 8 with the unit vectors ``u1 = j_s``, ``u2 = j_t``, ``u3 = j_u``, ``u4 = j_v``)."""
    scaled = [whiten @ b for b in B]
    q = whiten.shape[0]
    cov = numpy.zeros((q, q, q, q))
    for bi in scaled:
        for bj in scaled:
            G = bi @ theta @ bj.T
            cov += numpy.einsum('sv,tu->stuv', G, G) + numpy.einsum('su,tv->stuv', G, G)
    return cov


def satterthwaite_dfs(X: FloatArray, W: FloatArray, theta: FloatArray, cluster: IntArray, vcov: str) -> FloatArray:
    """The Satterthwaite degrees of freedom of the t test of each coefficient, ``eta = 2 / Var(d_11)`` (Tipton and Pustejovsky 2015, after equation 14;
    Pustejovsky and Tipton 2018, section 3), with ``Var(d_11)`` from equation 8 of the first paper at ``C`` the unit row of the coefficient."""
    p = X.shape[1]
    out = numpy.zeros(p)
    for s in range(p):
        C = numpy.eye(p)[[s]]
        B, _, whiten = club_components(X, W, theta, cluster, vcov, C)
        out[s] = 2 / covariance_of_d(B, theta, whiten)[0, 0, 0, 0]
    return out


@dataclasses.dataclass(frozen=True)
class WaldTest:
    """A cluster-robust Wald test: the statistic on the scale of ``QM`` (``F``), its degrees of freedom (the denominator ``inf`` for a chi-square) and
    its p-value."""

    QM: float
    df_num: int
    df_denom: float
    pval: float


def hotelling_eta(cov: FloatArray, test: str) -> float:
    """The degrees of freedom ``eta`` of the approximating Hotelling's T-squared distribution (Tipton and Pustejovsky 2015): ``HTA`` equation 12,
    ``HTB`` equation 13 and ``HTZ`` equation 14, from the covariances ``cov[s, t, u, v]`` of the entries of ``D``. The squares of the covariances in
    equations 12 and 13 are the least-squares criterion the equations solve: ``eta`` minimises the squared differences of ``eta Cov`` from the
    Wishart's ``I(s = u) I(t = v) + I(s = v) I(t = u)``."""
    q = cov.shape[0]
    var = numpy.array([[cov[s, t, s, t] for t in range(q)] for s in range(q)])
    if test == 'HTZ':
        return float(q * (q + 1) / var.sum())
    if test == 'HTA':
        return 2 * float(var.sum()) / float(numpy.sum(cov**2))
    top = 0.0
    bottom = 0.0
    for s in range(q):
        for t in range(s + 1):
            top += var[s, t]
            for u in range(s):
                for v in range(u + 1):
                    bottom += cov[s, t, u, v] ** 2
            for v in range(t + 1):
                bottom += cov[s, t, s, v] ** 2
    return 2 * top / bottom


def hill_transform(t: FloatArray, f: FloatArray) -> FloatArray:
    """Hill's (1970) normalising transformation of a t variate with ``f`` degrees of freedom (Tipton and Pustejovsky 2015, equation 18 and after)."""
    a = f - 0.5
    b = 48 * a**2
    c = numpy.sqrt(a * numpy.log(1 + t**2 / f))
    return numpy.asarray(c + (c**3 + 3 * c) / b - (4 * c**7 + 33 * c**5 + 240 * c**3 + 855 * c) / (10 * b**2 + 8 * b * c**4 + 1000 * b))


def hotelling_wald(
    X: FloatArray, W: FloatArray, theta: FloatArray, cluster: IntArray, vcov: str, beta: FloatArray, vb: FloatArray, C: FloatArray, test: str
) -> WaldTest:
    """The cluster-robust Wald test of ``C beta = 0`` (Tipton and Pustejovsky 2015, equations 10 to 18; Pustejovsky and Tipton 2018, section 3).

    ``vb`` is the robust variance of ``beta``. ``chi-sq`` compares ``Q`` to a chi-square with ``q`` degrees of freedom, ``Naive-F`` ``Q / q`` to
    ``F(q, m - 1)`` and ``Naive-Fp`` to ``F(q, m - p)`` (the ``clubSandwich`` documentation; Hedges et al. 2010 use ``m - p``); ``HTA``, ``HTB`` and
    ``HTZ`` compare ``(eta - q + 1) Q / (eta q)`` to ``F(q, eta - q + 1)`` (equation 10); ``EDF`` and ``EDT`` are the eigen-decomposition tests of
    equations 15 to 18. ``QM`` is ``Q / q`` for the chi-square and naive tests, as metafor reports it, and the scaled statistic of the test otherwise.
    """
    q = C.shape[0]
    m = int(cluster.max()) + 1
    p = X.shape[1]
    cb = C @ beta
    cvc = C @ vb @ C.T
    Q = float(cb @ numpy.linalg.solve(cvc, cb))
    if test == 'chi-sq':
        return WaldTest(Q / q, q, math.inf, float(scipy.stats.chi2.sf(Q, q)))
    if test in ('Naive-F', 'Naive-Fp'):
        df = m - 1 if test == 'Naive-F' else m - p
        return WaldTest(Q / q, q, float(df), float(scipy.stats.f.sf(Q / q, q, df)))
    B, _, whiten = club_components(X, W, theta, cluster, vcov, C)
    cov = covariance_of_d(B, theta, whiten)
    if test in ('HTA', 'HTB', 'HTZ'):
        eta = hotelling_eta(cov, test)
        df = eta - q + 1
        stat = df * Q / (eta * q)
        return WaldTest(stat, q, df, float(scipy.stats.f.sf(stat, q, df)))
    lam, vectors = numpy.linalg.eigh(whiten @ cvc @ whiten)
    scaled = [whiten @ b for b in B]
    f = numpy.zeros(q)
    for s in range(q):
        ps = vectors[:, s]
        f[s] = 1 / sum(float(ps @ bi @ theta @ bj.T @ ps) ** 2 for bi in scaled for bj in scaled)
    if test == 'EDF':
        g = numpy.maximum(f, EDF_MIN_DF)
        EQ = float(numpy.sum(g / (g - 2)))
        VQ = float(2 * numpy.sum(g**2 * (g - 1) / ((g - 2) ** 2 * (g - 4))))
        if q * VQ <= 2 * EQ**2:
            raise ValueError('equation 17 has no solution: q VQ <= 2 EQ^2')
        delta = (EQ**2 * (q - 2) + 2 * q * VQ) / (EQ * (VQ + EQ**2))  # the printed q in the denominator breaks E[delta Q / q] = nu / (nu - 2)
        df = 4 + 2 * EQ**2 * (q + 2) / (q * VQ - 2 * EQ**2)
        stat = delta * Q / q
        return WaldTest(stat, q, df, float(scipy.stats.f.sf(stat, q, df)))
    if test == 'EDT':
        t = (vectors.T @ whiten @ cb) / numpy.sqrt(lam)
        total = float(numpy.sum(hill_transform(t, f) ** 2))
        return WaldTest(total / q, q, math.inf, float(scipy.stats.chi2.sf(total, q)))
    raise ValueError(f'unknown Wald test {test!r}')


def saddlepoint_pvalue(X: FloatArray, W: FloatArray, theta: FloatArray, cluster: IntArray, vcov: str, beta: FloatArray, vb: FloatArray) -> FloatArray:
    """The saddlepoint p-value of the t test of each coefficient (Kuonen 1999; the application of section 5, equation 4, to ``T = c'b / sqrt(V^R)``).

    Under the working model ``e ~ N(0, Theta)`` and ``c'(b - beta) = a'e`` with ``a = W X M c``, and ``V^R = e'Q e`` with ``Q = sum_j b_j' b_j`` for
    the rows ``b_j`` of ``C M X_j' W_j A_j (I - H)_j``. The two-sided p-value of the observed ``t`` is ``pr(e'U e > 0)`` with ``U = a a' - t^2 Q``
    (equation 4 of Kuonen with ``T^2`` for ``T``), approximated by the Lugannani-Rice form Kuonen names after equation 3: ``1 - Phi(w) + phi(w) (1 / v
    - 1 / w)``, ``w = sign(s) sqrt(-2 K(s))``, ``v = s sqrt(K''(s))`` and ``K'(s) = 0``, for the central form ``K(s) = -1/2 sum log(1 - 2 l s)`` over
    the eigenvalues ``l`` of ``Theta^(1/2) U Theta^(1/2)`` (section 2). The form is Lugannani-Rice, the one clubSandwich uses, not equation 3 itself;
    the paper calls them "usually indistinguishable in practice".
    """
    p = X.shape[1]
    M = numpy.linalg.inv(X.T @ W @ X)
    root = numpy.linalg.cholesky(theta)
    out = numpy.zeros(p)
    for s in range(p):
        C = numpy.eye(p)[[s]]
        B, _, _ = club_components(X, W, theta, cluster, vcov, C)
        Q = sum(b.T @ b for b in B)
        a = W @ X @ M @ C.T
        t2 = float(beta[s] ** 2 / vb[s, s])
        lam = numpy.linalg.eigvalsh(root.T @ (a @ a.T - t2 * Q) @ root)
        lam = lam[numpy.abs(lam) > 1e-12 * numpy.abs(lam).max()]

        def slope(z: float, lam: FloatArray = lam) -> float:
            return float(numpy.sum(lam / (1 - 2 * lam * z)))

        lo, hi = 1 / (2 * float(lam.min())), 1 / (2 * float(lam.max()))
        gap = 1e-9 * (hi - lo)
        z = float(scipy.optimize.brentq(slope, lo + gap, hi - gap, xtol=1e-14, rtol=1e-14))
        w = math.copysign(math.sqrt(float(numpy.sum(numpy.log(1 - 2 * lam * z)))), z)
        v = z * math.sqrt(2 * float(numpy.sum(lam**2 / (1 - 2 * lam * z) ** 2)))
        out[s] = float(scipy.stats.norm.sf(w) + scipy.stats.norm.pdf(w) * (1 / v - 1 / w))
    return out


# --- Location-scale models: Viechtbauer and Lopez-Lopez (2022), Res Synth Methods 13:697-715, doi:10.1002/jrsm.1562 ---


LS_START_FACTORS = (1.0, 0.1, 10.0, 0.01, 100.0)  # the multiples of the moment-based intercept that ``ls_starts`` returns
LS_SCORE_TOLERANCE = 1e-6  # the largest absolute score at which ``ls_reference_fit`` accepts a log-link fit as stationary
LS_AGREEMENT = 1e-9  # two log-link starts whose log-likelihoods differ by no more than this end the search over starts
IDENTITY_SLACK = 1e-10  # the amount by which SLSQP may leave Z alpha >= 0 before the reference treats alpha as infeasible


def ls_tau2(Z: FloatArray, alpha: FloatArray, link: str) -> FloatArray:
    """The scale model ``tau2_i = exp(z_i' alpha)`` (log link) or ``z_i' alpha`` (identity link): equations 2 and 3 of Viechtbauer and Lopez-Lopez (2022).

    The log-linear form of the dispersion is that of Harvey (1976) and Smyth (1989).
    """
    eta = Z @ alpha
    if link == 'log':
        return numpy.exp(eta)
    if link == 'identity':
        return eta
    raise ValueError(f'unknown link {link!r}')


def ls_derivatives(Z: FloatArray, alpha: FloatArray, link: str) -> tuple[FloatArray, FloatArray]:
    """``d tau2_i / d eta_i`` and ``d2 tau2_i / d eta_i^2`` for ``eta = Z alpha``: both ``tau2_i`` for the log link, ``1`` and ``0`` for the identity link."""
    if link == 'log':
        t = numpy.exp(Z @ alpha)
        return t, t
    if link == 'identity':
        return numpy.ones(Z.shape[0]), numpy.zeros(Z.shape[0])
    raise ValueError(f'unknown link {link!r}')


def ls_components(
    yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray] | None:
    """``M = V + diag(tau2)`` (its diagonal), ``W = M^-1``, ``(X'WX)^-1``, ``beta`` (equation 6) and ``P`` (equation 8).

    ``None`` when ``M`` is not positive or ``X'WX`` is singular.
    """
    m = vi + ls_tau2(Z, alpha, link)
    if not bool(numpy.all(numpy.isfinite(m))) or not bool(numpy.all(m > 0)):
        return None
    W = numpy.diag(1 / m)
    try:
        A_inv = numpy.linalg.inv(X.T @ W @ X)
    except numpy.linalg.LinAlgError:
        return None
    beta = A_inv @ X.T @ W @ yi
    P = W - W @ X @ A_inv @ X.T @ W
    return m, W, A_inv, beta, P


def ls_loglik(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> float:
    """The profile log-likelihood (equation 7, ``beta`` at its value from equation 6) for ``ML`` or the restricted log-likelihood (equation 9) for ``REML``.

    ``ll_P = -k/2 ln 2pi - 1/2 ln|M| - 1/2 y'Py`` and
    ``ll_R = -(k-p)/2 ln 2pi + 1/2 ln|X'X| - 1/2 ln|M| - 1/2 ln|X'WX| - 1/2 y'Py`` with ``p`` the number of columns of ``X`` (the paper's ``p + 1``).
    The restricted form is that of Verbyla (1993). Returns ``-inf`` where ``M`` is not positive (identity link outside ``Z alpha >= 0``).
    """
    parts = ls_components(yi, vi, X, Z, alpha, link)
    if parts is None:
        return -math.inf
    m, W, _, _, P = parts
    k, p = X.shape
    quad = float(yi @ P @ yi)
    logdet_m = float(numpy.sum(numpy.log(m)))
    if method == 'ML':
        return float(-k / 2 * math.log(2 * math.pi) - logdet_m / 2 - quad / 2)
    if method == 'REML':
        logdet_xx = float(numpy.linalg.slogdet(X.T @ X)[1])
        logdet_xwx = float(numpy.linalg.slogdet(X.T @ W @ X)[1])
        return float(-(k - p) / 2 * math.log(2 * math.pi) + logdet_xx / 2 - logdet_m / 2 - logdet_xwx / 2 - quad / 2)
    raise ValueError(f'unknown method {method!r}')


def ls_score(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The analytic gradient of ``ls_loglik`` in ``alpha``, derived from equations 7 and 9 with ``dP = -P (dM) P``.

    ``d ll / d alpha_j = -1/2 tr(T D_j) + 1/2 y'P D_j P y`` with ``D_j = diag(tau2'_i z_ij)`` and ``T = W`` (ML) or ``T = P`` (REML), because
    ``d ln|X'WX| = -tr(W X (X'WX)^-1 X' W dM)`` turns the trace of ``W`` into the trace of ``P`` in the restricted case.
    """
    parts = ls_components(yi, vi, X, Z, alpha, link)
    if parts is None:
        raise ValueError('M is not positive at alpha')
    _, W, _, _, P = parts
    T = W if method == 'ML' else P
    d1, _ = ls_derivatives(Z, alpha, link)
    Py = P @ yi
    out = numpy.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        dj = d1 * Z[:, j]
        out[j] = -0.5 * float(numpy.sum(numpy.diag(T) * dj)) + 0.5 * float(numpy.sum(Py**2 * dj))
    return out


def ls_information(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The analytic observed information (minus the Hessian of ``ls_loglik``) in ``alpha``, differentiating ``ls_score`` once more.

    ``-d2 ll / d alpha_j d alpha_l = -1/2 tr(T D_l T D_j) + 1/2 tr(T D_jl) + y'P D_l P D_j P y - 1/2 y'P D_jl P y``,
    where ``D_jl = diag(tau2''_i z_ij z_il)`` and ``T`` is as in ``ls_score`` (``dW = -W dM W`` and ``dP = -P dM P``).
    The inverse of this matrix is the covariance of the scale coefficients.
    """
    parts = ls_components(yi, vi, X, Z, alpha, link)
    if parts is None:
        raise ValueError('M is not positive at alpha')
    _, W, _, _, P = parts
    T = W if method == 'ML' else P
    d1, d2 = ls_derivatives(Z, alpha, link)
    Py = P @ yi
    q = Z.shape[1]
    out = numpy.zeros((q, q))
    for j in range(q):
        for ell in range(q):
            dj, dl = d1 * Z[:, j], d1 * Z[:, ell]
            djl = d2 * Z[:, j] * Z[:, ell]
            tr_cross = float(numpy.trace(T @ numpy.diag(dl) @ T @ numpy.diag(dj)))
            out[j, ell] = 0.5 * tr_cross - 0.5 * float(numpy.sum(numpy.diag(T) * djl)) - float((Py * dl) @ P @ (dj * Py)) + 0.5 * float(numpy.sum(Py**2 * djl))
    return -out


def ls_starts(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str) -> list[FloatArray]:
    """Starting values for ``ls_reference_fit``: a moment-based intercept, scaled by each of ``LS_START_FACTORS``, with zero slopes."""
    resid = yi - X @ numpy.linalg.lstsq(X, yi, rcond=None)[0]
    t0 = max(float(numpy.sum(resid**2)) / max(len(yi) - X.shape[1], 1) - float(numpy.mean(vi)), 1e-3)
    out = []
    for factor in LS_START_FACTORS:
        alpha = numpy.zeros(Z.shape[1])
        alpha[0] = math.log(t0 * factor) if link == 'log' else t0 * factor
        out.append(alpha)
    return out


def ls_identity_starts(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, method: str) -> list[FloatArray]:
    """Starting values for the identity link from the log-link optimum: ``alpha`` that reproduces its ``tau2_i`` by least squares, and the same with a
    boundary-ward shift. The restricted likelihood of the identity link can have several optima, and the moment-based starts reach only one."""
    try:
        fitted = ls_reference_fit(yi, vi, X, Z, 'log', method)
    except RuntimeError:
        return []
    alpha = numpy.linalg.lstsq(Z, fitted['tau2'], rcond=None)[0]
    shifted = alpha.copy()
    shifted[0] -= float(numpy.min(Z @ alpha))
    return [alpha, shifted]


def ls_newton_polish(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str, maxiter: int) -> FloatArray:
    """Newton steps on the analytic ``ls_score`` with the analytic ``ls_information``, each accepted only when the log-likelihood does not fall.

    Stops when the largest absolute score is below ``1e-10``, when the information is not positive definite, or when a step cannot be halved into one
    that does not lower the log-likelihood.
    """
    current = alpha
    value = ls_loglik(yi, vi, X, Z, current, link, method)
    for _ in range(maxiter):
        score = ls_score(yi, vi, X, Z, current, link, method)
        if float(numpy.max(numpy.abs(score))) < 1e-10:
            break
        info = ls_information(yi, vi, X, Z, current, link, method)
        if not bool(numpy.all(numpy.linalg.eigvalsh(info) > 0)):
            break
        step = numpy.linalg.solve(info, score)
        for _ in range(30):
            trial = current + step
            trial_value = ls_loglik(yi, vi, X, Z, trial, link, method)
            if trial_value >= value:
                current, value = trial, trial_value
                break
            step = step / 2
        else:
            break
    return current


def ls_reference_fit(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, maxiter: int = 2000) -> dict[str, typing.Any]:
    """Maximise ``ls_loglik`` with ``scipy.optimize``, independently of metafopy.

    From each of ``ls_starts`` (and, for the identity link, ``ls_identity_starts``) BFGS with the analytic ``ls_score`` (SLSQP with the constraint
    ``Z alpha >= 0`` for the identity link, as Viechtbauer and Lopez-Lopez 2022 suggest after equation 8). The optimiser's message is not trusted: a stop
    at "precision loss" or a "positive directional derivative" can lie on the maximum, and a "success" can lie on a worse stationary point, and the
    platform's libm and BLAS decide which. For the log link every start is polished by ``ls_newton_polish`` and kept when its largest absolute score is
    at most ``LS_SCORE_TOLERANCE``; the search stops once two kept starts agree to ``LS_AGREEMENT``. For the identity link, where the score does not
    vanish on the boundary ``Z alpha = 0``, every feasible finite end point is kept. The kept point of highest log-likelihood is polished by
    Nelder-Mead. Returns ``alpha``, ``beta`` (equation 6), ``vb`` (equation 10), ``tau2``, ``ll`` and ``va`` (the inverse of the analytic observed
    information). Raises ``RuntimeError`` with each start's message when no start is kept.
    """

    def negative(alpha: FloatArray) -> float:
        if link == 'identity' and float(numpy.min(Z @ alpha)) < -IDENTITY_SLACK:
            return math.inf
        value = ls_loglik(yi, vi, X, Z, alpha, link, method)
        return math.inf if value == -math.inf else -value

    def gradient(alpha: FloatArray) -> FloatArray:
        if ls_components(yi, vi, X, Z, alpha, link) is None:
            return numpy.full(len(alpha), numpy.nan)
        return -ls_score(yi, vi, X, Z, alpha, link, method)

    kept: list[tuple[float, FloatArray]] = []
    messages: list[str] = []
    starts = ls_starts(yi, vi, X, Z, link)
    if link == 'identity':
        starts.extend(ls_identity_starts(yi, vi, X, Z, method))
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if link == 'log':
                res = scipy.optimize.minimize(negative, start, jac=gradient, method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-9})
            else:
                feasible = scipy.optimize.LinearConstraint(Z, 0.0, numpy.inf)
                res = scipy.optimize.minimize(
                    negative, start, jac=gradient, method='SLSQP', constraints=[feasible], options={'maxiter': maxiter, 'ftol': 1e-14}
                )
        point = numpy.asarray(res.x, dtype=float)
        if not bool(numpy.all(numpy.isfinite(point))):
            messages.append(str(res.message))
            continue
        if link == 'log':
            point = ls_newton_polish(yi, vi, X, Z, point, link, method, maxiter)
        value = negative(point)
        if not math.isfinite(value):
            messages.append(str(res.message))
            continue
        if link == 'log':
            score = float(numpy.max(numpy.abs(ls_score(yi, vi, X, Z, point, link, method))))
            if score > LS_SCORE_TOLERANCE:
                messages.append(f'{res.message} (score {score:.3g})')
                continue
        kept.append((value, point))
        if link == 'log' and len(kept) >= 2 and abs(sorted(v for v, _ in kept)[1] - min(v for v, _ in kept)) <= LS_AGREEMENT:
            break
    if not kept:
        raise RuntimeError(f'ls_reference_fit: no start converged ({link} link, {method}): {"; ".join(messages)}')
    best_value, best_point = min(kept, key=lambda item: item[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        polish = scipy.optimize.minimize(negative, best_point, method='Nelder-Mead', options={'maxiter': maxiter * 10, 'xatol': 1e-11, 'fatol': 1e-13})
    alpha = polish.x if polish.fun <= best_value else best_point
    parts = ls_components(yi, vi, X, Z, alpha, link)
    if parts is None:
        raise RuntimeError(f'ls_reference_fit: the best point leaves M not positive ({link} link, {method})')
    _, _, A_inv, beta, _ = parts
    return {
        'alpha': alpha,
        'beta': beta,
        'vb': A_inv,
        'tau2': ls_tau2(Z, alpha, link),
        'll': ls_loglik(yi, vi, X, Z, alpha, link, method),
        'va': numpy.linalg.inv(ls_information(yi, vi, X, Z, alpha, link, method)),
    }


def ls_i2(tau2: FloatArray, vt: float) -> FloatArray:
    """``I2_i = 100 tau2_i / (vt + tau2_i)`` per study, in percent: Higgins and Thompson (2002), equation 9, with the heterogeneity of study ``i``."""
    return 100 * tau2 / (vt + tau2)


def ls_h2(tau2: FloatArray, vt: float) -> FloatArray:
    """``H2_i = (vt + tau2_i) / vt`` per study: Higgins and Thompson (2002), equation 7, with the heterogeneity of study ``i``."""
    return (vt + tau2) / vt


def ls_fitstats(ll: float, parms: int, n: int) -> dict[str, float]:
    """The information criteria of a fit with log-likelihood ``ll`` and ``parms`` estimated parameters on ``n`` observations.

    ``AIC = -2 ll + 2 parms`` (Akaike, 1974), ``BIC = -2 ll + parms ln n`` (Schwarz, 1978) and
    ``AICc = -2 ll + 2 parms n / (n - parms - 1)`` (Hurvich and Tsai, 1989). ``n`` is ``k`` under ML and ``k - p`` under REML, as in Table 3 of
    Viechtbauer and Lopez-Lopez (2022).
    """
    return {
        'll': ll,
        'AIC': -2 * ll + 2 * parms,
        'BIC': -2 * ll + parms * math.log(n),
        'AICc': -2 * ll + 2 * parms * n / (n - parms - 1),
    }


def ls_wald(estimate: FloatArray, vcov: FloatArray, idx: list[int], ddf: float | None) -> tuple[float, float]:
    """The Wald statistic ``b' V^-1 b`` over the coefficients ``idx`` and its p-value: equations 12 (location) and 14 (scale) of Viechtbauer and
    Lopez-Lopez (2022).

    Without ``ddf`` the statistic is chi-square distributed with ``len(idx)`` degrees of freedom. With ``ddf`` (the t-type tests) the statistic is divided by
    ``len(idx)`` and has an F distribution with ``(len(idx), ddf)`` degrees of freedom, as in Knapp and Hartung (2003).
    """
    b = estimate[idx]
    m = len(idx)
    stat = float(b @ numpy.linalg.solve(vcov[numpy.ix_(idx, idx)], b))
    if ddf is None:
        return stat, float(scipy.stats.chi2.sf(stat, m))
    return stat / m, float(scipy.stats.f.sf(stat / m, m, ddf))


def ls_knha_s2w(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray, adhoc: bool) -> float:
    """The Knapp-Hartung factor of section 2.5.4 of Viechtbauer and Lopez-Lopez (2022): ``RSS / (k - p)``, the weighted residual sum of squares at ``tau2_i``.

    ``adhoc`` raises a factor below 1 to 1 (the ad hoc correction of Knapp and Hartung, 2003, Stat Med 22:2693-2710).
    """
    w = 1 / (vi + tau2)
    beta = numpy.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * yi))
    rss = float(numpy.sum(w * (yi - X @ beta) ** 2))
    s2w = rss / (X.shape[0] - X.shape[1])
    return float(max(s2w, 1.0) if adhoc else s2w)


def ls_profile_ll(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, j: int, value: float, start: FloatArray) -> float:
    """The log-likelihood maximised over the scale coefficients other than ``j``, with ``alpha_j`` held at ``value`` (the profile of section 2.5.3).

    BFGS with the analytic score (SLSQP with ``Z alpha >= 0`` for the identity link) from ``start``, then Nelder-Mead. A value that leaves no feasible
    ``tau2_i`` gives a large negative number, so that a root search treats it as far outside the interval.
    """
    free = [i for i in range(Z.shape[1]) if i != j]

    def full(par: FloatArray) -> FloatArray:
        alpha = start.copy()
        alpha[j] = value
        alpha[free] = par
        return alpha

    def negative(par: FloatArray) -> float:
        alpha = full(par)
        if link == 'identity' and float(numpy.min(Z @ alpha)) < -IDENTITY_SLACK:
            return math.inf
        ll = ls_loglik(yi, vi, X, Z, alpha, link, method)
        return math.inf if ll == -math.inf else -ll

    def gradient(par: FloatArray) -> FloatArray:
        alpha = full(par)
        if ls_components(yi, vi, X, Z, alpha, link) is None:
            return numpy.full(len(par), numpy.nan)
        return -ls_score(yi, vi, X, Z, alpha, link, method)[free]

    if not free:
        ll = ls_loglik(yi, vi, X, Z, full(numpy.zeros(0)), link, method)
        return ll if math.isfinite(ll) else -1e10
    # A start that keeps the fitted line of ``Z alpha`` as close as possible to that of ``start`` with alpha_j at ``value``: the profile of a coefficient
    # whose change the others can mostly absorb is flat, and BFGS from the unadjusted start can run to tau2 = 0.
    adjusted = numpy.linalg.lstsq(Z[:, free], Z @ start - Z[:, j] * value, rcond=None)[0]
    best = math.inf
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for first in (adjusted, start[free]):
            if link == 'log':
                res = scipy.optimize.minimize(negative, first, jac=gradient, method='BFGS', options={'gtol': 1e-9, 'maxiter': 2000})
            else:
                constraint = scipy.optimize.LinearConstraint(Z[:, free], -Z[:, j] * value, numpy.inf)
                res = scipy.optimize.minimize(negative, first, jac=gradient, method='SLSQP', constraints=[constraint], options={'ftol': 1e-14, 'maxiter': 2000})
            polish = scipy.optimize.minimize(negative, res.x, method='Nelder-Mead', options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 20000})
            best = min(best, float(res.fun), float(polish.fun))
    return -best if math.isfinite(best) else -1e10


def ls_profile_bounds(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, j: int, level: float) -> tuple[float, float]:
    """The profile-likelihood interval of ``alpha_j`` (section 2.5.3 of Viechtbauer and Lopez-Lopez, 2022; Hardy and Thompson, 1996, Stat Med 15:619-629).

    ``brentq`` finds the values ``a`` at which ``2 (ll_max - ll_j(a)) = chi2_{1, level}``, refitting the other coefficients at each ``a``. ``level`` is a
    proportion (``0.95``). A bound that the profile does not cross within 20 standard errors of the estimate is ``nan``.
    """
    fit = ls_reference_fit(yi, vi, X, Z, link, method)
    alpha_hat = fit['alpha']
    se = math.sqrt(float(fit['va'][j, j]))
    critical = float(scipy.stats.chi2.ppf(level, 1))

    def excess(a: float) -> float:
        return float(2 * (fit['ll'] - ls_profile_ll(yi, vi, X, Z, link, method, j, a, alpha_hat)) - critical)

    bounds = []
    for sign in (-1.0, 1.0):
        end = float(alpha_hat[j]) + sign * 20 * se
        if excess(end) < 0:
            bounds.append(math.nan)
            continue
        lo, hi = sorted((float(alpha_hat[j]), end))
        bounds.append(float(scipy.optimize.brentq(excess, lo, hi, xtol=1e-13, rtol=1e-13)))
    return bounds[0], bounds[1]


def ls_lrt(ll_full: float, ll_reduced: float, df: int) -> tuple[float, float]:
    """The likelihood-ratio statistic ``2 (ll_full - ll_reduced)`` of equation 15 of Viechtbauer and Lopez-Lopez (2022) and its chi-square p-value on ``df``."""
    stat = max(0.0, 2 * (ll_full - ll_reduced))
    return stat, float(scipy.stats.chi2.sf(stat, df))


def ls_predict(x0: FloatArray, z0: FloatArray, fit: dict[str, typing.Any], link: str, crit: float) -> dict[str, float]:
    """The prediction at one row ``x0`` of the location and ``z0`` of the scale model matrix (sections 2.5.2 and 2.7 of Viechtbauer and Lopez-Lopez, 2022).

    The average true effect is ``x0' beta`` with variance ``x0' vb x0`` and the interval ``pred +- crit se``; the prediction interval adds the predicted
    ``tau2`` of ``z0`` to that variance. The scale prediction ``z0' alpha`` has variance ``z0' va z0`` and an interval on the scale of the link; ``tau2``
    is that prediction under the inverse link (``exp`` for the log link).
    """
    pred = float(x0 @ fit['beta'])
    se = math.sqrt(float(x0 @ fit['vb'] @ x0))
    lin = float(z0 @ fit['alpha'])
    se_lin = math.sqrt(float(z0 @ fit['va'] @ z0))
    tau2 = math.exp(lin) if link == 'log' else max(lin, 0.0)
    pi_half = crit * math.sqrt(se**2 + tau2)
    return {
        'pred': pred,
        'se': se,
        'ci_lb': pred - crit * se,
        'ci_ub': pred + crit * se,
        'pi_lb': pred - pi_half,
        'pi_ub': pred + pi_half,
        'scale': lin,
        'scale_se': se_lin,
        'scale_lb': lin - crit * se_lin,
        'scale_ub': lin + crit * se_lin,
        'tau2': tau2,
    }


def ls_exact_permutation_p(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, part: str) -> dict[str, FloatArray | float]:
    """Exact permutation p-values of one part of a location-scale model, by enumerating every distinct permutation of the rows of its model matrix.

    ``part='location'`` permutes the rows of ``X`` and ``part='scale'`` those of ``Z`` (Higgins and Thompson, 2004; Follmann and Proschan, 1999), each
    permuted model refitted with ``ls_reference_fit``. The p-value of a coefficient is the share of permutations whose ``|z|`` reaches the observed
    ``|z|`` (less ``comptol``), and that of the omnibus test the share whose Wald statistic over the coefficients after the intercept reaches the
    observed one.

    A permuted model whose likelihood has no finite maximum (``tau2`` tends to 0 in a group, so ``alpha`` tends to ``-inf`` under the log link, or the
    information is singular) has no reference fit. Those permutations are counted in ``failed`` and bound each p-value instead: the reference share of
    all ``count`` permutations lies between ``pval_low`` (none of them reaches the observed statistic) and ``pval_high`` (all of them do). Returns
    ``pval_low`` and ``pval_high`` (per coefficient), ``omnibus_low`` and ``omnibus_high``, ``count`` (the distinct permutations) and ``failed``.
    """
    comptol = math.sqrt(numpy.finfo(float).eps)
    if part == 'location':
        matrix = X
    elif part == 'scale':
        matrix = Z
    else:
        raise ValueError(f"part must be 'location' or 'scale', not {part!r}")

    def statistics(order: list[int]) -> tuple[FloatArray, float] | None:
        try:
            fit = ls_reference_fit(yi, vi, X[order] if part == 'location' else X, Z[order] if part == 'scale' else Z, link, method)
            estimate, vcov = (fit['beta'], fit['vb']) if part == 'location' else (fit['alpha'], fit['va'])
            variances = numpy.diag(vcov)
            if not numpy.all(numpy.isfinite(variances) & (variances > 0)):
                return None
            return estimate / numpy.sqrt(variances), ls_wald(estimate, vcov, list(range(1, len(estimate))), None)[0]
        except RuntimeError, numpy.linalg.LinAlgError:
            return None

    observed = statistics(list(range(len(yi))))
    if observed is None:
        raise RuntimeError('ls_exact_permutation_p: the observed model has no reference fit')
    z_obs, q_obs = observed
    groups = {row: index for index, row in enumerate(sorted({tuple(r) for r in matrix.tolist()}))}
    labels = [groups[tuple(r)] for r in matrix.tolist()]
    members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    z_hits = numpy.zeros(len(z_obs))
    q_hits = 0
    count = failed = 0
    for sequence in distinct_label_orders(sorted(labels)):
        taken = {label: iter(rows) for label, rows in members.items()}
        count += 1
        stats = statistics([next(taken[label]) for label in sequence])
        if stats is None:
            failed += 1
            continue
        z_hits += numpy.abs(stats[0]) >= numpy.abs(z_obs) - comptol
        q_hits += stats[1] >= q_obs - comptol
    return {
        'pval_low': z_hits / count,
        'pval_high': (z_hits + failed) / count,
        'omnibus_low': q_hits / count,
        'omnibus_high': (q_hits + failed) / count,
        'count': float(count),
        'failed': float(failed),
    }


def distinct_label_orders(remaining: list[int]) -> collections.abc.Iterator[tuple[int, ...]]:
    """Every distinct ordering of the multiset ``remaining`` (a sorted list of labels)."""
    if not remaining:
        yield ()
        return
    for label in sorted(set(remaining)):
        rest = list(remaining)
        rest.remove(label)
        for tail in distinct_label_orders(rest):
            yield (label, *tail)


# Publication bias: regression tests for funnel plot asymmetry and the rank correlation test.


def regtest_predictor(vi: FloatArray, predictor: str, ni: FloatArray | None) -> FloatArray:
    """The measure of precision added to the model: the standard error ``sqrt(v)`` (Egger et al., 1997, BMJ 315:629-634), the sampling variance ``v``
    (Moreno et al., 2009, BMC Med Res Methodol 9:2), the sample size ``n`` or its inverse (Macaskill et al., 2001, Stat Med 20:641-654; Peters et al.,
    2006, JAMA 295:676-680), or the square root of ``n`` or its inverse (Pustejovsky and Rodgers, 2019, Res Synth Methods 10:57-71)."""
    if predictor == 'sei':
        return numpy.sqrt(vi)
    if predictor == 'vi':
        return vi
    if ni is None:
        raise ValueError(f'the predictor {predictor!r} needs the sample sizes')
    return {'ni': ni, 'ninv': 1 / ni, 'sqrtni': numpy.sqrt(ni), 'sqrtninv': 1 / numpy.sqrt(ni)}[predictor]


def egger_regression(yi: FloatArray, vi: FloatArray, extra: FloatArray) -> tuple[FloatArray, FloatArray, int]:
    """Egger's regression in its original form: the standardised effect ``yi / sei`` regressed by ordinary least squares on the precision ``1 / sei``
    and on ``extra / sei`` (Egger et al., 1997, equation 1, where ``extra`` is the standard error and ``extra / sei`` is 1; Sterne and Egger, 2005,
    for the other predictors). Returns the coefficients of ``(1 / sei, extra / sei)``, their standard errors from the residual variance ``RSS / (k - 2)``,
    and ``k - 2``. The coefficient of ``extra / sei`` is the intercept of the original equation, the bias; that of ``1 / sei`` the effect at ``extra = 0``."""
    sei = numpy.sqrt(vi)
    design = numpy.column_stack([1 / sei, extra / sei])
    response = yi / sei
    gram = numpy.linalg.inv(design.T @ design)
    coefficients = gram @ design.T @ response
    df = len(yi) - 2
    residual = response - design @ coefficients
    return coefficients, numpy.sqrt(numpy.diag(gram) * float(residual @ residual) / df), df


def meta_regression_with_predictor(
    yi: FloatArray, vi: FloatArray, extra: FloatArray, method: str, test: str, level: float
) -> tuple[FloatArray, FloatArray, float, float]:
    """The meta-regression of ``yi`` on an intercept and ``extra`` with the random-effects weights ``1 / (v + tau2)`` (Sterne and Egger, 2005; Thompson and
    Sharp, 1999, Stat Med 18:2693-2708), ``tau2`` of ``method`` estimated with the predictor in the model. Returns the coefficients, the standard errors,
    the degrees of freedom of the test (infinite for ``z``) and the critical value of the interval."""
    beta, cov, df = coefficient_covariance(yi, vi, method, test, '~ extra', {'extra': extra})
    return beta, numpy.sqrt(numpy.diag(cov)), df, critical_value(test, df, level)


def regtest_quantities(
    x: FloatArray, vi: FloatArray, model: str, predictor: str, ni: FloatArray | None, method: str, test: str, level: float
) -> dict[str, float]:
    """``zval``, ``pval``, ``est``, ``ci_lb`` and ``ci_ub`` of the regression test: the statistic and p-value of the coefficient of the predictor, and the
    limit estimate (the intercept, the effect at zero of the predictor) with its interval (Stanley and Doucouliagos, 2014, Res Synth Methods 5:60-78)."""
    extra = regtest_predictor(vi, predictor, ni)
    if model == 'lm':
        coefficients, errors, df = egger_regression(x, vi, extra)
        crit = float(scipy.stats.t.isf(alpha_of(level) / 2, df))
        beta, se = coefficients[[1, 0]], errors[[1, 0]]
        stat = beta / se
        pval = 2 * scipy.stats.t.sf(abs(stat[0]), df)
    else:
        coefficients, errors, test_df, crit = meta_regression_with_predictor(x, vi, extra, method, test, level)
        beta, se = coefficients[[1, 0]], errors[[1, 0]]
        stat = beta / se
        pval = 2 * (scipy.stats.norm.sf(abs(stat[0])) if test == 'z' else scipy.stats.t.sf(abs(stat[0]), test_df))
    # ``beta`` is ordered (coefficient of the predictor, intercept) in both branches: index 0 is tested, index 1 is the limit estimate
    return {'zval': float(stat[0]), 'pval': float(pval), 'est': float(beta[1]), 'ci_lb': float(beta[1] - crit * se[1]), 'ci_ub': float(beta[1] + crit * se[1])}


def regtest_zval(
    x: FloatArray,
    vi: FloatArray,
    model: str = 'rma',
    predictor: str = 'sei',
    ni: FloatArray | None = None,
    method: str = 'REML',
    test: str = 'z',
    level: float = 95,
) -> float:
    """The test statistic of the coefficient of the predictor."""
    return regtest_quantities(x, vi, model, predictor, ni, method, test, level)['zval']


def regtest_pval(
    x: FloatArray,
    vi: FloatArray,
    model: str = 'rma',
    predictor: str = 'sei',
    ni: FloatArray | None = None,
    method: str = 'REML',
    test: str = 'z',
    level: float = 95,
) -> float:
    """The two-sided p-value of the coefficient of the predictor."""
    return regtest_quantities(x, vi, model, predictor, ni, method, test, level)['pval']


def regtest_est(
    x: FloatArray,
    vi: FloatArray,
    model: str = 'rma',
    predictor: str = 'sei',
    ni: FloatArray | None = None,
    method: str = 'REML',
    test: str = 'z',
    level: float = 95,
) -> float:
    """The limit estimate: the intercept, the effect at zero of the predictor."""
    return regtest_quantities(x, vi, model, predictor, ni, method, test, level)['est']


def regtest_ci_lb(
    x: FloatArray,
    vi: FloatArray,
    model: str = 'rma',
    predictor: str = 'sei',
    ni: FloatArray | None = None,
    method: str = 'REML',
    test: str = 'z',
    level: float = 95,
) -> float:
    """The lower bound of the interval of the limit estimate."""
    return regtest_quantities(x, vi, model, predictor, ni, method, test, level)['ci_lb']


def regtest_ci_ub(
    x: FloatArray,
    vi: FloatArray,
    model: str = 'rma',
    predictor: str = 'sei',
    ni: FloatArray | None = None,
    method: str = 'REML',
    test: str = 'z',
    level: float = 95,
) -> float:
    """The upper bound of the interval of the limit estimate."""
    return regtest_quantities(x, vi, model, predictor, ni, method, test, level)['ci_ub']


def begg_deviates(yi: FloatArray, vi: FloatArray) -> FloatArray:
    """Begg and Mazumdar's standardised effects ``y*_i = (y_i - ybar) / sqrt(v*_i)`` with the inverse-variance mean ``ybar = sum(w y) / sum(w)`` and
    ``v*_i = v_i - 1 / sum(w)`` (Begg and Mazumdar, 1994, Biometrics 50:1088-1101, p. 1089)."""
    w = 1 / vi
    pooled = float(numpy.sum(w * yi) / numpy.sum(w))
    return (yi - pooled) / numpy.sqrt(vi - 1 / float(numpy.sum(w)))


def tie_sizes(values: FloatArray) -> list[int]:
    """The sizes of the groups of equal values that have more than one member."""
    _, counts = numpy.unique(values, return_counts=True)
    return [int(c) for c in counts if c > 1]


def kendall_counts(a: FloatArray, b: FloatArray) -> tuple[int, int, list[int], list[int]]:
    """The concordant and discordant pairs of ``a`` and ``b`` and the sizes of the groups of tied values of each."""
    concordant = discordant = 0
    for i in range(len(a) - 1):
        product = numpy.sign(a[i + 1 :] - a[i]) * numpy.sign(b[i + 1 :] - b[i])
        concordant += int(numpy.sum(product > 0))
        discordant += int(numpy.sum(product < 0))
    return concordant, discordant, tie_sizes(a), tie_sizes(b)


def begg_tau(x: FloatArray, vi: FloatArray) -> float:
    """Kendall's rank correlation between ``y*`` and ``v`` (Begg and Mazumdar, 1994, p. 1089; Kendall, 1970, Rank Correlation Methods, ch. 3):
    ``tau = (C - D) / sqrt((N - Tx) (N - Ty))`` over the ``N = k (k - 1) / 2`` pairs, ``Tx`` and ``Ty`` the tied pairs (``tau-b``; it is
    ``(C - D) / N`` without ties)."""
    concordant, discordant, ties_a, ties_b = kendall_counts(begg_deviates(x, vi), vi)
    k = len(x)
    pairs = k * (k - 1) / 2
    return (concordant - discordant) / math.sqrt((pairs - sum(t * (t - 1) / 2 for t in ties_a)) * (pairs - sum(t * (t - 1) / 2 for t in ties_b)))


def kendall_normal_variance(k: int, ties_a: list[int], ties_b: list[int]) -> float:
    """The variance of ``S = C - D`` under independence, with the correction for ties (Kendall, 1970, ch. 4); ``k (k - 1) (2k + 5) / 18`` without ties
    (Begg and Mazumdar, 1994, equation 2.1)."""
    v0 = k * (k - 1) * (2 * k + 5)
    vt = sum(t * (t - 1) * (2 * t + 5) for t in ties_a)
    vu = sum(u * (u - 1) * (2 * u + 5) for u in ties_b)
    v1 = sum(t * (t - 1) for t in ties_a) * sum(u * (u - 1) for u in ties_b) / (2 * k * (k - 1))
    v2 = sum(t * (t - 1) * (t - 2) for t in ties_a) * sum(u * (u - 1) * (u - 2) for u in ties_b) / (9 * k * (k - 1) * (k - 2))
    return (v0 - vt - vu) / 18 + v1 + v2


def begg_z(x: FloatArray, vi: FloatArray) -> float:
    """Begg and Mazumdar's normal deviate ``Z = (C - D) / sqrt(k (k - 1) (2k + 5) / 18)`` (1994, p. 1089), the variance corrected for ties if any."""
    concordant, discordant, ties_a, ties_b = kendall_counts(begg_deviates(x, vi), vi)
    return (concordant - discordant) / math.sqrt(kendall_normal_variance(len(x), ties_a, ties_b))


def inversion_distribution(k: int) -> list[int]:
    """The number of the ``k!`` orderings of ``k`` items with each number ``d`` of discordant pairs (index ``d``): the coefficients of the product over
    ``j = 1 .. k`` of ``1 + z + ... + z^(j-1)``, the distribution of Kendall's statistic under independence (Kendall, 1970, ch. 5)."""
    counts = [1]
    for j in range(2, k + 1):
        grown = [0] * (len(counts) + j - 1)
        for d, c in enumerate(counts):
            for step in range(j):
                grown[d + step] += c
        counts = grown
    return counts


def begg_pval(x: FloatArray, vi: FloatArray, exact: bool = True) -> float:
    """The two-sided p-value of the rank test. The exact value (without ties) doubles the smaller tail of the distribution of the number of discordant
    pairs, capped at one (Kendall, 1970); otherwise ``2 P(|Z| >= z)`` of the normal deviate (Begg and Mazumdar, 1994)."""
    _, discordant, ties_a, ties_b = kendall_counts(begg_deviates(x, vi), vi)
    if exact and not ties_a and not ties_b:
        counts = inversion_distribution(len(x))
        total = sum(counts)
        lower = sum(counts[: discordant + 1]) / total
        upper = sum(counts[discordant:]) / total
        return min(1.0, 2 * min(lower, upper))
    return float(2 * scipy.stats.norm.sf(abs(begg_z(x, vi))))


# Publication bias: the trim and fill method.


def trimfill_pooled(yi: FloatArray, vi: FloatArray) -> float:
    """The random-effects pooled estimate with the DerSimonian-Laird ``tau2`` (Duval and Tweedie, 2000, JASA 95:89-98, who use it in
    their examples)."""
    tau2 = rma_tau2(yi, vi, 'DL')
    weights = 1 / (vi + tau2)
    return float(numpy.sum(weights * yi) / numpy.sum(weights))


def trimfill_side(yi: FloatArray, vi: FloatArray) -> str:
    """The side of the funnel plot the studies are missing from: ``right`` when the slope of the random-effects meta-regression of the effects on their
    standard errors is negative (small studies report smaller effects, so the larger ones are the missing), else ``left`` (the side Duval and Tweedie,
    2000, take for their derivation; the rule that chooses between the two is the regression test of Egger et al., 1997, BMJ 315:629-634)."""
    beta, _, _, _ = meta_regression_with_predictor(yi, vi, numpy.sqrt(vi), 'DL', 'z', 95)
    return 'right' if beta[1] < 0 else 'left'


def trimfill_variance_of_t(n: int, k0: float) -> float:
    """``var[T_n]``, the variance of the sum of the ranks of the positive deviations of the ``n`` observed studies when ``k0`` are missing (Duval and
    Tweedie, 2000, proof of Theorem 3): given the number ``g`` of the positive ranks the trimmed studies mirror, ``T_n = W + g (2n - g + 1) / 2`` with
    ``W`` Wilcoxon-distributed over ``n - g`` ranks (mean ``m (m + 1) / 4``, variance ``m (m + 1) (2m + 1) / 24``), and ``g`` is negative binomial with
    mean ``k0`` and variance ``2 k0``. The total variance is the mean of the Wilcoxon variance plus the variance of the conditional mean, both taken
    over ``g`` from its raw moments, so ``k0`` need not be an integer."""
    moments = [float(scipy.stats.nbinom(k0, 0.5).moment(order)) for order in range(5)]
    moments[0] = 1.0

    def expectation(polynomial: numpy.polynomial.Polynomial) -> float:
        coefficients = numpy.asarray(polynomial.coef, dtype=numpy.float64)
        return float(numpy.dot(coefficients, moments[: len(coefficients)]))

    g = numpy.polynomial.Polynomial([0, 1])
    m = n - g
    wilcoxon_variance = m * (m + 1) * (2 * m + 1) / 24
    conditional_mean = m * (m + 1) / 4 + g * (2 * n - g + 1) / 2
    centred = conditional_mean - expectation(conditional_mean)
    return expectation(wilcoxon_variance) + expectation(centred * centred)


def trimfill_estimate(yi: FloatArray, vi: FloatArray, estimator: str, side: str | None) -> dict[str, typing.Any]:
    """Duval and Tweedie's trim and fill (2000, Biometrics 56:455-463, JASA 95:89-98; Duval, 2005, for ``Q0``), iterated to a fixed point of the
    rounded ``k0``: the effects are sorted so the missing studies are on the left, the ``k0`` largest are trimmed, the rest are pooled, the ranks of the
    absolute deviations from that estimate give ``R0`` (the length of the run of positive ranks at the right end less one, ``n - r - 1`` for the
    absolute rank ``r`` of the most negative deviation), ``L0 = (4 T - n (n + 1)) / (2n - 1)`` and ``Q0 = n - 1/2 - sqrt(2n^2 - 4T + 1/4)`` for the sum
    ``T`` of the positive ranks (equations 5 and 6). The standard errors are ``sqrt(2 R0 + 2)`` for ``R0`` (Theorem 1a) and, from ``var[T_n]``,
    ``4 sqrt(var) / (2n - 1)`` for ``L0`` and ``2 sqrt(var) / sqrt((n - 1/2)^2 - k0 (2n - k0 - 1))`` for ``Q0`` (equation 10). The filled studies mirror
    the trimmed ones about the final estimate (the imputed effect ``2 mu - y`` with the variance of the study it mirrors)."""
    chosen = trimfill_side(yi, vi) if side is None else side
    sign = -1.0 if chosen == 'right' else 1.0
    order = numpy.argsort(sign * yi, kind='stable')
    x, v = sign * yi[order], vi[order]
    n = len(x)
    k0, previous = 0, -1
    unrounded, se = 0.0, math.nan
    centre = 0.0
    while k0 != previous:
        previous = k0
        centre = trimfill_pooled(x[: n - k0], v[: n - k0])
        deviation = x - centre
        ranks = scipy.stats.rankdata(numpy.abs(deviation))
        if estimator == 'R0':
            negative = ranks[deviation < 0]
            unrounded = float(n - negative.max() - 1)
            se = math.sqrt(2 * max(0.0, unrounded) + 2)
        else:
            total = float(ranks[deviation > 0].sum())
            if estimator == 'L0':
                unrounded = (4 * total - n * (n + 1)) / (2 * n - 1)
            else:
                unrounded = n - 1 / 2 - math.sqrt(2 * n**2 - 4 * total + 1 / 4)
            variance = trimfill_variance_of_t(n, unrounded)
            if estimator == 'L0':
                se = 4 * math.sqrt(variance) / (2 * n - 1)
            else:
                se = 2 * math.sqrt(variance) / math.sqrt((n - 1 / 2) ** 2 - unrounded * (2 * n - unrounded - 1))
        k0 = max(0, round(unrounded))
    mirrored = sign * (2 * centre - x[n - k0 :]) if k0 > 0 else numpy.zeros(0)
    filled_v = v[n - k0 :] if k0 > 0 else numpy.zeros(0)
    pooled = trimfill_pooled(numpy.concatenate([yi, mirrored]), numpy.concatenate([vi, filled_v]))
    return {'k0': k0, 'se_k0': max(0.0, se), 'side': chosen, 'fill_yi': mirrored, 'fill_vi': filled_v, 'beta': pooled}


def trimfill_of(data: typing.Any, estimator: str, side: str | None) -> dict[str, typing.Any]:
    """``trimfill_estimate`` on the ``yi`` and ``vi`` columns of a frame."""
    return trimfill_estimate(data['yi'].to_numpy(dtype=numpy.float64), data['vi'].to_numpy(dtype=numpy.float64), estimator, side)


def trimfill_k0(data: typing.Any, estimator: str = 'L0', side: str | None = None) -> float:
    """The estimated number of missing studies of ``trimfill_estimate``."""
    return float(trimfill_of(data, estimator, side)['k0'])


def trimfill_se_k0(data: typing.Any, estimator: str = 'L0', side: str | None = None) -> float:
    """The standard error of the estimated number of missing studies of ``trimfill_estimate``."""
    return float(trimfill_of(data, estimator, side)['se_k0'])


def trimfill_beta(data: typing.Any, estimator: str = 'L0', side: str | None = None) -> float:
    """The pooled estimate over the observed and the filled studies of ``trimfill_estimate``."""
    return float(trimfill_of(data, estimator, side)['beta'])


def trimfill_p_k0(data: typing.Any, estimator: str = 'R0', side: str | None = None) -> float:
    """The p-value of the test of no missing studies with ``R0``: under ``k0 = 0`` the run ``gamma*`` of positive ranks is negative binomial with one
    tail (``P(gamma* = g) = 2^-(g+1)``; Duval and Tweedie, 2000, Theorem 2), ``R0 = gamma* - 1``, so the p-value is ``P(gamma* >= R0 + 1)``."""
    return float(scipy.stats.nbinom.sf(trimfill_of(data, estimator, side)['k0'], 1, 0.5))


# Publication bias: the fail-safe N.


def smallest_integer(satisfied: collections.abc.Callable[[int], bool]) -> int:
    """The smallest ``n >= 0`` with ``satisfied(n)``, for a predicate that stays true once it is true: the fail-safe N is the least number of added
    studies that reaches the criterion."""
    if satisfied(0):
        return 0
    upper = 1
    while not satisfied(upper):
        upper *= 2
    lower = upper // 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if satisfied(middle):
            upper = middle
        else:
            lower = middle
    return upper


def fsn_weights(x: FloatArray, vi: FloatArray) -> tuple[float, float, float]:
    """The sum of the inverse-variance weights, the weighted sum of the effects and the harmonic mean of the variances."""
    weights = 1 / vi
    return float(weights.sum()), float((weights * x).sum()), float(len(x) / weights.sum())


def fsn_stouffer_pval(x: FloatArray, vi: FloatArray, added: int) -> float:
    """The one-sided p-value of Stouffer's combination of the studies' normal deviates ``z = y / sqrt(v)`` with ``added`` studies of ``z = 0``
    (Stouffer et al., 1949; Rosenthal, 1979, doi:10.1037/0033-2909.86.3.638, p. 638): ``Z = sum(z) / sqrt(k + added)``."""
    z = x / numpy.sqrt(vi)
    return float(scipy.stats.norm.sf(abs(z.sum()) / math.sqrt(len(x) + added)))


def fsn_fisher_pval(x: FloatArray, vi: FloatArray, added: int) -> float:
    """The p-value of Fisher's combination of the studies' one-sided p-values with ``added`` studies of ``p = 0.5``: ``-2 sum(ln p)`` is chi-square
    with ``2 (k + added)`` degrees of freedom (Becker, 2005, Failsafe N or file-drawer number, in Rothstein et al., doi:10.1002/0470870168.ch7)."""
    p = scipy.stats.norm.sf(numpy.abs(x / numpy.sqrt(vi)))
    statistic = -2 * (numpy.log(p).sum() + added * math.log(0.5))
    return float(scipy.stats.chi2.sf(statistic, 2 * (len(x) + added)))


def fsn_equal_effects_pval(x: FloatArray, vi: FloatArray, added: int) -> float:
    """The two-sided p-value of the equal-effects estimate after adding ``added`` studies with an effect of 0 and the harmonic mean ``vbar`` of the
    variances (Rosenberg, 2005, doi:10.1111/j.0014-3820.2005.tb01004.x, equations 1 to 3; Viechtbauer, 2024, slide 16): the weights sum to
    ``sum(w) + added / vbar`` and the weighted sum of the effects is unchanged."""
    total, weighted_sum, vbar = fsn_weights(x, vi)
    return float(2 * scipy.stats.norm.sf(abs(weighted_sum) / math.sqrt(total + added / vbar)))


def fsn_orwin_target(x: FloatArray, vi: FloatArray, target: float | None) -> float:
    """The target effect of Orwin's fail-safe N (1983, doi:10.3102/10769986008002157): half the average effect if none is given, with the sign of the
    average effect; a target of the opposite sign is flipped."""
    total, weighted_sum, _ = fsn_weights(x, vi)
    average = weighted_sum / total
    chosen = average / 2 if target is None else target
    return -chosen if chosen != 0 and numpy.sign(chosen) != numpy.sign(average) else chosen


def fsn_general_fit(x: FloatArray, vi: FloatArray, method: str, added: int, mumiss: float) -> tuple[float, float, float]:
    """The estimate, ``tau2`` and p-value of a random-effects model of the studies and ``added`` file-drawer studies (Viechtbauer, 2024, slides 17 and
    18): the file-drawer effects have the mean ``mumiss`` and the variance ``vbar + tau2`` (the harmonic mean of the variances plus the ``tau2`` of the
    observed studies) and the sampling variance ``vbar``. They are standardised, so the model depends on them through their mean and variance
    alone."""
    tau2 = rma_tau2(x, vi, method)
    vbar = fsn_weights(x, vi)[2]
    if added > 1:
        base = numpy.arange(added, dtype=numpy.float64)
        spread = (base - base.mean()) / base.std(ddof=1)
    else:
        spread = numpy.zeros(added)
    values = numpy.concatenate([x, mumiss + spread * math.sqrt(vbar + tau2)])
    variances = numpy.concatenate([vi, numpy.full(added, vbar)])
    return float(rma_beta(values, variances, method)[0]), rma_tau2(values, variances, method), float(rma_pval(values, variances, method)[0])


def fsn_of(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> dict[str, float]:
    """The fail-safe N of ``type`` (and ``pool`` for Rosenthal's) with the quantities metafopy reports, from the criteria of the publications: the
    least number of added null studies for which the p-value is at least ``alpha`` (Rosenthal, Rosenberg and the general approach without a target) or
    the average effect is no further from zero than the target (Orwin and the general approach with a target)."""
    out = {'est': math.nan, 'tau2': math.nan, 'pval': math.nan, 'est_fsn': math.nan, 'tau2_fsn': math.nan, 'pval_fsn': math.nan}
    if type == 'Rosenthal':
        pval = fsn_stouffer_pval if pool == 'stouffer' else fsn_fisher_pval
        out['pval'] = pval(x, vi, 0)
        out['fsnum'] = smallest_integer(lambda n: pval(x, vi, n) >= alpha)
    elif type == 'Rosenberg':
        total, weighted_sum, _ = fsn_weights(x, vi)
        out['est'] = weighted_sum / total
        out['pval'] = fsn_equal_effects_pval(x, vi, 0)
        out['fsnum'] = smallest_integer(lambda n: fsn_equal_effects_pval(x, vi, n) >= alpha)
    elif type == 'Orwin':
        total, weighted_sum, vbar = fsn_weights(x, vi)
        out['est'] = weighted_sum / total
        goal = fsn_orwin_target(x, vi, target)
        out['fsnum'] = math.inf if goal == 0 else smallest_integer(lambda n: abs(weighted_sum / (total + n / vbar)) <= abs(goal))
    elif type == 'General':
        out['est'], out['tau2'], out['pval'] = fsn_general_fit(x, vi, method, 0, mumiss)
        if target is None:
            out['fsnum'] = smallest_integer(lambda n: fsn_general_fit(x, vi, method, n, mumiss)[2] >= alpha)
        else:
            goal = target if numpy.sign(target) == numpy.sign(out['est']) else -target
            out['fsnum'] = smallest_integer(lambda n: abs(fsn_general_fit(x, vi, method, n, mumiss)[0]) <= abs(goal))
        out['est_fsn'], out['tau2_fsn'], out['pval_fsn'] = fsn_general_fit(x, vi, method, int(out['fsnum']), mumiss)
    else:
        typing.assert_never(type)
    return out


def fsn_number(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The fail-safe N of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['fsnum']


def fsn_pval(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The observed p-value of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['pval']


def fsn_est(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The average effect of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['est']


def fsn_tau2(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The ``tau2`` of the observed studies of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['tau2']


def fsn_est_fsn(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The average effect with the file-drawer studies of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['est_fsn']


def fsn_tau2_fsn(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The ``tau2`` with the file-drawer studies of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['tau2_fsn']


def fsn_pval_fsn(
    x: FloatArray,
    vi: FloatArray,
    type: FsnType = 'Rosenthal',
    pool: str = 'stouffer',
    alpha: float = 0.05,
    target: float | None = None,
    method: str = 'REML',
    mumiss: float = 0.0,
) -> float:
    """The p-value with the file-drawer studies of ``fsn_of``."""
    return fsn_of(x, vi, type, pool, alpha, target, method, mumiss)['pval_fsn']


# Publication bias: the test of excess significance.


def tes_power(theta: float, vi: FloatArray, h0: float, alternative: str, alpha: float, tau2: float) -> FloatArray:
    """The power of each study's test at the true effect ``theta`` (Ioannidis and Trikalinos, 2007, doi:10.1177/1740774507079441, p. 246), averaged over
    true effects ``N(theta, tau2)``: the deviate ``(y - h0) / sqrt(v)`` is then normal with mean ``m = (theta - h0) / sqrt(v)`` and variance
    ``s2 = 1 + tau2 / v``, so the power is the normal probability of the rejection region."""
    mean = (theta - h0) / numpy.sqrt(vi)
    spread = numpy.sqrt(1 + tau2 / vi)
    if alternative == 'two.sided':
        crit = scipy.stats.norm.isf(alpha / 2)
        return numpy.asarray(scipy.stats.norm.sf((crit - mean) / spread) + scipy.stats.norm.cdf((-crit - mean) / spread), dtype=numpy.float64)
    if alternative == 'greater':
        return numpy.asarray(scipy.stats.norm.sf((scipy.stats.norm.isf(alpha) - mean) / spread), dtype=numpy.float64)
    return numpy.asarray(scipy.stats.norm.cdf((scipy.stats.norm.ppf(alpha) - mean) / spread), dtype=numpy.float64)


def poisson_binomial_pmf(p: FloatArray) -> FloatArray:
    """The distribution of the number of successes of independent trials with probabilities ``p``, from its characteristic function (Hong, 2013,
    Computational Statistics & Data Analysis 59:41-51, doi:10.1016/j.csda.2012.10.006): ``P(S = j) = (1 / (n + 1)) sum_l exp(-i w l j) prod_m (1 - p_m +
    p_m exp(i w l))`` with ``w = 2 pi / (n + 1)``."""
    n = len(p)
    frequency = 2j * math.pi * numpy.arange(n + 1) / (n + 1)
    characteristic = numpy.prod(1 - p[None, :] + p[None, :] * numpy.exp(frequency[:, None]), axis=1)
    pmf = numpy.real(numpy.exp(-numpy.outer(numpy.arange(n + 1), frequency)) @ characteristic) / (n + 1)
    return numpy.clip(pmf, 0.0, 1.0)


def tes_of(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
    theta: float | None = None,
    tau2: float | None = None,
    test: str = 'exact',
    tes_alternative: str = 'greater',
    tes_alpha: float = 0.1,
) -> dict[str, typing.Any]:
    """The test of excess significance (Ioannidis and Trikalinos, 2007, doi:10.1177/1740774507079441): the number ``O`` of significant studies, the
    expected number ``E`` (the sum of the powers at ``theta``, the inverse-variance weighted average when none is given), the chi-square statistic
    ``X2 = (O - E)^2 / E + (O - E)^2 / (k - E)`` and the p-value of ``test``: ``chi2`` (the normal deviate ``sign(O - E) sqrt(X2)`` for a one-sided
    alternative), ``binom`` (the binomial distribution with ``E / k``) or ``exact`` (the Poisson-binomial distribution of the powers; Francis, 2013,
    doi:10.1007/s11336-013-9331-1, for the two-sided rule of summing the outcomes no more probable than ``O``)."""
    k = len(x)
    heterogeneity = tau2 if tau2 is not None else 0.0
    weights = 1 / (vi + heterogeneity)
    centre = float((weights * x).sum() / weights.sum()) if theta is None else theta
    power = tes_power(centre, vi, H0, alternative, alpha, heterogeneity)
    z = (x - H0) / numpy.sqrt(vi)
    if alternative == 'two.sided':
        significant = numpy.abs(z) >= scipy.stats.norm.isf(alpha / 2)
    elif alternative == 'greater':
        significant = z >= scipy.stats.norm.isf(alpha)
    else:
        significant = z <= scipy.stats.norm.ppf(alpha)
    observed, expected = int(significant.sum()), float(power.sum())
    statistic = (observed - expected) ** 2 / expected + (observed - expected) ** 2 / (k - expected)
    deviate = math.copysign(math.sqrt(statistic), observed - expected)
    if test == 'chi2':
        pval = {'two.sided': scipy.stats.chi2.sf(statistic, 1), 'greater': scipy.stats.norm.sf(deviate), 'less': scipy.stats.norm.cdf(deviate)}[tes_alternative]
    else:
        pmf = scipy.stats.binom.pmf(numpy.arange(k + 1), k, expected / k) if test == 'binom' else poisson_binomial_pmf(power)
        slack = 1 + 1e-7
        pval = {
            'two.sided': pmf[pmf <= pmf[observed] * slack].sum(),
            'greater': pmf[observed:].sum(),
            'less': pmf[: observed + 1].sum(),
        }[tes_alternative]
    return {'O': observed, 'E': expected, 'X2': statistic, 'pval': min(1.0, float(pval)), 'theta': centre}


def tes_E(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
    theta: float | None = None,
    tau2: float | None = None,
) -> float:
    """The expected number of significant studies of ``tes_of``."""
    return float(tes_of(x, vi, H0, alternative, alpha, theta, tau2)['E'])


def tes_O(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
) -> float:
    """The observed number of significant studies of ``tes_of``."""
    return float(tes_of(x, vi, H0, alternative, alpha)['O'])


def tes_X2(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
    theta: float | None = None,
    tau2: float | None = None,
) -> float:
    """The chi-square statistic of ``tes_of``."""
    return float(tes_of(x, vi, H0, alternative, alpha, theta, tau2)['X2'])


def tes_pval(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
    theta: float | None = None,
    tau2: float | None = None,
    test: str = 'exact',
    tes_alternative: str = 'greater',
    tes_alpha: float = 0.1,
) -> float:
    """The p-value of the test of excess significance of ``tes_of``."""
    return float(tes_of(x, vi, H0, alternative, alpha, theta, tau2, test, tes_alternative, tes_alpha)['pval'])


def tes_theta_lim(
    x: FloatArray,
    vi: FloatArray,
    H0: float = 0.0,
    alternative: str = 'two.sided',
    alpha: float = 0.05,
    tau2: float | None = None,
    test: str = 'exact',
    tes_alternative: str = 'greater',
    tes_alpha: float = 0.1,
) -> float:
    """The limit estimate ``theta_lim`` of Ioannidis and Trikalinos (2007, p. 246) as metafor's ``tes`` help page (5.2-1) defines it: the true effect, on
    the side of the estimated effect, at which the p-value of the test of excess significance equals ``tes_alpha``, NaN when the p-value exceeds
    ``tes_alpha`` even at ``theta = H0``. Found by bracketing the root from ``H0`` outwards and bisecting it. One-sided ``greater`` test only."""
    if tes_alternative != 'greater':
        raise ValueError("tes_theta_lim is coded for tes_alternative = 'greater'")

    def gap(theta: float) -> float:
        return float(tes_of(x, vi, H0, alternative, alpha, theta, tau2, test, tes_alternative, tes_alpha)['pval']) - tes_alpha

    if gap(H0) >= 0:
        return math.nan
    estimate = float(tes_of(x, vi, H0, alternative, alpha, None, tau2, test, tes_alternative, tes_alpha)['theta'])
    outer = estimate
    while gap(outer) < 0:
        outer = H0 + 2 * (outer - H0)
    return float(scipy.optimize.brentq(gap, H0, outer, xtol=1e-13, rtol=1e-13))


# Publication bias: the Henmi and Copas interval.


def hc_interval(yi: FloatArray, vi: FloatArray, level: float = 95) -> tuple[float, float, float, float]:
    """The pooled estimate, the DerSimonian-Laird ``tau2`` and the confidence limits of Henmi and Copas (2010, Statistics in Medicine 29:2969-2983,
    doi:10.1002/sim.4029): the fixed-effects estimate (equation 8) with the interval ``estimate +- u sqrt(Vhat)`` (equations 14 to 16), ``u`` solving
    ``integral_u^inf G(u, r) p(r) dr = alpha / 2`` (equation 21) where ``p`` is the ``N(0, 1 + tau2 W2)`` density of ``R``, ``G`` is the gamma
    approximation (equation 20) of ``P(Q <= f^-1(r / u) | R = r)`` with the conditional mean (18) and variance (19) of ``Q`` and ``tau2`` replaced by its
    estimate."""
    n = len(yi)
    w = 1 / vi
    w1 = float(w.sum())
    w2, w3, w4 = (float((w**power).sum()) / w1 for power in (2, 3, 4))
    estimate = float((w * yi).sum()) / w1
    q = float((w * (yi - estimate) ** 2).sum())
    tau2 = max(0.0, (q - (n - 1)) / (w1 - w2))
    variance_r = 1 + tau2 * w2
    variance_hat = (tau2 * w2 + 1) / w1

    def conditional_mean(r: float) -> float:
        return (n - 1) + tau2 * (w1 - w2) + tau2**2 * (r**2 / variance_r**2 - 1 / variance_r) * (w3 - w2**2)

    def conditional_variance(r: float) -> float:
        shared = r**2 / variance_r**2 - 1 / variance_r
        return (
            2 * (n - 1)
            + 4 * tau2 * (w1 - w2)
            + 2 * tau2**2 * (w1 * w2 - 2 * w3 + w2**2)
            + 4 * tau2**2 * shared * (w3 - w2**2)
            + 4 * tau2**3 * shared * (w4 - 2 * w2 * w3 + w2**3)
            + 2 * tau2**4 * (1 / variance_r**2 - 2 * r**2 / variance_r**3) * (w3 - w2**2) ** 2
        )

    def inverse_f(ratio: float) -> float:
        return (ratio**2 - 1) * (w1 - w2) / w2 + (n - 1)

    def tail(u: float) -> float:
        def integrand(r: float) -> float:
            mean, variance = conditional_mean(r), conditional_variance(r)
            return float(
                scipy.stats.gamma.cdf(inverse_f(r / u), a=mean**2 / variance, scale=variance / mean) * scipy.stats.norm.pdf(r, scale=math.sqrt(variance_r))
            )

        return float(scipy.integrate.quad(integrand, u, numpy.inf, epsabs=1e-13, epsrel=1e-13, limit=200)[0])

    alpha = 1 - level / 100
    u = float(scipy.optimize.brentq(lambda value: tail(value) - alpha / 2, 1e-6, 10 * math.sqrt(variance_r), xtol=1e-13, rtol=1e-13))
    half = u * math.sqrt(variance_hat)
    return estimate, tau2, estimate - half, estimate + half


def hc_beta(data: typing.Any) -> float:
    """The fixed-effects estimate of ``hc_interval`` on the ``yi`` and ``vi`` columns of a frame."""
    return hc_interval(data['yi'].to_numpy(dtype=numpy.float64), data['vi'].to_numpy(dtype=numpy.float64))[0]


def hc_tau2(data: typing.Any) -> float:
    """The DerSimonian-Laird ``tau2`` of ``hc_interval`` on the ``yi`` and ``vi`` columns of a frame."""
    return hc_interval(data['yi'].to_numpy(dtype=numpy.float64), data['vi'].to_numpy(dtype=numpy.float64))[1]


# Publication bias: the selection-model likelihood.

type Weight = collections.abc.Callable[[float, float], float]


def precision_measure(name: str | None, vi: FloatArray, ni: FloatArray | None = None) -> FloatArray:
    """The measure of precision a selection weight may depend on, rescaled by its maximum (``None`` for none): the standard error ``sei``, the
    variance ``vi``, ``1 / ni`` or ``1 / sqrt(ni)`` for the sample sizes ``ni``."""
    if name is None:
        return numpy.ones(len(vi))
    if name == 'sei':
        measure = numpy.sqrt(vi)
    elif name == 'vi':
        measure = numpy.asarray(vi, dtype=numpy.float64)
    elif ni is None:
        raise ValueError(f'precision {name!r} needs the sample sizes')
    elif name == 'ninv':
        measure = 1 / ni
    elif name == 'sqrtninv':
        measure = 1 / numpy.sqrt(ni)
    else:
        raise ValueError(f'unknown precision {name!r}')
    return numpy.asarray(measure / measure.max(), dtype=numpy.float64)


def step_weight(steps: FloatArray, delta: FloatArray) -> Weight:
    """The step weight function of Hedges (1992, Journal of Educational Statistics 17:279-296, doi:10.3102/10769986017004279) and Hedges and Vevea
    (2005, in Rothstein et al., Publication Bias in Meta-Analysis, ch. 9): the weight of a study with one-sided p-value ``p`` is ``delta[j]``, ``j``
    being the interval ``steps[j - 1] < p <= steps[j]`` that holds ``p``, divided by the study's precision measure."""

    def weight(p: float, preci: float) -> float:
        j = int(numpy.searchsorted(steps, p, side='left'))
        return float(delta[min(j, len(delta) - 1)] / preci)

    return weight


def one_sided_p(y: float, v: float, alternative: str) -> float:
    """The p-value of an estimate ``y`` with sampling variance ``v`` against 0: ``1 - Phi(z)`` for ``greater``, ``Phi(z)`` for ``less`` and
    ``2 (1 - Phi(|z|))`` for ``two.sided``, ``z = y / sqrt(v)``."""
    z = y / math.sqrt(v)
    if alternative == 'greater':
        return 0.5 * math.erfc(z / math.sqrt(2))
    if alternative == 'less':
        return 0.5 * math.erfc(-z / math.sqrt(2))
    return math.erfc(abs(z) / math.sqrt(2))


def selection_loglik(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    beta: FloatArray,
    tau2: float,
    weight: Weight,
    alternative: str = 'greater',
    preci: FloatArray | None = None,
    cuts: collections.abc.Sequence[float] = (),
) -> float:
    """The log-likelihood of a random-effects selection model (Iyengar and Greenhouse, 1988, Statistical Science 3:109-135, doi:10.1214/ss/1177013012;
    Vevea and Hedges, 1995, Psychometrika 60:419-435, doi:10.1007/BF02294384).

    An estimate is observed with probability ``w(p)``, ``p`` its p-value, so the density of an observed ``y_i`` is ``w(p_i) f(y_i) / A_i`` with
    ``f`` the ``N(x_i' beta, v_i + tau2)`` density and ``A_i = integral w(p(y)) f(y) dy`` (Vevea and Hedges 1995, equation 5, and the weighted density
    of Iyengar and Greenhouse 1988, section 4). ``A_i`` is integrated numerically over ``x_i' beta`` +- 14 standard deviations, split at the values of
    ``y`` where ``p`` equals one of the ``cuts``, so a step function is integrated exactly."""
    ones = numpy.ones(len(yi)) if preci is None else preci
    total = 0.0
    for i in range(len(yi)):
        mean = float(X[i] @ beta)
        sd = math.sqrt(float(vi[i]) + tau2)
        sampling_sd = math.sqrt(float(vi[i]))
        lower, upper = mean - 14 * sd, mean + 14 * sd
        points = []
        for cut in cuts:
            if alternative == 'greater':
                found = [sampling_sd * float(scipy.stats.norm.isf(cut))]
            elif alternative == 'less':
                found = [sampling_sd * float(scipy.stats.norm.ppf(cut))]
            else:
                found = [sampling_sd * float(scipy.stats.norm.isf(cut / 2)), -sampling_sd * float(scipy.stats.norm.isf(cut / 2))]
            points += [y for y in found if math.isfinite(y) and lower < y < upper]

        def integrand(y: float, i: int = i, mean: float = mean, sd: float = sd) -> float:
            return weight(one_sided_p(y, float(vi[i]), alternative), float(ones[i])) * math.exp(-0.5 * ((y - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

        a = float(scipy.integrate.quad(integrand, lower, upper, points=sorted(points) or None, epsabs=0, epsrel=1e-13, limit=400)[0])
        w_i = weight(one_sided_p(float(yi[i]), float(vi[i]), alternative), float(ones[i]))
        total += math.log(w_i) + float(scipy.stats.norm.logpdf(yi[i], mean, sd)) - math.log(a)
    return total


def no_selection(p: float, preci: float) -> float:
    """The weight function of a model without selection: every study is observed with weight 1."""
    return 1.0 if p >= 0 and preci > 0 else math.nan


PARAMETRIC_PVAL_MIN = {'beta': 1e-5, 'power': 1e-5}
PARAMETRIC_KINDS = ('beta', 'halfnorm', 'logistic', 'negexp', 'negexppow', 'power')


def parametric_weight(kind: str, delta: FloatArray) -> Weight:
    """The weight function of one parametric selection model, with the p-value cut off at ``PARAMETRIC_PVAL_MIN`` (1e-5 for ``beta`` and ``power``,
    where the function is infinite or zero at 0 or 1; metafor's ``pval.min``, ``?selmodel`` 5.2-1) and ``preci`` the study's precision measure (1 for
    none):

    - ``beta``: ``p ** (delta[0] - 1) * (1 - p) ** (delta[1] - 1)`` (Citkowicz and Vevea, 2017, Psychological Methods 22:28-41,
      doi:10.1037/met0000119, as the 2012 dissertation "A parsimonious weight function for modeling publication bias" and Pustejovsky's 2024 post
      state it);
    - ``halfnorm``: ``exp(-delta * preci * p ** 2)`` (half-normal; Preston, Ashby and Smyth, 2004, Journal of Evaluation in Clinical Practice
      10:313-322, doi:10.1111/j.1365-2753.2003.00457.x);
    - ``negexp``: ``exp(-delta * preci * p)`` (negative exponential, Preston et al. 2004);
    - ``logistic``: ``2 exp(-delta * preci * p) / (1 + exp(-delta * preci * p))`` (logistic, Preston et al. 2004);
    - ``power``: ``(1 - p) ** (delta * preci)`` (Preston et al. 2004);
    - ``negexppow``: ``exp(-delta[0] * preci * p ** (1 / delta[1]))`` (metafor's ``?selmodel`` help, 5.2-1).

    Only the ratio of the weight to its integral enters the likelihood, so a constant factor is immaterial.
    """
    pmin = PARAMETRIC_PVAL_MIN.get(kind, 0.0)

    def weight(p: float, preci: float) -> float:
        q = min(max(p, pmin), 1 - pmin)
        if kind == 'beta':
            return float(q ** (delta[0] - 1) * (1 - q) ** (delta[1] - 1))
        if kind == 'halfnorm':
            return math.exp(-float(delta[0]) * preci * q * q)
        if kind == 'negexp':
            return math.exp(-float(delta[0]) * preci * q)
        if kind == 'logistic':
            e = math.exp(-float(delta[0]) * preci * q)
            return 2 * e / (1 + e)
        if kind == 'power':
            return float((1 - q) ** (preci * float(delta[0])))
        if kind == 'negexppow':
            return math.exp(-float(delta[0]) * preci * q ** (1 / float(delta[1])))
        raise ValueError(f'unknown parametric selection model {kind!r}')

    return weight


def parametric_cuts(kind: str) -> list[float]:
    """The p-values where ``parametric_weight`` has a kink, so ``selection_loglik`` integrates it in pieces."""
    pmin = PARAMETRIC_PVAL_MIN.get(kind, 0.0)
    return [pmin, 1 - pmin] if pmin > 0 else []


def truncation_loglik(
    yi: FloatArray, vi: FloatArray, X: FloatArray, beta: FloatArray, tau2: float, delta: float, cut: float, alternative: str = 'greater'
) -> float:
    """The log-likelihood of a truncated selection model (metafor's ``?selmodel`` 5.2-1, types ``trunc`` and ``truncest``; after the truncation
    model of Hedges 1984, Journal of Educational Statistics 9:61-85, doi:10.3102/10769986009001061): ``w(y) = 1`` for ``y > cut`` (``y < cut`` for
    ``alternative = 'less'``) and ``delta`` otherwise, so the density of an observed ``y_i`` is ``w(y_i) f(y_i) / A_i`` with
    ``A_i = P(Y > cut) + delta * P(Y <= cut)`` and ``Y ~ N(x_i' beta, v_i + tau2)``."""
    total = 0.0
    for i in range(len(yi)):
        mean = float(X[i] @ beta)
        sd = math.sqrt(float(vi[i]) + tau2)
        below = float(scipy.stats.norm.cdf(cut, mean, sd))
        if alternative == 'greater':
            kept = yi[i] > cut
            a = (1 - below) + delta * below
        else:
            kept = yi[i] < cut
            a = below + delta * (1 - below)
        w_i = 1.0 if kept else delta
        total += math.log(w_i) + float(scipy.stats.norm.logpdf(yi[i], mean, sd)) - math.log(a)
    return total
