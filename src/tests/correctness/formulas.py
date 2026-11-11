"""Independent re-derivations of the quantities metafopy computes, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as the value of an ``expect`` key. The function is called with the arguments of the case that it declares
(see ``cases.formula_value``).
"""

import collections.abc
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


def model_matrix(count: int, mods: str | None, data: object | None) -> FloatArray:
    """``X``: a column of ones, then the columns of the one-sided formula ``mods`` (``'~ ablat + year'``) read from the data frame ``data``."""
    columns = [numpy.ones(count)]
    if mods:
        for name in mods.replace('~', '').split('+'):
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


def rma_tau2(yi: FloatArray, vi: FloatArray, method: str, weights: FloatArray | None = None, mods: str | None = None, data: object | None = None) -> float:
    """The estimate of ``tau2`` of ``method``, written from the estimating equation or moment equation of its publication.

    ``DL``, ``HE`` (and ``CO``, ``VC``, which the metafor documentation lists as Hedges' variance component estimator), ``HS``, ``HSk``, ``SJ`` closed
    forms; ``DLIT``, ``SJIT``, ``ML``, ``REML``, ``EB``, ``PM`` (``MP``), ``PMM`` solved by root finding on their equations; ``GENQ`` and ``GENQM``
    with the given ``weights``; the equal-effects methods ``EE``, ``FE`` and ``CE`` fix ``tau2 = 0``.
    """
    k = len(yi)
    design = model_matrix(k, mods, data)
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
    if test in ('knha', 'adhoc'):
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


def rma_QE(yi: FloatArray, vi: FloatArray, mods: str | None = None, data: object | None = None) -> float:
    """Cochran's Q: the weighted residual sum of squares with the inverse-variance weights."""
    return random_effects_q(yi, vi, model_matrix(len(yi), mods, data), 0.0)


def rma_QM(yi: FloatArray, vi: FloatArray, method: str, test: str = 'z', mods: str | None = None, data: object | None = None) -> float:
    """The omnibus test statistic of the moderators (``chi2`` for ``z``, ``F`` for the t-based tests)."""
    beta, cov, _ = coefficient_covariance(yi, vi, method, test, mods, data)
    which = slice(1, None) if mods else slice(0, 1)
    b = beta[which]
    stat = float(b @ numpy.linalg.inv(cov[which, which]) @ b)
    return stat if test == 'z' else stat / len(b)


def typical_variance(yi: FloatArray, vi: FloatArray, mods: str | None, data: object | None) -> float:
    """``vt = (k - p) / tr(P(W(0)))``, the typical sampling variance."""
    design = model_matrix(len(yi), mods, data)
    return float(len(yi) - design.shape[1]) / float(numpy.trace(projection(1 / vi, design)))


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
