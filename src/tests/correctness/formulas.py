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
