"""``derived`` cases: expected values from ``formulas.py``, coded from the publications' equations."""

import collections.abc
import dataclasses
import itertools
import math
import re
import typing

import numpy
import numpy.typing
import pandas
import pytest
import scipy.optimize
import scipy.stats

import correctness.cases
import correctness.formulas
import correctness.glmm_formulas
import metafopy

type FloatArray = numpy.typing.NDArray[numpy.float64]


@pytest.mark.parametrize('case', correctness.cases.case_params('derived'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_derived(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def numbers_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any) -> None:
    allowed = (case.tolerance or correctness.cases.DERIVED_TOLERANCE) * numpy.maximum(1.0, numpy.abs(expected))
    if not numpy.all(numpy.abs(numpy.asarray(actual) - numpy.asarray(expected)) <= allowed):
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r}')


def confint_bounds(case: correctness.cases.Case, result: typing.Any, kind: str, formula: str) -> None:
    table = metafopy.confint(result, type=kind, tol=1e-12).random
    assert table is not None
    expected = correctness.cases.formula_value(case, f'formulas.{formula}', correctness.cases.case_frame(case))
    numbers_agree(case, f'confint {kind}', [table['ci.lb'].iloc[0], table['ci.ub'].iloc[0]], expected)


def confint_qp(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Q-profile interval of tau2 (Viechtbauer, 2007)."""
    confint_bounds(case, result, 'qp', 'qprofile_bounds')


def confint_genq(case: correctness.cases.Case, result: typing.Any) -> None:
    """The generalised Q interval of tau2 (Jackson, 2013)."""
    confint_bounds(case, result, 'genq', 'general_q_bounds')


def confint_pl(case: correctness.cases.Case, result: typing.Any) -> None:
    """The profile-likelihood interval of tau2 (Hardy and Thompson, 1996)."""
    confint_bounds(case, result, 'pl', 'profile_likelihood_bounds')


def confint_ht(case: correctness.cases.Case, result: typing.Any) -> None:
    """The test-based interval of tau2 (Higgins and Thompson, 2002)."""
    confint_bounds(case, result, 'ht', 'higgins_thompson_bounds')


def confint_wald(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2."""
    confint_bounds(case, result, 'wald', 'wald_tau2_bounds')


def confint_wald_log(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2 on the log scale."""
    confint_bounds(case, result, 'wald.log', 'wald_log_bounds')


def confint_wald_sqrt(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2 on the scale of tau."""
    confint_bounds(case, result, 'wald.sqrt', 'wald_sqrt_bounds')


def predict_pi(case: correctness.cases.Case, result: typing.Any) -> None:
    """The prediction interval at each row of the model matrix (Riley, Higgins and Deeks, 2011)."""
    frame = correctness.cases.case_frame(case)
    predicted = metafopy.predict(result)
    for name, formula, actual in (('pi_lb', 'prediction_lb', predicted.pi_lb), ('pi_ub', 'prediction_ub', predicted.pi_ub)):
        numbers_agree(case, name, actual, correctness.cases.formula_value(case, f'formulas.{formula}', frame))


def kendall_z(case: correctness.cases.Case, result: typing.Any) -> None:
    """Begg and Mazumdar's normal deviate from ``tau`` and the number of studies k (1994, p. 1089): ``Z = 3 tau sqrt(k (k - 1) / (2 (2k + 5)))``.

    It is compared with the deviate the formulas compute from the data, and its two-sided normal p-value with ``pval`` of the result.
    """
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    k = int(frame[case.args['x']].notna().sum())
    z = 3 * result.tau * math.sqrt(k * (k - 1) / (2 * (2 * k + 5)))
    numbers_agree(case, 'Z', z, correctness.cases.formula_value(case, 'formulas.begg_z', frame))
    numbers_agree(case, 'normal p-value from Z', 2 * scipy.stats.norm.sf(abs(z)), result.pval)


def trimfill_filled(case: correctness.cases.Case, result: typing.Any) -> None:
    """The side chosen, and the filled effects and variances (compared order-free) of Duval and Tweedie's trim and fill (2000)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    expected = correctness.formulas.trimfill_of(frame, case.args.get('estimator', 'L0'), case.args.get('side'))
    if result.side != expected['side']:
        raise AssertionError(f'{case.id}: side: metafopy gives {result.side!r}, the reference gives {expected["side"]!r}')
    filled = numpy.asarray(result.fill, dtype=bool)
    pairs = sorted(zip(numpy.asarray(result.yi)[filled], numpy.asarray(result.vi)[filled], strict=True))
    reference = sorted(zip(expected['fill_yi'], expected['fill_vi'], strict=True))
    if len(pairs) != len(reference):
        raise AssertionError(f'{case.id}: filled studies: metafopy fills {len(pairs)}, the reference fills {len(reference)}')
    numbers_agree(case, 'filled effects', [p[0] for p in pairs], [r[0] for r in reference])
    numbers_agree(case, 'filled variances', [p[1] for p in pairs], [r[1] for r in reference])


def hc_interval(case: correctness.cases.Case, result: typing.Any) -> None:
    """The limits of Henmi and Copas's confidence interval (2010, equations 14 to 21), from their integral on the case's ``yi`` and ``vi``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    _, _, lower, upper = correctness.formulas.hc_interval(frame['yi'].to_numpy(dtype=numpy.float64), frame['vi'].to_numpy(dtype=numpy.float64))
    numbers_agree(case, 'ci_lb', result.ci_lb, lower)
    numbers_agree(case, 'ci_ub', result.ci_ub, upper)


SELMODEL_LL_TOLERANCE = 1e-8
SELMODEL_FLAT_LL_TOLERANCE = 1e-6  # a case whose likelihood is flat in delta (tolerance above 1e-3) stops the optimisers 1e-7 apart in ll
SELMODEL_DELTA_FLOOR = 1e-6
SELMODEL_DELTA_CEILING = 20.0


def truncest_stretch_maximum(
    case: correctness.cases.Case,
    result: typing.Any,
    negative: collections.abc.Callable[[FloatArray], float],
    start: FloatArray,
    bounds: list[tuple[float | None, float | None]],
    cut_index: int,
    maxiter: int,
) -> scipy.optimize.OptimizeResult:
    """The best bounded maximum of the ``truncest`` likelihood over the stretches between consecutive observed effects.

    The likelihood is a step function of the truncation point, with a jump down at each observed effect (a study is selected only when ``y > cut``,
    ``y < cut`` for ``alternative='less'``), so it is maximised within each stretch with the cut bounded by the stretch. The end of a stretch at an
    observed effect, where the likelihood has a supremum and no maximum, is replaced by the point ``1e-8 * max(1, |end|)`` inside it.

    Raises:
        AssertionError: the optimiser converges in no stretch.
    """
    lower, upper = float(result.spec.delta_min[1]), float(result.spec.delta_max[1])
    observed = sorted({float(y) for y in numpy.asarray(result.base.yi)})
    edges = [lower, *(y for y in observed if lower < y < upper), upper]
    greater = result.alternative == 'greater'
    best: scipy.optimize.OptimizeResult | None = None
    messages: list[str] = []
    for low, high in itertools.pairwise(edges):
        if not high > low:
            continue
        step = 1e-8 * max(1.0, abs(high if greater else low))
        cut_low = low if greater or low not in observed else low + step
        cut_high = high - step if greater and high in observed else high
        stretch = list(bounds)
        stretch[cut_index] = (cut_low, cut_high)
        begin = start.copy()
        begin[cut_index] = (cut_low + cut_high) / 2
        fit = scipy.optimize.minimize(negative, begin, method='SLSQP', bounds=stretch, options={'maxiter': maxiter, 'ftol': 1e-15})
        if not fit.success:
            messages.append(str(fit.message))
            continue
        if best is None or fit.fun < best.fun:
            best = fit
    if best is None:
        raise AssertionError(f'{case.id}: the optimiser did not converge in any stretch: {messages[:1]}')
    return best


def selmodel_maximum(case: correctness.cases.Case, result: typing.Any, maxiter: int = 1000) -> tuple[FloatArray, float, FloatArray, float, float]:
    """Maximise the case's selection-model log-likelihood (``formulas.selection_loglik`` for a step function or a parametric weight function,
    ``formulas.truncation_loglik`` for ``trunc`` and ``truncest``) from metafopy's start values: the coefficients, ``tau2``, the selection
    parameters, the log-likelihood, and the log-likelihood of the model without selection (every weight 1).

    Raises:
        AssertionError: the optimiser does not converge, naming the case and the optimiser's message.
    """
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    base = result.base
    yi, vi, X = (numpy.asarray(v, dtype=numpy.float64) for v in (base.yi, base.vi, base.X))
    ni = frame['ni'].to_numpy(dtype=numpy.float64) if 'ni' in frame.columns else None
    preci = correctness.formulas.precision_measure(case.args.get('prec'), vi, ni)
    steps = numpy.asarray(result.steps, dtype=numpy.float64)
    fixed = numpy.asarray(result.delta_fix, dtype=bool)
    given = numpy.asarray(result.delta, dtype=numpy.float64)
    free = ~fixed
    p = X.shape[1]
    start = numpy.concatenate([base.beta, [base.tau2 + 1e-3], numpy.asarray(result.spec.delta_init)[free]])
    alternative = result.alternative
    kind = result.spec.type
    truncated = kind in ('trunc', 'truncest')
    stepped = kind in ('stepfun', 'stepcon')
    cuts = steps[:-1].tolist() if stepped else correctness.formulas.parametric_cuts(kind)

    def parts(par: FloatArray) -> tuple[FloatArray, float, FloatArray]:
        delta = given.copy()
        delta[free] = par[p + 1 :]
        return par[:p], float(par[p]), delta

    def loglik(beta: FloatArray, tau2: float, delta: FloatArray) -> float:
        if truncated:
            cut = float(delta[1]) if kind == 'truncest' else float(steps[0])
            return correctness.formulas.truncation_loglik(yi, vi, X, beta, tau2, float(delta[0]), cut, alternative)
        weight = correctness.formulas.step_weight(steps, delta) if stepped else correctness.formulas.parametric_weight(kind, delta)
        return correctness.formulas.selection_loglik(yi, vi, X, beta, tau2, weight, alternative, preci, cuts=cuts)

    def negative(par: FloatArray) -> float:
        return -loglik(*parts(par))

    bounds: list[tuple[float | None, float | None]] = [(None, None) for _ in range(p)]
    bounds += [(0.0, None)]
    for index in numpy.flatnonzero(free):
        if kind == 'truncest' and index == 1:
            bounds.append((float(result.spec.delta_min[1]), float(result.spec.delta_max[1])))
        else:
            bounds.append((SELMODEL_DELTA_FLOOR, SELMODEL_DELTA_CEILING if kind in correctness.formulas.PARAMETRIC_KINDS else None))
    constraints: typing.Any = []
    if result.decreasing:
        constraints = [{'type': 'ineq', 'fun': lambda par, j=j: parts(par)[2][j - 1] - parts(par)[2][j]} for j in range(1, len(given))]
    if kind == 'truncest' and free[1]:
        fit = truncest_stretch_maximum(case, result, negative, start, bounds, int(free[:1].sum()) + p + 1, maxiter)
    else:
        fit = scipy.optimize.minimize(negative, start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-15})
        if not fit.success:
            raise AssertionError(f'{case.id}: the optimiser did not converge: {fit.message}')
    beta, tau2, delta = parts(fit.x)

    def negative_plain(par: FloatArray) -> float:
        if truncated:
            return -correctness.formulas.truncation_loglik(yi, vi, X, par[:p], float(par[p]), 1.0, 0.0, alternative)
        return -correctness.formulas.selection_loglik(yi, vi, X, par[:p], float(par[p]), correctness.formulas.no_selection, alternative, preci)

    plain_bounds: list[tuple[float | None, float | None]] = bounds[: p + 1]
    plain = scipy.optimize.minimize(negative_plain, start[: p + 1], method='SLSQP', bounds=plain_bounds, options={'maxiter': maxiter, 'ftol': 1e-15})
    if not plain.success:
        raise AssertionError(f'{case.id}: the optimiser did not converge for the model without selection: {plain.message}')
    return beta, tau2, delta, float(-fit.fun), float(-plain.fun)


def selmodel_fit(case: correctness.cases.Case, result: typing.Any) -> None:
    """The fit of a selection model (Vevea and Hedges, 1995; Hedges and Vevea, 2005; the weight functions of ``formulas``), by maximising its
    log-likelihood with ``scipy.optimize`` from the start values of metafopy: ``beta``, ``tau2`` and ``delta`` at the case tolerance, ``ll`` at 1e-8
    and the likelihood ratio test statistic ``LRT`` against the fit without selection."""
    beta, tau2, delta, loglik, plain = selmodel_maximum(case, result)
    numbers_agree(case, 'beta', result.beta, beta)
    numbers_agree(case, 'tau2', result.tau2, tau2)
    numbers_agree(case, 'delta', result.delta, delta)
    allowed = (SELMODEL_FLAT_LL_TOLERANCE if (case.tolerance or 0.0) > 1e-3 else SELMODEL_LL_TOLERANCE) * max(1.0, abs(loglik))
    if abs(result.ll - loglik) > allowed:
        raise AssertionError(f'{case.id}: ll: metafopy gives {result.ll!r}, the reference gives {loglik!r}')
    lrt = max(0.0, 2 * (loglik - plain))
    numbers_agree(case, 'LRT', result.LRT, lrt)


def test_selmodel_fit_names_the_case_when_the_optimiser_does_not_converge() -> None:
    case = next(param.values[0] for param in correctness.cases.case_params('derived') if param.values[0].id == 'selmodel.stepfun.bangertdrowns2004')
    result = correctness.cases.run_case(case)
    with pytest.raises(AssertionError, match=r'selmodel\.stepfun\.bangertdrowns2004: the optimiser did not converge: .+'):
        selmodel_maximum(case, result, maxiter=1)


MV_TERM = re.compile(r'~\s*([\w.]+(?:\s*\+\s*[\w.]+)*)\s*\|\s*([\w.]+)(?:\s*/\s*([\w.]+))?')
MV_COMPONENT_TOLERANCE = 1e-6
MV_LOGLIK_EXCESS = 1e-8
MV_TIGHT_TOLERANCE = 1e-10
MV_FIXED_METHODS = ('CE', 'EE', 'FE')


def design_matrix(frame: pandas.DataFrame, mods: str | None) -> numpy.typing.NDArray[numpy.float64]:
    """The model matrix of ``mods``: a sum of terms, each ``0``, ``1``, a numeric column or a text column (dummy columns, the first level dropped
    unless the intercept is removed)."""
    names = ['1'] if mods is None else [part.strip() for part in mods.removeprefix('~').split('+')]
    intercept = '0' not in names
    columns: list[numpy.typing.NDArray[numpy.float64]] = [numpy.ones(len(frame))] if intercept else []
    for name in names:
        if name.isdecimal():
            continue
        column = frame[name]
        if pandas.api.types.is_numeric_dtype(column):
            columns.append(column.to_numpy(dtype=numpy.float64))
            continue
        dummies = pandas.get_dummies(column).sort_index(axis=1)
        dummies = dummies if not intercept else dummies.iloc[:, 1:]
        columns.extend(dummies.to_numpy(dtype=numpy.float64).T)
    return numpy.column_stack(columns)


def fixed_values(args: dict[str, typing.Any], name: str, length: int) -> numpy.typing.NDArray[numpy.float64] | None:
    """The values held fixed through the ``name`` argument (``nan`` where estimated), a single value repeated to ``length``."""
    if name not in args:
        return None
    values = numpy.atleast_1d(numpy.asarray(args[name], dtype=numpy.float64))
    return numpy.full(length, values[0]) if len(values) == 1 else values


def mv_terms(case: correctness.cases.Case, frame: pandas.DataFrame) -> list[correctness.formulas.RandomTerm]:
    """The random-effects terms of the case's ``random`` and ``struct``: ``~ 1 | a``, ``~ 1 | a/b`` (two ``sigma2`` terms) and ``~ inner | outer``.

    The inner part is one variable (the factor structures; ``CAR`` takes its values as time points), a sum of coordinates (the spatial structures,
    Euclidean distances) or the variables of the random coefficients (``GEN`` and ``GDIAG``, with an intercept). The ``sigma2``, ``tau2``, ``rho``,
    ``gamma2`` and ``phi`` arguments give fixed values, and ``setup.tree_r`` the known matrix of its factor, rescaled by ``Rscale``.
    """
    args = case.args
    formulas = [] if args.get('random') is None else [args['random']] if isinstance(args['random'], str) else args['random']
    structs = [] if args.get('struct') is None else [args['struct']] if isinstance(args['struct'], str) else list(args['struct'])
    names = [('sigma2', None), ('tau2', 'rho'), ('gamma2', 'phi')]
    sigma2 = iter(numpy.atleast_1d(numpy.asarray(args['sigma2'], dtype=numpy.float64)).tolist() if 'sigma2' in args else [])
    terms: list[correctness.formulas.RandomTerm] = []

    def sigma2_term(codes: numpy.typing.NDArray[numpy.intp], known: numpy.typing.NDArray[numpy.float64] | None) -> correctness.formulas.RandomTerm:
        fixed = next(sigma2, None)
        return correctness.formulas.RandomTerm(
            correctness.formulas.SIGMA2,
            codes,
            numpy.zeros(len(frame), dtype=numpy.intp),
            1,
            R=known,
            fixed_variances=None if fixed is None else numpy.array([fixed]),
        )

    for text in formulas:
        match = MV_TERM.fullmatch(text.strip())
        if match is None:
            raise ValueError(f'unsupported random formula {text!r}')
        inner, outer, nested = match.groups()
        codes, labels = pandas.factorize(frame[outer], sort=True)
        codes = codes.astype(numpy.intp)
        if inner == '1':
            known = None
            if case.setup is not None and 'tree_r' in case.setup and case.setup['tree_r']['factor'] == outer:
                tree = correctness.cases.build_tree_r(case.setup['tree_r']['tree'])
                known = correctness.formulas.rescale_r(tree.loc[labels.astype(str), labels.astype(str)].to_numpy(), str(args.get('Rscale', 'cor')))
            terms.append(sigma2_term(codes, known))
            if nested is not None:
                pairs = frame[outer].astype(str) + '/' + frame[nested].astype(str)
                terms.append(sigma2_term(pandas.factorize(pairs, sort=True)[0].astype(numpy.intp), None))
            continue
        struct = structs[sum(t.struct != correctness.formulas.SIGMA2 for t in terms)]
        position = sum(t.struct != correctness.formulas.SIGMA2 for t in terms) + 1
        variances_name, correlations_name = names[position]
        columns = [name.strip() for name in inner.split('+')]
        zeros = numpy.zeros(len(frame), dtype=numpy.intp)
        if struct in correctness.formulas.SPATIAL_STRUCTS:
            points = frame[columns].to_numpy(dtype=numpy.float64)
            dist = numpy.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
            term = correctness.formulas.RandomTerm(struct, codes, zeros, 1, dist=dist)
        elif struct in correctness.formulas.COEFFICIENT_STRUCTS:
            design = design_matrix(frame, '~ ' + inner)
            term = correctness.formulas.RandomTerm(struct, codes, zeros, design.shape[1], design=design)
        else:
            inner_codes, inner_labels = pandas.factorize(frame[inner], sort=True)
            levels = numpy.asarray(inner_labels, dtype=numpy.float64) if struct == 'CAR' else None
            term = correctness.formulas.RandomTerm(struct, codes, inner_codes.astype(numpy.intp), len(inner_labels), levels=levels)
        fixed_v = fixed_values(args, variances_name, term.n_variances())
        fixed_c = fixed_values(args, correlations_name, term.n_correlations()) if correlations_name else None
        terms.append(dataclasses.replace(term, fixed_variances=fixed_v, fixed_correlations=fixed_c))
    return terms


def mv_components(
    result: typing.Any, terms: list[correctness.formulas.RandomTerm]
) -> tuple[list[numpy.typing.NDArray[numpy.float64]], list[numpy.typing.NDArray[numpy.float64]]]:
    """The variances and correlations of each term in metafopy's fit: ``sigma2`` in order, then ``tau2`` and ``rho`` of the ``G`` term, ``gamma2`` and
    ``phi`` of the ``H`` term."""
    variances: list[numpy.typing.NDArray[numpy.float64]] = []
    correlations: list[numpy.typing.NDArray[numpy.float64]] = []
    sigma2 = iter(numpy.atleast_1d(result.sigma2).tolist() if result.sigma2 is not None else [])
    structured = [(result.tau2, result.rho), (result.gamma2, result.phi)]
    position = 0
    for term in terms:
        if term.struct == correctness.formulas.SIGMA2:
            variances.append(numpy.array([next(sigma2)]))
            correlations.append(numpy.zeros(0))
            continue
        tau, rho = structured[position]
        position += 1
        variances.append(numpy.atleast_1d(tau).astype(numpy.float64)[: term.n_variances()])
        correlations.append(numpy.atleast_1d(rho).astype(numpy.float64)[: term.n_correlations()] if term.n_correlations() else numpy.zeros(0))
    return variances, correlations


def mv_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any, tolerance: float) -> None:
    allowed = tolerance * numpy.maximum(1.0, numpy.abs(expected))
    if not numpy.all(numpy.abs(numpy.asarray(actual) - numpy.asarray(expected)) <= allowed):
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r} (tolerance {tolerance:g})')


@dataclasses.dataclass(frozen=True)
class MVSetup:
    """The marginal model of a case rebuilt from its data: ``y``, ``V``, ``X``, the optional ``W``, the terms and the components of metafopy's fit."""

    frame: pandas.DataFrame
    yi: numpy.typing.NDArray[numpy.float64]
    V: numpy.typing.NDArray[numpy.float64]
    X: numpy.typing.NDArray[numpy.float64]
    W: numpy.typing.NDArray[numpy.float64] | None
    method: str
    scored: str
    terms: list[correctness.formulas.RandomTerm]
    variances: list[numpy.typing.NDArray[numpy.float64]]
    correlations: list[numpy.typing.NDArray[numpy.float64]]

    def extra(self) -> numpy.typing.NDArray[numpy.float64]:
        """The sum of the random-effects matrices of the terms at metafopy's components."""
        total = numpy.zeros_like(self.V)
        for term, var, cor in zip(self.terms, self.variances, self.correlations, strict=True):
            total = total + correctness.formulas.term_matrix(term, var, cor)
        return total


def mv_setup(case: correctness.cases.Case, result: typing.Any) -> MVSetup:
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi = frame[case.args['yi']].to_numpy(dtype=numpy.float64)
    if case.setup is not None and ('vblocks' in case.setup or 'vconst' in case.setup):
        V = correctness.cases.build_v(frame, case.setup, case.id)
    else:
        V = numpy.diag(frame[case.args['V']].to_numpy(dtype=numpy.float64))
    X = design_matrix(frame, case.args.get('mods'))
    W = numpy.diag(frame[case.args['W']].to_numpy(dtype=numpy.float64)) if 'W' in case.args else None
    method = case.args.get('method', 'REML')
    fixed = method in MV_FIXED_METHODS
    terms = [] if fixed else mv_terms(case, frame)
    variances, correlations = mv_components(result, terms)
    return MVSetup(frame, yi, V, X, W, method, 'ML' if fixed else method, terms, variances, correlations)


def mv_model(case: correctness.cases.Case, result: typing.Any) -> None:
    """The likelihood, the generalised least squares estimate and the optimum of an ``rma_mv`` fit (Harville 1977; Wolfinger 1993).

    The case's ``random``, ``struct``, ``mods``, ``method`` and data give ``y``, ``V``, ``X`` and the terms; the checks are the log-likelihood at
    metafopy's components (1e-10), ``beta`` and ``se`` by generalised least squares at them (1e-10), and the optimum of ``formulas.mv_fit``
    (components within the case's tolerance, 1e-6 by default, and a log-likelihood no more than 1e-8 above metafopy's). A user ``W`` changes only
    ``beta`` and ``se``: the components are those of the fit without it (the software's convention; its R parity fixture gives the same ``sigma2``
    with and without ``W``).
    """
    s = mv_setup(case, result)
    yi, V, X, W, terms, scored = s.yi, s.V, s.X, s.W, s.terms, s.scored
    variances, correlations = s.variances, s.correlations
    stats = result.fit_stats_reml if scored == 'REML' else result.fit_stats_ml
    extra = s.extra()
    mv_agree(case, 'log-likelihood', correctness.formulas.mv_loglik(yi, V, X, extra, scored, W), stats.ll, MV_TIGHT_TOLERANCE)
    b, vb = correctness.formulas.gls(yi, V + extra, X, W)
    mv_agree(case, 'beta', result.beta, b, MV_TIGHT_TOLERANCE)
    mv_agree(case, 'se', result.se, numpy.sqrt(numpy.diag(vb)), MV_TIGHT_TOLERANCE)
    fit = correctness.formulas.mv_fit(yi, V, X, terms, scored)
    structure = ', '.join(term.struct for term in terms) or s.method
    if not fit.success:
        raise AssertionError(f'{case.id}: mv_fit ({structure}) did not converge: gradient norm {fit.gradient_norm:g}; optimiser: {fit.message}')
    tolerance = case.tolerance or MV_COMPONENT_TOLERANCE
    no_values = [numpy.zeros(0)]
    mv_agree(case, 'variances at the optimum', numpy.concatenate(variances or no_values), numpy.concatenate(list(fit.variances) or no_values), tolerance)
    mv_agree(
        case, 'correlations at the optimum', numpy.concatenate(correlations or no_values), numpy.concatenate(list(fit.correlations) or no_values), tolerance
    )
    optimum = numpy.zeros_like(V)
    for term, var, cor in zip(terms, fit.variances, fit.correlations, strict=True):
        optimum = optimum + correctness.formulas.term_matrix(term, var, cor)
    reference = correctness.formulas.mv_loglik(yi, V, X, optimum, scored, W)
    if reference - stats.ll > MV_LOGLIK_EXCESS:
        raise AssertionError(
            f"{case.id}: the reference optimum {reference!r} is above metafopy's log-likelihood {stats.ll!r} by more than {MV_LOGLIK_EXCESS:g}"
        )


def mv_expected_tests(case: correctness.cases.Case, fit: typing.Any) -> tuple[MVSetup, correctness.formulas.MVTests, str]:
    """The tests of the coefficients of ``fit`` from ``formulas.mv_tests`` at metafopy's components, with the test used: ``z`` with ``dfs`` other than
    ``residual`` is a ``t`` test, as the software defines it. The degrees of freedom are ``k - p`` for ``residual`` and ``formulas.contain_ddf`` over
    the grouping of each random term for ``contain``."""
    s = mv_setup(case, fit)
    test = case.args.get('test', 'z')
    dfs = case.args.get('dfs', 'residual')
    if test == 'z' and dfs != 'residual':
        test = 't'
    k, p = s.X.shape
    if dfs == 'residual':
        ddf = numpy.full(p, float(k - p))
    else:
        groupings: list[numpy.typing.NDArray[numpy.intp]] = []
        for term in s.terms:
            groupings.append(term.outer)
            if term.struct != correctness.formulas.SIGMA2:
                pairs = pandas.Series(list(zip(term.inner.tolist(), term.outer.tolist(), strict=True)))
                groupings.append(pandas.factorize(pairs)[0].astype(numpy.intp))
        ddf = correctness.formulas.contain_ddf(s.X, groupings)
    tests = correctness.formulas.mv_tests(s.yi, s.V + s.extra(), s.X, test, ddf, list(fit.btt), float(case.args.get('level', 95)))
    return s, tests, test


def mv_tests(case: correctness.cases.Case, result: typing.Any) -> None:
    """The tests of the coefficients, ``QM`` and ``QE`` of an ``rma_mv`` fit (Wald, t, Knapp-Hartung and ad hoc; containment degrees of freedom;
    Cochran 1954), by ``formulas.mv_tests`` and ``formulas.mv_qe`` at metafopy's components."""
    s, tests, test = mv_expected_tests(case, result)
    for name, actual, expected in (
        ('se', result.se, tests.se),
        ('zval', result.zval, tests.zval),
        ('pval', result.pval, tests.pval),
        ('ci_lb', result.ci_lb, tests.ci_lb),
        ('ci_ub', result.ci_ub, tests.ci_ub),
        ('QM', result.QM, tests.QM),
        ('QMp', result.QMp, tests.QMp),
    ):
        mv_agree(case, name, actual, expected, MV_TIGHT_TOLERANCE)
    if test != 'z':
        mv_agree(case, 'ddf', result.ddf, tests.ddf, MV_TIGHT_TOLERANCE)
    qe, qep = correctness.formulas.mv_qe(s.yi, s.V, s.X)
    mv_agree(case, 'QE', result.QE, qe, MV_TIGHT_TOLERANCE)
    mv_agree(case, 'QEp', result.QEp, qep, MV_TIGHT_TOLERANCE)


def refit(case: correctness.cases.Case) -> typing.Any:
    """The ``rma_mv`` fit of a case that has a ``then`` follow-on call."""
    return correctness.cases.run_case(dataclasses.replace(case, then=None))


def mv_confint_fixed(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald-type intervals of the coefficients from ``confint(fit, fixed=True)``: the t distribution with the degrees of freedom of each
    coefficient unless the test is ``z``."""
    _, tests, _ = mv_expected_tests(case, refit(case))
    mv_agree(case, 'ci.lb', result.fixed['ci.lb'].to_numpy(), tests.ci_lb, MV_TIGHT_TOLERANCE)
    mv_agree(case, 'ci.ub', result.fixed['ci.ub'].to_numpy(), tests.ci_ub, MV_TIGHT_TOLERANCE)


def mv_confint_pl(case: correctness.cases.Case, result: typing.Any) -> None:
    """The profile-likelihood interval of the ``sigma2``, ``tau2`` or ``rho`` component the case's ``then`` names (Hardy and Thompson 1996)."""
    assert case.then is not None
    fit = refit(case)
    s = mv_setup(case, fit)
    (name,) = [n for n in ('sigma2', 'tau2', 'rho') if n in case.then.args]
    position = int(case.then.args[name])
    if name == 'sigma2':
        index, kind, where = [i for i, t in enumerate(s.terms) if t.struct == correctness.formulas.SIGMA2][position], 'variance', 0
    else:
        index = next(i for i, t in enumerate(s.terms) if t.struct != correctness.formulas.SIGMA2)
        kind, where = ('variance' if name == 'tau2' else 'correlation'), position
    bounds = correctness.formulas.mv_profile_bounds(s.yi, s.V, s.X, s.terms, s.scored, index, kind, where)
    assert result.random is not None
    tolerance = case.tolerance or MV_COMPONENT_TOLERANCE
    mv_agree(case, f'confint {name}', [result.random['ci.lb'].iloc[0], result.random['ci.ub'].iloc[0]], bounds, tolerance)


def mv_predict_pi(case: correctness.cases.Case, result: typing.Any) -> None:
    """The prediction interval at each row of the model matrix of an ``rma_mv`` fit with ``sigma2`` components only."""
    fit = refit(case)
    s, tests, test = mv_expected_tests(case, fit)
    if any(t.struct != correctness.formulas.SIGMA2 for t in s.terms):
        raise ValueError(f'{case.id}: mv_predict_pi covers sigma2 components only')
    total = float(sum(v.sum() for v in s.variances))
    lb, ub = correctness.formulas.mv_prediction_interval(s.yi, s.V + s.extra(), s.X, test, tests.ddf, s.X, total)
    mv_agree(case, 'pi_lb', result.pi_lb, lb, MV_TIGHT_TOLERANCE)
    mv_agree(case, 'pi_ub', result.pi_ub, ub, MV_TIGHT_TOLERANCE)


ROBUST_TOLERANCE = 1e-8


def robust_inputs(
    case: correctness.cases.Case,
) -> tuple[typing.Any, FloatArray, FloatArray, FloatArray, FloatArray, pandas.DataFrame]:
    """The fit of a ``robust`` case and the working model it implies: ``y``, ``X``, the weights ``W`` (the inverse of the marginal covariance, or the user's
    ``W``), the working covariance ``Theta`` (the marginal covariance: ``V`` plus the random-effects matrices, or ``vi + tau2`` on the diagonal) and the
    data."""
    fit = refit(case)
    if case.function == 'rma_mv':
        s = mv_setup(case, fit)
        theta = s.V + s.extra()
        return fit, s.yi, s.X, s.W if s.W is not None else numpy.linalg.inv(theta), theta, s.frame
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    theta = numpy.diag(frame[case.args['vi']].to_numpy(dtype=numpy.float64) + float(fit.tau2))
    return fit, frame[case.args['yi']].to_numpy(dtype=numpy.float64), design_matrix(frame, case.args.get('mods')), numpy.linalg.inv(theta), theta, frame


ROBUST_ADJUSTMENTS: dict[bool | str, str] = {False: 'CR0', True: 'Hedges', 'Stata': 'Stata1', 'Stata1': 'Stata1', 'Stata2': 'Stata2'}


def robust_sandwich(case: correctness.cases.Case, result: typing.Any) -> None:
    """The default ``robust`` fit (the cluster-robust variance, its tests, intervals and omnibus test) of an ``rma_uni`` or ``rma_mv`` fit, from
    ``formulas.cluster_robust_vcov`` and ``formulas.robust_tests`` (Sidik and Jonkman 2005; Hedges, Tipton and Johnson 2010; Cameron and Miller 2015).

    ``y``, ``X`` and the weights are rebuilt from the case at metafopy's variance components: ``1 / (vi + tau2)`` for ``rma_uni``, the inverse of
    ``V`` plus the random-effects matrices (or the user's ``W``) for ``rma_mv``.
    """
    assert case.then is not None
    fit, yi, X, W, _, frame = robust_inputs(case)
    beta = numpy.linalg.solve(X.T @ W @ X, X.T @ W @ yi)
    codes = pandas.factorize(frame[case.then.args['cluster']], sort=True)[0].astype(numpy.intp)
    adjust = ROBUST_ADJUSTMENTS[case.then.args.get('adjust', True)]
    vb = correctness.formulas.cluster_robust_vcov(X, W, yi - X @ beta, codes, adjust)
    n = int(codes.max()) + 1
    tests = correctness.formulas.robust_tests(beta, vb, n, list(fit.btt), float(case.args.get('level', 95)))
    for name, actual, expected in (
        ('vb', result.vb, vb),
        ('se', result.se, tests.se),
        ('zval', result.zval, tests.zval),
        ('pval', result.pval, tests.pval),
        ('ci_lb', result.ci_lb, tests.ci_lb),
        ('ci_ub', result.ci_ub, tests.ci_ub),
        ('QM', result.QM, tests.QM),
        ('QMp', result.QMp, tests.QMp),
        ('ddf', result.ddf, tests.ddf),
    ):
        mv_agree(case, name, actual, expected, MV_TIGHT_TOLERANCE)


def infinite_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any, tolerance: float) -> None:
    """``mv_agree`` where infinite values (the degrees of freedom of a z or chi-square test) must be infinite in both."""
    got, want = numpy.asarray(actual, dtype=numpy.float64), numpy.asarray(expected, dtype=numpy.float64)
    if not numpy.array_equal(numpy.isinf(got), numpy.isinf(want)):
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r}')
    finite = ~numpy.isinf(want)
    mv_agree(case, name, got[finite], want[finite], tolerance)


def robust_club(case: correctness.cases.Case, result: typing.Any) -> None:
    """The ``club_sandwich`` fit of ``robust``: the variance of ``vcov``, the degrees of freedom, p-values and intervals of ``coef_test`` and
    ``conf_test`` and the omnibus ``wald_test`` of Tipton and Pustejovsky (2015) and Pustejovsky and Tipton (2018), from ``formulas``. The working model
    is the fit's own (``Theta`` the marginal covariance, ``W`` its inverse)."""
    assert case.then is not None
    args = case.then.args
    vcov = args.get('vcov', 'CR2')
    coef_test = args.get('coef_test', 'Satterthwaite')
    conf_test = args.get('conf_test', 'Satterthwaite' if coef_test == 'saddlepoint' else coef_test)
    wald_test = args.get('wald_test', 'HTZ')
    fit, yi, X, W, theta, frame = robust_inputs(case)
    beta = numpy.linalg.solve(X.T @ W @ X, X.T @ W @ yi)
    codes = pandas.factorize(frame[args['cluster']], sort=True)[0].astype(numpy.intp)
    vb = correctness.formulas.cluster_robust_vcov(X, W, yi - X @ beta, codes, vcov, theta)
    m, p = int(codes.max()) + 1, X.shape[1]
    se = numpy.sqrt(numpy.diag(vb))
    zval = beta / se
    satterthwaite = correctness.formulas.satterthwaite_dfs(X, W, theta, codes, vcov)
    named = {'z': numpy.full(p, numpy.inf), 'naive-t': numpy.full(p, m - 1.0), 'naive-tp': numpy.full(p, m - p * 1.0), 'Satterthwaite': satterthwaite}
    if coef_test == 'saddlepoint':
        pval = correctness.formulas.saddlepoint_pvalue(X, W, theta, codes, vcov, beta, vb)
        coef_df = numpy.full(p, numpy.nan)
    else:
        coef_df = named[coef_test]
        pval = numpy.asarray(2 * scipy.stats.t.sf(numpy.abs(zval), coef_df), dtype=numpy.float64)
    half = (1 - float(case.args.get('level', 95)) / 100) / 2
    crit = scipy.stats.t.isf(half, named[conf_test])
    btt = list(fit.btt)
    wald = correctness.formulas.hotelling_wald(X, W, theta, codes, vcov, beta, vb, numpy.eye(p)[btt], wald_test)
    tolerance = case.tolerance or ROBUST_TOLERANCE
    for name, actual, expected in (
        ('vb', result.vb, vb),
        ('se', result.se, se),
        ('zval', result.zval, zval),
        ('ci_lb', result.ci_lb, beta - crit * se),
        ('ci_ub', result.ci_ub, beta + crit * se),
        ('QM', result.QM, wald.QM),
        ('QMdf', result.QMdf[1], wald.df_denom),
    ):
        infinite_agree(case, name, actual, expected, ROBUST_TOLERANCE)
    if coef_test == 'saddlepoint':
        assert numpy.isnan(result.robust.dfs).all(), case.id
    else:
        infinite_agree(case, 'dfs', result.robust.dfs, coef_df, ROBUST_TOLERANCE)
    mv_agree(case, 'pval', result.pval, pval, tolerance)
    mv_agree(case, 'QMp', result.QMp, wald.pval, ROBUST_TOLERANCE)


GLMM_LIKELIHOOD_RTOL = 1e-8
GLMM_GRADIENT_TOLERANCE = 1e-4
GLMMADAPTIVE_LIKELIHOOD_RTOL = 1e-5
GLMM_CONDITIONAL_DEVIANCE_FITS = frozenset({('CM.AL', 'ML'), ('UM.FS', 'ML'), ('UM.RS', 'FE'), ('UM.RS', 'ML')})


def glmm_likelihood_at_estimate(case: correctness.cases.Case, result: typing.Any) -> None:
    """The formulas' log-likelihood at the reported estimates equals the formulas' maximum and, where ``rma_glmm`` reports the marginal
    log-likelihood, ``result.fit_stats_ml.ll``; the score there is small.

    ``lme4`` fits (the mixed models other than ``CM.EL``) report metafor's saturated model log-likelihood minus half the conditional deviance, not the
    marginal log-likelihood, so ``ll`` is compared only where it is the marginal one (``CM.EL``, ``GLMMadaptive`` and ``glmmTMB``);
    the other two checks hold for every fit.
    """
    frame = correctness.cases.case_frame(case)
    problem, rule, points = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_problem', frame)
    reference = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_fit', frame)
    method = case.args.get('method', 'ML')
    sigma2 = float(result.sigma2) if problem.model == 'UM.RS' else math.nan
    rho = float(result.rho) if problem.cor and problem.model == 'UM.RS' else math.nan
    loglik, score = correctness.glmm_formulas.profile_check(problem, method, result.beta, float(result.tau2), sigma2, rho, rule, points)
    where = f"{case.id}: the formulas' log-likelihood at the estimates of metafopy"
    package = (case.args.get('control') or {}).get('package', 'lme4')
    # GLMMadaptive stops its EM and quasi-Newton iterations at relative tolerances of 1e-5 to 1e-8 (R's own stopping rule, which metafopy follows),
    # so its estimates sit near, not at, the maximum: the log-likelihood there is compared at 1e-5 and the score is not required to vanish.
    rtol = GLMMADAPTIVE_LIKELIHOOD_RTOL if package == 'GLMMadaptive' else GLMM_LIKELIHOOD_RTOL
    if not loglik <= reference.loglik + rtol * abs(reference.loglik):
        raise AssertionError(f"{where} is {loglik!r}, above the formulas' maximum {reference.loglik!r}")
    if not reference.loglik - loglik <= rtol * abs(reference.loglik):
        raise AssertionError(f"{where} is {loglik!r}, not the formulas' maximum {reference.loglik!r} (rtol {rtol:g})")
    if package != 'GLMMadaptive' and not score <= GLMM_GRADIENT_TOLERANCE:
        raise AssertionError(f'{where} has a score of {score:.3g}, not below {GLMM_GRADIENT_TOLERANCE:g}')
    lme4 = (case.args.get('control') or {}).get('package', 'lme4') == 'lme4'
    conditional_deviance = lme4 and ((problem.model, method) in GLMM_CONDITIONAL_DEVIANCE_FITS or problem.kind == 'one_group' or problem.family == 'poisson')
    if not conditional_deviance:
        reported = float(result.fit_stats_ml.ll)
        if not abs(reported - loglik) <= GLMM_LIKELIHOOD_RTOL * abs(loglik):
            raise AssertionError(f'{case.id}: metafopy reports ll {reported!r}, the formulas give {loglik!r} at its estimates')


def glmm_score_at_estimate(case: correctness.cases.Case, result: typing.Any) -> None:
    """The score of the formulas' log-likelihood at the reported estimates is small, for a fit whose own maximum the formulas cannot find."""
    frame = correctness.cases.case_frame(case)
    problem, rule, points = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_problem', frame)
    sigma2 = float(result.sigma2) if problem.model == 'UM.RS' else math.nan
    rho = float(result.rho) if problem.cor and problem.model == 'UM.RS' else math.nan
    _, score = correctness.glmm_formulas.profile_check(problem, case.args.get('method', 'ML'), result.beta, float(result.tau2), sigma2, rho, rule, points)
    if not score <= GLMM_GRADIENT_TOLERANCE:
        raise AssertionError(f'{case.id}: the formulas have a score of {score:.3g} at the estimates of metafopy, not below {GLMM_GRADIENT_TOLERANCE:g}')


def irr_cm_al_is_cm_el(case: correctness.cases.Case, result: typing.Any) -> None:
    """``IRR`` with ``CM.AL`` is fitted as ``CM.EL`` (the conditional model of two Poisson counts is binomial for both): the same coefficients,
    standard errors, ``tau2`` and log-likelihood as the call with ``model='CM.EL'``."""
    frame = correctness.cases.case_frame(case)
    exact = metafopy.rma_glmm(**{**case.args, 'data': frame, 'model': 'CM.EL'})
    # auditme: accept[literal-duplication] the attributes of a fit that two calls must share; a test module names its own
    for name in ('beta', 'se', 'tau2'):
        if not numpy.array_equal(getattr(result, name), getattr(exact, name)):
            raise AssertionError(f'{case.id}: {name} of CM.AL is {getattr(result, name)!r}, of CM.EL {getattr(exact, name)!r}')
    if result.fit_stats_ml.ll != exact.fit_stats_ml.ll:
        raise AssertionError(f'{case.id}: ll of CM.AL is {result.fit_stats_ml.ll!r}, of CM.EL {exact.fit_stats_ml.ll!r}')


LS_RELATIVE = 1e-6
LS_LOGLIK = 1e-9


def ls_inputs(case: correctness.cases.Case) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``yi``, ``vi`` and the location and scale model matrices of a location-scale case, built from the case's data and its ``mods`` and ``scale``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = frame[case.args['yi']].to_numpy(dtype=numpy.float64), frame[case.args['vi']].to_numpy(dtype=numpy.float64)
    return yi, vi, design_matrix(frame, case.args.get('mods')), design_matrix(frame, case.args.get('scale'))


def ls_reference(case: correctness.cases.Case) -> dict[str, typing.Any]:
    """``formulas.ls_reference_fit`` for the case's data, ``method`` and ``link``; a failure to converge names the case and the step."""
    yi, vi, X, Z = ls_inputs(case)
    try:
        return correctness.formulas.ls_reference_fit(yi, vi, X, Z, case.args.get('link', 'log'), case.args['method'])
    except RuntimeError as err:
        raise AssertionError(f'{case.id}: the reference fit did not converge (the optimiser messages are in the chained error)') from err


def ls_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any, tolerance: float) -> None:
    """``actual`` equals ``expected`` to ``tolerance`` of the largest absolute value in ``expected``."""
    actual_array, expected_array = numpy.asarray(actual, dtype=numpy.float64), numpy.asarray(expected, dtype=numpy.float64)
    allowed = tolerance * float(numpy.max(numpy.abs(expected_array)))
    if not numpy.all(numpy.abs(actual_array - expected_array) <= allowed):
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r} (tolerance {tolerance:g})')


def ls_fit(case: correctness.cases.Case, result: typing.Any) -> None:
    """The coefficients, their covariance, ``tau2_i`` and the log-likelihood against the independent maximisation of equations 7 and 9."""
    reference = ls_reference(case)
    ls_agree(case, 'alpha', result.alpha, reference['alpha'], LS_RELATIVE)
    ls_agree(case, 'beta', result.beta, reference['beta'], LS_RELATIVE)
    ls_agree(case, 'vb', result.vb, reference['vb'], LS_RELATIVE)
    ls_agree(case, 'tau2', result.tau2, reference['tau2'], LS_RELATIVE)
    ls_agree(case, 'logLik', float(result.logLik(REML=case.args['method'] == 'REML')), reference['ll'], LS_LOGLIK)


def ls_optbeta(case: correctness.cases.Case, result: typing.Any) -> None:
    """With ``optbeta=True`` the joint and the profile ML maximum coincide, so ``alpha``, ``beta``, ``tau2_i`` and the log-likelihood match the profile fit.

    ``vb`` is not compared: the joint fit takes it from the information matrix of ``beta`` and ``alpha`` together, not from equation 10, which is the
    covariance of ``beta`` with ``alpha`` held at its estimate.
    """
    reference = ls_reference(case)
    for name in ('alpha', 'beta', 'tau2'):
        ls_agree(case, name, getattr(result, name), reference[name], LS_RELATIVE)
    ls_agree(case, 'logLik', float(result.logLik(REML=False)), reference['ll'], LS_LOGLIK)


def ls_fitstats(case: correctness.cases.Case, result: typing.Any) -> None:
    """Log-likelihood, AIC, BIC and AICc against the formulas at the reference optimum; the deviance is not compared, as the paper defines none."""
    reference = ls_reference(case)
    yi, _, X, Z = ls_inputs(case)
    reml = case.args['method'] == 'REML'
    expected = correctness.formulas.ls_fitstats(reference['ll'], X.shape[1] + Z.shape[1], len(yi) - X.shape[1] if reml else len(yi))
    actual = result.fit_stats_reml if reml else result.fit_stats_ml
    for name, value in expected.items():
        ls_agree(case, name, getattr(actual, name), value, LS_LOGLIK)


def ls_heterogeneity(case: correctness.cases.Case, result: typing.Any) -> None:
    """``QE`` as Cochran's ``y' P y`` with the inverse-variance weights, and ``I2`` and ``H2`` per study at the reference ``tau2_i``."""
    reference = ls_reference(case)
    yi, vi, X, _ = ls_inputs(case)
    vt = correctness.formulas.design_typical_variance(vi, X)
    ls_agree(case, 'QE', result.QE, correctness.formulas.random_effects_q(yi, vi, X, 0.0), LS_RELATIVE)
    ls_agree(case, 'I2', result.I2, correctness.formulas.ls_i2(reference['tau2'], vt), LS_RELATIVE)
    ls_agree(case, 'H2', result.H2, correctness.formulas.ls_h2(reference['tau2'], vt), LS_RELATIVE)


LS_NUMERIC_HESSIAN = 1e-4  # metafopy's covariance of alpha is a numerical Hessian (Richardson, r = 4), good to about 1e-5 of the information
LS_PROFILE = 1e-5
LS_TDIST = ('adhoc', 'knha', 't')
LS_CONFIDENCE_FIELDS = ('ci_lb', 'ci_ub', 'pred', 'se')
LS_PREDICT_FIELDS = (*LS_CONFIDENCE_FIELDS, 'pi_lb', 'pi_ub')  # the prediction interval needs a predicted tau2


def ls_location(case: correctness.cases.Case, reference: dict[str, typing.Any]) -> tuple[FloatArray, FloatArray, float]:
    """``beta``, its covariance after the Knapp-Hartung factor of the case's ``test`` (section 2.5.4), and the factor, at the reference ``tau2_i``."""
    yi, vi, X, _ = ls_inputs(case)
    test = case.args.get('test', 'z')
    s2w = correctness.formulas.ls_knha_s2w(yi, vi, X, reference['tau2'], test == 'adhoc') if test in correctness.formulas.KNAPP_HARTUNG_TESTS else 1.0
    return reference['beta'], s2w * reference['vb'], s2w


def ls_test(case: correctness.cases.Case, result: typing.Any) -> None:
    """The location and scale tests of the case's ``test``: standard errors, statistics, p-values and omnibus tests against equations 12 and 14."""
    reference = ls_reference(case)
    _, _, X, Z = ls_inputs(case)
    beta, vb, s2w = ls_location(case, reference)
    k, p, q = X.shape[0], X.shape[1], Z.shape[1]
    tdist = case.args.get('test', 'z') in LS_TDIST
    ls_agree(case, 's2w', result.s2w, s2w, LS_RELATIVE)
    ls_agree(case, 'se', result.se, numpy.sqrt(numpy.diag(vb)), LS_RELATIVE)
    zval = beta / numpy.sqrt(numpy.diag(vb))
    pval = 2 * (scipy.stats.t.sf(numpy.abs(zval), k - p) if tdist else scipy.stats.norm.sf(numpy.abs(zval)))
    ls_agree(case, 'pval', result.pval, pval, LS_RELATIVE)
    qm, qmp = correctness.formulas.ls_wald(beta, vb, list(range(1, p)), k - p if tdist else None)
    ls_agree(case, 'QM', result.QM, qm, LS_RELATIVE)
    ls_agree(case, 'QMp', result.QMp, qmp, LS_RELATIVE)
    se_alpha = numpy.sqrt(numpy.diag(reference['va']))
    pval_alpha = 2 * (
        scipy.stats.t.sf(numpy.abs(reference['alpha'] / se_alpha), k - q) if tdist else scipy.stats.norm.sf(numpy.abs(reference['alpha'] / se_alpha))
    )
    ls_agree(case, 'pval_alpha', result.pval_alpha, pval_alpha, LS_NUMERIC_HESSIAN)
    qs, qsp = correctness.formulas.ls_wald(reference['alpha'], reference['va'], list(range(1, q)), k - q if tdist else None)
    ls_agree(case, 'QS', result.QS, qs, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'QSp', result.QSp, qsp, LS_NUMERIC_HESSIAN)


def ls_vcov_alpha(case: correctness.cases.Case, result: typing.Any) -> None:
    """``va`` against the inverse of the analytic observed information, not a numerical Hessian."""
    ls_agree(case, 'va', result.va, ls_reference(case)['va'], LS_NUMERIC_HESSIAN)


def ls_qs(case: correctness.cases.Case, result: typing.Any) -> None:
    """The omnibus Wald test of the scale coefficients other than the intercept (equation 14) at the reference estimate and information."""
    reference = ls_reference(case)
    q = reference['alpha'].shape[0]
    stat, pval = correctness.formulas.ls_wald(reference['alpha'], reference['va'], list(range(1, q)), None)
    ls_agree(case, 'QS', result.QS, stat, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'QSp', result.QSp, pval, LS_NUMERIC_HESSIAN)


def ls_confint_scale(case: correctness.cases.Case, result: typing.Any) -> None:
    """``confint(result, alpha=j, tol=1e-12)`` against the profile-likelihood bounds of ``formulas.ls_profile_bounds`` for each scale coefficient."""
    yi, vi, X, Z = ls_inputs(case)
    link, method = case.args.get('link', 'log'), case.args['method']
    for j in range(Z.shape[1]):
        interval = metafopy.confint(result, alpha=j, tol=1e-12)
        table = interval.random
        assert table is not None
        expected = correctness.formulas.ls_profile_bounds(yi, vi, X, Z, link, method, j, 0.95)
        # a bound shown with a '<' or '>' sign is not crossed within the interval searched, which the reference reports as nan
        actual = [math.nan if interval.lb_sign else table['ci.lb'].iloc[0], math.nan if interval.ub_sign else table['ci.ub'].iloc[0]]
        for name, bound, reference in zip(('lower', 'upper'), actual, expected, strict=True):
            if math.isnan(reference) != math.isnan(bound):
                raise AssertionError(f'{case.id}: confint alpha.{j + 1} {name} bound: metafopy gives {bound!r}, the reference gives {reference!r}')
        found = [i for i in range(2) if not math.isnan(expected[i])]
        if found:
            ls_agree(case, f'confint alpha.{j + 1}', [actual[i] for i in found], [expected[i] for i in found], LS_PROFILE)


def ls_confint_location(case: correctness.cases.Case, result: typing.Any) -> None:
    """``confint(result, fixed=True)`` against Wald intervals ``beta +- crit se``, with the t critical value for the t-type tests."""
    reference = ls_reference(case)
    yi, _, X, _ = ls_inputs(case)
    beta, vb, _ = ls_location(case, reference)
    se = numpy.sqrt(numpy.diag(vb))
    crit = float(scipy.stats.t.isf(0.025, len(yi) - X.shape[1]) if case.args.get('test', 'z') in LS_TDIST else scipy.stats.norm.isf(0.025))
    table = metafopy.confint(result, fixed=True).fixed
    assert table is not None
    ls_agree(case, 'ci.lb', table['ci.lb'].to_numpy(), beta - crit * se, LS_RELATIVE)
    ls_agree(case, 'ci.ub', table['ci.ub'].to_numpy(), beta + crit * se, LS_RELATIVE)


def ls_critical(case: correctness.cases.Case, ddf: int) -> float:
    """The two-sided 95% critical value of the case's ``test``: t on ``ddf`` degrees of freedom for the t-type tests, else normal."""
    return float(scipy.stats.t.isf(0.025, ddf) if case.args.get('test', 'z') in LS_TDIST else scipy.stats.norm.isf(0.025))


def ls_fit_dict(case: correctness.cases.Case) -> dict[str, typing.Any]:
    """The reference fit with the Knapp-Hartung factor of the case's ``test`` applied to ``vb``."""
    reference = ls_reference(case)
    beta, vb, _ = ls_location(case, reference)
    return {**reference, 'beta': beta, 'vb': vb}


def ls_new_values(X: FloatArray, Z: FloatArray) -> tuple[list[float], list[float]]:
    """New moderator and scale values spanning the data: the smallest, the mean and the largest value of the first non-intercept column of each part."""
    return [float(f(X[:, 1])) for f in (numpy.min, numpy.mean, numpy.max)], [float(f(Z[:, 1])) for f in (numpy.min, numpy.mean, numpy.max)]


def ls_new_rows(values: list[float]) -> FloatArray:
    """Model matrix rows (an intercept and one variable) for the new values."""
    return numpy.column_stack([numpy.ones(len(values)), numpy.asarray(values)])


def ls_predict_check(case: correctness.cases.Case, result: typing.Any, with_x: bool, with_z: bool) -> None:
    """``predict`` for new moderator values and/or new scale values against ``formulas.ls_predict`` row by row.

    Both parts are compared where given: ``pred``, ``se`` and the confidence interval for ``newmods``, the prediction interval when ``newscale`` is given
    too (the predicted ``tau2`` of its row), and for ``newscale`` alone ``pred`` and the interval on the scale of the link, then ``tau2`` and its bounds
    through ``transf=numpy.exp`` (the back-transform).
    """
    yi, _, X, Z = ls_inputs(case)
    link = case.args.get('link', 'log')
    fit = ls_fit_dict(case)
    k, p, q = len(yi), X.shape[1], Z.shape[1]
    new_x, new_z = ls_new_values(X, Z)
    x_rows, z_rows = ls_new_rows(new_x), ls_new_rows(new_z)
    crit_x, crit_z = ls_critical(case, k - p), ls_critical(case, k - q)
    expected = [correctness.formulas.ls_predict(x_rows[i], z_rows[i], fit, link, crit_x) for i in range(len(new_x))]
    scale_expected = [correctness.formulas.ls_predict(x_rows[i], z_rows[i], fit, link, crit_z) for i in range(len(new_x))]
    if with_x and with_z:
        predicted = metafopy.predict(result, newmods=new_x, newscale=new_z)
        for name in LS_PREDICT_FIELDS:
            ls_agree(case, name, getattr(predicted, name), [e[name] for e in expected], LS_RELATIVE)
    elif with_x:
        predicted = metafopy.predict(result, newmods=new_x)
        for name in LS_CONFIDENCE_FIELDS:
            ls_agree(case, name, getattr(predicted, name), [e[name] for e in expected], LS_RELATIVE)
    else:
        predicted = metafopy.predict(result, newscale=new_z)
        for name, key in (('pred', 'scale'), ('se', 'scale_se'), ('ci_lb', 'scale_lb'), ('ci_ub', 'scale_ub')):
            ls_agree(case, name, getattr(predicted, name), [e[key] for e in scale_expected], LS_NUMERIC_HESSIAN)
        if link == 'log':
            back = metafopy.predict(result, newscale=new_z, transf=numpy.exp)
            ls_agree(case, 'tau2', back.pred, [e['tau2'] for e in scale_expected], LS_NUMERIC_HESSIAN)
            ls_agree(case, 'tau2 ci_lb', back.ci_lb, numpy.exp([e['scale_lb'] for e in scale_expected]), LS_NUMERIC_HESSIAN)
            ls_agree(case, 'tau2 ci_ub', back.ci_ub, numpy.exp([e['scale_ub'] for e in scale_expected]), LS_NUMERIC_HESSIAN)


def ls_predict_newmods(case: correctness.cases.Case, result: typing.Any) -> None:
    """``predict(result, newmods=...)``: the average true effects with their standard errors and confidence intervals."""
    ls_predict_check(case, result, True, False)


def ls_predict_newscale(case: correctness.cases.Case, result: typing.Any) -> None:
    """``predict(result, newscale=...)``: the predicted scale on the scale of the link and ``tau2`` with its interval after ``transf=numpy.exp``."""
    ls_predict_check(case, result, False, True)


def ls_predict_both(case: correctness.cases.Case, result: typing.Any) -> None:
    """``predict(result, newmods=..., newscale=...)``: the confidence and the prediction intervals with the predicted ``tau2`` of each row."""
    ls_predict_check(case, result, True, True)


def ls_predict_fitted(case: correctness.cases.Case, result: typing.Any) -> None:
    """``predict(result)``: the fitted values of the studies, with the prediction interval that uses each study's ``tau2_i``."""
    yi, _, X, Z = ls_inputs(case)
    fit = ls_fit_dict(case)
    crit = ls_critical(case, len(yi) - X.shape[1])
    expected = [correctness.formulas.ls_predict(X[i], Z[i], fit, case.args.get('link', 'log'), crit) for i in range(len(yi))]
    predicted = metafopy.predict(result)
    for name in LS_PREDICT_FIELDS:
        ls_agree(case, name, getattr(predicted, name), [e[name] for e in expected], LS_RELATIVE)


def ls_wald_check(case: correctness.cases.Case, actual: typing.Any, estimate: FloatArray, vcov: FloatArray, idx: list[int], ddf: int) -> None:
    """The omnibus statistic and p-value of an ``anova`` result against equation 12 or 14 on the coefficients ``idx`` (F with ``ddf`` for the t-type tests)."""
    stat, pval = correctness.formulas.ls_wald(estimate, vcov, idx, ddf if case.args.get('test', 'z') in LS_TDIST else None)
    ls_agree(case, 'stat', actual.stat, stat, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'pval_omnibus', actual.pval_omnibus, pval, LS_NUMERIC_HESSIAN)


def ls_anova_btt(case: correctness.cases.Case, result: typing.Any) -> None:
    """``anova(result, btt=...)``: the omnibus Wald test of the location coefficients other than the intercept (equation 12)."""
    yi, _, X, _ = ls_inputs(case)
    fit = ls_fit_dict(case)
    idx = list(range(1, X.shape[1]))
    ls_wald_check(case, metafopy.anova(result, btt=idx), fit['beta'], fit['vb'], idx, len(yi) - X.shape[1])


def ls_anova_att(case: correctness.cases.Case, result: typing.Any) -> None:
    """``anova(result, att=...)``: the omnibus Wald test of the scale coefficients other than the intercept (equation 14)."""
    yi, _, _, Z = ls_inputs(case)
    fit = ls_fit_dict(case)
    idx = list(range(1, Z.shape[1]))
    ls_wald_check(case, metafopy.anova(result, att=idx), fit['alpha'], fit['va'], idx, len(yi) - Z.shape[1])


def ls_anova_combinations(case: correctness.cases.Case, actual: typing.Any, estimate: FloatArray, vcov: FloatArray, L: FloatArray, ddf: int) -> None:
    """The linear combinations ``L b`` of an ``anova`` result: estimates, standard errors, statistics, p-values and the omnibus test of all of them."""
    est = L @ estimate
    cov = L @ vcov @ L.T
    se = numpy.sqrt(numpy.diag(cov))
    tdist = case.args.get('test', 'z') in LS_TDIST
    pval = 2 * (scipy.stats.t.sf(numpy.abs(est / se), ddf) if tdist else scipy.stats.norm.sf(numpy.abs(est / se)))
    ls_agree(case, 'estimate', actual.estimate, est, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'se', actual.se, se, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'zval', actual.zval, est / se, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'pval', actual.pval, pval, LS_NUMERIC_HESSIAN)
    ls_wald_check(case, actual, est, cov, list(range(L.shape[0])), ddf)


def ls_anova_X(case: correctness.cases.Case, result: typing.Any) -> None:
    """``anova(result, X=...)``: the sum and the difference of the intercept and slope of the location part."""
    yi, _, X, _ = ls_inputs(case)
    fit = ls_fit_dict(case)
    L = numpy.array([[1.0, 1.0], [1.0, -1.0]])
    ls_anova_combinations(case, metafopy.anova(result, X=L), fit['beta'], fit['vb'], L, len(yi) - X.shape[1])


def ls_anova_Z(case: correctness.cases.Case, result: typing.Any) -> None:
    """``anova(result, Z=...)``: the sum and the difference of the intercept and slope of the scale part."""
    yi, _, _, Z = ls_inputs(case)
    fit = ls_fit_dict(case)
    L = numpy.array([[1.0, 1.0], [1.0, -1.0]])
    ls_anova_combinations(case, metafopy.anova(result, Z=L), fit['alpha'], fit['va'], L, len(yi) - Z.shape[1])


def ls_anova_lrt(case: correctness.cases.Case, result: typing.Any) -> None:
    """``anova(full, reduced)``: equation 15 against the reference log-likelihoods of both models.

    The reduced model has a constant scale (``scale='~1'``) and, under ML, a constant location too; under REML it keeps the location part, since REML
    likelihoods are comparable only for the same fixed effects (the paper's own condition).
    """
    yi, vi, X, Z = ls_inputs(case)
    method, link = case.args['method'], case.args.get('link', 'log')
    keep_location = method == 'REML'
    reduced_args = {**case.args, 'scale': '~1'}
    reduced_X = X
    if not keep_location:
        reduced_args.pop('mods')
        reduced_X = numpy.ones((len(yi), 1))
    reduced = metafopy.rma_uni(**reduced_args, data=correctness.cases.case_frame(case))
    reduced_ll = correctness.formulas.ls_reference_fit(yi, vi, reduced_X, numpy.ones((len(yi), 1)), link, method)['ll']
    stat, pval = correctness.formulas.ls_lrt(ls_reference(case)['ll'], reduced_ll, (X.shape[1] - reduced_X.shape[1]) + (Z.shape[1] - 1))
    actual = metafopy.anova(result, reduced)
    ls_agree(case, 'LRT', actual.LRT, stat, LS_NUMERIC_HESSIAN)
    ls_agree(case, 'pval', actual.pval, pval, LS_NUMERIC_HESSIAN)


def ls_permutest_exact(case: correctness.cases.Case, result: typing.Any) -> None:
    """``permutest(..., exact=True)`` of one part against ``formulas.ls_exact_permutation_p``, which enumerates and refits every distinct permutation.

    The p-values are shares of the same permutations. The reference has no fit for a permuted model whose likelihood has no finite maximum, so each
    p-value is checked to lie between the share without those permutations and the share with all of them.
    """
    yi, vi, X, Z = ls_inputs(case)
    part = 'scale' if result.skip_beta else 'location'
    expected = correctness.formulas.ls_exact_permutation_p(yi, vi, X, Z, case.args.get('link', 'log'), case.args['method'], part)
    actual_pval, actual_omnibus, count = (
        (result.pval, result.QMp, result.exact_iter) if part == 'location' else (result.pval_alpha, result.QSp, result.Z_exact_iter)
    )
    ls_agree(case, 'permutations', count, expected['count'], 1e-12)
    slack = 1e-9
    for name, actual, low, high in (
        (f'pval ({part})', numpy.asarray(actual_pval), expected['pval_low'], expected['pval_high']),
        (f'omnibus p ({part})', numpy.asarray(actual_omnibus), expected['omnibus_low'], expected['omnibus_high']),
    ):
        if not numpy.all((actual >= numpy.asarray(low) - slack) & (actual <= numpy.asarray(high) + slack)):
            raise AssertionError(
                f'{case.id}: {name}: metafopy gives {actual!r}, the reference lies between {low!r} and {high!r} '
                f'({expected["failed"]:g} of {expected["count"]:g} permutations have no reference fit)'
            )
