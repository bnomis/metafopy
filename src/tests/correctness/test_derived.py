"""``derived`` cases: expected values from ``formulas.py``, coded from the publications' equations."""

import dataclasses
import math
import re
import typing

import numpy
import numpy.typing
import pandas
import pytest
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
GLMM_CONDITIONAL_DEVIANCE_FITS = frozenset({('CM.AL', 'ML'), ('UM.FS', 'ML'), ('UM.RS', 'FE'), ('UM.RS', 'ML')})


def glmm_likelihood_at_estimate(case: correctness.cases.Case, result: typing.Any) -> None:
    """The formulas' log-likelihood at the reported estimates equals the formulas' maximum and, where ``rma_glmm`` reports the marginal
    log-likelihood, ``result.fit_stats_ml.ll``; the score there is small.

    ``lme4`` fits (the mixed models other than ``CM.EL``) report metafor's saturated model log-likelihood minus half the conditional deviance, not the
    marginal log-likelihood, so ``ll`` is compared only where it is the marginal one; the other two checks hold for every fit.
    """
    frame = correctness.cases.case_frame(case)
    problem, rule, points = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_problem', frame)
    reference = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_fit', frame)
    method = case.args.get('method', 'ML')
    sigma2 = float(result.sigma2) if problem.model == 'UM.RS' else math.nan
    rho = float(result.rho) if problem.cor and problem.model == 'UM.RS' else math.nan
    loglik, score = correctness.glmm_formulas.profile_check(problem, method, result.beta, float(result.tau2), sigma2, rho, rule, points)
    where = f"{case.id}: the formulas' log-likelihood at the estimates of metafopy"
    if not loglik <= reference.loglik + GLMM_LIKELIHOOD_RTOL * abs(reference.loglik):
        raise AssertionError(f"{where} is {loglik!r}, above the formulas' maximum {reference.loglik!r}")
    if not reference.loglik - loglik <= GLMM_LIKELIHOOD_RTOL * abs(reference.loglik):
        raise AssertionError(f"{where} is {loglik!r}, not the formulas' maximum {reference.loglik!r} (rtol {GLMM_LIKELIHOOD_RTOL:g})")
    if not score <= GLMM_GRADIENT_TOLERANCE:
        raise AssertionError(f'{where} has a score of {score:.3g}, not below {GLMM_GRADIENT_TOLERANCE:g}')
    if (problem.model, method) not in GLMM_CONDITIONAL_DEVIANCE_FITS:
        reported = float(result.fit_stats_ml.ll)
        if not abs(reported - loglik) <= GLMM_LIKELIHOOD_RTOL * abs(loglik):
            raise AssertionError(f'{case.id}: metafopy reports ll {reported!r}, the formulas give {loglik!r} at its estimates')
