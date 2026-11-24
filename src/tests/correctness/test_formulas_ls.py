"""Unit checks of the location-scale functions of ``formulas``: the analytic score and information against numerical derivatives, and the failure mode.

The data come from the bundled datasets only; the effect sizes of ``dat.bcg`` are computed with ``formulas``, not with metafopy.
"""

import collections.abc
import itertools
import math
import pathlib
import tomllib
import typing

import numpy
import numpy.typing
import pytest
import scipy.optimize

import correctness.formulas
import metafopy.datasets

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Data = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def bcg() -> Data:
    f = metafopy.datasets.load('dat.bcg')
    a, b = f['tpos'].to_numpy(dtype=float), f['tneg'].to_numpy(dtype=float)
    c, d = f['cpos'].to_numpy(dtype=float), f['cneg'].to_numpy(dtype=float)
    yi = correctness.formulas.log_risk_ratio_yi(a, b, c, d)
    vi = correctness.formulas.log_risk_ratio_vi(a, b, c, d)
    X = numpy.column_stack([numpy.ones(len(yi)), f['ablat'].to_numpy(dtype=float)])
    Z = numpy.column_stack([numpy.ones(len(yi)), (f['year'].to_numpy(dtype=float) - 1950) / 10])
    return yi, vi, X, Z


def bcg_ablat() -> Data:
    """The design of the case ``ls.fit.REML.identity.bcg``: absolute latitude in both the location and the scale model."""
    yi, vi, _, _ = bcg()
    ablat = metafopy.datasets.load('dat.bcg')['ablat'].to_numpy(dtype=float)
    X = numpy.column_stack([numpy.ones(len(yi)), ablat])
    return yi, vi, X, X


def bcg2() -> Data:
    """The design of the cases ``ls.vif.REML.bcg2``: absolute latitude and the uncentred year in both the location and the scale model."""
    yi, vi, _, _ = bcg()
    f = metafopy.datasets.load('dat.bcg')
    X = numpy.column_stack([numpy.ones(len(yi)), f['ablat'].to_numpy(dtype=float), f['year'].to_numpy(dtype=float)])
    return yi, vi, X, X.copy()


def bangert() -> Data:
    f = metafopy.datasets.load('dat.bangertdrowns2004').dropna(subset=['length']).reset_index(drop=True)
    yi, vi = f['yi'].to_numpy(dtype=float), f['vi'].to_numpy(dtype=float)
    length = f['length'].to_numpy(dtype=float)
    scaled = (length - length.mean()) / length.std()
    X = numpy.column_stack([numpy.ones(len(yi)), scaled])
    Z = numpy.column_stack([numpy.ones(len(yi)), scaled])
    return yi, vi, X, Z


DATASETS = {'bangertdrowns2004': bangert, 'bcg': bcg}
ALPHAS = {
    'identity': [numpy.array([0.3, 0.02]), numpy.array([0.5, 0.0]), numpy.array([0.2, 0.05])],
    'log': [numpy.array([-1.0, 0.2]), numpy.array([-2.0, -0.3]), numpy.array([0.0, 0.1])],
}
METHODS = sorted({'REML', 'ML'})
CASES = list(itertools.product(DATASETS, METHODS, ('log', 'identity')))


def numerical_hessian(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The Jacobian of the analytic score, by forward differences."""
    step = 1e-7
    base = correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method)
    columns = []
    for ell in range(len(alpha)):
        shifted = alpha.copy()
        shifted[ell] += step
        columns.append((correctness.formulas.ls_score(yi, vi, X, Z, shifted, link, method) - base) / step)
    return numpy.column_stack(columns)


def numerical_score(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The gradient of the log-likelihood by forward differences."""
    step = 1e-7
    base = correctness.formulas.ls_loglik(yi, vi, X, Z, alpha, link, method)
    out = numpy.zeros(len(alpha))
    for j in range(len(alpha)):
        shifted = alpha.copy()
        shifted[j] += step
        out[j] = (correctness.formulas.ls_loglik(yi, vi, X, Z, shifted, link, method) - base) / step
    return out


@pytest.mark.parametrize(('name', 'method', 'link'), CASES)
def test_score_and_information_match_numerical_derivatives(name: str, method: str, link: str) -> None:
    yi, vi, X, Z = DATASETS[name]()
    checked = 0
    for alpha in ALPHAS[link]:
        if correctness.formulas.ls_loglik(yi, vi, X, Z, alpha, link, method) == -numpy.inf:
            continue
        checked += 1
        score = correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method)
        assert score == pytest.approx(numerical_score(yi, vi, X, Z, alpha, link, method), rel=1e-4, abs=1e-4)
        info = correctness.formulas.ls_information(yi, vi, X, Z, alpha, link, method)
        assert info == pytest.approx(-numerical_hessian(yi, vi, X, Z, alpha, link, method), rel=1e-4, abs=1e-4)
    assert checked > 0


def test_information_of_the_wrong_sign_is_caught() -> None:
    yi, vi, X, Z = bcg()
    alpha = ALPHAS['log'][0]
    info = correctness.formulas.ls_information(yi, vi, X, Z, alpha, 'log', 'ML')
    assert info != pytest.approx(numerical_hessian(yi, vi, X, Z, alpha, 'log', 'ML'), rel=1e-4, abs=1e-4)


@pytest.mark.parametrize('method', METHODS)
def test_reference_fit_has_a_zero_score_and_a_covariance(method: str) -> None:
    yi, vi, X, Z = bangert()
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    assert correctness.formulas.ls_score(yi, vi, X, Z, fit['alpha'], 'log', method) == pytest.approx(0.0, abs=1e-5)
    assert fit['ll'] == pytest.approx(correctness.formulas.ls_loglik(yi, vi, X, Z, fit['alpha'], 'log', method), rel=1e-12)
    assert numpy.all(numpy.linalg.eigvalsh(fit['va']) > 0)


def relabel(monkeypatch: pytest.MonkeyPatch, method: str, success: collections.abc.Callable[[scipy.optimize.OptimizeResult], bool], message: str) -> None:
    """Make ``formulas`` report each ``method`` result as the optimiser would on another platform: ``success`` says whether it is called a success."""
    real = scipy.optimize.minimize

    def minimize(*args: typing.Any, **kwargs: typing.Any) -> scipy.optimize.OptimizeResult:
        res: scipy.optimize.OptimizeResult = real(*args, **kwargs)
        if kwargs.get('method') == method:
            res['success'] = success(res)
            res['message'] = 'Optimization terminated successfully' if res['success'] else message
        return res

    monkeypatch.setattr(scipy.optimize, 'minimize', minimize)


@pytest.mark.parametrize('method', METHODS)
def test_reference_fit_accepts_starts_that_report_precision_loss(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """A BFGS stop called "precision loss" lies on the maximum on some platforms; the fit keeps it because its score is small."""
    yi, vi, X, Z = bcg()
    expected = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    relabel(monkeypatch, 'BFGS', lambda res: False, 'Desired error not necessarily achieved due to precision loss.')
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    assert fit['ll'] == pytest.approx(expected['ll'], abs=1e-10)
    assert float(numpy.max(numpy.abs(correctness.formulas.ls_score(yi, vi, X, Z, fit['alpha'], 'log', method)))) < correctness.formulas.LS_SCORE_TOLERANCE


def test_reference_fit_does_not_choose_a_worse_optimum_that_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The identity-link restricted likelihood of the case has two optima (log-likelihood -7.98877 and -8.08078): the fit returns the higher whatever the
    optimiser calls successful."""
    yi, vi, X, Z = bcg_ablat()
    relabel(monkeypatch, 'SLSQP', lambda res: float(-res.fun) < -8.0, 'Positive directional derivative for linesearch')
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'identity', 'REML')
    assert fit['ll'] == pytest.approx(-7.988765268, abs=1e-8)


def test_identity_polish_stays_feasible_and_finds_the_face_optimum() -> None:
    """The identity link's polish never leaves ``Z alpha >= 0``. On a boundary optimum (the ``ls.fit.ML.identity.bcg`` design, where the score across the
    constraint is 742) it ends with a score along the constraint below 1e-6: about 1e-7 in the arm64 container, 2e-5 without the polish, which left
    ``alpha`` about 2e-6 off the other platforms' reference.
    """
    yi, vi, X, Z = bangert()
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'identity', 'REML')
    assert float(numpy.min(Z @ fit['alpha'])) >= -correctness.formulas.IDENTITY_SLACK
    yi, vi, X, Z = bcg_ablat()
    ml = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'identity', 'ML')
    assert float(numpy.min(Z @ ml['alpha'])) >= -correctness.formulas.IDENTITY_SLACK
    basis = correctness.formulas.ls_face_basis(Z, ml['alpha'], 'identity')
    assert basis.shape[1] == 1
    assert float(numpy.max(numpy.abs(basis.T @ correctness.formulas.ls_score(yi, vi, X, Z, ml['alpha'], 'identity', 'ML')))) < 1e-6


def test_ls_profile_ll_identity_link_starts_feasible() -> None:
    """At ``alpha.2 = -0.00615`` of the ``ls.fit.ML.identity.bcg`` fit the identity-link profile is -8.045643 (the maximum of a brute-force profile
    over the feasible interval of ``alpha.1``), not the -1e10 of a refit that never leaves an infeasible start."""
    yi, vi, X, Z = bcg_ablat()
    ml = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'identity', 'ML')
    value = -0.00615
    profile = correctness.formulas.ls_profile_ll(yi, vi, X, Z, 'identity', 'ML', 1, value, ml['alpha'])
    assert profile == pytest.approx(-8.045643, abs=1e-6)
    lower = float(numpy.max(-value * Z[:, 1]))  # Z alpha >= 0 for every row

    def negative(a0: float) -> float:
        ll = correctness.formulas.ls_loglik(yi, vi, X, Z, numpy.array([a0, value]), 'identity', 'ML')
        return math.inf if ll == -math.inf else -ll

    grid = lower + numpy.linspace(0.0, 0.2, 2001)
    values = [negative(float(a0)) for a0 in grid]
    k = int(numpy.argmin(values))
    refined = scipy.optimize.minimize_scalar(
        negative, bounds=(float(grid[max(k - 1, 0)]), float(grid[min(k + 1, len(grid) - 1)])), method='bounded', options={'xatol': 1e-13}
    )
    brute = -min(float(refined.fun), values[k])
    assert profile == pytest.approx(brute, abs=1e-6)


def test_reference_fit_raises_when_nothing_converges() -> None:
    yi, vi, X, Z = bcg()
    with pytest.raises(RuntimeError, match='no start converged'):
        correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'ML', maxiter=1)


def test_inline_columns_are_the_recoding_of_the_dataset() -> None:
    """The inline ``yi``, ``vi`` and ``ni100`` of the no-intercept heterogeneity case are the dataset's ``yi`` and ``vi`` and ``ni / 100``."""
    path = pathlib.Path(__file__).parent / 'cases' / 'location_scale.toml'
    cases = tomllib.loads(path.read_text(encoding='utf-8'))['case']
    inline = next(c['data'] for c in cases if c['id'] == 'ls.heterogeneity.bdni')
    frame = metafopy.datasets.load('dat.bangertdrowns2004')
    assert inline['yi'] == frame['yi'].tolist()
    assert inline['vi'] == frame['vi'].tolist()
    assert inline['ni100'] == pytest.approx((frame['ni'] / 100).tolist(), rel=1e-15)


def test_inline_columns_of_the_published_cases_are_the_authors_recoding() -> None:
    """Every ``published.vl2022`` case's inline columns are the dataset's ``yi``, ``vi``, ``ni / 100`` and the paper's three subject groups."""
    path = pathlib.Path(__file__).parent / 'cases' / 'location_scale.toml'
    cases = [c for c in tomllib.loads(path.read_text(encoding='utf-8'))['case'] if c['id'].startswith('published.vl2022.')]
    assert cases
    frame = metafopy.datasets.load('dat.bangertdrowns2004')
    math = {'Algebra', 'Calculus', 'Math', 'Math in Science', 'Statistics'}
    science = {'Chemistry', 'Comp Science and Chemistry', 'Biology', 'Earth Science', 'Natural Resources', 'Nursing', 'Science'}
    expected = {
        'yi': frame['yi'],
        'vi': frame['vi'],
        'ni100': frame['ni'] / 100,
        'sci': frame['subject'].isin(science).astype(int),
        'soc': (~frame['subject'].isin(math | science)).astype(int),
    }
    assert [int(expected[name].sum()) for name in ('sci', 'soc')] == [9, 11]
    for case in cases:
        for name, column in case['data'].items():
            assert column == pytest.approx(expected[name].tolist(), rel=1e-15), f'{case["id"]}: {name}'


def test_inline_counts_of_the_permutation_cases_are_the_dataset_with_a_binary_moderator() -> None:
    """The inline counts of the exact-permutation cases are those of ``dat.bcg``, and ``lat`` is ``ablat >= 33``."""
    path = pathlib.Path(__file__).parent / 'cases' / 'location_scale.toml'
    cases = [c for c in tomllib.loads(path.read_text(encoding='utf-8'))['case'] if c['id'].startswith('ls.permutest.') and '.exact.' in c['id']]
    assert cases
    frame = metafopy.datasets.load('dat.bcg')
    for case in cases:
        for name, column in case['data'].items():
            expected = (frame['ablat'] >= 33).astype(int) if name == 'lat' else frame[name]
            assert column == expected.tolist()


def test_distinct_label_orders_are_the_multiset_permutations() -> None:
    orders = list(correctness.formulas.distinct_label_orders([0, 0, 1, 1, 1]))
    assert len(orders) == len(set(orders)) == 10
    assert all(sorted(order) == [0, 0, 1, 1, 1] for order in orders)


def test_exact_permutation_p_bounds_the_share_of_permutations_that_reach_the_observed_statistic() -> None:
    yi, vi, X, _ = bcg()
    lat = (X[:, 1] >= 33).astype(float)[:8]
    design = numpy.column_stack([numpy.ones(8), lat])
    exact = correctness.formulas.ls_exact_permutation_p(yi[:8], vi[:8], design, design, 'log', 'ML', 'location')
    assert exact['count'] == 70 or exact['count'] < 70
    low, high = numpy.asarray(exact['pval_low']), numpy.asarray(exact['pval_high'])
    assert numpy.all(low >= 1 / exact['count'] - 1e-12)
    assert numpy.all(high >= low)
    assert numpy.allclose(high - low, exact['failed'] / exact['count'])
    with pytest.raises(ValueError, match='part must be'):
        correctness.formulas.ls_exact_permutation_p(yi[:8], vi[:8], design, design, 'log', 'ML', 'both')


def joint_ll_at(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, theta: FloatArray, link: str) -> float:
    p = X.shape[1]
    return correctness.formulas.ls_joint_loglik(yi, vi, X, Z, theta[:p], theta[p:], link)


@pytest.mark.parametrize('name', sorted(DATASETS))
@pytest.mark.parametrize('link', sorted(ALPHAS))
def test_joint_information_matches_central_differences_of_equation_5(name: str, link: str) -> None:
    yi, vi, X, Z = DATASETS[name]()
    beta, alpha = numpy.array([0.1, -0.05]), ALPHAS[link][0]
    theta = numpy.concatenate([beta, alpha])
    analytic = correctness.formulas.ls_joint_information(yi, vi, X, Z, beta, alpha, link)
    step = 1e-4
    numerical = numpy.zeros_like(analytic)
    for i in range(len(theta)):
        for j in range(len(theta)):
            e_i, e_j = numpy.eye(len(theta))[i] * step, numpy.eye(len(theta))[j] * step
            numerical[i, j] = -(
                joint_ll_at(yi, vi, X, Z, theta + e_i + e_j, link)
                - joint_ll_at(yi, vi, X, Z, theta + e_i - e_j, link)
                - joint_ll_at(yi, vi, X, Z, theta - e_i + e_j, link)
                + joint_ll_at(yi, vi, X, Z, theta - e_i - e_j, link)
            ) / (4 * step**2)
    assert numpy.max(numpy.abs(analytic - numerical)) / max(1.0, float(numpy.max(numpy.abs(analytic)))) < 1e-5


def test_joint_information_of_the_wrong_sign_is_caught() -> None:
    yi, vi, X, Z = bangert()
    beta, alpha = numpy.array([0.1, -0.05]), ALPHAS['log'][0]
    analytic = correctness.formulas.ls_joint_information(yi, vi, X, Z, beta, alpha, 'log')
    assert float(numpy.max(numpy.abs(analytic + analytic.T))) > 1e-3


def test_gvif_of_two_moderators_is_the_variance_ratio_of_the_ols_fit() -> None:
    rng = numpy.random.default_rng(7)
    x1 = rng.normal(size=40)
    x2 = 0.6 * x1 + rng.normal(size=40)
    full = numpy.column_stack([numpy.ones(40), x1, x2])
    alone = numpy.column_stack([numpy.ones(40), x1])
    vcov = numpy.linalg.inv(full.T @ full)
    ratio = vcov[1, 1] / numpy.linalg.inv(alone.T @ alone)[1, 1]
    assert correctness.formulas.gvif(vcov, [0]) == pytest.approx(ratio, rel=1e-10)
    assert correctness.formulas.gvif(vcov, [0, 1]) == pytest.approx(1.0, rel=1e-10)


def test_ls_blups_with_a_constant_tau2_is_the_uni_blups() -> None:
    yi, vi, X, _ = bangert()
    tau2 = 0.07
    ours = correctness.formulas.ls_blups(yi, vi, X, numpy.full(len(yi), tau2))
    theirs = correctness.formulas.uni_blups(yi, vi, X, tau2)
    assert numpy.max(numpy.abs(ours['pred'] - theirs.pred)) < 1e-12
    assert numpy.max(numpy.abs(ours['pred_se'] - theirs.se)) < 1e-12


def test_ls_blups_refuses_a_zero_tau2() -> None:
    yi, vi, X, _ = bangert()
    with pytest.raises(ValueError, match='above zero'):
        correctness.formulas.ls_blups(yi, vi, X, numpy.zeros(len(yi)))


@pytest.mark.parametrize('method', METHODS)
def test_ls_rstandard_types_share_z_and_agree_without_heterogeneity(method: str) -> None:
    yi, vi, X, Z = bcg()
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    marginal = correctness.formulas.ls_rstandard(yi, vi, X, fit['tau2'], False)
    conditional = correctness.formulas.ls_rstandard(yi, vi, X, fit['tau2'], True)
    assert numpy.max(numpy.abs(marginal[2] - conditional[2])) < 1e-12
    homogeneous = correctness.formulas.ls_rstandard(yi, vi, X, numpy.zeros(len(yi)), True)
    plain = correctness.formulas.ls_rstandard(yi, vi, X, numpy.zeros(len(yi)), False)
    assert numpy.max(numpy.abs(homogeneous[0] - plain[0])) == 0.0


@pytest.mark.parametrize('name', sorted(DATASETS))
@pytest.mark.parametrize('method', METHODS)
def test_constrained_fit_with_nothing_held_is_the_reference_fit(name: str, method: str) -> None:
    yi, vi, X, Z = DATASETS[name]()
    reference = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    constrained = correctness.formulas.ls_constrained_fit(yi, vi, X, Z, 'log', method, numpy.full(2, numpy.nan))
    assert abs(constrained['ll'] - reference['ll']) < 1e-9
    assert numpy.max(numpy.abs(constrained['alpha'] - reference['alpha'])) < 1e-6 * max(1.0, float(numpy.max(numpy.abs(reference['alpha']))))
    assert numpy.max(numpy.abs(constrained['va'] - reference['va'])) < 1e-5 * float(numpy.max(numpy.abs(reference['va'])))


def test_constrained_fit_holds_a_coefficient_and_respects_a_binding_bound() -> None:
    yi, vi, X, Z = bangert()
    held = correctness.formulas.ls_constrained_fit(yi, vi, X, Z, 'log', 'ML', numpy.array([numpy.nan, 0.5]))
    assert held['alpha'][1] == 0.5
    assert held['va'].shape == (1, 1)
    free = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'ML')
    assert held['ll'] < free['ll']
    cap = float(free['alpha'][1]) - 0.3
    boxed = correctness.formulas.ls_constrained_fit(
        yi, vi, X, Z, 'log', 'ML', numpy.full(2, numpy.nan), numpy.array([-numpy.inf, -numpy.inf]), numpy.array([numpy.inf, cap])
    )
    assert boxed['alpha'][1] == pytest.approx(cap, abs=1e-8)
    assert boxed['ll'] < free['ll']


def test_joint_fit_has_a_zero_score_and_matches_the_profile_fit() -> None:
    yi, vi, X, Z = bcg()
    joint = correctness.formulas.ls_joint_fit(yi, vi, X, Z, 'log', numpy.full(2, numpy.nan))
    profile = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'ML')
    assert abs(joint['ll'] - profile['ll']) < 1e-9
    assert numpy.max(numpy.abs(joint['alpha'] - profile['alpha'])) < 1e-6
    assert numpy.max(numpy.abs(joint['va'] - profile['va'])) < 1e-5 * float(numpy.max(numpy.abs(profile['va'])))
    held = correctness.formulas.ls_joint_fit(yi, vi, X, Z, 'log', numpy.array([numpy.nan, 0.02]))
    assert held['free_beta'] == [0]
    assert held['beta'][1] == 0.02
    assert held['vb'].shape == (1, 1)
    assert held['ll'] < joint['ll']


def test_joint_fit_refuses_the_identity_link() -> None:
    yi, vi, X, Z = bangert()
    with pytest.raises(ValueError, match='log link'):
        correctness.formulas.ls_joint_fit(yi, vi, X, Z, 'identity', numpy.full(2, numpy.nan))


@pytest.mark.parametrize('tolerance', [5e-7, 1e-9])
@pytest.mark.parametrize('method', METHODS)
def test_reference_fit_of_uncentred_columns_does_not_depend_on_the_score_tolerance(monkeypatch: pytest.MonkeyPatch, method: str, tolerance: float) -> None:
    """In the raw ``Z`` the Newton polish meets a singular information on some platforms; in standardised coordinates it does not."""
    yi, vi, X, Z = bcg2()
    reference = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    monkeypatch.setattr(correctness.formulas, 'LS_SCORE_TOLERANCE', tolerance)
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    assert fit['ll'] == pytest.approx(reference['ll'], abs=1e-9)
    T = correctness.formulas.ls_standardise(Z)
    assert correctness.formulas.ls_is_maximum(yi, vi, X, Z @ numpy.linalg.inv(T), T @ fit['alpha'], 'log', method)


def test_is_maximum_rejects_an_underflowed_tau2() -> None:
    """The REML start that reaches ``year`` slope -1.00986 has ``tau2 = 0`` and a zero score; it is not a maximum (in the standardised coordinates)."""
    yi, vi, X, Z = bcg2()
    T = correctness.formulas.ls_standardise(Z)
    Zs = Z @ numpy.linalg.inv(T)
    degenerate = T @ numpy.array([0.89967, -0.01675, -1.00986])
    assert not correctness.formulas.ls_is_maximum(yi, vi, X, Zs, degenerate, 'log', 'REML')
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'REML')
    assert correctness.formulas.ls_is_maximum(yi, vi, X, Zs, T @ fit['alpha'], 'log', 'REML')
