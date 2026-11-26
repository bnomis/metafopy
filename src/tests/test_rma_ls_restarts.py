"""The restarts of a log-link location-scale fit (agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md)."""

import itertools
import warnings

import correctness.formulas
import numpy
import pandas
import pytest

import metafopy
import metafopy.models.rma_ls_fit
import metafopy.models.rma_ls_opt

TPOS = [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27]
TNEG = [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886]
CPOS = [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29]
CNEG = [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825]
LAT = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
RIDGE = 11  # a permutation whose default fit lies on the ridge where tau2 tends to 0 in one group (alpha about [-26, 25])


def permuted(index: int, **control: object) -> metafopy.RMALSResult:
    ai, bi, ci, di = (numpy.array(v, dtype=float) for v in (TPOS, TNEG, CPOS, CNEG))
    n1, n2 = ai + bi, ci + di
    order = next(itertools.islice(correctness.formulas.distinct_label_orders(sorted(LAT)), index, None))
    data = pandas.DataFrame(
        {
            'yi': numpy.log(ai / n1) - numpy.log(ci / n2),
            'vi': 1 / ai - 1 / n1 + 1 / ci - 1 / n2,
            'lat': numpy.array(order, dtype=float),
            'grp': numpy.array(LAT, dtype=float),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', mods='~ lat', scale='~ grp', data=data, method='ML', link='log', test='z', control=control or None)
    assert isinstance(res, metafopy.RMALSResult)
    return res


def test_restart_reaches_the_higher_maximum() -> None:
    """Permutation 433 of the bcglat data: the single start stops at ll -12.332 (R's fit), the restart reaches ll -11.968."""
    res = permuted(433)
    assert res.fit_stats.loc['ll', 'ML'] == pytest.approx(-11.96789514, abs=1e-6)
    assert res.zval[1] == pytest.approx(1.9024, abs=1e-4)


def test_single_start_without_restarts() -> None:
    res = permuted(433, restarts=False)
    assert res.fit_stats.loc['ll', 'ML'] == pytest.approx(-12.33206543, abs=1e-6)
    assert res.zval[1] == pytest.approx(-2.5047, abs=1e-4)


def test_ridge_fit_is_unchanged() -> None:
    """No restart is accepted where the higher likelihood is on the ridge: the fit equals the single start's."""
    on, off = permuted(RIDGE), permuted(RIDGE, restarts=False)
    assert numpy.abs(off.alpha).max() > 10
    assert on.fit_stats.loc['ll', 'ML'] == off.fit_stats.loc['ll', 'ML']
    assert numpy.array_equal(on.alpha, off.alpha)


def test_restart_starts_one_per_non_constant_column() -> None:
    Z = numpy.column_stack([numpy.ones(4), [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 0.0, 0.0]])
    alpha = numpy.array([0.5, -1.0, 0.0])
    starts = metafopy.models.rma_ls_fit.restart_starts(alpha, Z)
    assert len(starts) == 2
    step = metafopy.models.rma_ls_fit.RESTART_STEP
    assert starts[0] == pytest.approx([0.5, -1.0 + step / numpy.std(Z[:, 1], ddof=1), 0.0])
    assert starts[1] == pytest.approx([0.5, -1.0, -step / numpy.std(Z[:, 2], ddof=1)])


def test_restart_starts_of_an_intercept_only_model_is_empty() -> None:
    assert metafopy.models.rma_ls_fit.restart_starts(numpy.array([0.3]), numpy.ones((5, 1))) == []


def test_is_regular_maximum_separates_interior_from_ridge() -> None:
    interior, ridge = permuted(433), permuted(RIDGE, restarts=False)
    for res, expected in ((interior, True), (ridge, False)):
        problem = metafopy.models.rma_ls_fit.LSProblem(
            yi=res.yi,
            vi=res.vi,
            X=res.X,
            Z=res.Z,
            reml=False,
            REMLf=True,
            link='log',
            alpha_min=numpy.full(2, -numpy.inf),
            alpha_max=numpy.full(2, numpy.inf),
            tau2_min=0.0,
            tau2_max=numpy.inf,
            optbeta=False,
        )
        assert metafopy.models.rma_ls_fit.is_regular_maximum(problem, res.alpha) is expected


def identity_problem(Z: numpy.ndarray, yi: numpy.ndarray, vi: numpy.ndarray, X: numpy.ndarray) -> metafopy.models.rma_ls_fit.LSProblem:
    return metafopy.models.rma_ls_fit.LSProblem(
        yi=yi,
        vi=vi,
        X=X,
        Z=Z,
        reml=False,
        REMLf=True,
        link='identity',
        alpha_min=numpy.full(Z.shape[1], -numpy.inf),
        alpha_max=numpy.full(Z.shape[1], numpy.inf),
        tau2_min=0.0,
        tau2_max=numpy.inf,
        optbeta=False,
    )


def permuted_identity(index: int, *, restarts: bool = False) -> metafopy.RMALSResult:
    """A permutation of the bcglat data with ``lat`` permuted in the location model only (``grp`` is the scale model's ``lat``).

    Single start unless ``restarts``.
    """
    ai, bi, ci, di = (numpy.array(v, dtype=float) for v in (TPOS, TNEG, CPOS, CNEG))
    n1, n2 = ai + bi, ci + di
    order = next(itertools.islice(correctness.formulas.distinct_label_orders(sorted(LAT)), index, None))
    data = pandas.DataFrame(
        {
            'yi': numpy.log(ai / n1) - numpy.log(ci / n2),
            'vi': 1 / ai - 1 / n1 + 1 / ci - 1 / n2,
            'lat': numpy.array(order, dtype=float),
            'grp': numpy.array(LAT, dtype=float),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', mods='~ lat', scale='~ grp', data=data, method='ML', link='identity', test='z', control={'restarts': restarts})
    assert isinstance(res, metafopy.RMALSResult)
    return res


def test_identity_restart_starts_on_a_binary_column() -> None:
    Z = numpy.column_stack([numpy.ones(6), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    alpha = numpy.array([0.4, 0.7])  # tau2 0.4 in group 0 and 1.1 in group 1
    starts = metafopy.models.rma_ls_fit.identity_restart_starts(alpha, Z)
    pairs = [(float(Z[0] @ s), float(Z[3] @ s)) for s in starts]
    assert any(p[0] == 0.0 and p[1] == pytest.approx(1.1) for p in pairs)
    assert any(p[1] == 0.0 and p[0] == pytest.approx(0.4) for p in pairs)
    assert any(p[0] == pytest.approx(1.1) and p[1] == pytest.approx(0.4) for p in pairs)


def test_identity_restart_starts_are_feasible_on_a_continuous_column() -> None:
    Z = numpy.column_stack([numpy.ones(9), numpy.linspace(-1.0, 1.0, 9)])
    starts = metafopy.models.rma_ls_fit.identity_restart_starts(numpy.array([0.5, 0.3]), Z)
    assert starts
    for start in starts:
        assert numpy.min(Z @ start) >= 0.0


def test_identity_restart_start_zeros_are_exact() -> None:
    """A dummy face's ``tau2 = 0`` entry is exactly 0.0, with no negative residue (printed ``-0.``)."""
    Z = numpy.column_stack([numpy.ones(6), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    alpha = numpy.array([0.1 + 0.2, 0.3 - 0.1])
    for start in metafopy.models.rma_ls_fit.identity_restart_starts(alpha, Z):
        assert numpy.min(Z @ start) >= 0.0
        assert not numpy.signbit(start[numpy.abs(start) < 1e-12]).any()


def test_is_face_maximum_passes_a_face_maximum() -> None:
    res = permuted_identity(18)
    problem = identity_problem(res.Z, res.yi, res.vi, res.X)
    nan = numpy.full(2, numpy.nan)
    assert res.fit_stats.loc['ll', 'ML'] == pytest.approx(-12.62494606, abs=1e-6)
    best_ll, best = -numpy.inf, None
    for start in metafopy.models.rma_ls_fit.identity_restart_starts(res.alpha, res.Z):
        end = metafopy.models.rma_ls_opt.polish_identity(
            start,
            lambda a: problem.loglik(a, nan, nan, alpha_transf=False),
            problem.ls_score,
            problem.ls_information,
            res.Z,
            numpy.ones(2, dtype=bool),
        )
        ll = problem.loglik(end, nan, nan, alpha_transf=False)
        if ll > best_ll:
            best_ll, best = ll, end
    assert best is not None
    assert best_ll == pytest.approx(-11.42937466, abs=1e-6)
    assert metafopy.models.rma_ls_fit.is_face_maximum(problem, best)


def test_is_face_maximum_refuses_a_singular_face() -> None:
    """Two identical columns make the information singular on the face, where the active row leaves both free."""
    x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    Z = numpy.column_stack([numpy.ones(6), x, x])
    rng = numpy.random.default_rng(1)
    problem = identity_problem(Z, rng.normal(size=6), numpy.full(6, 0.2), numpy.ones((6, 1)))
    assert not metafopy.models.rma_ls_fit.is_face_maximum(problem, numpy.array([0.0, 0.5, 0.5]))


def test_identity_restart_reaches_the_face_maximum() -> None:
    """Permutation 18: R's single start stops at ll -12.625, the restart reaches the face maximum with tau2 of the grp = 0 group at 0."""
    res = permuted_identity(18, restarts=True)
    assert res.fit_stats.loc['ll', 'ML'] == pytest.approx(-11.42937466, abs=1e-6)
    assert numpy.min(res.tau2) == pytest.approx(0.0, abs=1e-6)


def test_identity_single_start_without_restarts() -> None:
    res = permuted_identity(18)
    assert res.fit_stats.loc['ll', 'ML'] == pytest.approx(-12.62494606, abs=1e-6)
    assert res.zval[1] == pytest.approx(0.2490, abs=1e-4)
