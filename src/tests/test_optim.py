"""The ports of R's optimize(), optim(method="Nelder-Mead") and optimhess must reproduce R's iterates."""

import math

import hypothesis
import hypothesis.strategies
import numpy
import pytest
import rstrategies
import scipy.stats

import metafopy.core.bfgs
import metafopy.core.bvn
import metafopy.core.optim

INF = math.inf


def test_fmin_matches_r() -> None:
    # R: optimize(function(x) (x-1/3)^2 + sin(5*x), c(-1, 2))$minimum
    assert metafopy.core.optim.fmin(lambda x: (x - 1 / 3) ** 2 + math.sin(5 * x), -1, 2) == 0.89699447458266623


def rosenbrock(x: numpy.ndarray) -> float:
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def test_nelder_mead_matches_r() -> None:
    # R: optim(c(-1.2, 1), fr, hessian=TRUE) with fr the Rosenbrock function
    res = metafopy.core.optim.nelder_mead(rosenbrock, [-1.2, 1.0])
    numpy.testing.assert_allclose(res.par, [1.0002601387256695, 1.0005059993037650], rtol=1e-14)
    assert math.isclose(res.value, 8.8252410967227472e-08, rel_tol=1e-12)
    assert res.fncount == 195
    assert res.fail == 0
    hess = metafopy.core.optim.optimhess(rosenbrock, res.par)
    numpy.testing.assert_allclose(hess, [[802.42281442667115, -400.10405549025830], [-400.10405549025830, 199.99999999998931]], rtol=1e-10)


def test_bvn_matches_mvtnorm() -> None:
    # R: mvtnorm::pmvnorm(..., corr=matrix(c(1, r, r, 1), 2))
    assert math.isclose(metafopy.core.bvn.bvn_rect((-INF, -INF), (0.3, -0.2), 0.5), 0.33619843701551877, rel_tol=1e-14)
    assert math.isclose(metafopy.core.bvn.bvn_rect((0.3, -INF), (INF, -0.2), 0.95), 0.0030896714018696747, rel_tol=1e-13)
    assert math.isclose(metafopy.core.bvn.bvn_rect((-INF, -0.2), (0.3, INF), -0.97), 0.5576283936677815, rel_tol=1e-14)
    assert math.isclose(metafopy.core.bvn.bvn_rect((0.3, -0.2), (INF, INF), 0.1), 0.23621914480234804, rel_tol=1e-14)


def test_bvn_degenerate_limits() -> None:
    assert metafopy.core.bvn.bvn_rect((-INF, -INF), (INF, INF), 0.3) == 1.0
    assert math.isclose(metafopy.core.bvn.bvn_rect((-INF, -1.0), (INF, 1.0), 0.3), 0.6826894921370859, rel_tol=1e-14)


def test_bfgs_analytic_gradient_and_parscale() -> None:
    """An analytic gradient and a parameter scale reach the minimum of a badly scaled quadratic."""
    res = metafopy.core.bfgs.optim_bfgs(
        lambda x: float((x[0] - 3) ** 2 + 1e4 * (x[1] - 0.01) ** 2),
        numpy.array([0.0, 0.0]),
        gr=lambda x: numpy.array([2 * (x[0] - 3), 2e4 * (x[1] - 0.01)]),
        parscale=[1.0, 0.01],
    )
    assert numpy.allclose(res.par, [3.0, 0.01], atol=1e-5)


LIMITS = rstrategies.finite_floats(-6.0, 6.0)
CORRELATIONS = rstrategies.finite_floats(-0.95, 0.95)
BVN_SLACK = 4 * math.ulp(1.0)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(h=LIMITS, k=LIMITS, r=CORRELATIONS)
def test_bvnu_property_symmetric_bounded_and_matches_scipy(h: float, k: float, r: float) -> None:
    """Candidate ``src/metafopy/core/bvn.py::bvnu``: symmetric in (h, k), in [0, 1], and the scipy bivariate normal upper orthant probability."""
    value = metafopy.core.bvn.bvnu(h, k, r)
    assert value == metafopy.core.bvn.bvnu(k, h, r)
    assert -BVN_SLACK <= value <= 1 + BVN_SLACK
    expected = float(scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]]).cdf([-h, -k]))
    assert value == pytest.approx(expected, abs=1e-6)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(h=LIMITS, k=LIMITS)
def test_bvnu_property_independent_at_zero_correlation(h: float, k: float) -> None:
    """Candidate ``src/metafopy/core/bvn.py::bvnu``: at r = 0 the probability is ``phi(-h) * phi(-k)``."""
    expected = float(scipy.stats.norm.sf(h) * scipy.stats.norm.sf(k))
    assert metafopy.core.bvn.bvnu(h, k, 0.0) == pytest.approx(expected, rel=1e-8, abs=1e-14)


@hypothesis.strategies.composite
def BOXES(draw: hypothesis.strategies.DrawFn) -> tuple[tuple[float, float], tuple[float, float]]:
    """A box ``(lower, upper)`` whose LIMITS may be infinite."""
    result = []
    for _ in range(2):
        a = draw(hypothesis.strategies.one_of(LIMITS, hypothesis.strategies.just(-math.inf)))
        b = draw(hypothesis.strategies.one_of(LIMITS, hypothesis.strategies.just(math.inf)))
        result.append((min(a, b), max(a, b)))
    return (result[0][0], result[1][0]), (result[0][1], result[1][1])


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(box=BOXES(), r=CORRELATIONS)
def test_bvn_rect_property_bounded_and_symmetric_under_swapping_coordinates(box: tuple[tuple[float, float], tuple[float, float]], r: float) -> None:
    """Candidate ``src/metafopy/core/bvn.py::bvn_rect``: in [0, 1] and unchanged when both coordinates are swapped."""
    (l1, l2), (u1, u2) = box
    value = metafopy.core.bvn.bvn_rect((l1, l2), (u1, u2), r)
    assert -1e-12 <= value <= 1 + 1e-12
    assert metafopy.core.bvn.bvn_rect((l2, l1), (u2, u1), r) == pytest.approx(value, abs=1e-12)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(box=BOXES(), r=CORRELATIONS)
def test_bvn_rect_property_monotone_in_the_box(box: tuple[tuple[float, float], tuple[float, float]], r: float) -> None:
    """Candidate ``src/metafopy/core/bvn.py::bvn_rect``: a box that contains another has no smaller probability; the whole plane has 1."""
    (l1, l2), (u1, u2) = box
    inner = metafopy.core.bvn.bvn_rect((l1, l2), (u1, u2), r)
    outer = metafopy.core.bvn.bvn_rect((-math.inf, l2), (u1, u2), r)
    assert outer >= inner - 1e-12
    assert metafopy.core.bvn.bvn_rect((-math.inf, -math.inf), (math.inf, math.inf), r) == 1.0
