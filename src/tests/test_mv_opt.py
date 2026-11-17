"""Tests of the Newton polish that follows the BFGS stop of :func:`metafopy.models.mv_opt.maximise`."""

import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing
import pytest
import test_rma_mv

import metafopy
import metafopy.models.mv_lik
import metafopy.models.mv_opt

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True)
class Stop:
    problem: metafopy.models.mv_lik.Problem
    par: FloatArray
    free: numpy.typing.NDArray[numpy.intp]
    reml: bool
    fit: metafopy.RMAMVResult


def capture(monkeypatch: pytest.MonkeyPatch, fit: collections.abc.Callable[[], metafopy.RMAMVResult], polish: bool = True) -> Stop:
    """Run ``fit`` and keep what ``maximise`` was called with and the parameters it returned."""
    seen: list[tuple[typing.Any, ...]] = []
    original = metafopy.models.mv_opt.maximise
    if not polish:
        monkeypatch.setattr(
            metafopy.models.mv_opt,
            'polish_score',
            lambda problem, par, free, reml: (par, metafopy.models.mv_lik.negative_loglik(problem, par, reml)),
        )

    def wrapper(
        problem: metafopy.models.mv_lik.Problem,
        par0: FloatArray,
        free: numpy.typing.NDArray[numpy.intp],
        reml: bool,
        **kwargs: typing.Any,
    ) -> metafopy.models.mv_opt.OptResult:
        result = original(problem, par0, free, reml, **kwargs)
        seen.append((problem, free, reml, result))
        return result

    monkeypatch.setattr(metafopy.models.mv_opt, 'maximise', wrapper)
    res = fit()
    problem, free, reml, result = seen[-1]
    return Stop(problem=problem, par=result.par, free=free, reml=reml, fit=res)


def kon(**control: typing.Any) -> metafopy.RMAMVResult:
    return test_rma_mv.konst(mods='~ year', method='REML', control=control or None)


def largest_gradient(stop: Stop) -> float:
    return float(numpy.max(numpy.abs(metafopy.models.mv_lik.gradient(stop.problem, stop.par, stop.reml, stop.free))))


def test_polish_finishes_an_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    early = capture(monkeypatch, lambda: kon(gtol=1e-4))
    assert largest_gradient(early) <= metafopy.models.mv_opt.POLISH_GRADIENT
    default = capture(monkeypatch, kon)
    numpy.testing.assert_allclose(numpy.asarray(early.fit.sigma2), numpy.asarray(default.fit.sigma2), rtol=1e-9, atol=0)


def test_polish_takes_no_step_at_the_optimum(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = capture(monkeypatch, lambda: kon(gtol=1e-4))
    par, _ = metafopy.models.mv_opt.polish_score(stop.problem, stop.par, stop.free, stop.reml)
    assert numpy.array_equal(par, stop.par)


def test_polish_is_capped_on_a_boundary_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = capture(monkeypatch, test_rma_mv.CASES['sim_R_known'], polish=False)
    par, _ = metafopy.models.mv_opt.polish_score(stop.problem, stop.par, stop.free, stop.reml)
    assert numpy.array_equal(par, stop.par)


def test_score_jacobian_is_the_second_difference_of_the_likelihood(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = capture(monkeypatch, kon)
    hessian = metafopy.models.mv_opt.score_jacobian(stop.problem, stop.par, stop.free, stop.reml)
    assert hessian.shape == (stop.free.size, stop.free.size)

    def value(par: FloatArray) -> float:
        return metafopy.models.mv_lik.negative_loglik(stop.problem, par, stop.reml)

    step = 1e-3
    expected = numpy.empty_like(hessian)
    for i, a in enumerate(stop.free):
        for j, b in enumerate(stop.free):
            points = []
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                par = stop.par.copy()
                par[a] += sa * step
                par[b] += sb * step
                points.append(value(par))
            expected[i, j] = (points[0] - points[1] - points[2] + points[3]) / (4 * step * step)
    numpy.testing.assert_allclose(hessian, expected, rtol=1e-5, atol=1e-5 * float(numpy.max(numpy.abs(expected))))
