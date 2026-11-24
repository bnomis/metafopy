"""The Newton polish of the log-link location-scale fit (``metafopy.models.rma_ls_opt.polish_log``) and the score and information it uses."""

import collections.abc
import typing
import warnings

import correctness.formulas
import numpy
import numpy.typing
import pandas
import pytest

import metafopy
import metafopy.models.rma_ls_fit
import metafopy.models.rma_ls_opt

type FloatArray = numpy.typing.NDArray[numpy.float64]

DRAWS = 5
NOISE = 1e-15
AGREEMENT = 1e-8
METHODS = ('ML', 'REML')


def bcg_frame(perturb: int | None) -> pandas.DataFrame:
    frame = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    if perturb is not None:
        draw = numpy.random.default_rng(perturb).standard_normal(len(frame))
        frame['yi'] = frame['yi'].to_numpy(dtype=float) * (1 + NOISE * draw)
    return frame


def location_vif(frame: pandas.DataFrame, method: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.rma(yi='yi', vi='vi', mods='~ ablat + year', scale='~ ablat + year', method=method, link='log', data=frame)
    vifs: typing.Any = metafopy.vif(fit)
    return float(vifs.beta.vif[0].vif)


def reference_vif(method: str) -> float:
    frame = bcg_frame(None)
    yi, vi = frame['yi'].to_numpy(dtype=float), frame['vi'].to_numpy(dtype=float)
    X = numpy.column_stack([numpy.ones(len(yi)), frame['ablat'].to_numpy(dtype=float), frame['year'].to_numpy(dtype=float)])
    ref = correctness.formulas.ls_reference_fit(yi, vi, X, X.copy(), 'log', method)
    return float(correctness.formulas.gvif(ref['vb'], [0]))


def test_location_vif_of_a_flat_reml_fit_does_not_depend_on_rounding_noise() -> None:
    """``ls.vif.REML.bcg2``: the stop of BFGS and Nelder-Mead moves the location vif by up to 2e-6 relative when ``yi`` changes by 1e-15."""
    unperturbed = location_vif(bcg_frame(None), 'REML')
    expected = reference_vif('REML')
    assert unperturbed == pytest.approx(expected, rel=AGREEMENT)
    for draw in range(1, DRAWS + 1):
        vif = location_vif(bcg_frame(draw), 'REML')
        assert vif == pytest.approx(unperturbed, rel=AGREEMENT)
        assert vif == pytest.approx(expected, rel=AGREEMENT)


def problem_of(method: str) -> tuple[metafopy.models.rma_ls_fit.LSProblem, FloatArray]:
    frame = bcg_frame(None)
    yi, vi = frame['yi'].to_numpy(dtype=float), frame['vi'].to_numpy(dtype=float)
    ablat = frame['ablat'].to_numpy(dtype=float)
    Z = numpy.column_stack([numpy.ones(len(yi)), (ablat - ablat.mean()) / ablat.std(ddof=1)])
    problem = metafopy.models.rma_ls_fit.LSProblem(
        yi=yi,
        vi=vi,
        X=Z.copy(),
        Z=Z,
        reml=method == 'REML',
        REMLf=True,
        link='log',
        alpha_min=numpy.full(2, -numpy.inf),
        alpha_max=numpy.full(2, numpy.inf),
        tau2_min=0.0,
        tau2_max=numpy.inf,
        optbeta=False,
    )
    return problem, numpy.array([-1.2, 0.3])


def central_difference(function: collections.abc.Callable[[FloatArray], FloatArray], alpha: FloatArray) -> FloatArray:
    step = 1e-6
    columns = []
    for j in range(alpha.shape[0]):
        up, down = alpha.copy(), alpha.copy()
        up[j] += step
        down[j] -= step
        columns.append((function(up) - function(down)) / (2 * step))
    return numpy.column_stack(columns)


@pytest.mark.parametrize('method', METHODS)
def test_log_score_is_the_gradient_of_the_loglik(method: str) -> None:
    problem, alpha = problem_of(method)
    nothing = numpy.full(2, numpy.nan)

    def loglik(a: FloatArray) -> FloatArray:
        return numpy.array([problem.loglik(a, nothing, nothing, alpha_transf=False)])

    numeric = central_difference(loglik, alpha)[0]
    assert problem.log_score(alpha) == pytest.approx(numeric, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize('method', METHODS)
def test_log_information_is_minus_the_jacobian_of_the_score(method: str) -> None:
    problem, alpha = problem_of(method)
    numeric = -central_difference(problem.log_score, alpha)
    assert problem.log_information(alpha) == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_polish_log_returns_a_degenerate_start_unchanged() -> None:
    """``tau2`` heading to 0 in one group: the information is not positive definite or the step is capped, so the start stays."""
    problem, _ = problem_of('REML')
    nothing = numpy.full(2, numpy.nan)
    start = numpy.array([-37.0, 0.0])
    polished = metafopy.models.rma_ls_opt.polish_log(
        start,
        lambda a: problem.loglik(a, nothing, nothing, alpha_transf=False),
        problem.log_score,
        problem.log_information,
        numpy.ones(2, dtype=bool),
    )
    assert numpy.array_equal(polished, start)


def test_polish_log_reaches_the_optimum_from_a_point_near_it() -> None:
    problem, alpha = problem_of('ML')
    nothing = numpy.full(2, numpy.nan)
    stop = metafopy.models.rma_ls_opt.minimise(lambda a: -problem.loglik(a, nothing, nothing, alpha_transf=False), alpha).par
    optimum = metafopy.models.rma_ls_opt.polish_log(
        stop,
        lambda a: problem.loglik(a, nothing, nothing, alpha_transf=False),
        problem.log_score,
        problem.log_information,
        numpy.ones(2, dtype=bool),
    )
    assert float(numpy.max(numpy.abs(problem.log_score(optimum)))) <= metafopy.models.rma_ls_opt.POLISH_LOG_SCORE * 10


def test_polished_fit_with_bounds_matches_the_unbounded_fit() -> None:
    """The chain rule through ``mapfun``: a box that does not bind gives the unbounded optimum."""
    frame = bcg_frame(None)
    kwargs: dict[str, typing.Any] = {'yi': 'yi', 'vi': 'vi', 'mods': '~ ablat', 'scale': '~ ablat', 'method': 'REML', 'link': 'log', 'data': frame}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        free: typing.Any = metafopy.rma(**kwargs)
        boxed: typing.Any = metafopy.rma(**kwargs, control={'alpha_min': [-20, -20], 'alpha_max': [20, 20]})
    assert numpy.asarray(boxed.alpha) == pytest.approx(numpy.asarray(free.alpha), rel=1e-4)
