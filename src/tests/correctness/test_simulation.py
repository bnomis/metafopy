"""``simulation`` cases: seeded Monte Carlo checks of results that are exact in theory.

Each test draws from ``numpy.random.default_rng(seed)`` with a fixed seed and a stated number of replicates, and asserts at ``STANDARD_ERRORS``
standard errors of the Monte Carlo mean or proportion. Only what holds exactly is tested, not asymptotic behaviour.
"""

import typing

import numpy
import numpy.typing
import pandas
import pytest

import metafopy

type FloatArray = numpy.typing.NDArray[numpy.float64]

STANDARD_ERRORS = 4
# the truncation of tau2 at zero is metafopy's ``tau2_min`` control setting; a very negative value turns it off
UNTRUNCATED = {'tau2_min': -1e6}


def assert_mean_within(name: str, draws: FloatArray, truth: float) -> None:
    """The Monte Carlo mean of ``draws`` is within ``STANDARD_ERRORS`` of its standard error from ``truth``."""
    mean = float(numpy.mean(draws))
    error = float(numpy.std(draws, ddof=1) / numpy.sqrt(len(draws)))
    assert abs(mean - truth) <= STANDARD_ERRORS * error, f'{name}: mean {mean!r} over {len(draws)} replicates, truth {truth!r}, standard error {error!r}'


def tau2_estimates(method: str, control: dict[str, float], *, seed: int, replicates: int) -> FloatArray:
    """``tau2`` of ``replicates`` random-effects meta-analyses of 50 studies with true ``tau2 = 0.04``, as ``method`` estimates it.

    The sampling variances are near 1 so that an untruncated estimate below ``-min(vi)``, which the library rejects as a negative marginal variance, is
    some four standard deviations out and does not occur in the seeded draws.
    """
    rng = numpy.random.default_rng(seed)
    vi = numpy.linspace(0.8, 1.2, 50)
    tau2 = 0.04
    estimates = numpy.empty(replicates)
    for index in range(replicates):
        yi = rng.normal(0.3, numpy.sqrt(vi + tau2))
        estimates[index] = metafopy.rma_uni(yi=yi, vi=vi, method=method, control=control).tau2
    return estimates


@pytest.mark.slow
@pytest.mark.parametrize('method', ['DL', 'HE'])
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_untruncated_tau2_is_unbiased(method: str) -> None:
    """The DerSimonian-Laird (1986) and Hedges (1983) estimators are unbiased for ``tau2`` until they are truncated at zero (2000 replicates)."""
    assert_mean_within(f'{method} without truncation', tau2_estimates(method, UNTRUNCATED, seed=20261112, replicates=2000), 0.04)


@pytest.mark.slow
def test_hedges_g_is_unbiased() -> None:
    """Hedges' ``g`` with the exact ``J`` (Hedges, 1981) is unbiased for the standardised mean difference (200000 replicates, 8 per group)."""
    rng = numpy.random.default_rng(20261113)
    replicates, n, delta = 200_000, 8, 0.5
    frame = pandas.DataFrame(
        {
            'm1i': rng.normal(delta, 1 / numpy.sqrt(n), replicates),
            'm2i': rng.normal(0.0, 1 / numpy.sqrt(n), replicates),
            'sd1i': numpy.sqrt(rng.chisquare(n - 1, replicates) / (n - 1)),
            'sd2i': numpy.sqrt(rng.chisquare(n - 1, replicates) / (n - 1)),
            'n1i': n,
            'n2i': n,
        }
    )
    g = metafopy.escalc(measure='SMD', data=frame, m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i')
    assert_mean_within('Hedges g', numpy.asarray(g['yi'], dtype=float), delta)


@pytest.mark.slow
def test_equal_effects_interval_covers_exactly() -> None:
    """With known sampling variances the 95% equal-effects interval covers the true effect 95% of the time (4000 replicates, 10 studies)."""
    rng = numpy.random.default_rng(20261114)
    vi = numpy.linspace(0.05, 0.3, 10)
    replicates, truth, level = 4000, 0.3, 0.95
    covered = 0
    for _ in range(replicates):
        fit = metafopy.rma_uni(yi=rng.normal(truth, numpy.sqrt(vi)), vi=vi, method='EE')
        covered += bool(fit.ci_lb <= truth <= fit.ci_ub)
    proportion = covered / replicates
    error = float(numpy.sqrt(level * (1 - level) / replicates))
    assert abs(proportion - level) <= STANDARD_ERRORS * error, (
        f'coverage {proportion!r} over {replicates} replicates, expected {level!r}, standard error {error!r}'
    )


COEFFICIENTS = ('intercept', 'slope')
TRUTH = numpy.array([0.3, -0.2])


class FixedEffectsDraws(typing.NamedTuple):
    """The estimates and 95% interval bounds of each coefficient over the replicates, one row per replicate."""

    beta: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray


def fixed_effects_draws(*, seed: int, replicates: int) -> FixedEffectsDraws:
    """``replicates`` fits of the ``method='FE'`` model ``y = 0.3 - 0.2 x + e`` with known, correlated ``e ~ N(0, V)``.

    Eight clusters of three rows; within a cluster the correlation is 0.5 and the variances differ, so ``V`` is block diagonal and not a multiple
    of the identity.
    """
    rng = numpy.random.default_rng(seed)
    clusters, size = 8, 3
    variances = numpy.linspace(0.05, 0.25, clusters * size)
    sd = numpy.sqrt(variances)
    V = numpy.zeros((clusters * size, clusters * size))
    for cluster in range(clusters):
        rows = slice(cluster * size, (cluster + 1) * size)
        block = 0.5 * numpy.outer(sd[rows], sd[rows])
        numpy.fill_diagonal(block, variances[rows])
        V[rows, rows] = block
    frame = pandas.DataFrame({'x': numpy.linspace(-1, 1, clusters * size)})
    mean = TRUTH[0] + TRUTH[1] * frame['x'].to_numpy()
    factor = numpy.linalg.cholesky(V)
    beta, ci_lb, ci_ub = (numpy.empty((replicates, 2)) for _ in range(3))
    for index in range(replicates):
        frame['yi'] = mean + factor @ rng.standard_normal(len(mean))
        fit = metafopy.rma_mv(yi='yi', V=V, mods='~ x', data=frame, method='FE')
        beta[index], ci_lb[index], ci_ub[index] = fit.beta, fit.ci_lb, fit.ci_ub
    return FixedEffectsDraws(beta, ci_lb, ci_ub)


@pytest.mark.slow
def test_gls_estimate_is_unbiased() -> None:
    """The generalised least squares estimate of a fixed-effects ``rma_mv`` model with known correlated errors is unbiased for each coefficient
    (Aitken 1935; 2000 replicates)."""
    draws = fixed_effects_draws(seed=20261115, replicates=2000)
    for position, name in enumerate(COEFFICIENTS):
        assert_mean_within(f'GLS {name}', draws.beta[:, position], float(TRUTH[position]))


@pytest.mark.slow
def test_fixed_effects_interval_covers_exactly_with_known_v() -> None:
    """With known ``V`` the 95% Wald interval of each coefficient of the ``method='FE'`` fit covers its true value 95% of the time: the estimate is
    exactly normal with the variance the fit reports (2000 replicates)."""
    replicates, level = 2000, 0.95
    draws = fixed_effects_draws(seed=20261116, replicates=replicates)
    covered = ((draws.ci_lb <= TRUTH) & (draws.ci_ub >= TRUTH)).mean(axis=0)
    error = float(numpy.sqrt(level * (1 - level) / replicates))
    for position, name in enumerate(COEFFICIENTS):
        assert abs(covered[position] - level) <= STANDARD_ERRORS * error, (
            f'{name}: coverage {covered[position]!r} over {replicates} replicates, expected {level!r}, standard error {error!r}'
        )
