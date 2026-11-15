"""``simulation`` cases: seeded Monte Carlo checks of results that are exact in theory.

Each test draws from ``numpy.random.default_rng(seed)`` with a fixed seed and a stated number of replicates, and asserts at ``STANDARD_ERRORS``
standard errors of the Monte Carlo mean or proportion. Only what holds exactly is tested, not asymptotic behaviour.
"""

import typing

import numpy
import numpy.typing
import pandas
import pytest

import correctness.cases
import correctness.formulas
import metafopy
import metafopy.methods.ls_permutest

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


@pytest.mark.slow
def test_random_permutation_p_values_follow_the_exact_ones() -> None:
    """The random-permutation p-values of a location-scale model lie within 4 binomial standard errors of the exact ones (2000 permutations per part).

    The exact p-value is the reference's share of all 1287 distinct permutations; the few permutations with no reference fit make it an interval,
    and the distance to that interval is what is held to the standard error. The generator is metafopy's R-exact Mersenne-Twister, seeded.
    """
    case = next(c for c in correctness.cases.load_cases('derived') if c.id == 'ls.permutest.location.exact.ML.z.bcglat')
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = frame['yi'].to_numpy(dtype=numpy.float64), frame['vi'].to_numpy(dtype=numpy.float64)
    X = numpy.column_stack([numpy.ones(len(yi)), frame['lat'].to_numpy(dtype=numpy.float64)])
    fit = metafopy.rma_uni(**case.args, data=frame)
    iterations = 2000
    drawn = metafopy.permutest(fit, iter=iterations, seed=20261117)
    assert isinstance(drawn, metafopy.methods.ls_permutest.LSPermutestResult)
    for part, actual in (('location', [*drawn.pval, drawn.QMp]), ('scale', [*drawn.pval_alpha, drawn.QSp])):
        expected = correctness.formulas.ls_exact_permutation_p(yi, vi, X, X, 'log', 'ML', part)
        low = [*numpy.asarray(expected['pval_low']), expected['omnibus_low']]
        high = [*numpy.asarray(expected['pval_high']), expected['omnibus_high']]
        for index, value in enumerate(actual):
            middle = (low[index] + high[index]) / 2
            error = float(numpy.sqrt(middle * (1 - middle) / iterations))
            distance = max(low[index] - value, value - high[index], 0.0)
            assert distance <= STANDARD_ERRORS * error, (
                f'{part} p-value {index}: {value!r} from {iterations} random permutations, exact between {low[index]!r} and {high[index]!r}, '
                f'standard error {error!r}'
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


def binomial_error(proportion: float, replicates: int) -> float:
    """The Monte Carlo standard error of a proportion near ``proportion`` over ``replicates`` replicates."""
    return float(numpy.sqrt(proportion * (1 - proportion) / replicates))


def regtest_lm_rejections(alpha: float, *, seed: int, replicates: int) -> float:
    """The share of ``replicates`` studies sets in which ``regtest(model='lm')`` rejects at ``alpha`` when the weighted normal linear model it assumes
    holds with no asymmetry: ``yi = 0.2 + e`` with ``e ~ N(0, phi vi)``, 20 studies, dispersion ``phi = 1.7`` (the model's multiplicative dispersion)."""
    rng = numpy.random.default_rng(seed)
    vi = numpy.linspace(0.02, 0.2, 20)
    phi = 1.7
    rejected = 0
    for _ in range(replicates):
        yi = rng.normal(0.2, numpy.sqrt(phi * vi))
        rejected += metafopy.regtest(yi, vi, model='lm').pval < alpha
    return rejected / replicates


@pytest.mark.slow
def test_regtest_lm_has_its_nominal_size() -> None:
    """With no asymmetry the ``regtest(model='lm')`` t-test (Egger et al. 1997) rejects at ``alpha = 0.05`` 5% of the time, within 4 binomial standard
    errors: under the weighted normal linear model with unknown dispersion the t statistic is exactly t-distributed (4000 replicates, 20 studies)."""
    replicates, alpha = 4000, 0.05
    proportion = regtest_lm_rejections(alpha, seed=20261118, replicates=replicates)
    error = binomial_error(alpha, replicates)
    assert abs(proportion - alpha) <= STANDARD_ERRORS * error, (
        f'rejection rate {proportion!r} over {replicates} replicates, expected {alpha!r}, standard error {error!r}'
    )


def tes_exact_rejections(tes_alpha: float, *, seed: int, replicates: int) -> float:
    """The share of ``replicates`` studies sets in which the exact ``tes`` test rejects at ``tes_alpha`` with ``theta`` known (``0.3``) and the studies
    drawn from it: ``yi ~ N(theta, vi)``, 12 studies with spread sampling variances, each tested two-sided at 0.05."""
    rng = numpy.random.default_rng(seed)
    vi = numpy.linspace(0.02, 0.25, 12)
    rejected = 0
    for _ in range(replicates):
        yi = rng.normal(0.3, numpy.sqrt(vi))
        result = metafopy.tes(yi, vi, theta=0.3, test='exact', tes_alpha=tes_alpha, find_lim=False)
        rejected += result.pval < tes_alpha
    return rejected / replicates


@pytest.mark.slow
def test_tes_exact_size_is_at_most_alpha() -> None:
    """With the true ``theta`` the exact test of excess significance (Ioannidis and Trikalinos 2007) rejects at most ``tes_alpha = 0.1`` of the time, to
    within 4 binomial standard errors: the number of significant studies is a sum of independent Bernoulli variables with the estimated powers, and
    the exact test is the tail of that sum, so it is conservative (4000 replicates, 12 studies)."""
    replicates, tes_alpha = 4000, 0.1
    proportion = tes_exact_rejections(tes_alpha, seed=20261119, replicates=replicates)
    error = binomial_error(tes_alpha, replicates)
    assert proportion <= tes_alpha + STANDARD_ERRORS * error, (
        f'rejection rate {proportion!r} over {replicates} replicates, expected at most {tes_alpha!r}, standard error {error!r}'
    )
