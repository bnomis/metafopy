"""Property tests of the Farebrother and Ruben port: bounds, monotonicity and the chi-square oracle."""

import hypothesis
import hypothesis.strategies
import pytest
import rstrategies
import scipy.stats

import metafopy.core.farebrother

FAREBROTHER_TOLERANCE = 1e-6
LAMBDAS = hypothesis.strategies.lists(hypothesis.strategies.floats(min_value=0.1, max_value=5.0), min_size=1, max_size=4)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(q=rstrategies.finite_floats(0.01, 100.0), lam=LAMBDAS)
def test_farebrother_property_bounds_and_non_increasing_in_q(q: float, lam: list[float]) -> None:
    """Candidate ``src/metafopy/core/farebrother.py::farebrother``: the upper tail is in [0, 1] and non-increasing in q."""
    tail = metafopy.core.farebrother.farebrother(q, lam).Qq
    assert -FAREBROTHER_TOLERANCE <= tail <= 1 + FAREBROTHER_TOLERANCE
    assert metafopy.core.farebrother.farebrother(q + 1.0, lam).Qq <= tail + FAREBROTHER_TOLERANCE


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(q=rstrategies.finite_floats(0.01, 100.0), k=hypothesis.strategies.integers(min_value=1, max_value=20))
def test_farebrother_property_single_lambda_one_is_chi_square_tail(q: float, k: int) -> None:
    """Candidate ``src/metafopy/core/farebrother.py::farebrother``: one lambda of 1 with ``k`` degrees of freedom is the chi-square upper tail."""
    tail = metafopy.core.farebrother.farebrother(q, [1.0], [k]).Qq
    assert tail == pytest.approx(float(scipy.stats.chi2.sf(q, k)), abs=FAREBROTHER_TOLERANCE)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(c=rstrategies.finite_floats(0.01, 100.0), k=hypothesis.strategies.integers(min_value=1, max_value=20))
def test_ruben_property_single_lambda_one_is_chi_square_cdf(c: float, k: int) -> None:
    """Candidate ``src/metafopy/core/farebrother.py::ruben``: the returned ``P(Q < c)`` is in [0, 1] and equals the chi-square cdf for lambda 1."""
    _, ifault, res = metafopy.core.farebrother.ruben([1.0], [k], [0.0], 1, c, 1.0, 100000, 1e-10)
    # ifault 5 flags a series sum marginally outside [0, 1] (about -3e-17 for c = 0.0625, k = 16). R: CompQuadForm::farebrother(0.0625, 1, 16)
    # gives ifault 5 and Qq 1, so R keeps the property only within the tolerance.
    assert ifault in (0, 5)
    assert -FAREBROTHER_TOLERANCE <= res <= 1 + FAREBROTHER_TOLERANCE
    assert res == pytest.approx(float(scipy.stats.chi2.cdf(c, k)), abs=FAREBROTHER_TOLERANCE)
