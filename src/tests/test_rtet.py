"""Property tests of the tetrachoric correlation of one 2x2 table."""

import math

import hypothesis
import hypothesis.strategies
import pytest

import metafopy.effects.rtet

counts = hypothesis.strategies.integers(min_value=0, max_value=40)
# Nelder-Mead stops at its own tolerance, so a table and its mirror image give estimates that agree only to about this much.
MIRROR_TOLERANCE = 1e-3


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(ai=counts, bi=counts, ci=counts, di=counts)
@pytest.mark.filterwarnings('ignore:Could not estimate')
def test_rtet_property_estimate_bounded_variance_non_negative_and_sign_flips_under_swaps(ai: int, bi: int, ci: int, di: int) -> None:
    """Candidate ``src/metafopy/effects/rtet.py::rtet``: ``yi`` in [-1, 1], ``vi`` non-negative or NA, and swapping rows or columns flips ``yi``."""
    yi, vi = metafopy.effects.rtet.rtet(ai, bi, ci, di)
    hypothesis.assume(not math.isnan(yi))
    assert -1.0 <= yi <= 1.0
    assert math.isnan(vi) or vi >= 0.0
    for swapped in ((ci, di, ai, bi), (bi, ai, di, ci)):
        flipped, _ = metafopy.effects.rtet.rtet(*swapped)
        hypothesis.assume(not math.isnan(flipped))
        assert flipped == pytest.approx(-yi, abs=MIRROR_TOLERANCE)
