"""Property tests for the number text of ``metafopy.models.mv_factor``."""

import hypothesis
import rstrategies

import metafopy.models.mv_factor


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e300, 1e300))
def test_format_number_is_fifteen_digits_and_integers_are_plain(x: float) -> None:
    """Candidate ``src/metafopy/models/mv_factor.py::format_number``."""
    text = metafopy.models.mv_factor.format_number(x)
    assert abs(float(text) - x) <= 1e-14 * abs(x)
    if x == int(x) and abs(x) < 1e15:
        assert text == str(int(x))
