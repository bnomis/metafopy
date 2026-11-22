"""Property tests for metafopy.methods.ls_common (T2 candidate center_stars)."""

import math

import hypothesis
import hypothesis.strategies

import metafopy.methods.ls_common
import metafopy.printing


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    pvals=hypothesis.strategies.lists(
        hypothesis.strategies.floats(min_value=0.0, max_value=1.0) | hypothesis.strategies.sampled_from([0.001, 0.01, 0.05, 0.1, math.nan]),
        max_size=8,
    )
)
def test_center_stars_centres_the_codes_in_one_width(pvals: list[float]) -> None:
    """methods/ls_common.py::center_stars: every cell has the common width, holds its code, and an odd space falls on the right."""
    out = metafopy.methods.ls_common.center_stars(pvals)
    assert len(out) == len(pvals)
    assert len({len(s) for s in out}) <= 1
    for p, cell in zip(pvals, out, strict=True):
        stars = '' if math.isnan(p) else metafopy.printing.signif_stars(p)
        assert cell.strip() == stars
        if stars:
            left = len(cell) - len(cell.lstrip())
            right = len(cell) - len(cell.rstrip())
            assert right - left in (0, 1)
