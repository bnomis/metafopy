"""``formulas.ls_reference_fit`` reaches the highest regular interior maximum of a permuted location-scale model."""

import itertools

import numpy
import pytest

import correctness.formulas

TPOS = [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27]
TNEG = [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886]
CPOS = [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29]
CNEG = [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825]
LAT = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

# permutation index (0-based, in ``distinct_label_orders`` order of ``LAT``) -> log-likelihood of the highest interior maximum
# (the best of 40 random starts per permutation, 2026-11-25)
HIGHER_MAXIMUM = {
    68: -12.57323737,
    264: -11.64011395,
    433: -11.96789514,
    651: -12.60144281,
    805: -12.13242026,
    979: -11.67846163,
    1110: -12.58432080,
    1278: -12.17150179,
}


def permuted_fit(index: int) -> dict[str, object]:
    ai, bi, ci, di = (numpy.array(v, dtype=float) for v in (TPOS, TNEG, CPOS, CNEG))
    n1, n2 = ai + bi, ci + di
    yi = numpy.log(ai / n1) - numpy.log(ci / n2)
    vi = 1 / ai - 1 / n1 + 1 / ci - 1 / n2
    order = next(itertools.islice(correctness.formulas.distinct_label_orders(sorted(LAT)), index, None))
    lat = numpy.array(order, dtype=float)
    X = numpy.column_stack([numpy.ones(13), lat])
    Z = numpy.column_stack([numpy.ones(13), numpy.array(LAT, dtype=float)])
    return correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'ML')


@pytest.mark.parametrize('index', sorted(HIGHER_MAXIMUM))
def test_reference_fit_reaches_the_higher_maximum(index: int) -> None:
    assert permuted_fit(index)['ll'] == pytest.approx(HIGHER_MAXIMUM[index], abs=1e-7)
