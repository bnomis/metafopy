"""The studentized range port must reproduce R's ptukey()/qtukey() including qtukey's 1e-4 stopping rule."""

import math

import pytest

import metafopy.core.tukey

# R: ptukey(q, nmeans, df, lower.tail = FALSE)
PTUKEY = (
    (3.5, 4, 17, 0.10035225382452329),
    (2, 3, math.inf, 0.33349932504014979),
    (0.5, 2, 5, 0.73809260189397108),
    (6, 10, 40, 0.0044612259640748331),
    (4.2, 3.5, 120, 0.014066105206207435),
    (3, 5, 3000, 0.21116097102210762),
    (2, 3, 30000, 0.33349932504014979),
    (9, 5, 10, 0.00060614902739197785),
)
# R: qtukey(p, nmeans, df)
QTUKEY = (
    (0.95, 3, 9, 3.9484922022687434),
    (0.95, 3.5, math.inf, 3.4900430372296976),
    (0.99, 6, 25, 5.3467873126681358),
    (0.5, 2, 10, 0.98968391110987097),
    (0.9, 4, 2.5, 5.7668044719689773),
    (0.95, 20, 200, 5.0801223682304109),
)


@pytest.mark.parametrize(('q', 'nmeans', 'df', 'expected'), PTUKEY)
def test_ptukey_matches_r(q: float, nmeans: float, df: float, expected: float) -> None:
    assert metafopy.core.tukey.ptukey(q, nmeans, df, lower_tail=False) == pytest.approx(expected, rel=1e-12)
    assert metafopy.core.tukey.ptukey(q, nmeans, df) == pytest.approx(1 - expected, abs=1e-12)


@pytest.mark.parametrize(('p', 'nmeans', 'df', 'expected'), QTUKEY)
def test_qtukey_matches_r(p: float, nmeans: float, df: float, expected: float) -> None:
    assert metafopy.core.tukey.qtukey(p, nmeans, df) == pytest.approx(expected, rel=1e-12)


def test_boundaries() -> None:
    assert metafopy.core.tukey.ptukey(0, 3, 10) == 0
    assert metafopy.core.tukey.ptukey(-1, 3, 10, lower_tail=False) == 1
    assert metafopy.core.tukey.ptukey(math.inf, 3, 10) == 1
    assert metafopy.core.tukey.ptukey(math.inf, 3, 10, lower_tail=False) == 0
    assert metafopy.core.tukey.qtukey(0, 3, 10) == 0
    assert metafopy.core.tukey.qtukey(1, 3, 10) == math.inf


@pytest.mark.parametrize('args', [(1.0, 3, 1.5), (1.0, 1, 10), (math.nan, 3, 10), (1.0, 3, math.nan)])
def test_invalid_arguments_give_nan(args: tuple[float, float, float]) -> None:
    assert math.isnan(metafopy.core.tukey.ptukey(*args))


def test_qtukey_invalid_arguments_give_nan() -> None:
    assert math.isnan(metafopy.core.tukey.qtukey(0.95, 3, 1.5))
    assert math.isnan(metafopy.core.tukey.qtukey(1.5, 3, 10))
    assert math.isnan(metafopy.core.tukey.qtukey(math.nan, 3, 10))
