"""print() of a labelled matrix against base R 4.6 (``print(matrix(c(1, 2.5, 3, 4), 2))``)."""

import math

import hypothesis
import hypothesis.strategies
import numpy
import rstrategies

import metafopy.core.labelled

VALUES = numpy.array([[1.0, 3.0], [2.5, 4.0]])


def test_str_without_names() -> None:
    expected = '     [,1] [,2]\n[1,]  1.0    3\n[2,]  2.5    4'
    assert str(metafopy.core.labelled.LabelledMatrix(VALUES)).strip('\n') == expected
    assert metafopy.core.labelled.matrix_text(VALUES).strip('\n') == expected


def test_str_with_names() -> None:
    out = metafopy.core.labelled.LabelledMatrix(VALUES, ['top', 'bottom'], ['left', 'right'])
    assert str(out).strip('\n') == '       left right\ntop     1.0     3\nbottom  2.5     4'


def test_str_of_a_vector_is_numpys() -> None:
    out = metafopy.core.labelled.LabelledMatrix(VALUES)[0]
    assert str(out) == str(numpy.asarray(out))


magnitudes = hypothesis.strategies.builds(
    lambda mantissa, exponent, negative: float(f'{"-" if negative else ""}{mantissa}e{exponent}'),
    hypothesis.strategies.integers(min_value=1, max_value=9_999_999),
    hypothesis.strategies.integers(min_value=-8, max_value=8),
    hypothesis.strategies.booleans(),
)
columns = hypothesis.strategies.lists(magnitudes, min_size=1, max_size=6)
digits_counts = hypothesis.strategies.integers(min_value=1, max_value=10)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(values=columns, digits=digits_counts)
def test_decimals_keeps_the_significant_digits(values: list[float], digits: int) -> None:
    """Candidate ``src/metafopy/core/labelled.py::decimals``."""
    places = metafopy.core.labelled.decimals(values, digits)
    assert places >= 0
    for x in values:
        bound = 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - digits + 1)
        assert abs(float(f'{x:.{places}f}') - x) <= bound * (1 + 1e-9)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    values=hypothesis.strategies.lists(magnitudes | hypothesis.strategies.sampled_from([math.nan, math.inf, -math.inf, 0.0]), min_size=0, max_size=6),
    digits=digits_counts,
)
def test_format_real_marks_missing_and_infinite_and_shares_decimals(values: list[float], digits: int) -> None:
    """Candidate ``src/metafopy/core/labelled.py::format_real``."""
    out = metafopy.core.labelled.format_real(values, digits)
    assert len(out) == len(values)
    fixed: list[str] = []
    for x, text in zip(values, out, strict=True):
        if math.isnan(x):
            assert text == 'NA'
        elif math.isinf(x):
            assert text == ('Inf' if x > 0 else '-Inf')
        elif 'e' not in text:
            fixed.append(text)
    assert len({len(text.partition('.')[2]) for text in fixed}) <= 1


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    mantissas=hypothesis.strategies.lists(hypothesis.strategies.integers(min_value=1, max_value=9_999_999), min_size=1, max_size=5),
    exponents=hypothesis.strategies.lists(hypothesis.strategies.integers(min_value=-12, max_value=12), min_size=5, max_size=5),
    digits=digits_counts,
)
def test_scientific_places_is_unchanged_by_a_power_of_ten(mantissas: list[int], exponents: list[int], digits: int) -> None:
    """Candidate ``src/metafopy/core/labelled.py::scientific_places``."""
    values = [float(f'{m}e{e}') for m, e in zip(mantissas, exponents, strict=False)]
    scaled = [float(f'{m}e{e + 1}') for m, e in zip(mantissas, exponents, strict=False)]
    places = metafopy.core.labelled.scientific_places(values, digits)
    assert places is None or places >= 0
    again = metafopy.core.labelled.scientific_places(scaled, digits)
    if places is not None and again is not None:
        assert places == again


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e6, 1e6), digits=hypothesis.strategies.integers(min_value=0, max_value=8))
def test_trim_zeros_is_idempotent_and_keeps_the_value(x: float, digits: int) -> None:
    """Candidate ``src/metafopy/core/labelled.py::trim_zeros``."""
    text = f'{x:.{digits}f}'
    once = metafopy.core.labelled.trim_zeros(text, digits)
    assert metafopy.core.labelled.trim_zeros(once, digits) == once
    if '.' in text:
        assert float(once) == float(text)
