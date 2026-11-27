"""print() of a labelled matrix against base R 4.6 (``print(matrix(c(1, 2.5, 3, 4), 2))``)."""

import math

import hypothesis
import hypothesis.strategies
import numpy
import pandas
import refutil
import rstrategies
import scipy.sparse

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


# The expected texts below are R 4.6's output (metafor 5.2-1, Matrix), run on 2026-11-23 with LC_ALL=C.UTF-8; R pads lines with spaces that
# refutil.r_print strips from both sides.
def test_rframe_text_prints_as_r_data_frame() -> None:
    """Candidate ``src/metafopy/core/labelled.py::RFrame.text``, a frame without digits: R's
    ``print(data.frame(a = c(1, 2.5, -0.125), b = c(3L, 4L, 10L), row.names = c("x", "y", "z")))``."""
    frame = metafopy.core.labelled.RFrame({'a': [1.0, 2.5, -0.125], 'b': [3, 4, 10]}, index=['x', 'y', 'z'])
    assert refutil.r_print(frame.text()) == '       a  b\nx  1.000  3\ny  2.500  4\nz -0.125 10'


def test_rframe_text_rounds_to_the_digits_of_the_result() -> None:
    """Candidate ``src/metafopy/core/labelled.py::RFrame.text``, a frame with ``attrs['digits']``: R's
    ``print(data.frame(est = round(c(0.123456, -1.98765), 2), n = c(12L, 7L), row.names = c("s1", "s2")))``."""
    frame = metafopy.core.labelled.RFrame({'est': [0.123456, -1.98765], 'n': [12, 7]}, index=['s1', 's2'])
    frame.attrs['digits'] = 2
    assert refutil.r_print(frame.text()) == '     est  n\ns1  0.12 12\ns2 -1.99  7'


def test_rframe_text_of_a_vcovmat_prints_zeros_as_dots() -> None:
    """Candidate ``src/metafopy/core/labelled.py::RFrame.text``, a frame with ``attrs['vcovmat']``: metafor's ``print.vcovmat`` of
    ``structure(matrix(c(0.40191, 0, 0, 0.25), 2, dimnames = list(c("1", "2"), c("1", "2"))), class = c("vcovmat", "matrix"))``."""
    names = [str(i) for i in range(1, 3)]
    frame = metafopy.core.labelled.RFrame(pandas.DataFrame([[0.40191, 0.0], [0.0, 0.25]], index=names, columns=names))
    frame.attrs['vcovmat'] = True
    assert refutil.r_print(frame.text()).strip('\n') == '       1      2\n1 0.4019      .\n2      . 0.2500'


def test_sparse_vcovmat_text_prints_as_r_dgcmatrix() -> None:
    """Candidate ``src/metafopy/core/labelled.py::SparseVCovMat.text``: R's
    ``print(Matrix::sparseMatrix(i = c(1, 2, 1, 2, 3), j = c(1, 1, 2, 2, 3), x = c(0.5, 0.1, 0.1, 0.25, 1)))``."""
    matrix = metafopy.core.labelled.SparseVCovMat(scipy.sparse.csr_array([[0.5, 0.1, 0.0], [0.1, 0.25, 0.0], [0.0, 0.0, 1.0]]))
    expected = '3 x 3 sparse Matrix of class "dgCMatrix"\n\n[1,] 0.5 0.10 .\n[2,] 0.1 0.25 .\n[3,] .   .    1'
    assert refutil.r_print(matrix.text()) == expected


MAGNITUDES = hypothesis.strategies.builds(
    lambda mantissa, exponent, negative: float(f'{"-" if negative else ""}{mantissa}e{exponent}'),
    hypothesis.strategies.integers(min_value=1, max_value=9_999_999),
    hypothesis.strategies.integers(min_value=-8, max_value=8),
    hypothesis.strategies.booleans(),
)
COLUMNS = hypothesis.strategies.lists(MAGNITUDES, min_size=1, max_size=6)
DIGITS_COUNTS = hypothesis.strategies.integers(min_value=1, max_value=10)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(values=COLUMNS, digits=DIGITS_COUNTS)
def test_decimals_keeps_the_significant_digits(values: list[float], digits: int) -> None:
    """Candidate ``src/metafopy/core/labelled.py::decimals``."""
    places = metafopy.core.labelled.decimals(values, digits)
    assert places >= 0
    for x in values:
        bound = 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - digits + 1)
        assert abs(float(f'{x:.{places}f}') - x) <= bound * (1 + 1e-9)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    values=hypothesis.strategies.lists(MAGNITUDES | hypothesis.strategies.sampled_from([math.nan, math.inf, -math.inf, 0.0]), min_size=0, max_size=6),
    digits=DIGITS_COUNTS,
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
    digits=DIGITS_COUNTS,
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
