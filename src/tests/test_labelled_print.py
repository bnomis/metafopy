"""print() of a labelled matrix against base R 4.6 (``print(matrix(c(1, 2.5, 3, 4), 2))``)."""

import numpy

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
