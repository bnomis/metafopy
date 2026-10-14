"""Missing values in V for rma_mv (metafor's isSymmetric() and .anyNAv())."""

import numpy
import numpy.typing
import scipy.sparse

import metafopy.models.mv_inputs

NA = numpy.nan


def v_with_missing() -> numpy.typing.NDArray[numpy.float64]:
    """Row 1 has an NA variance; row 2 a present variance but an NA covariance with row 3, whose variance is present."""
    return numpy.array(
        [
            [1.0, NA, 0.2, 0.1],
            [NA, NA, NA, NA],
            [0.2, NA, 1.0, NA],
            [0.1, NA, NA, 1.0],
        ]
    )


def test_is_symmetric_ignores_mirrored_missing_values() -> None:
    V = v_with_missing()
    assert metafopy.models.mv_inputs.is_symmetric(V)
    assert metafopy.models.mv_inputs.is_symmetric(scipy.sparse.csc_array(V))
    V[0, 1] = 0.5
    assert not metafopy.models.mv_inputs.is_symmetric(V)


def test_any_na_v_ignores_columns_of_dropped_rows() -> None:
    V = v_with_missing()
    numpy.testing.assert_array_equal(metafopy.models.mv_inputs.any_na_v(V), [False, True, True, True])
    V[2, 3] = V[3, 2] = 0.3
    numpy.testing.assert_array_equal(metafopy.models.mv_inputs.any_na_v(V), [False, True, False, False])
    numpy.testing.assert_array_equal(metafopy.models.mv_inputs.any_na_v(scipy.sparse.csc_array(V)), [False, True, False, False])
