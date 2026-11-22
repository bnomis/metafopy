import numpy
import pandas
import pytest

import metafopy
import metafopy.metadata


def test_make_unique_matches_r() -> None:
    assert metafopy.metadata.make_unique(['a', 'b', 'a', 'a', 'a.1']) == ('a', 'b', 'a.2', 'a.3', 'a.1')


def test_defaults_add_intercept_only() -> None:
    md = metafopy.MetaData.from_inputs([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    assert md.k == 3
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    assert md.X_names == ('intrcpt',)
    numpy.testing.assert_array_equal(md.X, numpy.ones((3, 1)))
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert md.slab == ('1', '2', '3')


def test_sei_squared_to_vi() -> None:
    md = metafopy.MetaData.from_inputs([0.1, 0.2], sei=[0.1, 0.2])
    numpy.testing.assert_allclose(md.vi, [0.01, 0.04])


def test_escalc_style_frame_is_picked_up() -> None:
    dat = pandas.DataFrame({'yi': [0.1, 0.2], 'vi': [0.01, 0.02], 'x': [1.0, 2.0]})
    md = metafopy.MetaData.from_inputs(data=dat, mods='~ x')
    assert md.X_names == ('intrcpt', 'x')


def test_matrix_mods_get_intercept_unless_disabled() -> None:
    x = numpy.array([1.0, 2.0, 3.0])
    assert metafopy.MetaData.from_inputs([1, 2, 3], [1, 1, 1], mods=x).X_names == ('intrcpt', 'mods')
    md = metafopy.MetaData.from_inputs([1, 2, 3], [1, 1, 1], mods=x, intercept=False)
    assert md.X_names == ('mods',)
    assert not md.intercept


def test_formula_without_intercept() -> None:
    dat = pandas.DataFrame({'yi': [1.0, 2.0, 3.0], 'vi': [1.0, 1.0, 1.0], 'g': ['a', 'b', 'a']})
    md = metafopy.MetaData.from_inputs(data=dat, mods='~ 0 + g')
    assert md.X_names == ('ga', 'gb')
    assert not md.intercept


def test_na_omit_then_subset_bookkeeping() -> None:
    md = metafopy.MetaData.from_inputs([0.1, numpy.nan, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1], subset=[True, True, True, False])
    assert md.k == 2
    assert md.k_all == 4
    numpy.testing.assert_array_equal(md.keep, [True, False, True, False])
    assert md.slab == ('1', '3')


def test_na_fail() -> None:
    with pytest.raises(ValueError, match='missing'):
        metafopy.MetaData.from_inputs([0.1, numpy.nan], [0.1, 0.1], na_action='fail')


def test_duplicate_slabs_made_unique() -> None:
    md = metafopy.MetaData.from_inputs([1, 2], [1, 1], slab=['Smith', 'Smith'])
    assert md.slab == ('Smith', 'Smith.1')


def test_column_name_without_data_errors() -> None:
    with pytest.raises(ValueError, match='no data'):
        metafopy.MetaData.from_inputs('yi', 'vi')


def test_row_level_views_keep_subset_and_missing_rows() -> None:
    # the fit uses rows 1 and 3; metafor's ni.f and weights.f keep the missing row 2 and drop the subset-out row 4
    md = metafopy.MetaData.from_inputs(
        [0.1, numpy.nan, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1], ni=[10, 20, 30, 40], weights=[1, 2, 3, 4], subset=[True, True, True, False]
    )
    assert md.names_all == ('1', '2', '3', '4')
    assert md.ni_f is not None
    assert md.weights_f is not None
    numpy.testing.assert_array_equal(md.ni_f, [10, 20, 30])
    numpy.testing.assert_array_equal(md.weights_f, [1, 2, 3])


def test_row_level_views_without_ni_and_weights() -> None:
    md = metafopy.MetaData.from_inputs([0.1, 0.2], [0.1, 0.1])
    assert md.names_all == ('1', '2')
    assert md.ni_f is None
    assert md.weights_f is None


def test_design_without_mods_is_an_intercept() -> None:
    X, names, missing, has_intercept = metafopy.MetaData.design(None, True, None, 3)
    numpy.testing.assert_array_equal(X, numpy.ones((3, 1)))
    assert names == ('intrcpt',)
    assert not missing.any()
    assert has_intercept


def test_design_of_a_vector_adds_an_intercept() -> None:
    X, names, missing, has_intercept = metafopy.MetaData.design(numpy.array([1.0, 2.0, 3.0]), True, None, 3)
    numpy.testing.assert_array_equal(X, [[1, 1], [1, 2], [1, 3]])
    assert names == ('intrcpt', 'mods')
    assert not missing.any()
    assert has_intercept
