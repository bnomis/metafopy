from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metafopy import MetaData
from metafopy._data import make_unique


def test_make_unique_matches_r() -> None:
    assert make_unique(["a", "b", "a", "a", "a.1"]) == ("a", "b", "a.1", "a.2", "a.1.1")


def test_defaults_add_intercept_only() -> None:
    md = MetaData.from_inputs([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    assert md.k == 3
    assert md.X_names == ("intrcpt",)
    np.testing.assert_array_equal(md.X, np.ones((3, 1)))
    assert md.slab == ("1", "2", "3")


def test_sei_squared_to_vi() -> None:
    md = MetaData.from_inputs([0.1, 0.2], sei=[0.1, 0.2])
    np.testing.assert_allclose(md.vi, [0.01, 0.04])


def test_escalc_style_frame_is_picked_up() -> None:
    dat = pd.DataFrame({"yi": [0.1, 0.2], "vi": [0.01, 0.02], "x": [1.0, 2.0]})
    md = MetaData.from_inputs(data=dat, mods="~ x")
    assert md.X_names == ("intrcpt", "x")


def test_matrix_mods_get_intercept_unless_disabled() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert MetaData.from_inputs([1, 2, 3], [1, 1, 1], mods=x).X_names == ("intrcpt", "X1")
    md = MetaData.from_inputs([1, 2, 3], [1, 1, 1], mods=x, intercept=False)
    assert md.X_names == ("X1",)
    assert not md.intercept


def test_formula_without_intercept() -> None:
    dat = pd.DataFrame({"yi": [1.0, 2.0, 3.0], "vi": [1.0, 1.0, 1.0], "g": ["a", "b", "a"]})
    md = MetaData.from_inputs(data=dat, mods="~ 0 + g")
    assert md.X_names == ("ga", "gb")
    assert not md.intercept


def test_na_omit_then_subset_bookkeeping() -> None:
    md = MetaData.from_inputs(
        [0.1, np.nan, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1], subset=[True, True, True, False]
    )
    assert md.k == 2
    assert md.k_all == 4
    np.testing.assert_array_equal(md.keep, [True, False, True, False])
    assert md.slab == ("1", "3")


def test_na_fail() -> None:
    with pytest.raises(ValueError, match="missing"):
        MetaData.from_inputs([0.1, np.nan], [0.1, 0.1], na_action="fail")


def test_duplicate_slabs_made_unique() -> None:
    md = MetaData.from_inputs([1, 2], [1, 1], slab=["Smith", "Smith"])
    assert md.slab == ("Smith", "Smith.1")


def test_column_name_without_data_errors() -> None:
    with pytest.raises(ValueError, match="no data"):
        MetaData.from_inputs("yi", "vi")
