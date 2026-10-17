"""Values that audit 8 found different from R metafor 5.2-1 (checked against R 4.3.3, metafor 5.2-1)."""

import warnings

import pytest

import metafopy
import metafopy.datasets


def test_escalc_dataset_carries_its_measure_into_rma() -> None:
    # R: rma(yi, vi, data = dat.konstantopoulos2011)$measure is "SMD" because the data set is an escalc object
    frame = metafopy.datasets.load('dat.konstantopoulos2011')
    assert frame.attrs['measure'] == 'SMD'
    assert metafopy.rma(yi='yi', vi='vi', data=frame).measure == 'SMD'


def test_dfs_equals_ddf_for_knha() -> None:
    # R: res$dfs == res$ddf for test = "knha" (11 for 13 studies and 2 coefficients)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=data, test='knha')
        pred = metafopy.predict(res)
    assert res.dfs == pytest.approx(11.0)
    assert res.ddf == 11
    # R labels the predictions 1..k when the fit has no study labels
    assert pred.slab == tuple(str(i) for i in range(1, 14))
