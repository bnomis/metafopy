"""Values that audit 8 found different from R metafor 5.2-1 (checked against R 4.3.3, metafor 5.2-1)."""

import warnings

import numpy
import pandas
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


def test_k_all_counts_the_rows_before_subsetting_for_counts() -> None:
    # R: rma(measure = "OR", ai, bi, ci, di, data = dat.bcg, subset = 1:8)$k.all is 13 (the 8 studies are $k)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame, subset=list(range(8)))
    assert (res.k, res.k_all) == (8, 13)


def test_simulate_pads_to_the_subset_unless_na_omit() -> None:
    # tests/testthat/test_misc_handling_nas.r: length(simulate(res, seed = 1234)[[1]]) is 5 under na.omit and 7 under na.pass and na.exclude
    frame = pandas.DataFrame({'yi': [numpy.nan, 1, 3, 2, 5, 4, 6], 'vi': [1, numpy.nan, 1, 1, 1, 1, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(yi='yi', vi='vi', data=frame)
    assert len(metafopy.simulate(res, seed=1234)) == 5
    for action in ('pass', 'exclude'):
        sim = metafopy.simulate(res, seed=1234, na_action=action)
        assert len(sim) == 7
        assert sim['sim_1'].isna().to_list() == [True, True, False, False, False, False, False]
