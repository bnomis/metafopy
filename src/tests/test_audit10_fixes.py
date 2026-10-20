"""Audit 10: the dimnames of ``Z.G1``/``Z.G2`` of ``rma.mv`` after ``subset=`` and the removal of missing values, the column names of
``summary.escalc``, and an R time limit caught by a ``try()`` of metafor reported as a timeout by the audit harness. The expected names are what
R metafor 5.2-1 (R 4.3.3) gives."""

import typing
import warnings

import numpy
import pandas

import audit.pyrun
import metafopy
import metafopy.datasets


def fit_quietly(**kwargs: typing.Any) -> typing.Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.rma_mv('yi', 'vi', **kwargs)


def test_design_matrices_are_numbered_over_the_subset() -> None:
    # R: rma.mv(yi, vi, random = ~ factor(meta) | id, struct = "DIAG", subset = !is.na(meta), data = dat) with dat$yi[2] <- NA
    # rownames(res$Z.G1)[1:4] is c("1", "3", "4", "5") and the last is "46": the 46 rows of the subset, less row 2
    frame = metafopy.datasets.load('dat.bangertdrowns2004').copy()
    frame.loc[frame.index[1], 'yi'] = numpy.nan
    fit = fit_quietly(data=frame, random='~ factor(meta) | id', struct='DIAG', subset=~frame['meta'].isna().to_numpy())
    z1, z2 = typing.cast(typing.Any, fit.Z_G1), typing.cast(typing.Any, fit.Z_G2)
    assert z1.shape == (45, 2)
    assert tuple(z1.rownames[:4]) == ('1', '3', '4', '5')
    assert z1.rownames[-1] == '46'
    assert tuple(z1.colnames) == ('mf.g[[1]]0', 'mf.g[[1]]1')
    assert tuple(z2.rownames) == tuple(z1.rownames)
    assert tuple(z2.colnames[:3]) == ('mf.g[[nvars]]1', 'mf.g[[nvars]]2', 'mf.g[[nvars]]3')


def test_design_matrices_keep_the_levels_of_rows_with_missing_values() -> None:
    # R: dat$yi[1:4] <- NA; rma.mv(yi, vi, random = ~ factor(study) | district, data = dat): Z.G1 is 52 x 56 (every study),
    # Z.G2 is 52 x 11 (district 11 has no rows left) and both have the row names "5", "6", ...
    frame = metafopy.datasets.load('dat.konstantopoulos2011').copy()
    frame.loc[frame.index[:4], 'yi'] = numpy.nan
    fit = fit_quietly(random='~ factor(study) | district', data=frame)
    z1, z2 = typing.cast(typing.Any, fit.Z_G1), typing.cast(typing.Any, fit.Z_G2)
    assert z1.shape == (52, 56)
    assert tuple(z1.rownames[:2]) == ('5', '6')
    assert tuple(z1.colnames[:2]) == ('mf.g[[1]]1', 'mf.g[[1]]2')
    assert z2.shape == (52, 11)
    assert tuple(z2.colnames[:4]) == ('mf.g[[nvars]]11', 'mf.g[[nvars]]12', 'mf.g[[nvars]]18', 'mf.g[[nvars]]27')


def test_summary_escalc_names_the_interval_like_r() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    out = metafopy.summary_escalc(dat, append=False)
    assert list(out.columns) == ['yi', 'vi', 'sei', 'zi', 'pval', 'ci.lb', 'ci.ub']
    assert list(metafopy.summary_escalc(dat, transf=numpy.exp, append=False).columns) == ['yi', 'ci.lb', 'ci.ub']


def test_r_time_limit_inside_the_call_is_a_timeout() -> None:
    r = {'status': 'error', 'error': 'reached elapsed time limit inside the call, which carried on with the step it interrupted counted as failed'}
    py = {'status': 'ok', 'warnings': []}
    assert [f['kind'] for f in audit.pyrun.classify(r, py, pandas.DataFrame())] == ['timeout']


def test_r_shard_that_stops_answering_is_a_timeout() -> None:
    r = {'status': 'error', 'error': 'R did not return within 270s (timeout)'}
    py = {'status': 'ok', 'warnings': []}
    assert [f['kind'] for f in audit.pyrun.classify(r, py, pandas.DataFrame())] == ['timeout']
