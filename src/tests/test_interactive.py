import subprocess
import sys

import numpy
import pandas
import pytest

import metafopy
import metafopy.errors

pytest.importorskip('altair')

import metafopy.interactive
import metafopy.interactive.data


@pytest.fixture(scope='module')
def bcg() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def fit(bcg: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma(bcg['yi'], bcg['vi'], data=bcg)


def test_pooled_row_and_weights(fit: metafopy.RMAResult) -> None:
    data = metafopy.interactive.chart_data(fit, transf=numpy.exp)
    pred = metafopy.predict(fit, transf=numpy.exp)
    row = data.pooled.iloc[0]
    assert row['estimate'] == pred.pred[0]
    assert (row['ci_lb'], row['ci_ub']) == (pred.ci_lb[0], pred.ci_ub[0])
    assert (row['pi_lb'], row['pi_ub']) == (pred.pi_lb[0], pred.pi_ub[0])
    assert row['k'] == fit.k
    assert row['tau2'] == fit.tau2
    assert data.transformed
    assert data.measure == 'RR'
    numpy.testing.assert_array_equal(data.studies['weight'], fit.weights_())


def test_study_intervals(fit: metafopy.RMAResult) -> None:
    data = metafopy.interactive.chart_data(fit)
    z = metafopy.interactive.data.zcrit(None, fit)
    numpy.testing.assert_allclose(data.studies['ci_lb'], fit.yi - z * numpy.sqrt(fit.vi))
    numpy.testing.assert_allclose(data.studies['ci_ub'], fit.yi + z * numpy.sqrt(fit.vi))
    exp = metafopy.interactive.chart_data(fit, transf=numpy.exp)
    numpy.testing.assert_allclose(exp.studies['ci_lb'], numpy.exp(data.studies['ci_lb']))
    assert list(data.studies.columns) == list(metafopy.interactive.data.STUDY_COLUMNS)
    assert list(data.pooled.columns) == list(metafopy.interactive.data.POOLED_COLUMNS)


def test_missing_yi_omitted_and_indexed_before_omission(bcg: pandas.DataFrame) -> None:
    dat = bcg.copy()
    dat.loc[2, 'yi'] = numpy.nan
    with pytest.warns(UserWarning, match='NAs omitted'):
        res = metafopy.rma(dat['yi'], dat['vi'], data=dat)
    hrefs: list[str | None] = [f'#{i}' for i in range(len(dat))]
    hrefs[3] = None
    data = metafopy.interactive.chart_data(res, href=hrefs, slab=[f's{i}' for i in range(len(dat))])
    assert len(data.studies) == len(dat) - 1
    assert 's2' not in set(data.studies['slab'])
    row = data.studies[data.studies['slab'] == 's3'].iloc[0]
    assert row['href'] is None
    assert row['order'] == 3
    assert data.studies[data.studies['slab'] == 's4'].iloc[0]['href'] == '#4'


def test_group_fits(bcg: pandas.DataFrame, fit: metafopy.RMAResult) -> None:
    groups = ['a' if i % 2 else 'b' for i in range(len(bcg))]
    fits = {g: metafopy.rma(bcg['yi'][[x == g for x in groups]], bcg['vi'][[x == g for x in groups]]) for g in ('b', 'a')}
    data = metafopy.interactive.chart_data(fit, groups=groups, group_fits=fits)
    assert list(data.pooled['label']) == ['Overall', 'b', 'a']
    assert list(data.studies['group']) == groups


def test_multilevel_fit() -> None:
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('dat.konstantopoulos2011'))
    data = metafopy.interactive.chart_data(mv)
    assert len(data.studies) == mv.k
    assert len(data.pooled) == 1


@pytest.mark.parametrize('name', ['slab', 'href', 'groups'])
def test_wrong_length_arguments(fit: metafopy.RMAResult, name: str) -> None:
    kwargs: dict[str, list[str]] = {name: ['x']}
    with pytest.raises(metafopy.errors.InputError, match=name):
        metafopy.interactive.chart_data(fit, slab=kwargs.get('slab'), href=kwargs.get('href'), groups=kwargs.get('groups'))


def test_group_fits_without_groups(fit: metafopy.RMAResult) -> None:
    with pytest.raises(metafopy.errors.InputError, match='groups'):
        metafopy.interactive.chart_data(fit, group_fits={'a': fit})


def test_moderators_refused(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma(bcg['yi'], bcg['vi'], mods=bcg['ablat'], data=bcg)
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        metafopy.interactive.chart_data(res)


def test_check_chart_data_names_missing_column(fit: metafopy.RMAResult) -> None:
    data = metafopy.interactive.chart_data(fit)
    metafopy.interactive.data.check_chart_data(data)
    bad = metafopy.interactive.ChartData(data.studies.drop(columns=['weight']), data.pooled, data.measure, data.transformed)
    with pytest.raises(metafopy.errors.InputError, match='weight'):
        metafopy.interactive.data.check_chart_data(bad)
    bad = metafopy.interactive.ChartData(data.studies, data.pooled.drop(columns=['pi_lb']), data.measure, data.transformed)
    with pytest.raises(metafopy.errors.InputError, match='pi_lb'):
        metafopy.interactive.data.check_chart_data(bad)


def test_import_metafopy_does_not_load_altair() -> None:
    code = 'import sys, metafopy; sys.exit(1 if "altair" in sys.modules else 0)'
    assert subprocess.run([sys.executable, '-I', '-c', code], check=False, env={'PYTHONPATH': ':'.join(sys.path)}).returncode == 0
