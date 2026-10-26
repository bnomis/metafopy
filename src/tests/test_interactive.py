import json
import subprocess
import sys
import typing

import numpy
import pandas
import pytest

import metafopy
import metafopy.errors

pytest.importorskip('altair')

import altair

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


def mark_types(spec: dict[str, typing.Any]) -> set[str]:
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            mark = node.get('mark')
            if isinstance(mark, dict):
                found.add(str(mark.get('type')))
            elif isinstance(mark, str):
                found.add(mark)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec)
    return found


def datasets(spec: dict[str, typing.Any], column: str) -> list[dict[str, typing.Any]]:
    """The inlined rows of the dataset whose rows carry ``column``."""
    rows = [r for d in spec['datasets'].values() for r in d] if 'datasets' in spec else []
    inline: list[dict[str, typing.Any]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            data = node.get('data')
            if isinstance(data, dict) and 'values' in data:
                inline.extend(data['values'])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec)
    return [r for r in rows + inline if column in r]


def test_forest_chart_validates_and_inlines(fit: metafopy.RMAResult) -> None:
    spec = metafopy.interactive.forest_chart(fit, transf=numpy.exp, refline=1.0).to_dict()
    assert '"url"' not in json.dumps(spec)
    assert {'rule', 'square', 'point'} <= mark_types(spec)
    assert len(datasets(spec, 'slab')) >= len(fit.yi)


def test_forest_chart_orders_rows(fit: metafopy.RMAResult) -> None:
    data = metafopy.interactive.chart_data(fit)
    expected = {
        'input': list(data.studies['slab']),
        'obs': list(data.studies.sort_values('yi', kind='stable')['slab']),
        'weight': list(data.studies.sort_values('weight', ascending=False, kind='stable')['slab']),
    }
    for order, slabs in expected.items():
        spec = metafopy.interactive.forest_chart(fit, order=order).to_dict()  # type: ignore[arg-type]
        assert slabs == spec['vconcat'][0]['layer'][-1]['encoding']['y']['sort']


def test_forest_chart_refline(fit: metafopy.RMAResult) -> None:
    with_line = metafopy.interactive.forest_chart(fit, refline=1.0).to_dict()
    without = metafopy.interactive.forest_chart(fit, refline=None).to_dict()
    assert len(with_line['vconcat'][0]['layer']) == len(without['vconcat'][0]['layer']) + 1
    assert {'x': 1.0} in datasets(with_line, 'x')


def test_forest_chart_href(fit: metafopy.RMAResult, bcg: pandas.DataFrame) -> None:
    hrefs: list[str | None] = [f'#{i}' for i in range(len(bcg))]
    hrefs[1] = None
    spec = metafopy.interactive.forest_chart(fit, href=hrefs).to_dict()
    rows = datasets(spec, 'slab')
    assert sum('href' in r for r in rows) == len(bcg) - 1
    assert 'href' not in next(r for r in rows if r['order'] == 1)
    assert spec['vconcat'][0]['layer'][-1]['encoding']['href']['field'] == 'href'


def test_forest_chart_multilevel() -> None:
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('dat.konstantopoulos2011'))
    spec = metafopy.interactive.forest_chart(mv).to_dict()
    assert len(datasets(spec, 'slab')) == mv.k


def test_forest_chart_group_rows(bcg: pandas.DataFrame, fit: metafopy.RMAResult) -> None:
    groups = ['a' if i % 2 else 'b' for i in range(len(bcg))]
    fits = {g: metafopy.rma(bcg['yi'][[x == g for x in groups]], bcg['vi'][[x == g for x in groups]]) for g in ('b', 'a')}
    spec = metafopy.interactive.forest_chart(fit, groups=groups, group_fits=fits).to_dict()
    assert [r['label'] for r in datasets(spec, 'label')] == ['Overall', 'b', 'a']


def test_forest_chart_from_chart_data(fit: metafopy.RMAResult) -> None:
    data = metafopy.interactive.chart_data(fit, transf=numpy.exp)
    assert metafopy.interactive.forest_chart(data, transf=numpy.exp).to_dict()
    with pytest.raises(metafopy.errors.InputError, match='transf'):
        metafopy.interactive.forest_chart(data)
    with pytest.raises(metafopy.errors.InputError, match='level'):
        metafopy.interactive.forest_chart(data, transf=numpy.exp, level=90)
    bad = metafopy.interactive.ChartData(data.studies.drop(columns=['weight']), data.pooled, data.measure, data.transformed)
    with pytest.raises(metafopy.errors.InputError, match='weight'):
        metafopy.interactive.forest_chart(bad, transf=numpy.exp)


def test_forest_chart_renders_svg(fit: metafopy.RMAResult) -> None:
    vl_convert = pytest.importorskip('vl_convert')
    svg = vl_convert.vegalite_to_svg(metafopy.interactive.forest_chart(fit, transf=numpy.exp, refline=1.0).to_dict())
    assert svg.startswith('<svg')


def test_forest_chart_leaves_data_transformer_alone(fit: metafopy.RMAResult) -> None:
    before = altair.data_transformers.active
    metafopy.interactive.forest_chart(fit).to_dict()
    assert altair.data_transformers.active == before
