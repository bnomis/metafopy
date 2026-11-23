"""Tests of ``bin/platform-probe.py``, the probe of the cross-platform numerics spec."""

import importlib.util
import json
import math
import pathlib
import sys
import types

import numpy
import pytest

import metafopy.models.mv_opt

PROBE_PATH = pathlib.Path(__file__).resolve().parents[2] / 'bin' / 'platform-probe.py'


def load_probe() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('platform_probe', PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()
GROUP = 'cm-el-tight-se'


def test_measure_groups_returns_finite_values_for_a_group() -> None:
    measured = probe.measure_groups([GROUP])
    assert list(measured) == [GROUP]
    group = measured[GROUP]
    assert isinstance(group, probe.GroupMeasurement)
    assert group.error is None
    assert group.values
    assert all(math.isfinite(v) for values in group.values.values() for v in values)


def test_measure_groups_rejects_an_unknown_group() -> None:
    with pytest.raises(KeyError, match='no-such-group'):
        probe.measure_groups(['no-such-group'])


def test_a_group_that_raises_is_recorded_and_the_others_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fails() -> None:
        raise ValueError('boom')

    # auditme: accept[literal-duplication] each test names its own group
    monkeypatch.setitem(probe.GROUPS, 'cm-el-tight-se', fails)
    measured = probe.measure_groups(['cm-el-tight-se', 'mv-reml-score'])
    assert measured['cm-el-tight-se'].error == 'ValueError: boom'
    assert measured['mv-reml-score'].error is None
    assert measured['mv-reml-score'].values


def test_main_writes_the_environment_and_the_groups(tmp_path: pathlib.Path) -> None:
    out = tmp_path / 'probe.json'
    assert probe.main(['--group', 'mv-reml-score', '--out', str(out)]) == 0
    document = json.loads(out.read_text(encoding='utf-8'))
    assert set(document) == {'environment', 'groups'}
    assert document['environment']['sys.platform'] == sys.platform
    assert set(document['groups']) == {'mv-reml-score'}
    assert document['groups']['mv-reml-score']['error'] is None


def test_main_exits_one_when_a_group_raised(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fails() -> None:
        raise ValueError('boom')

    monkeypatch.setitem(probe.GROUPS, 'mv-reml-score', fails)
    out = tmp_path / 'probe.json'
    assert probe.main(['--group', 'mv-reml-score', '--out', str(out)]) == 1
    assert json.loads(out.read_text(encoding='utf-8'))['groups']['mv-reml-score']['error'] == 'ValueError: boom'


def test_mv_reml_score_records_the_polish() -> None:
    values = probe.measure_groups(['mv-reml-score'])['mv-reml-score'].values
    assert {'polish.hessians', 'polish.gradient_before', 'polish.gradient_after'} <= set(values)
    assert values['polish.gradient_after']
    assert all(math.isfinite(v) for v in values['polish.gradient_before'] + values['polish.gradient_after'])
    assert max(values['polish.gradient_after']) <= metafopy.models.mv_opt.POLISH_GRADIENT


def location_scale_inputs() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """A small location-scale problem: eight effect sizes, an intercept for the mean and an intercept and a slope for the scale."""
    rng = numpy.random.default_rng(1)
    yi = rng.normal(size=8)
    vi = rng.uniform(0.1, 0.5, size=8)
    X = numpy.ones((8, 1))
    Z = numpy.column_stack([numpy.ones(8), numpy.linspace(-1.0, 1.0, 8)])
    return yi, vi, X, Z


def test_score_norm_is_the_largest_absolute_score_and_nan_where_the_model_is_undefined() -> None:
    """``score_norm``: a finite non-negative number at a point of the model, NaN where ``ls_score`` raises (an identity link with ``Z @ alpha < 0``)."""
    yi, vi, X, Z = location_scale_inputs()
    assert math.isfinite(probe.score_norm(yi, vi, X, Z, numpy.zeros(2), 'log', 'ML'))
    assert probe.score_norm(yi, vi, X, Z, numpy.zeros(2), 'log', 'ML') >= 0.0
    assert math.isnan(probe.score_norm(yi, vi, X, Z, numpy.array([-5.0, 0.0]), 'identity', 'ML'))


@pytest.mark.parametrize(('link', 'starts'), [('log', 5), ('identity', 7)])
def test_ls_runs_records_every_start_and_the_reference(link: str, starts: int) -> None:
    """``ls_runs``: a message and four figures per start (the identity link has its own extra starts), then the reference fit's alpha, ll and score."""
    yi, vi, X, Z = location_scale_inputs()
    record = probe.Record()
    probe.ls_runs(record, 'lbl', yi, vi, X, Z, link, 'ML')
    assert len(record.messages) == starts
    for number in range(starts):
        assert {f'lbl.start{number}.{field}' for field in ('alpha', 'negloglik', 'score', 'success')} <= set(record.values)
    assert {'lbl.reference.alpha', 'lbl.reference.ll', 'lbl.reference.score'} <= set(record.values)


def test_ls_runs_says_so_when_the_reference_fit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ls_runs``: a ``RuntimeError`` of ``ls_reference_fit`` is recorded as a message and no reference figures are put."""

    def fails(*args: object) -> None:
        raise RuntimeError('no fit')

    monkeypatch.setattr(probe.correctness.formulas, 'ls_reference_fit', fails)
    yi, vi, X, Z = location_scale_inputs()
    record = probe.Record()
    probe.ls_runs(record, 'lbl', yi, vi, X, Z, 'log', 'ML')
    assert record.messages[-1] == 'lbl.reference: RuntimeError: no fit'
    assert 'lbl.reference.alpha' not in record.values


@pytest.mark.slow
def test_main_runs_the_groups_that_reach_ls_runs_and_measure_selmodel_decreasing(tmp_path: pathlib.Path) -> None:
    """``main`` on ``ls-bcg-reference`` (reaches ``ls_runs``) and ``selmodel-decreasing`` (is ``measure_selmodel_decreasing``): both measured, no error."""
    out = tmp_path / 'probe.json'
    assert probe.main(['--group', 'ls-bcg-reference', '--group', 'selmodel-decreasing', '--out', str(out)]) == 0
    groups = json.loads(out.read_text(encoding='utf-8'))['groups']
    assert groups['ls-bcg-reference']['error'] is None
    assert 'ls.fit.ML.log.bcg.start0.alpha' in groups['ls-bcg-reference']['values']
    assert 'ls.fit.ML.log.bcg.reference.ll' in groups['ls-bcg-reference']['values']
    decreasing = groups['selmodel-decreasing']
    assert decreasing['error'] is None
    assert len(decreasing['values']['ll_along_ray']) == len(probe.RAY_FRACTIONS)
    assert {'metafopy.point', 'reference.end_point', 'll_at_metafopy_point'} <= set(decreasing['values'])
