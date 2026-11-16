"""Tests of ``bin/platform-probe.py``, the probe of the cross-platform numerics spec."""

import importlib.util
import json
import math
import pathlib
import sys
import types

import pytest

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
