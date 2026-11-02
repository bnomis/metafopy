"""The scripts step of the audit harness (``src/audit/scripts.py``): a process that dies in a script must not stop the step."""

import json
import pathlib
import subprocess
import sys

import pytest

import audit.config
import audit.rdriver
import audit.scripts

R_STAND_IN = """
import json, sys
out = open(sys.argv[-2], 'w')
scripts = json.load(open(sys.argv[-3]))
die = sys.argv[-1] == 'die'
def line(x):
    out.write(json.dumps(x) + '\\n')
    out.flush()
for s in scripts:
    line({'start': s['id']})
    line({'part': s['id'], 'unit': {'status': 'ok', 'src': 'a', 'ast': None, 'warnings': []}})
    if die and s['id'] == 'rd/t0':
        sys.exit(1)
    line({'part': s['id'], 'unit': {'status': 'ok', 'src': 'b', 'ast': None, 'warnings': []}})
    line({'id': s['id'], 'kind': s['kind'], 'topic': s['topic']})
"""


def scripts() -> list[audit.scripts.Script]:
    return [audit.scripts.Script(f'rd/t{i}', 'rd', f't{i}', f'man/t{i}.Rd') for i in range(3)]


def stand_in(mode: str):  # type: ignore[no-untyped-def]
    def start(script: pathlib.Path, log: pathlib.Path, *args: str) -> subprocess.Popen[bytes]:
        # args: 'run', source, scripts path, out path, timeout
        return subprocess.Popen([sys.executable, '-c', R_STAND_IN, args[2], args[3], mode], stdout=log.open('ab'), stderr=subprocess.STDOUT)

    return start


@pytest.mark.parametrize('mode', ['finish', 'die'])
def test_r_side_writes_a_record_for_every_script(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.05)
    monkeypatch.setattr(audit.rdriver, 'start', stand_in(mode))
    audit.scripts.run_r_scripts(tmp_path, 0, tmp_path, scripts(), 10)
    records = json.loads((tmp_path / 'r-0.json').read_text())['scripts']
    assert [r['id'] for r in records] == ['rd/t0', 'rd/t1', 'rd/t2']
    if mode == 'finish':
        assert all(len(r['units']) == 2 and 'lost' not in r and 'parts' not in r for r in records)
        return
    first = records[0]
    assert first['lost'] is True
    assert len(first['units']) == 2
    assert first['units'][0]['status'] == 'ok'
    assert first['units'][1]['status'] == 'error'
    assert first['units'][1]['error'].endswith('(killed)')
    assert all(len(r['units']) == 2 and 'lost' not in r for r in records[1:])


def test_list_failure_names_its_log(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.rdriver, 'rscript_command', lambda script, *args: [sys.executable, '-c', 'print("boom")'])
    with pytest.raises(RuntimeError, match=r'list\.log'):
        audit.scripts.list_scripts(tmp_path, tmp_path, None)
