"""The scripts step of the audit harness (``src/audit/scripts.py``): a process that dies in a script must not stop the step."""

import collections
import json
import pathlib
import subprocess
import sys

import pytest

import audit.config
import audit.engine
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


def failed_unit(error: str) -> dict[str, object]:
    return {'status': 'error', 'error': error, 'warnings': [], 'src': 'x', 'ast': {'t': 'num', 'v': 1}}


def stub_records() -> list[dict[str, object]]:
    return [
        {'id': 'rd/t0', 'kind': 'rd', 'topic': 't0', 'units': [failed_unit('could not find function "f"'), failed_unit('boom')]},
        {'id': 'rd/t1', 'kind': 'rd', 'topic': 't1', 'units': [failed_unit('other')]},
    ]


def lost_script(units: list[dict[str, object]]) -> dict[str, object]:
    return {**stub_records()[0], 'lost': True, 'units': units}


def test_lost_script_gives_r_killed_at_the_dead_unit_and_one_not_run() -> None:
    dead = failed_unit('R process died on this unit (killed)')
    dead['ast'] = None
    script = lost_script([failed_unit('boom'), dead])
    parts = list(audit.scripts.run_script(script, set(), 10))
    assert len(parts) == 2
    assert parts[0][0] == []
    assert len(parts[1][0]) == 1
    assert parts[1][0][0]['kind'] == 'r-killed'
    assert parts[1][0][0]['id'] == 'scripts/t0/rd/1'
    assert parts[1][0][0]['detail'] == 'R: R process died on this unit (killed)'
    total: collections.Counter[str] = collections.Counter()
    for _, tally in parts:
        total.update(tally)
    assert total['not run: R process died partway through the script'] == 1
    assert total['units'] == 2


def test_lost_script_stall_counts_as_an_r_failure() -> None:
    dead = failed_unit('R did not return within 50s (timeout)')
    dead['ast'] = None
    script = lost_script([dead])
    ((findings, tally),) = audit.scripts.run_script(script, set(), 10)
    assert findings == []
    assert tally['R failed (skipped)'] == 1
    assert tally['not run: R process died partway through the script'] == 1


def test_stream_scripts_lines_read_back_and_sum_to_run_script(tmp_path: pathlib.Path) -> None:
    records = stub_records()
    (tmp_path / 'r-0.json').write_text(json.dumps({'scripts': records}))
    scripts_path = tmp_path / 'scripts.json'
    audit.scripts.write_scripts(scripts_path, [audit.scripts.Script(str(r['id']), 'rd', str(r['topic']), '') for r in records])
    out_path = tmp_path / 'py-0.jsonl'
    job = {
        'r_path': str(tmp_path / 'r-0.json'),
        'scripts_path': str(scripts_path),
        'out_path': str(out_path),
        'nse': [],
        'timeout': 10,
        'functions': [],
    }
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(job))
    done = subprocess.run([sys.executable, '-m', 'audit.scripts', str(job_path)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    finished, running, parts = audit.engine.read_outcomes(out_path)
    assert running is None
    assert parts == []
    assert list(finished) == ['rd/t0', 'rd/t1']
    assert [len(finished[k]['parts']) for k in finished] == [2, 1]
    streamed: collections.Counter[str] = collections.Counter()
    streamed_findings: list[dict[str, object]] = []
    for record in finished.values():
        for part in record['parts']:
            streamed.update(part['tally'])
            streamed_findings += part['findings']
    direct: collections.Counter[str] = collections.Counter()
    direct_findings: list[dict[str, object]] = []
    for record in records:
        for findings, tally in audit.scripts.run_script(record, set(), 10):
            direct.update(tally)
            direct_findings += findings
    assert streamed == direct
    assert streamed_findings == direct_findings
    assert streamed['R failed (skipped)'] == 3
