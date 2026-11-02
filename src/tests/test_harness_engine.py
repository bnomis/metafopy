"""The runner of the audit harness (``src/audit/engine.py``): a case that never returns must not stop an audit."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

import audit.cases
import audit.config
import audit.engine
import audit.pyrun
import audit.rdriver


def test_a_process_whose_output_stands_still_is_killed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.1)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert audit.engine.watch(proc, tmp_path / 'out.jsonl', 0.5, 1 << 40) == 'stalled'
    assert proc.returncode is not None


def test_a_process_that_finishes_is_left_alone(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.1)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)'])
    assert audit.engine.watch(proc, tmp_path / 'out.jsonl', 30, 1 << 40) is None
    assert proc.returncode == 0


def test_a_process_that_exits_non_zero_by_itself_died(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.1)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3); raise SystemExit(1)'])
    assert audit.engine.watch(proc, tmp_path / 'out.jsonl', 30, 1 << 40) == 'died'


def test_a_process_past_the_memory_limit_is_killed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.1)
    code = 'import time; data = bytearray(600 * 1024**2); time.sleep(60)'
    proc = subprocess.Popen([sys.executable, '-c', code])
    assert audit.engine.watch(proc, tmp_path / 'out.jsonl', 60, 300 * 1024**2) == 'memory'
    assert proc.returncode is not None


def test_resident_bytes_of_this_process_is_positive_and_of_a_finished_one_none() -> None:
    size = audit.engine.resident_bytes(os.getpid())
    assert size is not None
    assert size > 0
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    assert audit.engine.resident_bytes(proc.pid) is None


def test_the_case_running_when_r_stopped_is_the_one_reported(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'r.jsonl'
    path.write_text('{"start": "a"}\n{"id": "a", "status": "ok"}\n{"start": "b"}\n')
    done, running = audit.engine.read_outcomes(path)
    assert set(done) == {'a'}
    assert running == 'b'


def test_a_process_starts_where_the_platform_refuses_the_memory_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = pytest.importorskip('resource')

    def refuse(resource_id: int, limits: tuple[int, int]) -> None:
        raise ValueError('current limit exceeds maximum limit')

    monkeypatch.setattr(resource, 'setrlimit', refuse)
    audit.config.limit_memory()


def test_the_memory_cap_is_set_where_the_platform_allows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = pytest.importorskip('resource')
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, 'setrlimit', lambda resource_id, limits: calls.append((resource_id, limits)))
    audit.config.limit_memory()
    assert calls == [(resource.RLIMIT_AS, (audit.config.MEMORY_LIMIT, audit.config.MEMORY_LIMIT))]


def test_an_r_shard_process_starts_on_this_platform() -> None:
    proc = subprocess.run([sys.executable, '-c', 'pass'], check=False, preexec_fn=audit.rdriver.limit_memory)
    assert proc.returncode == 0


STAND_IN = """
import json, os, sys
cases, out, die = sys.argv[1:4]
if die == '*':
    os._exit(1)
with open(out, 'a') as f:
    for case in json.load(open(cases)):
        f.write(json.dumps({'start': case['id']}) + '\\n')
        f.flush()
        if case['id'] == die:
            os._exit(1)
        f.write(json.dumps({'id': case['id'], 'status': 'ok'}) + '\\n')
        f.flush()
"""


def run_stand_in(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, die: str) -> tuple[list[audit.engine.Json], list[int]]:
    monkeypatch.setattr(audit.config, 'POLL_SECONDS', 0.05)
    shard = [audit.cases.Case(id=name, axis='x', call={}) for name in 'abc']
    starts: list[int] = []

    def start(cases_path: pathlib.Path, out_path: pathlib.Path) -> subprocess.Popen[bytes]:
        starts.append(1)
        return subprocess.Popen([sys.executable, '-c', STAND_IN, str(cases_path), str(out_path), die])

    def dead(case_id: str, stop: audit.engine.Stop) -> audit.engine.Json:
        return {'id': case_id, 'status': 'dead', 'stop': stop}

    return audit.engine.run_shard(tmp_path, 'p', shard, 1.0, start, dead), starts


def test_a_shard_whose_process_dies_midway_returns_every_case(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, starts = run_stand_in(tmp_path, monkeypatch, 'b')
    assert ''.join(r['id'] + r['status'][0] for r in results) == 'aobdco'
    assert results[1]['stop'] == 'died'
    assert len(starts) == 2
    assert (tmp_path / 'p-attempt1.jsonl').exists()


def test_a_process_that_dies_before_its_first_case_does_not_restart(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, starts = run_stand_in(tmp_path, monkeypatch, '*')
    assert ''.join(r['id'] + r['status'][0] for r in results) == 'adbdcd'
    assert len(starts) == 1


def test_a_stalled_r_shard_reason_is_still_recognised_as_a_timeout() -> None:
    assert audit.pyrun.r_timed_out('R did not return within 90s (timeout)')


def test_a_python_shard_process_streams_a_start_line_before_each_result(tmp_path: pathlib.Path) -> None:
    first, second = 'one', 'two'
    cases = [{'id': name, 'axis': 'x', 'call': {'fn': 'nosuchfunction'}} for name in (first, second)]
    (tmp_path / 'fixtures.json').write_text('[]')
    (tmp_path / 'cases.json').write_text(json.dumps(cases))
    (tmp_path / 'r.json').write_text(json.dumps({'results': [{'id': name, 'status': 'ok', 'value': {}} for name in (first, second)]}))
    job = {
        'fixtures_path': str(tmp_path / 'fixtures.json'),
        'cases_path': str(tmp_path / 'cases.json'),
        'r_path': str(tmp_path / 'r.json'),
        'out_path': str(tmp_path / 'out.jsonl'),
        'timeout': 30.0,
    }
    (tmp_path / 'job.json').write_text(json.dumps(job))
    subprocess.run([sys.executable, '-m', 'audit.pyrun', str(tmp_path / 'job.json')], check=True, env=os.environ)
    lines = [json.loads(line) for line in (tmp_path / 'out.jsonl').read_text().splitlines()]
    assert [line.get('start', line.get('id')) for line in lines] == [first, first, second, second]
    assert 'start' in lines[0]
    assert 'start' not in lines[1]
    done, running = audit.engine.read_outcomes(tmp_path / 'out.jsonl')
    assert sorted(done) == [first, second]
    assert running is None
