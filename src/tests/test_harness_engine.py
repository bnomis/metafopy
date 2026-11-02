"""The runner of the audit harness (``src/audit/engine.py``): a case that never returns must not stop an audit."""

import os
import pathlib
import subprocess
import sys

import pytest

import audit.config
import audit.engine
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
