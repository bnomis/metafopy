"""The runner of the audit harness (``src/audit/engine.py``): a case that never returns must not stop an audit."""

import pathlib
import subprocess
import sys

import audit.engine


def test_a_process_whose_output_stands_still_is_killed(tmp_path: pathlib.Path) -> None:
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert audit.engine.wait_or_kill(proc, tmp_path / 'out.jsonl', 0.5)
    assert proc.returncode is not None


def test_a_process_that_finishes_is_left_alone(tmp_path: pathlib.Path) -> None:
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    assert not audit.engine.wait_or_kill(proc, tmp_path / 'out.jsonl', 30)
    assert proc.returncode == 0


def test_the_case_running_when_r_stopped_is_the_one_reported(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'r.jsonl'
    path.write_text('{"start": "a"}\n{"id": "a", "status": "ok"}\n{"start": "b"}\n')
    done, running = audit.engine.read_outcomes(path)
    assert set(done) == {'a'}
    assert running == 'b'
