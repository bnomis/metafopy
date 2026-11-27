"""The audit harness's file and process helpers (``audit.harness_io``)."""

import pathlib
import sys

import pytest

import audit.errors
import audit.harness_io


def test_text_round_trips(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'a.txt'
    audit.harness_io.write_text(path, 'héllo', 'testing')
    assert audit.harness_io.read_text(path, 'testing') == 'héllo'


def test_bytes_are_written(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'a.bin'
    audit.harness_io.write_bytes(path, b'\x00\x01', 'testing')
    assert path.read_bytes() == b'\x00\x01'


def test_ensure_dir_creates_parents_and_accepts_an_existing_one(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'a' / 'b'
    audit.harness_io.ensure_dir(path, 'testing')
    audit.harness_io.ensure_dir(path, 'testing')
    assert path.is_dir()


def test_run_command_returns_the_result() -> None:
    proc = audit.harness_io.run_command([sys.executable, '-c', 'print(1)'], 'testing', capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.strip() == '1'


def test_start_process_returns_a_popen() -> None:
    proc = audit.harness_io.start_process([sys.executable, '-c', 'pass'], 'testing')
    assert proc.wait() == 0


def test_read_of_a_missing_file_names_the_path(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'missing.txt'
    with pytest.raises(audit.errors.HarnessError, match=r'missing.txt.*reading the thing'):
        audit.harness_io.read_text(path, 'reading the thing')


def test_writes_into_a_missing_directory_name_the_path(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'no' / 'a.txt'
    with pytest.raises(audit.errors.HarnessError, match=r'no.*a.txt'):
        audit.harness_io.write_text(path, 'x', 'testing')
    with pytest.raises(audit.errors.HarnessError, match=r'no.*a.txt'):
        audit.harness_io.write_bytes(path, b'x', 'testing')


def test_ensure_dir_under_a_file_names_the_path(tmp_path: pathlib.Path) -> None:
    blocker = tmp_path / 'file'
    blocker.write_text('x', encoding='utf-8')
    with pytest.raises(audit.errors.HarnessError, match=r'file.*sub'):
        audit.harness_io.ensure_dir(blocker / 'sub', 'testing')


def test_a_missing_executable_names_the_command(tmp_path: pathlib.Path) -> None:
    command = str(tmp_path / 'no-such-tool')
    with pytest.raises(audit.errors.HarnessError, match=r'no-such-tool'):
        audit.harness_io.run_command([command], 'testing')
    with pytest.raises(audit.errors.HarnessError, match=r'no-such-tool'):
        audit.harness_io.start_process([command], 'testing')


def test_a_timeout_is_translated() -> None:
    with pytest.raises(audit.errors.HarnessError, match=r'sleep'):
        audit.harness_io.run_command([sys.executable, '-c', 'import time; time.sleep(5)  # sleep'], 'testing', timeout=0.2)


def test_append_text_adds_to_a_file_and_names_the_path_on_failure(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'a.txt'
    audit.harness_io.append_text(path, 'one', 'test')
    audit.harness_io.append_text(path, 'two', 'test')
    assert path.read_text() == 'onetwo'
    missing = tmp_path / 'no' / 'a.txt'
    with pytest.raises(audit.errors.HarnessError, match='no'):
        audit.harness_io.append_text(missing, 'x', 'test')


def test_read_text_replaces_undecodable_bytes_on_request(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'b.txt'
    path.write_bytes(b'a\xffb')
    assert audit.harness_io.read_text(path, 'test', errors='replace') == 'a�b'
    with pytest.raises(UnicodeDecodeError):
        audit.harness_io.read_text(path, 'test')
