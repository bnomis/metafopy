"""The audit harness's file and process envelope: each helper translates ``OSError`` and ``subprocess`` failures to ``HarnessError``."""

import collections.abc
import pathlib
import subprocess
import typing

import audit.errors
import audit.harness_io

__all__ = [
    'append_text',
    'ensure_dir',
    'read_text',
    'run_command',
    'start_process',
    'write_bytes',
    'write_text',
]


def read_text(path: pathlib.Path, what: str, errors: str = 'strict') -> str:
    """Read a UTF-8 text file.

    Args:
        path: The file to read.
        what: What the harness was doing, for the error message.
        errors: How to treat undecodable bytes, as for ``str.decode``.

    Returns:
        The file's text.

    Raises:
        audit.errors.HarnessError: the file cannot be read.
    """
    try:
        return path.read_text(encoding='utf-8', errors=errors)
    except OSError as e:
        raise audit.errors.HarnessError(f'cannot read {path} ({what}): {type(e).__name__}') from e


def write_text(path: pathlib.Path, text: str, what: str) -> None:
    """Write a UTF-8 text file.

    Args:
        path: The file to write.
        text: The text to write.
        what: What the harness was doing, for the error message.

    Raises:
        audit.errors.HarnessError: the file cannot be written.
    """
    try:
        path.write_text(text, encoding='utf-8')
    except OSError as e:
        raise audit.errors.HarnessError(f'cannot write {path} ({what}): {type(e).__name__}') from e


def append_text(path: pathlib.Path, text: str, what: str) -> None:
    """Append UTF-8 text to a file, creating it if need be.

    Args:
        path: The file to append to.
        text: The text to append.
        what: What the harness was doing, for the error message.

    Raises:
        audit.errors.HarnessError: the file cannot be written.
    """
    try:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(text)
    except OSError as e:
        raise audit.errors.HarnessError(f'cannot write {path} ({what}): {type(e).__name__}') from e


def write_bytes(path: pathlib.Path, data: bytes, what: str) -> None:
    """Write a binary file.

    Args:
        path: The file to write.
        data: The bytes to write.
        what: What the harness was doing, for the error message.

    Raises:
        audit.errors.HarnessError: the file cannot be written.
    """
    try:
        path.write_bytes(data)
    except OSError as e:
        raise audit.errors.HarnessError(f'cannot write {path} ({what}): {type(e).__name__}') from e


def ensure_dir(path: pathlib.Path, what: str) -> None:
    """Create a directory and its parents; an existing one is fine.

    Args:
        path: The directory to create.
        what: What the harness was doing, for the error message.

    Raises:
        audit.errors.HarnessError: the directory cannot be created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise audit.errors.HarnessError(f'cannot create directory {path} ({what}): {type(e).__name__}') from e


def run_command(args: collections.abc.Sequence[str], what: str, **kwargs: typing.Any) -> subprocess.CompletedProcess[typing.Any]:
    """Run a command with ``subprocess.run``.

    Args:
        args: The command and its arguments.
        what: What the harness was doing, for the error message.
        **kwargs: Passed to ``subprocess.run``.

    Returns:
        The completed process.

    Raises:
        audit.errors.HarnessError: the command cannot be started, or ``subprocess`` reports a failure such as a timeout.
    """
    try:
        return subprocess.run(args, **kwargs)
    except (OSError, subprocess.SubprocessError) as e:
        raise audit.errors.HarnessError(f'cannot run {" ".join(map(str, args))} ({what}): {type(e).__name__}') from e


def start_process(args: collections.abc.Sequence[str], what: str, **kwargs: typing.Any) -> subprocess.Popen[typing.Any]:
    """Start a command with ``subprocess.Popen``.

    Args:
        args: The command and its arguments.
        what: What the harness was doing, for the error message.
        **kwargs: Passed to ``subprocess.Popen``.

    Returns:
        The started process.

    Raises:
        audit.errors.HarnessError: the command cannot be started.
    """
    try:
        return subprocess.Popen(args, **kwargs)
    except (OSError, subprocess.SubprocessError) as e:
        raise audit.errors.HarnessError(f'cannot start {" ".join(map(str, args))} ({what}): {type(e).__name__}') from e
