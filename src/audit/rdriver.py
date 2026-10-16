"""Run R scripts of the audit with the audit's private library."""

import os
import pathlib
import subprocess

import audit.config


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env['R_LIBS'] = str(audit.config.R_LIB)
    env['LC_ALL'] = 'C.UTF-8'
    env['R_DEFAULT_PACKAGES'] = 'datasets,utils,grDevices,graphics,stats,methods'
    return env


def rscript_command(script: pathlib.Path, *args: str) -> list[str]:
    return ['Rscript', '--vanilla', str(script), *args]


def run(script: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an R script to completion; a non-zero exit raises with R's stderr."""
    proc = subprocess.run(rscript_command(script, *args), env=environment(), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'{script.name} failed:\n{proc.stderr[-3000:]}')
    return proc


def limit_memory() -> None:
    audit.config.limit_memory()


def start(script: pathlib.Path, log: pathlib.Path, *args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        rscript_command(script, *args),
        env=environment(),
        stdout=log.open('ab'),
        stderr=subprocess.STDOUT,
        preexec_fn=limit_memory,
    )
