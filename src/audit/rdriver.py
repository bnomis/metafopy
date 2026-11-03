"""Run R scripts of the audit with the audit's private library."""

import os
import pathlib
import socket
import subprocess
import threading

import audit.config

PORTS_LOCK = threading.Lock()
PORTS_GIVEN: set[int] = set()


def cluster_port() -> int:
    """A TCP port free on this machine now and given to no other R process of this run, for R's ``parallel`` clusters.

    Left to itself, R draws the port from its random seed and the clock when ``parallel`` loads; every audit script calls ``set.seed(1234)``,
    so R processes started side by side drew the same port and one failed with 'creation of server socket failed'."""
    with PORTS_LOCK:
        while True:
            with socket.socket() as probe:
                probe.bind(('', 0))
                port: int = probe.getsockname()[1]
            if port not in PORTS_GIVEN:
                PORTS_GIVEN.add(port)
                return port


def environment() -> dict[str, str]:
    """The environment of one R process: the audit's library, a fixed locale and packages, and a cluster port of its own (``R_PARALLEL_PORT``)."""
    env = dict(os.environ)
    env['R_LIBS'] = str(audit.config.R_LIB)
    env['LC_ALL'] = 'C.UTF-8'
    env['R_DEFAULT_PACKAGES'] = 'datasets,utils,grDevices,graphics,stats,methods'
    env['R_PARALLEL_PORT'] = str(cluster_port())
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
