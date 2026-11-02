"""Run cases in R and in Python and collect the findings."""

import collections.abc
import concurrent.futures
import json
import pathlib
import subprocess
import sys
import time
import typing

import audit.cases
import audit.config
import audit.rdriver

Json = dict[str, typing.Any]
Stop = typing.Literal['died', 'memory', 'stalled']


class Item(typing.Protocol):
    """Anything a shard runs: a case, a script."""

    @property
    def id(self) -> str: ...


def read_outcomes(path: pathlib.Path) -> tuple[dict[str, Json], str | None, list[Json]]:
    """The finished items of a runner's JSON-lines file, the item running when it stopped (None when it finished cleanly), and the part lines
    that item wrote before it stopped. A line ``{"part": id, ...}`` belongs to item ``id``; a finished item's parts are in its record under
    ``"parts"``, in order, and only when it has any."""
    done: dict[str, Json] = {}
    running: str | None = None
    parts: dict[str, list[Json]] = {}
    if not path.exists():
        return done, None, []
    for line in path.read_text(errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'start' in doc:
            running = doc['start']
        elif 'part' in doc:
            parts.setdefault(doc['part'], []).append(doc)
        elif 'id' in doc:
            if doc['id'] in parts:
                doc['parts'] = parts[doc['id']]
            done[doc['id']] = doc
            running = None
    return done, running, parts.get(running, []) if running is not None else []


def resident_bytes(pid: int) -> int | None:
    """The resident size of process ``pid`` in bytes (``ps -o rss= -p <pid>`` reports KiB), or None when the process has gone or ``ps`` is missing."""
    try:
        done = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True, check=False)
    except OSError:
        return None
    text = done.stdout.strip()
    if done.returncode != 0 or not text.isdigit():
        return None
    return int(text) * 1024


def watch(proc: subprocess.Popen[bytes], out: pathlib.Path, stall: float, limit: int) -> Stop | None:
    """Wait for a shard process. Kill it when its output has not moved for ``stall`` seconds ('stalled'; compiled code ignores in-process time
    limits) or its resident size passes ``limit`` bytes ('memory'). A process that exits non-zero by itself is 'died'; None when it finished cleanly."""
    last_size = -1
    last_move = time.monotonic()
    while proc.poll() is None:
        time.sleep(audit.config.POLL_SECONDS)
        if proc.poll() is not None:
            break
        resident = resident_bytes(proc.pid)
        if resident is not None and resident > limit:
            proc.kill()
            proc.wait()
            return 'memory'
        size = out.stat().st_size if out.exists() else 0
        if size != last_size:
            last_size, last_move = size, time.monotonic()
        elif time.monotonic() - last_move > stall:
            proc.kill()
            proc.wait()
            return 'stalled'
    return 'died' if proc.returncode != 0 else None


def run_shard[T: Item](
    work: pathlib.Path,
    name: str,
    shard: list[T],
    timeout: float,
    start: collections.abc.Callable[[pathlib.Path, pathlib.Path], subprocess.Popen[bytes]],
    dead: collections.abc.Callable[[str, Stop, list[Json]], Json],
    write: collections.abc.Callable[[pathlib.Path, list[T]], None],
) -> list[Json]:
    """Run ``shard`` in processes made by ``start(items_path, out_path)``, restarting on the items left whenever one stops early;
    ``write(path, remaining)`` writes the items a process is started on. The item running when a process stopped gets
    ``dead(item_id, stop, parts)`` as its result, ``parts`` being the part lines it wrote first. Returns one result per finished item, in shard order."""
    out = work / f'{name}.jsonl'
    finished: dict[str, Json] = {}
    remaining = list(shard)
    attempts = 0
    while remaining and attempts <= len(shard):
        attempts += 1
        part = work / f'{name}-rest.json'
        write(part, remaining)
        proc = start(part, out)
        stop = watch(proc, out, 2 * timeout + 30, audit.config.MEMORY_LIMIT)
        done, running, running_parts = read_outcomes(out)
        finished.update(done)
        if running is not None and stop is None:
            stop = 'died'
        if running is not None:
            finished[running] = dead(running, stop or 'died', running_parts)
        remaining = [c for c in remaining if c.id not in finished]
        if stop is None and running is None:
            break
        if running is None and not done:
            for case in remaining:
                finished[case.id] = dead(case.id, stop or 'died', [])
            remaining = []
        if out.exists():
            out.rename(work / f'{name}-attempt{attempts}.jsonl')
    return [finished[c.id] for c in shard if c.id in finished]


def r_stop_reason(stop: Stop, timeout: float, unit: str) -> str:
    """Why R's process stopped on ``unit`` ('case', 'unit'), in the words ``audit.pyrun.r_killed`` and the report's stall check read."""
    if stop == 'stalled':
        return f'R did not return within {2 * timeout + 30:.0f}s (timeout)'
    if stop == 'memory':
        return f'R passed the memory limit of {audit.config.MEMORY_LIMIT / 1024**3:g} GiB (killed)'
    return f'R process died on this {unit} (killed)'


def run_r_shard(work: pathlib.Path, i: int, fixtures_path: pathlib.Path, shard: list[audit.cases.Case], timeout: float) -> None:
    """Run a shard in R; when R dies on a case (memory, a crash) record that case as an R failure and restart on the cases that remain."""

    def start(cases_path: pathlib.Path, out_path: pathlib.Path) -> subprocess.Popen[bytes]:
        return audit.rdriver.start(audit.config.R_DIR / 'runcases.R', work / f'r-{i}.log', str(fixtures_path), str(cases_path), str(out_path), str(timeout))

    def dead(case_id: str, stop: Stop, parts: list[Json]) -> Json:
        return {'id': case_id, 'status': 'error', 'error': r_stop_reason(stop, timeout, 'case'), 'warnings': []}

    results = run_shard(work, f'r-{i}', shard, timeout, start, dead, audit.cases.write_cases)
    (work / f'r-{i}.json').write_text(json.dumps({'results': results}))


def py_stop(stop: Stop, timeout: float, returncode: int | None, log: pathlib.Path, unit: str) -> tuple[str, str]:
    """The finding kind and detail for a Python process that stopped on ``unit`` ('case', 'script')."""
    if stop == 'stalled':
        return 'timeout', f'Python did not return within {2 * timeout + 30:.0f}s'
    if stop == 'memory':
        return 'memory', f'Python passed the memory limit of {audit.config.MEMORY_LIMIT / 1024**3:g} GiB'
    return 'process-died', f'Python process died on this {unit} (exit status {returncode}); see {log}'


class PyLauncher:
    """Starts ``python -m <module> <job.json>`` for a shard, appending its output to ``log``, and remembers the last process."""

    def __init__(self, work: pathlib.Path, name: str, module: str, log: pathlib.Path) -> None:
        self.job_path = work / f'{name}-job.json'
        self.module = module
        self.log = log
        self.last: subprocess.Popen[bytes] | None = None

    def start(self, job: Json) -> subprocess.Popen[bytes]:
        self.job_path.write_text(json.dumps(job))
        with self.log.open('ab') as handle:
            self.last = subprocess.Popen([sys.executable, '-m', self.module, str(self.job_path)], stdout=handle, stderr=subprocess.STDOUT)
        return self.last

    def returncode(self) -> int | None:
        return self.last.returncode if self.last else None


def run_py_shard(work: pathlib.Path, i: int, fixtures_path: pathlib.Path, shard: list[audit.cases.Case], timeout: float) -> None:
    """Python side of shard ``i``, run in its own process so that one dying costs a case, not the step; writes ``py-{i}.json`` as
    ``{"results": [...]}`` in shard order."""
    r_path = work / f'r-{i}.json'
    r_by_id = {r['id']: r for r in json.loads(r_path.read_text())['results']}
    log = work / f'py-{i}.log'
    launcher = PyLauncher(work, f'py-{i}', 'audit.pyrun', log)

    def start(cases_path: pathlib.Path, out_path: pathlib.Path) -> subprocess.Popen[bytes]:
        return launcher.start(
            {'fixtures_path': str(fixtures_path), 'cases_path': str(cases_path), 'r_path': str(r_path), 'out_path': str(out_path), 'timeout': timeout}
        )

    def dead(case_id: str, stop: Stop, parts: list[Json]) -> Json:
        kind, detail = py_stop(stop, timeout, launcher.returncode(), log, 'case')
        r = r_by_id.get(case_id) or {}
        return {
            'id': case_id,
            'r_status': r.get('status'),
            'py_status': 'error',
            'r_error': r.get('error'),
            'py_error': detail,
            'secs_r': r.get('secs'),
            'secs_py': None,
            'findings': [{'kind': kind, 'path': '', 'detail': detail}],
        }

    results = run_shard(work, f'py-{i}', shard, timeout, start, dead, audit.cases.write_cases)
    (work / f'py-{i}.json').write_text(json.dumps({'results': results}))


def run_cases(cases: list[audit.cases.Case], fixtures: list[Json], work: pathlib.Path, jobs: int, timeout: float) -> list[Json]:
    """Execute every case in R, then in Python against R's outcome; returns one result per case, in case order."""
    work.mkdir(parents=True, exist_ok=True)
    fixtures_path = work / 'fixtures.json'
    fixtures_path.write_text(json.dumps(fixtures))
    n = max(1, min(jobs, len(cases)))
    shards = audit.cases.shards(cases, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as threads:
        list(threads.map(lambda i: run_r_shard(work, i, fixtures_path, shards[i], timeout), range(len(shards))))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as threads:
        list(threads.map(lambda i: run_py_shard(work, i, fixtures_path, shards[i], timeout), range(len(shards))))
    by_id: dict[str, Json] = {}
    for i in range(len(shards)):
        for result in json.loads((work / f'py-{i}.json').read_text())['results']:
            by_id[result['id']] = result
    return [by_id[c.id] for c in cases if c.id in by_id]
