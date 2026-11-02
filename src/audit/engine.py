"""Run cases in R and in Python and collect the findings."""

import concurrent.futures
import json
import pathlib
import subprocess
import time
import typing

import audit.cases
import audit.config
import audit.pyrun
import audit.rdriver

Json = dict[str, typing.Any]
Stop = typing.Literal['died', 'memory', 'stalled']


def read_outcomes(path: pathlib.Path) -> tuple[dict[str, Json], str | None]:
    """The finished cases of a runner's JSON-lines file and the case it was running when it stopped (None when it finished cleanly)."""
    done: dict[str, Json] = {}
    running: str | None = None
    if not path.exists():
        return done, None
    for line in path.read_text(errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'start' in doc:
            running = doc['start']
        else:
            done[doc['id']] = doc
            running = None
    return done, running


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


def run_r_shard(work: pathlib.Path, i: int, fixtures_path: pathlib.Path, shard: list[audit.cases.Case], timeout: float) -> None:
    """Run a shard in R; when R dies on a case (memory, a crash) record that case as an R failure and restart on the cases that remain."""
    out = work / f'r-{i}.jsonl'
    finished: dict[str, Json] = {}
    remaining = list(shard)
    attempts = 0
    while remaining and attempts <= len(shard):
        attempts += 1
        part = work / f'cases-{i}-rest.json'
        audit.cases.write_cases(part, remaining)
        proc = audit.rdriver.start(audit.config.R_DIR / 'runcases.R', work / f'r-{i}.log', str(fixtures_path), str(part), str(out), str(timeout))
        stop = watch(proc, out, 2 * timeout + 30, audit.config.MEMORY_LIMIT)
        done, running = read_outcomes(out)
        finished.update(done)
        if running is not None:
            reason = f'R did not return within {2 * timeout + 30:.0f}s (timeout)' if stop == 'stalled' else 'R process died on this case'
            finished[running] = {'id': running, 'status': 'error', 'error': reason, 'warnings': []}
        remaining = [c for c in remaining if c.id not in finished]
        if stop is None and running is None:
            break
        out.rename(work / f'r-{i}-attempt{attempts}.jsonl')
    (work / f'r-{i}.json').write_text(json.dumps({'results': [finished[c.id] for c in shard if c.id in finished]}))


def run_cases(cases: list[audit.cases.Case], fixtures: list[Json], work: pathlib.Path, jobs: int, timeout: float) -> list[Json]:
    """Execute every case in R, then in Python against R's outcome; returns one result per case, in case order."""
    work.mkdir(parents=True, exist_ok=True)
    fixtures_path = work / 'fixtures.json'
    fixtures_path.write_text(json.dumps(fixtures))
    n = max(1, min(jobs, len(cases)))
    shards = audit.cases.shards(cases, n)
    case_paths = []
    for i, shard in enumerate(shards):
        path = work / f'cases-{i}.json'
        audit.cases.write_cases(path, shard)
        case_paths.append(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as threads:
        list(threads.map(lambda i: run_r_shard(work, i, fixtures_path, shards[i], timeout), range(len(shards))))

    jobs_py = [(str(fixtures_path), str(path), str(work / f'r-{i}.json'), str(work / f'py-{i}.json'), timeout) for i, path in enumerate(case_paths)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
        outs = list(pool.map(audit.pyrun.run_shard, jobs_py))
    by_id: dict[str, Json] = {}
    for out in outs:
        for result in json.loads(pathlib.Path(out).read_text())['results']:
            by_id[result['id']] = result
    return [by_id[c.id] for c in cases if c.id in by_id]
