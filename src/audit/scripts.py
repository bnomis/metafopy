"""The scripts step: metafor's own help examples and test files, run in R and, translated, in Python, one top-level expression at a time.

R runs each expression (``r/runscripts.R``) and records its parse tree and value; ``audit.transpile`` turns the tree into Python, ``audit.rlib`` runs
it, and the values are compared exactly as in the other steps. Expressions that cannot be translated or run are counted and listed as coverage.
"""

import collections
import collections.abc
import concurrent.futures
import dataclasses
import functools
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import tarfile
import typing
import urllib.request

import numpy

import audit.cases
import audit.config
import audit.engine
import audit.pyrun
import audit.rdriver
import audit.rlib
import audit.transpile
import metafopy
import metafopy.errors

Json = dict[str, typing.Any]

# auditme: accept[non-alphabetical-literal] the primary URL is tried first, the archive second
SOURCE_URLS = (
    'https://cran.r-project.org/src/contrib/metafor_5.2-1.tar.gz',
    'https://cran.r-project.org/src/contrib/Archive/metafor/metafor_5.2-1.tar.gz',
)
ENVIRONMENT_ERRORS = re.compile(r'there is no package|could not find function|unable to find an inherited method|cannot open|no applicable method')


def fetch_source(work: pathlib.Path) -> pathlib.Path:
    """Download the pinned source of metafor (its help pages and tests are the scripts), verified against ``data/metafor-source.sha256``."""
    target = work / 'source'
    done = target / 'metafor'
    if done.exists():
        return done
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'metafor.tar.gz'
    expected = (audit.config.DATA_DIR / 'metafor-source.sha256').read_text().split()[0]
    last: Exception | None = None
    for url in SOURCE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        except OSError as e:
            last = e
            continue
        if hashlib.sha256(data).hexdigest() == expected:
            archive.write_bytes(data)
            break
        last = RuntimeError(f'{url} does not match the pinned checksum')
    else:
        raise RuntimeError(f'cannot fetch the source of metafor {audit.config.METAFOR_VERSION}: {last}')
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter='data')
    return done


def namespace(env: audit.rlib.Env) -> dict[str, object]:
    return {
        'E': env,
        'EMPTY': audit.rlib.EMPTY,
        'OP': audit.rlib.OPERATORS,
        'block': audit.rlib.block,
        'math': math,
        'numpy': numpy,
        'rc_pairs': audit.rlib.r_c_pairs,
        'rassign_dollar': audit.rlib.r_assign_dollar,
        'rassign_into': audit.rlib.r_assign_into,
        'rassign_index': audit.rlib.r_assign_index,
        'rassign_names': audit.rlib.r_assign_names,
        'rcall': audit.rlib.rcall,
        'rdollar': audit.rlib.r_dollar,
        'rfor': audit.rlib.rfor,
        'rindex': audit.rlib.r_index,
        'rindex2': audit.rlib.r_index2,
        'rsym': audit.rlib.r_symbol,
        'rnse': audit.rlib.rnse,
        'rwith': audit.rlib.r_with,
        'truth': audit.rlib.truth,
    }


R_STALL = re.compile(r'^R did not return within \d+s \(timeout\)$')


def run_script(script: Json, nse: set[str], timeout: float) -> collections.abc.Iterator[tuple[list[Json], collections.Counter[str]]]:
    """Run the units of one script and yield each unit's findings and tally. A lost script (R died in it) ends with its dead unit, which is
    reported as ``r-killed`` (or counted as an R failure for a stall) and not translated, and the script's remainder is counted once."""
    env = audit.rlib.Env()
    audit.rlib.SETTINGS.clear()
    ns = namespace(env)
    assigned: set[str] = set()
    stale: set[str] = set()
    units = script['units']
    for i, unit in enumerate(units):
        case_id = f'scripts/{script["topic"]}/{script["kind"]}/{i}'
        findings: list[Json] = []
        tally: collections.Counter[str] = collections.Counter()
        tally['units'] += 1
        if script.get('lost') and i == len(units) - 1:
            error = unit.get('error', '')
            if audit.pyrun.r_killed(error):
                findings.append({'id': case_id, 'kind': 'r-killed', 'path': '', 'detail': f'R: {error[:200]}'})
            elif R_STALL.match(error):
                tally['R failed (skipped)'] += 1
            tally['not run: R process died partway through the script'] += 1
            yield findings, tally
            continue
        metafopy.set_seed(1234)
        targets = audit.transpile.assignment_targets(unit['ast'])
        try:
            code = audit.transpile.translate_unit(unit['ast'], assigned, nse)
        except audit.transpile.UntranslatableError as e:
            # auditme: accept[except-interpolation] the exception message is the tally key that the report lists as coverage
            tally[f'untranslated: {e}'] += 1
            stale |= targets
            yield findings, tally
            continue
        if audit.transpile.symbols(unit['ast']) & stale:
            # it reads a variable that an earlier unit changed in R but not in Python (a unit that could not be translated or that failed)
            tally['skipped: reads a variable that differs between R and Python'] += 1
            stale |= targets
            yield findings, tally
            continue
        stale -= targets
        py, value = audit.pyrun.run_guarded(functools.partial(eval, code, ns), timeout)
        if unit['status'] != 'ok' or py.get('error') is not None:
            # the two sides no longer hold the same value in what the unit assigns to
            stale |= targets
        if unit['status'] != 'ok':
            tally['R failed (skipped)'] += 1
            if not ENVIRONMENT_ERRORS.search(unit.get('error', '')):
                tally[f'R error: {unit.get("error", "")[:60]}'] += 1
            yield findings, tally
            continue
        cls = py.get('error_class', '')
        if cls == 'UnsupportedError':
            tally[f'unsupported: {py["error"][:70]}'] += 1
            yield findings, tally
            continue
        tally['compared'] += 1
        for f in audit.pyrun.classify(unit, py, value):
            snippet = unit['src'].replace('\n', ' ')[:90]
            findings.append({'id': case_id, 'kind': f['kind'], 'path': f['path'], 'detail': f'{f["detail"]} [{snippet}]'.strip()})
        yield findings, tally


def stream_scripts(r_path: pathlib.Path, scripts_path: pathlib.Path, out_path: pathlib.Path, nse: list[str], timeout: float) -> None:
    """Run the scripts in ``scripts_path`` against R's records in ``r_path``, appending to ``out_path`` a ``{"start": id}`` line before each script,
    one ``{"part": id, "findings": [...], "tally": {...}}`` line per unit and a closing ``{"id": id}`` line, flushed each time."""
    audit.config.limit_memory()
    by_id = {s['id']: s for s in json.loads(r_path.read_text())['scripts']}
    wanted = [entry['id'] for entry in json.loads(scripts_path.read_text())]
    with out_path.open('a') as out:

        def write(doc: Json) -> None:
            out.write(json.dumps(doc) + '\n')
            out.flush()

        for script_id in wanted:
            write({'start': script_id})
            for findings, tally in run_script(by_id[script_id], set(nse), timeout):
                write({'part': script_id, 'findings': findings, 'tally': tally})
            write({'id': script_id})


@dataclasses.dataclass(frozen=True)
class Script:
    id: str  # 'rd/<topic>' or 'test/<topic>'
    kind: str  # 'rd' or 'test'
    topic: str
    file: str  # path inside the metafor source: 'man/<topic>.Rd' or 'tests/testthat/test_<topic>.R'


def list_scripts(source: pathlib.Path, work: pathlib.Path, only: list[str] | None) -> list[Script]:
    """Every script ``runscripts.R list`` finds in the metafor source, in R's order, filtered by ``only`` (substrings of a script id)."""
    out_path = work / 'scripts.json'
    log = work / 'list.log'
    out_path.unlink(missing_ok=True)
    env = audit.rdriver.environment()
    if only:
        env['AUDIT_ONLY'] = '\x1f'.join(only)
    with log.open('wb') as handle:
        subprocess.run(
            audit.rdriver.rscript_command(audit.config.R_DIR / 'runscripts.R', 'list', str(source), str(out_path)),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    try:
        return [Script(**entry) for entry in json.loads(out_path.read_text())]
    except (OSError, ValueError, TypeError) as e:
        raise RuntimeError(f'the list of scripts could not be read ({type(e).__name__}), see {log}') from e


def write_scripts(path: pathlib.Path, scripts: list[Script]) -> None:
    path.write_text(json.dumps([dataclasses.asdict(s) for s in scripts]))


def run_r_scripts(work: pathlib.Path, i: int, source: pathlib.Path, shard: list[Script], timeout: float) -> None:
    """R side of scripts shard ``i`` through ``engine.run_shard``; when R dies in a script, the units it finished are kept, the dead unit is an error
    and the script is marked lost. Writes ``r-{i}.json`` as ``{"scripts": [...]}`` in shard order."""
    by_id = {s.id: s for s in shard}

    def start(scripts_path: pathlib.Path, out_path: pathlib.Path) -> subprocess.Popen[bytes]:
        return audit.rdriver.start(audit.config.R_DIR / 'runscripts.R', work / f'r-{i}.log', 'run', str(source), str(scripts_path), str(out_path), str(timeout))

    def dead(script_id: str, stop: audit.engine.Stop, parts: list[Json]) -> Json:
        script = by_id[script_id]
        unit: Json = {'status': 'error', 'error': audit.engine.r_stop_reason(stop, timeout, 'unit'), 'warnings': [], 'src': '', 'ast': None}
        return {'id': script.id, 'kind': script.kind, 'topic': script.topic, 'lost': True, 'units': [p['unit'] for p in parts] + [unit]}

    results = audit.engine.run_shard(work, f'r-{i}', shard, timeout, start, dead, write_scripts)
    for record in results:
        if 'parts' in record:
            record['units'] = [p['unit'] for p in record.pop('parts')]
        record.setdefault('units', [])
    (work / f'r-{i}.json').write_text(json.dumps({'scripts': results}))


def run_py_scripts(work: pathlib.Path, i: int, shard: list[Script], nse: list[str], functions: list[str], timeout: float) -> None:
    """Python side of scripts shard ``i`` through ``engine.run_shard``, in its own process so that one dying costs the rest of a script, not the
    step. Writes ``py-{i}.json`` as ``{"findings": [...], "tally": {...}}`` from the parts."""
    r_path = work / f'r-{i}.json'
    unit_counts = {s['id']: len(s['units']) for s in json.loads(r_path.read_text())['scripts']}
    log = work / f'py-{i}.log'
    launcher = audit.engine.PyLauncher(work, f'py-{i}', 'audit.scripts', log)

    def start(scripts_path: pathlib.Path, out_path: pathlib.Path) -> subprocess.Popen[bytes]:
        job = {'r_path': str(r_path), 'scripts_path': str(scripts_path), 'out_path': str(out_path), 'nse': nse, 'functions': functions, 'timeout': timeout}
        return launcher.start(job)

    def dead(script_id: str, stop: audit.engine.Stop, parts: list[Json]) -> Json:
        kind, detail = audit.engine.py_stop(stop, timeout, launcher.returncode(), log, 'script')
        script = next(s for s in shard if s.id == script_id)
        left = max(0, unit_counts.get(script_id, 0) - len(parts) - 1)
        tally: collections.Counter[str] = collections.Counter()
        findings: list[Json] = []
        for part in parts:
            findings += part['findings']
            tally.update(part['tally'])
        tally['units'] += 1 + left
        if left:
            tally['not run: Python process died partway through the script'] += left
        findings.append({'id': f'scripts/{script.topic}/{script.kind}/{len(parts)}', 'kind': kind, 'path': '', 'detail': detail})
        return {'id': script_id, 'findings': findings, 'tally': dict(tally)}

    results = audit.engine.run_shard(work, f'py-{i}', shard, timeout, start, dead, write_scripts)
    findings: list[Json] = []
    tally: collections.Counter[str] = collections.Counter()
    for record in results:
        if 'parts' in record:
            for part in record['parts']:
                findings += part['findings']
                tally.update(part['tally'])
        else:
            findings += record.get('findings', [])
            tally.update(record.get('tally', {}))
    (work / f'py-{i}.json').write_text(json.dumps({'findings': findings, 'tally': tally}))


def run(
    work: pathlib.Path, jobs: int, timeout: float, universe_functions: dict[str, Json], only: list[str] | None
) -> tuple[list[Json], collections.Counter[str]]:
    source = fetch_source(audit.config.WORK)
    out = work / 'scripts'
    out.mkdir(parents=True, exist_ok=True)
    n = max(1, jobs)
    shards = [s for s in audit.cases.shards(list_scripts(source, out, only), n) if s]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as threads:
        list(threads.map(lambda i: run_r_scripts(out, i, source, shards[i], timeout), range(len(shards))))
    nse = sorted(name for name, entry in universe_functions.items() if any(f['name'] == 'data' for f in entry['formals']))
    functions = sorted(universe_functions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as threads:
        list(threads.map(lambda i: run_py_scripts(out, i, shards[i], nse, functions, timeout), range(len(shards))))
    findings: list[Json] = []
    tally: collections.Counter[str] = collections.Counter()
    for i in range(len(shards)):
        doc = json.loads((out / f'py-{i}.json').read_text())
        findings += doc['findings']
        tally.update(doc['tally'])
    return findings, tally


if __name__ == '__main__':
    # The shard process: argv[1] is a job file holding stream_scripts's arguments and the metafor function names (an internal entry point, not a CLI).
    job_doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
    audit.rlib.METAFOR_FUNCTIONS.update(job_doc['functions'])
    stream_scripts(
        pathlib.Path(job_doc['r_path']),
        pathlib.Path(job_doc['scripts_path']),
        pathlib.Path(job_doc['out_path']),
        job_doc['nse'],
        job_doc['timeout'],
    )
