"""The scripts step: metafor's own help examples and test files, run in R and, translated, in Python, one top-level expression at a time.

R runs each expression (``r/runscripts.R``) and records its parse tree and value; ``audit.transpile`` turns the tree into Python, ``audit.rlib`` runs
it, and the values are compared exactly as in the other steps. Expressions that cannot be translated or run are counted and listed as coverage.
"""

import collections
import concurrent.futures
import functools
import hashlib
import json
import pathlib
import re
import tarfile
import typing
import urllib.request

import audit.config
import audit.pyrun
import audit.rdriver
import audit.rlib
import audit.transpile
import metafopy
import metafopy.errors

Json = dict[str, typing.Any]

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
    import math

    import numpy

    return {
        'E': env,
        'EMPTY': audit.rlib.EMPTY,
        'OP': audit.rlib.OPERATORS,
        'block': audit.rlib.block,
        'math': math,
        'numpy': numpy,
        'rassign_dollar': audit.rlib.r_assign_dollar,
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


def run_script(script: Json, nse: set[str], timeout: float) -> tuple[list[Json], collections.Counter[str]]:
    env = audit.rlib.Env()
    audit.rlib.OPTIONS.clear()
    ns = namespace(env)
    assigned: set[str] = set()
    stale: set[str] = set()
    findings: list[Json] = []
    tally: collections.Counter[str] = collections.Counter()
    for i, unit in enumerate(script['units']):
        case_id = f'scripts/{script["topic"]}/{script["kind"]}/{i}'
        tally['units'] += 1
        metafopy.set_seed(1234)
        targets = audit.transpile.assignment_targets(unit['ast'])
        try:
            code = audit.transpile.translate_unit(unit['ast'], assigned, nse)
        except audit.transpile.UntranslatableError as e:
            tally[f'untranslated: {e}'] += 1
            stale |= targets
            continue
        if audit.transpile.symbols(unit['ast']) & stale:
            # it reads a variable that an earlier unit changed in R but not in Python (a unit that could not be translated or that failed)
            tally['skipped: reads a variable that differs between R and Python'] += 1
            stale |= targets
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
            continue
        cls = py.get('error_class', '')
        if cls == 'UnsupportedError':
            tally[f'unsupported: {py["error"][:70]}'] += 1
            continue
        tally['compared'] += 1
        for f in audit.pyrun.classify(unit, py, value):
            snippet = unit['src'].replace('\n', ' ')[:90]
            findings.append({'id': case_id, 'kind': f['kind'], 'path': f['path'], 'detail': f'{f["detail"]} [{snippet}]'.strip()})
    return findings, tally


def run_shard(job: tuple[str, str, list[str], float]) -> str:
    """Process entry point: (R results of the shard, output path, names of the functions with a ``data`` argument, timeout) -> output path."""
    r_path, out_path, nse_names, timeout = job
    audit.config.limit_memory()
    audit.rlib.METAFOR_FUNCTIONS.update(json.loads(pathlib.Path(out_path + '.functions').read_text()))
    doc = json.loads(pathlib.Path(r_path).read_text())
    findings: list[Json] = []
    tally: collections.Counter[str] = collections.Counter()
    for script in doc['scripts']:
        f, t = run_script(script, set(nse_names), timeout)
        findings += f
        tally.update(t)
    pathlib.Path(out_path).write_text(json.dumps({'findings': findings, 'tally': tally}))
    return out_path


def run(
    work: pathlib.Path, jobs: int, timeout: float, universe_functions: dict[str, Json], only: list[str] | None
) -> tuple[list[Json], collections.Counter[str]]:
    source = fetch_source(audit.config.WORK)
    out = work / 'scripts'
    out.mkdir(parents=True, exist_ok=True)
    n = max(1, jobs)
    env = audit.rdriver.environment()
    if only:
        env['AUDIT_ONLY'] = '\x1f'.join(only)
    import subprocess

    procs = []
    for i in range(n):
        path = out / f'r-{i}.json'
        log = (out / f'r-{i}.log').open('wb')
        procs.append(
            (
                path,
                subprocess.Popen(
                    audit.rdriver.rscript_command(audit.config.R_DIR / 'runscripts.R', str(source), str(path), str(i), str(n), str(timeout)),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=audit.rdriver.limit_memory,
                ),
            )
        )
    for path, proc in procs:
        proc.wait()
        if not path.exists():
            raise RuntimeError(f'R script shard failed, see {path.with_suffix(".log")}')
    nse = sorted(name for name, entry in universe_functions.items() if any(f['name'] == 'data' for f in entry['formals']))
    (out / 'functions.json').write_text(json.dumps(sorted(universe_functions)))
    shard_jobs = []
    for i in range(n):
        target = out / f'py-{i}.json'
        pathlib.Path(str(target) + '.functions').write_text(json.dumps(sorted(universe_functions)))
        shard_jobs.append((str(out / f'r-{i}.json'), str(target), nse, timeout))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
        outs = list(pool.map(run_shard, shard_jobs))
    findings: list[Json] = []
    tally: collections.Counter[str] = collections.Counter()
    for shard_out in outs:
        doc = json.loads(pathlib.Path(shard_out).read_text())
        findings += doc['findings']
        tally.update(doc['tally'])
    return findings, tally
