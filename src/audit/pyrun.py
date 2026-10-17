"""Runs a shard of cases in Python and classifies each against the outcome R gave for the same case."""

import contextlib
import io
import json
import pathlib
import signal
import time
import typing
import warnings

import matplotlib
import matplotlib.pyplot
import numpy
import pandas

import audit.config
import metafopy
import metafopy.errors
from audit import printdiff, pycall, rprint, valuetree

matplotlib.use('Agg')

Json = dict[str, typing.Any]

DOMAIN_ERRORS = frozenset(
    {
        metafopy.errors.ArgumentError.__name__,
        metafopy.errors.EstimationError.__name__,
        metafopy.errors.InputError.__name__,
        metafopy.errors.NotFoundError.__name__,
        metafopy.errors.NotSupportedError.__name__,
    }
)


class CaseTimeoutError(Exception):
    pass


def raise_timeout(signum: int, frame: object) -> typing.NoReturn:
    raise CaseTimeoutError


def quiet() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def build_env(fixtures: list[Json]) -> tuple[pycall.Env, dict[str, str]]:
    env: pycall.Env = {}
    errors: dict[str, str] = {}
    for fx in fixtures:
        try:
            with warnings.catch_warnings(), quiet():
                warnings.simplefilter('ignore')
                env[fx['name']] = pycall.run_call(fx, env)
        except Exception as e:
            errors[fx['name']] = f'{type(e).__name__}: {e}'
    return env, errors


def print_text(value: object) -> str:
    """What a Python user sees when printing ``value``, with plain pandas and numpy values in R's layout (see :mod:`audit.rprint`)."""
    if isinstance(value, pandas.DataFrame):
        return rprint.frame_text(value) + '\n'
    if isinstance(value, pandas.Series):
        return rprint.vector_text(value) + '\n'
    if isinstance(value, numpy.ndarray) and value.dtype.kind in 'fiub' and value.ndim <= 2:
        return rprint.matrix_text(value) + '\n'
    if isinstance(value, dict) and value and all(isinstance(k, str) for k in value):
        return ''.join(f'${k}\n{print_text(v)}\n' for k, v in value.items())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(value)
    return buf.getvalue()


def run_guarded(call: typing.Callable[[], object], timeout: float) -> tuple[Json, object]:
    """Call under a timeout, capturing warnings and output; returns a record with status 'ok' or 'error' (and the error's class) and the value."""
    record: Json = {'status': 'ok', 'warnings': []}
    value: object = None
    caught: list[warnings.WarningMessage] = []
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    start = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught, quiet():
            warnings.simplefilter('always')
            value = call()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        record.update(status='error', error_class=type(e).__name__, error=str(e)[:300])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        matplotlib.pyplot.close('all')
    record['secs'] = time.monotonic() - start
    record['warnings'] = sorted({str(w.message)[:200] for w in caught})
    return record, value


def execute(case: Json, env: pycall.Env, fixture_errors: dict[str, str], timeout: float) -> tuple[Json, object]:
    for name in case.get('needs', []):
        if name in fixture_errors:
            return {'status': 'error', 'warnings': [], 'error_class': 'FixtureError', 'error': f'fixture {name} failed: {fixture_errors[name]}'}, None
    return run_guarded(lambda: pycall.run_call(case['call'], env), timeout)


PLAIN_CLASSES = frozenset({'list', 'data.frame', 'matrix', 'array', 'numeric', 'character', 'logical', 'integer'})


def printed_class(node: Json) -> bool:
    """Whether the R value has a class of its own, hence a print method whose layout is part of metafor."""
    return bool(set(node.get('cls') or []) - PLAIN_CLASSES)


def classify(r: Json, py: Json, value: object) -> list[Json]:
    """The findings of one case: where Python and R disagree."""
    findings: list[Json] = []

    def add(kind: str, path: str = '', detail: str = '') -> None:
        findings.append({'kind': kind, 'path': path, 'detail': detail})

    r_ok = r['status'] == 'ok'
    py_ok = py['status'] == 'ok'
    cls = py.get('error_class', '')
    if r_ok and not py_ok:
        if cls == 'MissingFunctionError':
            add('missing-function', detail=py['error'])
        elif cls == 'MissingArgumentError':
            add('missing-arg', detail=py['error'])
        elif cls == 'CaseTimeoutError':
            add('timeout')
        elif cls == 'FixtureError':
            add('fixture-error', detail=py['error'])
        elif cls == 'MissingFunction':
            add('missing-function', detail=py['error'])
        elif cls == 'AttributeError' and ' has no field ' in py['error']:
            add('missing-field', detail=py['error'])
        elif cls in DOMAIN_ERRORS:
            add('refused-where-R-works', detail=f'{cls}: {py["error"]}')
        else:
            add('crash', detail=f'{cls}: {py["error"]}')
    elif not r_ok and py_ok:
        add('accepts-where-R-refuses', detail=f'R: {r.get("error", "")[:200]}')
    elif r_ok and py_ok:
        missing: list[str] = []
        tree = valuetree.to_node(value, r['value'], '', missing)
        for path in missing:
            add('missing-field', path)
        diffs: list[valuetree.Diff] = []
        valuetree.compare(r['value'], tree, '', diffs)
        for d in diffs:
            add(d.kind, d.path, d.detail)
        if 'text' in r and printed_class(r['value']):
            line = printdiff.first_difference(r['text'], print_text(value))
            if line is not None:
                add('print-diff', detail=line)
        elif 'text' in r and isinstance(value, dict) and r['text'].lstrip().startswith('$') and print_text(value).lstrip().startswith('{'):
            add('print-diff', detail='R prints the elements under $names, Python prints the repr of a dict')
        if r.get('warnings') and not py['warnings']:
            add('warning-missing', detail=str(r['warnings'][0])[:200])
        elif py['warnings'] and not r.get('warnings'):
            add('warning-extra', detail=str(py['warnings'][0])[:200])
    return findings


def run_shard(job: tuple[str, str, str, str, float]) -> str:
    """Process entry point: (fixtures.json, cases.json, r-results.json, out.json, timeout) -> out path."""
    fixtures_path, cases_path, r_path, out_path, timeout = job
    audit.config.limit_memory()
    fixtures = json.loads(pathlib.Path(fixtures_path).read_text())
    cases = json.loads(pathlib.Path(cases_path).read_text())
    r_doc = json.loads(pathlib.Path(r_path).read_text())
    r_by_id = {r['id']: r for r in r_doc['results']}
    metafopy.set_seed(1234)
    env, fixture_errors = build_env(fixtures)
    results = []
    for case in cases:
        r = r_by_id.get(case['id'])
        if r is None:
            results.append({'id': case['id'], 'findings': [{'kind': 'runner-error', 'path': '', 'detail': 'R produced no result'}]})
            continue
        metafopy.set_seed(1234)
        py, value = execute(case, env, fixture_errors, timeout)
        try:
            findings = classify(r, py, value)
        except Exception as e:  # a comparison that fails is the harness's fault, never silently a match
            findings = [{'kind': 'runner-error', 'path': '', 'detail': f'{type(e).__name__}: {e}'}]
        results.append(
            {
                'id': case['id'],
                'r_status': r['status'],
                'py_status': py['status'],
                'r_error': r.get('error'),
                'py_error': py.get('error'),
                'secs_r': r.get('secs'),
                'secs_py': py.get('secs'),
                'findings': findings,
            }
        )
    pathlib.Path(out_path).write_text(json.dumps({'results': results}))
    return out_path
