"""Helpers for the tests of the audit harness's R-library shims and transpiler against the R snippets recorded in reference/audit/snippets.json.

The fixture is written by tests/r/generate_audit_snippets.R. ``load_snippet`` refuses a fixture recorded under another R or metafor than the one
``src/audit/data/r-versions.json`` pins.
"""

import dataclasses
import functools
import json
import pathlib
import typing

import audit.config
import audit.rlib
import audit.scripts
import audit.transpile
import audit.valuetree
import metafopy

SNIPPETS_PATH = pathlib.Path(__file__).parent / 'reference' / 'audit' / 'snippets.json'
SNIPPET_RTOL = 1e-12
SNIPPET_SEED = 1234


@dataclasses.dataclass(frozen=True)
class Snippet:
    """One R snippet of reference/audit/snippets.json: its id, R text, R's parse tree of each top-level expression, R's value of the last."""

    id: str
    r_text: str
    units: list[dict[str, typing.Any]]
    value: audit.valuetree.Node
    deparsed: str | None
    error: str | None = None


@functools.cache
def read_fixture(path: pathlib.Path) -> dict[str, typing.Any]:
    document: dict[str, typing.Any] = json.loads(path.read_text(encoding='utf-8'))
    return document


def require_pinned_versions(meta: dict[str, typing.Any], path: pathlib.Path) -> None:
    pinned: dict[str, str] = json.loads((audit.config.DATA_DIR / 'r-versions.json').read_text(encoding='utf-8'))
    recorded_r = f'{meta["R_major"]}.{meta["R_minor"]}'
    if recorded_r != pinned['R'] or meta['metafor'] != pinned['metafor']:
        raise ValueError(
            f'{path} was recorded under R {recorded_r} and metafor {meta["metafor"]}, but r-versions.json pins R {pinned["R"]} '
            f'and metafor {pinned["metafor"]}; regenerate it under the pin with src/tests/r/generate_audit_snippets.R'
        )


def load_snippet(snippet_id: str, path: pathlib.Path = SNIPPETS_PATH) -> Snippet:
    """The recorded snippet; fails naming both versions when the fixture's metafor or R differs from src/audit/data/r-versions.json."""
    document = read_fixture(path)
    require_pinned_versions(document['meta'], path)
    record = document['snippets'][snippet_id]
    return Snippet(
        id=record['id'],
        r_text=record['r_text'],
        units=record['units'],
        value=record['value'],
        deparsed=record.get('deparsed'),
        error=record.get('error'),
    )


def run_in_python(snippet: Snippet) -> typing.Any:
    """Each unit through audit.transpile.translate_unit, run in one audit.rlib.Env as audit.scripts.run_script runs a script; the last value."""
    env = audit.rlib.Env()
    audit.rlib.SETTINGS.clear()
    ns = audit.scripts.namespace(env)
    assigned: set[str] = set()
    metafopy.set_seed(SNIPPET_SEED)
    value: typing.Any = None
    for unit in snippet.units:
        # eval is safe here: the code is the transpiler's output for the committed R snippets, run as audit.scripts.run_script runs it
        value = eval(audit.transpile.translate_unit(unit, assigned, set()), ns)
        assigned |= audit.transpile.assignment_targets(unit)
    return value


def numeric_mismatches(r: audit.valuetree.Node, py: audit.valuetree.Node, path: str, out: list[str]) -> None:
    """The paths where a numeric cell of the two trees differs by more than SNIPPET_RTOL."""
    if r['k'] == 'num' and py['k'] == 'num':
        for i, (a, b) in enumerate(zip(r['v'], py['v'], strict=True)):
            # R writes 2.0 as 2, so a cell loads as int or float; "Inf" and null are not compared here
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) > SNIPPET_RTOL * max(abs(a), abs(b)):
                out.append(f'{path}[{i}]: R {a!r}, Python {b!r}')
    elif r['k'] == 'df':
        for rc, pc in zip(r['cols'], py['cols'], strict=True):
            numeric_mismatches(rc['node'], pc['node'], f'{path}${rc["name"]}', out)
    elif r['k'] == 'list':
        for ri, pi in zip(r['items'], py['items'], strict=True):
            numeric_mismatches(ri['node'], pi['node'], f'{path}${ri.get("name")}', out)


def assert_same_as_r(snippet: Snippet, value: typing.Any) -> None:
    """Fail unless audit.valuetree.compare finds no Diff between R's node and to_node(value), and numeric cells agree to SNIPPET_RTOL."""
    assert snippet.error is None, f'{snippet.id}: R raises "{snippet.error}", so there is no value to compare with'
    missing: list[str] = []
    tree = audit.valuetree.to_node(value, snippet.value, '', missing)
    assert not missing, f'{snippet.id}: fields of the R value missing in Python: {missing}'
    diffs: list[audit.valuetree.Diff] = []
    audit.valuetree.compare(snippet.value, tree, '', diffs)
    assert not diffs, f'{snippet.id}: differs from R: {diffs}'
    mismatches: list[str] = []
    numeric_mismatches(snippet.value, tree, '', mismatches)
    assert not mismatches, f'{snippet.id}: numeric cells differ from R by more than {SNIPPET_RTOL}: {mismatches}'
