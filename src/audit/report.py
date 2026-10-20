"""The report: only what is wrong or missing, grouped so that one cause reads as one line."""

import collections
import json
import pathlib
import re
import subprocess
import typing

import audit.cases
import audit.config
import audit.ledger
import audit.steps
import audit.universe

SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ('Missing functions and arguments', ('missing-function', 'missing-arg')),
    ('Different defaults', ('default-diff',)),
    ('Missing fields of results', ('missing-field',)),
    ('Crashes', ('crash', 'fixture-error', 'runner-error')),
    ('Timeouts', ('timeout',)),
    ('Refused by metafopy where R works', ('refused-where-R-works',)),
    ('Accepted by metafopy where R refuses', ('accepts-where-R-refuses',)),
    ('Wrong values', ('value-diff', 'shape-diff', 'type-diff')),
    ('Wrong labels', ('label-diff',)),
    ('Printed output', ('print-diff',)),
    ('Warnings', ('warning-missing', 'warning-extra')),
    ('Other', ('universe-changed',)),
]

Json = dict[str, typing.Any]


DIGEST_ROWS = 15

Finding = audit.steps.Finding


def function_of(f: Finding, fixture_function: dict[str, str] | None = None) -> str:
    """The function the finding is about: the R function of the case, for a fixture the function that fits it."""
    parts = f.id.split('/')
    name = parts[1] if len(parts) > 1 else f.id
    if f.step == 'construct' and fixture_function and name in fixture_function:
        return fixture_function[name]
    return name


def result_class(f: Finding, classes: dict[str, str]) -> str:
    """The class of the result whose field is missing: the fit class for matrix and options cases, the fixture's for construct cases."""
    parts = f.id.split('/')
    if f.step in ('matrix', 'options') and len(parts) > 2:
        return classes.get(re.sub(r'#\d+$', '', parts[2]), parts[1])
    if f.step == 'construct' and len(parts) > 1:
        return classes.get(parts[1], parts[1])
    return 'results of the metafor examples and tests' if f.step == 'scripts' else function_of(f)


def normal_path(path: str) -> str:
    """The path of a difference without positions and without names that come from the data (``$Alegret (2001)``)."""
    path = re.sub(r'\[\d+\]', '[]', path)
    return '$'.join(re.sub(r'^.*[ ()].*$', '<name>', part) for part in path.split('$'))


def cause_key(f: Finding) -> tuple[str, str]:
    """What makes two findings the same cause: the kind and the path, or for findings without a path the message with its numbers masked."""
    if f.kind in ('missing-arg', 'missing-function') and f.path:
        return f.kind, f.path.replace('_', '.') if '_' in f.path and '.' not in f.path else f.path
    if f.path:
        return f.kind, normal_path(f.path)
    if f.kind == 'print-diff':
        return f.kind, re.sub(r'\d+(\.\d+)?', '#', f.detail)[:70]
    return f.kind, re.sub(r"\d+(\.\d+)?|'[^']*'|\"[^\"]*\"", '#', f.detail)[:90]


def line_of(kind: str, key: str, items: list[Finding], functions: list[str]) -> str:
    cases = len({i.id for i in items})
    shown = ', '.join(f'`{n}`' for n in functions[:4]) + (f' and {len(functions) - 4} more' if len(functions) > 4 else '')
    if kind in ('missing-arg', 'missing-function') and re.fullmatch(r'[\w.]+', key):
        return f'- `{key}` is missing from {shown}'
    label = f'`{key}`' if key and (kind in ('value-diff', 'shape-diff', 'type-diff', 'label-diff', 'missing-field')) else ''
    example = next((i.detail for i in items if i.detail), '')
    return '- ' + ' '.join(x for x in (label, example[:140], f'({cases} cases; {shown})') if x)


def git_head() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=audit.config.METAFOPY_SRC.parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return 'unknown'


def write(ctx: audit.steps.Context, outcomes: dict[str, audit.steps.Outcome], out: pathlib.Path) -> pathlib.Path:
    ledger = audit.ledger.Ledger([]) if ctx.no_ledger else audit.ledger.Ledger.load()
    new: list[audit.steps.Finding] = []
    known: list[tuple[audit.steps.Finding, audit.ledger.Entry]] = []
    drift = 0
    for outcome in outcomes.values():
        for f in outcome.findings:
            if f.kind == 'value-drift':
                drift += 1
                continue
            entry = ledger.classify(f.id, {'kind': f.kind, 'path': f.path, 'detail': f.detail})
            if entry is None:
                new.append(f)
            else:
                known.append((f, entry))
    ran_all = ctx.only is None and not ctx.quick and {s.name for s in audit.steps.STEPS} <= set(outcomes)

    fixtures = audit.cases.load_fixtures()
    fixture_function = {fx['name']: fx['fn'] for fx in fixtures if 'fn' in fx}

    causes: dict[tuple[str, str], list[Finding]] = collections.defaultdict(list)
    for f in new:
        causes[cause_key(f)].append(f)

    listed = {kind for _, kinds in SECTIONS for kind in kinds}

    def render(limit: int | None) -> list[str]:
        out_lines = [
            '# metafopy vs metafor: what is wrong or missing',
            '',
            f'metafopy `{git_head()}`, metafor {audit.config.METAFOR_VERSION}. One line is one cause; the same difference in many cases is one line. '
            f'Known differences (`data/known.toml`) are left out: {len(known)} findings matched them and {drift} numeric differences '
            f'below the reporting threshold were not listed. {len(new)} findings make {len(causes)} causes; every finding is in `findings.json`.',
            '',
        ]
        for title, kinds in SECTIONS:
            rows = []
            for (kind, key), items in causes.items():
                if kind not in kinds and not (title == 'Other' and kind not in listed):
                    continue
                functions = sorted({function_of(i, fixture_function) for i in items if i.step != 'scripts'}) or ['the examples and tests of metafor']
                rows.append((-len(functions), -len({i.id for i in items}), kind, key, items, functions))
            if not rows:
                continue
            rows.sort(key=lambda r: r[:4])
            out_lines += [f'## {title} ({len(rows)})', '']
            if limit is not None and kinds == ('missing-field',):
                names = [row[3].removeprefix('$') for row in rows]
                out_lines.append(
                    ', '.join(f'`{n}`' for n in names[:40]) + (f' and {len(names) - 40} more, in `wrong-or-missing-full.md`' if len(names) > 40 else '')
                )
                out_lines.append('')
                continue
            for row in rows[:limit]:
                out_lines.append(line_of(row[2], row[3], row[4], row[5]))
            if limit is not None and len(rows) > limit:
                out_lines.append(f'- and {len(rows) - limit} more causes, in `wrong-or-missing-full.md`')
            out_lines.append('')
        return out_lines

    lines = render(DIGEST_ROWS)
    full_lines = render(None)

    coverage = ctx.coverage
    notes: list[str] = []
    if coverage.get('uncovered_arguments'):
        notes.append(f'Arguments no case varies: {", ".join(coverage["uncovered_arguments"][:60])}')
    if coverage.get('unaccounted'):
        notes.append('; '.join(coverage['unaccounted'][:40]))
    if ran_all and ledger.stale():
        notes.append('Stale ledger entries: ' + ', '.join(f'{e.match} {e.kind}' for e in ledger.stale()))
    tail = ['## Not covered', ''] + [f'- {n}' for n in notes] + [''] if notes else []
    tail += ['## Steps', ''] + [f'- {name}: {len(o.findings)} findings; {"; ".join(o.notes)}' for name, o in outcomes.items()] + ['']

    path = out / 'wrong-or-missing.md'
    path.write_text('\n'.join(lines + tail))
    (out / 'wrong-or-missing-full.md').write_text('\n'.join(full_lines + tail))
    (out / 'findings.json').write_text(
        json.dumps(
            {
                'new': [dict(vars(f)) for f in new],
                'known': [{**vars(f), 'reason': e.reason} for f, e in known],
                'drift': drift,
                'full': ran_all,
                'coverage': coverage,
                'steps': {name: {'findings': len(o.findings), 'notes': o.notes} for name, o in outcomes.items()},
            },
            indent=1,
        )
    )
    return path


def rerender(directory: pathlib.Path, no_ledger: bool = False) -> pathlib.Path:
    """Write the report of an earlier run again from its ``findings.json`` with today's ledger (triage in seconds instead of hours)."""
    doc: Json = json.loads((directory / 'findings.json').read_text())
    by_step: dict[str, audit.steps.Outcome] = {}
    for row in [*doc['new'], *doc['known']]:
        finding = audit.steps.Finding(row['step'], row['id'], row['kind'], row['path'], row['detail'])
        by_step.setdefault(row['step'], audit.steps.Outcome()).findings.append(finding)
    for name, info in doc.get('steps', {}).items():
        by_step.setdefault(name, audit.steps.Outcome()).notes = info['notes']
    for _ in range(doc.get('drift', 0)):
        by_step.setdefault('drift', audit.steps.Outcome()).findings.append(audit.steps.Finding('drift', '', 'value-drift', '', ''))
    universe = audit.universe.Universe.load(audit.universe.snapshot_path())
    ctx = audit.steps.Context(
        work=directory, jobs=1, timeout=0.0, quick=False, only=None if doc.get('full', True) else ['partial'], no_ledger=no_ledger, universe=universe
    )
    ctx.coverage = doc.get('coverage', {})
    return write(ctx, by_step, directory)
