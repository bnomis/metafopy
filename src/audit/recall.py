"""The recall check: does the harness reproduce what the audits found by hand?

``data/recall.toml`` lists every item of the audits of 2026-10-15 and 2026-10-16 with the finding the harness must produce for it. ``--recall`` checks
out metafopy as it was when each audit looked, runs the harness against it with the ledger switched off, and reports which items it reproduced. The
differences between rounds that prompted the harness then cannot recur unnoticed: an item the harness misses is a hole in the harness.
"""

import dataclasses
import fnmatch
import json
import pathlib
import re
import subprocess
import sys
import tomllib
import typing

import audit.config

Json = dict[str, typing.Any]

STEPS = ('matrix', 'options', 'construct', 'standalone', 'datasets', 'scripts', 'api')


@dataclasses.dataclass
class Item:
    id: str
    audit: int
    text: str
    find: list[Json]
    hint: list[str]
    reach: str
    why: str

    def matches(self, finding: Json) -> bool:
        for detector in self.find:
            if all(fnmatch.fnmatchcase(str(finding.get(key, '')), str(detector.get(key, '*'))) for key in ('id', 'kind', 'path', 'detail')):
                return True
        return False


@dataclasses.dataclass
class Corpus:
    audits: dict[int, Json]
    items: list[Item]

    @classmethod
    def load(cls) -> Corpus:
        doc = tomllib.loads((audit.config.DATA_DIR / 'recall.toml').read_text())
        items = [Item(i['id'], i['audit'], i['text'], i.get('find', []), i.get('hint', []), i.get('reach', ''), i.get('why', '')) for i in doc['item']]
        ids = [i.id for i in items]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f'duplicate recall items: {duplicates}')
        return cls({int(k): v for k, v in doc['audit'].items()}, items)


def literal_prefix(pattern: str) -> str:
    return re.split(r'[*?\[]', pattern, maxsplit=1)[0]


def hints(items: list[Item]) -> list[str]:
    """Substrings of case ids (and of script topics) that select every case the items need, so a recall run costs a fraction of a full run."""
    out: set[str] = set()
    for item in items:
        if item.hint:
            out.update(item.hint)
            continue
        for detector in item.find:
            pattern = detector.get('id', '')
            if pattern.startswith('scripts/'):
                out.add(pattern.split('/')[1])
            elif pattern:
                out.add(pattern)
    return sorted(out)


def steps_for(items: list[Item]) -> list[str]:
    found = {literal_prefix(d.get('id', '')).split('/')[0] for item in items for d in item.find}
    return [s for s in STEPS if s in found]


def evaluate(items: list[Item], findings: list[Json]) -> dict[str, bool]:
    return {item.id: any(item.matches(f) for f in findings) for item in items}


def checkout(commit: str) -> pathlib.Path:
    """A detached worktree of metafopy at ``commit``, reused between runs."""
    path = audit.config.WORK / 'worktrees' / commit
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'worktree', 'add', '--detach', str(path), commit], cwd=audit.config.ROOT, check=True, capture_output=True)
    return path


def run_audit(number: int, spec: Json, items: list[Item], out: pathlib.Path, jobs: int, full: bool) -> list[Json]:
    """Run the harness against metafopy as of the audit's commit; the findings of the run, unfiltered."""
    worktree = checkout(spec['commit'])
    run_dir = out / f'audit{number}'
    command = [
        sys.executable,
        str(audit.config.ROOT / 'bin' / 'metafor-audit.py'),
        '--metafopy-src',
        str(worktree / 'src'),
        '--out',
        str(run_dir),
        '--jobs',
        str(jobs),
        '--no-ledger',
    ]
    if not full:
        command += ['--steps', ','.join(steps_for(items))]
        for hint in hints(items):
            command += ['--only', hint]
    subprocess.run(command, check=True)
    doc: Json = json.loads((run_dir / 'findings.json').read_text())
    findings: list[Json] = doc['new'] + doc['known']
    return findings


def write(out: pathlib.Path, corpus: Corpus, results: dict[int, dict[str, bool]], commits: dict[int, str]) -> tuple[pathlib.Path, bool]:
    lines = ['# Recall: what the harness reproduces of the earlier audits', '']
    ok = True
    total_hit = total = 0
    for number, hit in sorted(results.items()):
        items = [i for i in corpus.items if i.audit == number]
        got = [i for i in items if hit[i.id]]
        missed = [i for i in items if not hit[i.id]]
        total_hit += len(got)
        total += len(items)
        lines += [f'## Audit {number} (metafopy `{commits[number]}`): {len(got)} of {len(items)} reproduced', '']
        for i in missed:
            reason = f' ({i.why})' if i.reach == 'none' else ''
            lines.append(f'- missed `{i.id}`: {i.text}{reason}')
            if i.reach != 'none':
                ok = False
        lines.append('')
    lines.insert(2, f'{total_hit} of {total} items reproduced.')
    lines.insert(3, '')
    path = out / 'recall.md'
    path.write_text('\n'.join(lines))
    return path, ok
