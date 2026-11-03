"""The steps of the audit, in the order the one command runs them. Each step returns findings; none decides what counts as a gap."""

import collections.abc
import dataclasses
import fnmatch
import json
import pathlib
import platform
import re
import typing

import audit.apidiff
import audit.cases
import audit.config
import audit.domains
import audit.engine
import audit.generate
import audit.pyrun
import audit.rdriver
import audit.scripts
import audit.universe
import metafopy.core.dbinom

Json = dict[str, typing.Any]


@dataclasses.dataclass
class Finding:
    step: str
    id: str
    kind: str
    path: str
    detail: str

    def key(self) -> str:
        return f'{self.id}|{self.kind}|{self.path}'


@dataclasses.dataclass
class Context:
    work: pathlib.Path
    jobs: int
    timeout: float
    quick: bool
    only: list[str] | None
    update_universe: bool = False
    no_ledger: bool = False
    allow_r_drift: bool = False
    universe: audit.universe.Universe | None = None
    generated: audit.generate.Generated | None = None
    coverage: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    results: dict[str, list[Json]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Outcome:
    findings: list[Finding] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Step:
    name: str
    title: str
    run: collections.abc.Callable[[Context], Outcome]


def library_drift(lock: Json, installed: Json) -> dict[str, tuple[str | None, str | None]]:
    """Every package (and ``R``) whose installed version differs from the pin, as name -> (pinned, installed); a package missing on either
    side has None there."""
    return {name: (lock.get(name), installed.get(name)) for name in sorted(lock.keys() | installed.keys()) if lock.get(name) != installed.get(name)}


def check_r(ctx: Context) -> Outcome:
    """Check the installed R library against ``data/r-versions.json``. A missing pin file or any drift raises RuntimeError naming each
    difference and how to proceed, unless ``ctx.allow_r_drift``, when the drift becomes a note in the report."""
    audit.rdriver.run(
        audit.config.R_DIR / 'versions.R',
        str(ctx.work / 'r-versions.json'),
    )
    versions: Json = json.loads((ctx.work / 'r-versions.json').read_text())
    out = Outcome()
    if versions.get('metafor') != audit.config.METAFOR_VERSION:
        raise RuntimeError(f'metafor {versions.get("metafor")} is installed, the oracle is {audit.config.METAFOR_VERSION}')
    lock_path = audit.config.DATA_DIR / 'r-versions.json'
    if not lock_path.exists():
        raise RuntimeError(
            'no data/r-versions.json: pin the R library first by copying the r-versions.json that `Rscript src/audit/r/versions.R <out.json>` '
            'writes to src/audit/data/r-versions.json'
        )
    lock: Json = json.loads(lock_path.read_text())
    drift = library_drift(lock, versions)
    pinned_fma = lock.get('R_fma')
    python_fma = 'fused' if metafopy.core.dbinom.FUSED else 'unfused'
    if pinned_fma is not None and pinned_fma != python_fma:
        message = (
            f'the pinned R_fma is {pinned_fma} but metafopy.core.dbinom.FUSED is {python_fma} (platform {platform.machine()}): '
            'random gosh subsets would differ from R. Either run on the platform the pin was made on, or reinstall an R build that matches '
            "this platform and re-pin by copying a run's r-versions.json over src/audit/data/r-versions.json; or run once with --allow-r-drift"
        )
        if not ctx.allow_r_drift:
            raise RuntimeError(message)
        out.notes.append(f'{message} (--allow-r-drift)')
    if drift:
        listing = ', '.join(f'{name} {pinned} -> {installed}' for name, (pinned, installed) in drift.items())
        if not ctx.allow_r_drift:
            raise RuntimeError(
                f'the R library differs from data/r-versions.json (pinned -> installed): {listing}. Either reinstall the pinned versions with '
                "`Rscript src/audit/r/setup.R <library> <versions.json>`; or re-pin by copying a run's r-versions.json over "
                'src/audit/data/r-versions.json after reviewing what the new versions change; or run once with --allow-r-drift'
            )
        out.notes.append(f'R library differs from the pin (--allow-r-drift): {listing}')
    out.notes.append(f'R {versions.get("R")}, metafor {versions.get("metafor")}; {len(versions)} packages')
    return out


def build_universe(ctx: Context) -> Outcome:
    path = ctx.work / 'universe.json'
    # auditme: accept[build-name-has-side-effect] the step runs R to enumerate metafor; the name is the step id
    audit.rdriver.run(audit.config.R_DIR / 'universe.R', str(audit.config.DATA_DIR / 'fixtures.json'), str(path))
    ctx.universe = audit.universe.Universe.load(path)
    out = Outcome()
    snapshot = audit.universe.snapshot_path()
    if ctx.update_universe:
        snapshot.write_text(path.read_text())
    elif snapshot.exists() and json.loads(snapshot.read_text()) != json.loads(path.read_text()):
        out.findings.append(
            Finding(
                'universe',
                'universe',
                'universe-changed',
                '',
                'the R universe differs from data/universe.json; refresh it with `bin/metafor-audit.sh --update-universe`',
            )
        )
    out.notes.append(f'{len(ctx.universe.functions)} functions, {len(ctx.universe.methods)} S3 methods, {len(ctx.universe.datasets)} datasets')
    return out


def generate_cases(ctx: Context) -> audit.generate.Generated:
    if ctx.generated is None:
        assert ctx.universe is not None
        fixtures = audit.cases.load_fixtures()
        ctx.generated = audit.generate.generate(ctx.universe, audit.domains.Domains.load(), audit.domains.Recipes.load(), fixtures)
        ctx.coverage['uncovered_arguments'] = ctx.generated.uncovered
        ctx.coverage['unaccounted'] = ctx.generated.unaccounted
    return ctx.generated


def selected(case_id: str, only: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(case_id, part) if re.search(r'[*?]', part) else part in case_id for part in only)


def case_step(axis: str) -> collections.abc.Callable[[Context], Outcome]:
    def run(ctx: Context) -> Outcome:
        gen = generate_cases(ctx)
        cases = [c for c in gen.cases if c.axis == axis]
        if ctx.only:
            cases = [c for c in cases if selected(c.id, ctx.only)]
        if ctx.quick:
            cases = cases[::10]
        out = Outcome()
        if not cases:
            out.notes.append('no cases')
            return out
        results = audit.engine.run_cases(cases, audit.cases.load_fixtures(), ctx.work / axis, ctx.jobs, ctx.timeout)
        ctx.results[axis] = results
        for r in results:
            for f in r['findings']:
                out.findings.append(Finding(axis, r['id'], f['kind'], f['path'], f['detail']))
        out.notes.append(f'{len(results)} cases')
        return out

    return run


def api_step(ctx: Context) -> Outcome:
    assert ctx.universe is not None
    env, errors = audit.pyrun.build_env(audit.cases.load_fixtures())
    out = Outcome()
    doms = audit.domains.Domains.load()
    generics = set(ctx.universe.generics())
    methods = {m['method'] for m in ctx.universe.methods}
    standalone = sorted(n for n in ctx.universe.functions if n not in generics and n not in methods)
    found = audit.apidiff.run(
        ctx.universe, env, doms, frozenset(fx['name'] for fx in audit.cases.load_fixtures() if fx.get('meta', {}).get('derived'))
    ) + audit.apidiff.run_functions(ctx.universe, standalone, doms)
    for case_id, kind, path, detail in found:
        out.findings.append(Finding('api', case_id, kind, path, detail))
    for name, message in errors.items():
        out.findings.append(Finding('api', f'api/fixture/{name}', 'fixture-error', '', message))
    out.notes.append(f'{len(found)} differences in formals')
    return out


def scripts_step(ctx: Context) -> Outcome:
    assert ctx.universe is not None
    out = Outcome()
    only: list[str] | None = None
    if ctx.only:
        stripped = [part.removeprefix('scripts/').replace('*', '') for part in ctx.only]
        if any(part.startswith('scripts/') and not stripped[i] for i, part in enumerate(ctx.only)):
            only = None
        else:
            only = [part for part in stripped if part]
    findings, tally = audit.scripts.run(ctx.work, ctx.jobs, ctx.timeout, ctx.universe.functions, only)
    for f in findings:
        out.findings.append(Finding('scripts', f['id'], f['kind'], f['path'], f['detail']))
    ctx.coverage['scripts'] = dict(tally.most_common(60))
    out.notes.append(f'{tally["units"]} expressions, {tally["compared"]} compared')
    return out


STEPS = (
    Step('check-r', 'The R oracle is installed and pinned', check_r),
    Step('universe', 'Enumerate metafor from R itself', build_universe),
    Step('matrix', 'Every generic and model function on every fit class', case_step('matrix')),
    Step('options', 'One argument at a time on every swept fit', case_step('options')),
    Step('construct', 'One argument at a time on every fitting function', case_step('construct')),
    Step('standalone', 'Functions that take no model', case_step('standalone')),
    Step('emmeans', 'The emmeans package on the reference grids of metafor fits', case_step('emmeans')),
    Step('datasets', 'Every dataset of metafor and metadat', case_step('datasets')),
    Step('scripts', 'metafor help examples and testthat files, run in R and in translated Python', scripts_step),
    Step('api', 'Formals of each function and method against the Python callable of each fit class', api_step),
)


ALWAYS = frozenset({'check-r', 'universe'})
