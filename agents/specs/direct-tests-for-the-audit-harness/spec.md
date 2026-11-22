# Direct tests for the audit harness

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-22

Every function under `src/audit` and `bin` that the auditme T1 question (a branching function has no test) or T2 question (a pure function
with a well-defined domain has no hypothesis test) finds untested is either tested or dismissed in the ledger with its reason. The R-library
shims and the transpiler come first, tested against R itself; the pure comparison and report helpers next; the code that drives R, git and
subprocesses is dismissed as covered by the recall check.

## Motivation

<!-- doc-counts: accept-start found on 2026-11-21 and measured on 2026-11-22 - a dated record, not a claim about the present -->
The `TODO.md` item "Audit-harness and bin helpers with branches have no direct test" was found on 2026-11-21 by auditme review-1
(`agents/audit-reviews/review-1.md`, question T1, at `31d809f`): 108 functions under `src/audit` and 9 in `bin/logo.py`,
`bin/metafor-audit.py` and `bin/platform-probe.py`. Decision D4 of that review (answered 2026-11-21) filed it. A note of 2026-11-22 added the
T2 findings under `src/audit` and `bin`, moved there by decision D2 of `agents/reviews/property-tests-for-metafopy-helpers-spec.md`.

[measurements.md](measurements.md), measured on 2026-11-22 at `73ffe43`, lists every open candidate and its group:

| Group | Rule (first match wins) | T1 | T2 |
|---|---|---|---|
| A. R-library shims | in `src/audit/rlib.py` | 37 | 0 |
| B. Transpiler | in `src/audit/transpile.py` | 11 | 0 |
| D. Drivers | it, or a function it calls on every path, starts a process, opens a socket or URL, installs a signal handler or timer, or writes outside a directory it is given | 13 | 0 |
| E. Scripts | under `bin/` | 9 | 2 |
| C. Pure helpers | every other candidate under `src/audit` | 47 | 10 |

Every verdict predates the current briefs (T1 brief version 6, T2 version 3), so `phaseb.py plan` re-plans all of them; the open set is
re-judged before any test is written.
<!-- doc-counts: accept-end -->

The cause: the harness is exercised end to end by `bin/metafor-audit.sh --recall`, and by step tests, not function by function. The recall
check runs only when the operator starts it; neither CI nor a hook runs it, since it needs the pinned R library.

## Module location

- `src/tests/r/generate_audit_snippets.R` (new): records R's parse tree and value for each R snippet, into
  `src/tests/reference/audit/snippets.json` (new)
- `src/tests/auditsnippets.py` (new): the snippet helpers of the API section, imported as `import auditsnippets`, like `refutil`
- `src/tests/test_harness_rlib.py` (new): group A
- `src/tests/test_harness_transpile.py`: group B
- Group C, in the harness test file that already imports the module: `test_harness_compare.py` (`printdiff`, `valuetree`, `report`, `steps`),
  `test_harness_pycall.py` (`pycall`), `test_harness_accounts.py` (`generate`, `domains`, `ledger`, `universe`), `test_harness_engine.py`
  (`engine`), `test_harness_errors.py` (`pyrun`); `test_harness_apidiff.py` (new) for `apidiff`
- Group E: `src/tests/test_platform_probe.py`; `src/tests/test_metafor_audit_script.py` (new)
- `src/audit/r/common.R`: gains `ast()`, moved unchanged from `src/audit/r/runscripts.R`, so the generator writes the trees the scripts
  step reads. No behaviour change. I chose a move over a copy so there is one parse-tree writer; it is reversible.
- `src/audit/*.py`: an `__all__` per module, only if decision D3 is answered yes
- `agents/audit-reviews/ledger/T1.jsonl`, `T2.jsonl`: the re-judged verdicts and the dismissals

## API

No harness API changes. `src/tests/auditsnippets.py` exposes:

```python
import dataclasses
import typing

import audit.valuetree


@dataclasses.dataclass(frozen=True)
class Snippet:
    """One R snippet of reference/audit/snippets.json: its id, R text, R's parse tree of each top-level expression, R's value of the last."""

    id: str
    r_text: str
    units: list[dict[str, typing.Any]]
    value: audit.valuetree.Node
    deparsed: str | None


def load_snippet(snippet_id: str) -> Snippet:
    """The recorded snippet; fails naming both versions when the fixture's metafor or R differs from src/audit/data/r-versions.json."""


def run_in_python(snippet: Snippet) -> typing.Any:
    """Each unit through audit.transpile.translate_unit, run in one audit.rlib.Env as audit.scripts.run_script runs a script; the last value."""


def assert_same_as_r(snippet: Snippet, value: typing.Any) -> None:
    """Fail unless audit.valuetree.compare finds no Diff between R's node and to_node(value), and numeric cells agree to SNIPPET_RTOL."""
```

Tests are plain pytest functions in the project's style, naming the candidate id in the docstring:

```python
@pytest.mark.parametrize('snippet_id', ['rep_each', 'rep_times_vector', 'rep_times_scalar', 'rep_length_out'])
def test_rep_matches_r(snippet_id: str) -> None:
    """src/audit/rlib.py::r_rep against R's rep() on each branch the T1 finding names."""
```

## Error handling

- **A test finds a defect** (a shim or translation that disagrees with R, a comparator that hides or invents a difference). The task stops as
  `blocked`, naming the function and the input. The defect goes to `TODO.md` with `/todo-add` and a strict `xfail` test (`reason=` the item's
  title); it is never fixed in this work, since a harness change changes what every audit reports and is followed by `--recall`. The branches
  that agree with R get their own passing test, so the candidate is still covered (an `xfail` test does not count under T1).
- **R or Rscript is missing.** Under D2's recommended option the tests read the committed fixture and need no R. The generator needs
  `Rscript` with the pinned metafor (5.2-1) and `jsonlite`; it stops with an error naming the installed and the pinned version. Under the live
  option, each R-dependent test carries `refutil.needs_rscript`, which skips with "Rscript is not on the path (CI installs no R)".
- **The fixture is from another R or metafor.** `load_snippet` fails, naming the recorded and the pinned versions; the fix is to
  regenerate under the pin. The generator is not added to `.github/workflows/r-fixtures.yml`, which installs the latest R.
- **A branch R text cannot reach** (`truth` on a length-zero condition raising `audit.rlib.UnsupportedError`, `accepts` on a callable
  without a signature, `r_par` against matplotlib axes): a hand-built expected value, with a comment saying why R is not the oracle.
- **A test runs for more than 10 seconds:** the existing `slow` marker. Harness code that needs `SIGALRM` or `resource`: `refutil.posix_only`.

## Implementation

1. **Privatise (only if D3 is yes).** Give each `src/audit` module other than `rlib.py` an `__all__` naming exactly the names another file
   under `src/audit`, `bin` or `src/tests` reads as `audit.<module>.<name>` at the start of the task. Run `mypy` and the harness tests.
2. **Re-judge T1 and T2.** `phaseb.py plan . --question T1 --out <dir>`; judge each batch with `~/.claude/skills/auditme/phaseb/briefs/T1.md`,
   then `check` and `merge`; the same for T2. The open `src/audit` and `bin` list from `report` is the work. A candidate not in
   measurements.md is grouped by the rules above.
3. **Snippet fixture.** Move `ast()` into `common.R`; `runscripts.R` sources it from there. Write the generator: a named list of R snippets;
   for each, `parse(text = )`, `ast()` of every expression, evaluation in a fresh environment after `set.seed(1234)` with `metafor` and
   `metadat` attached, `node_of()` of the last value (or the error message), `deparse()` for formula snippets, and `meta` with
   `R.version.string`, `R.version$major`/`minor` and `packageVersion("metafor")`. Run it with `LC_ALL=C.UTF-8 Rscript`. Write `auditsnippets.py`.
4. **Group A.** For each open shim, one snippet per case its finding names ("rep with each, a vector times, a scalar times and
   length.out" gives four), run through `run_in_python` and checked with `assert_same_as_r`. Inputs come from R text on bundled datasets
   (`dat$yi` is a Series in Python, `rma(...)` a fit, `split(dat, dat$alloc)` a dict), on fits whose values the parity fixtures already match.
   The private helpers (`accepts`, `as_array`, `format_cell`, `Named.__array_wrap__`, `recycled`, `truth`) are covered by the snippet that
   reaches them, or by a direct call where none does.
5. **Group B.** Snippets for every case each finding names: each node type, `$<-`, `[<-` and `f<-` replacements, plain, nested and
   superassignment targets, operators, `if`, `for`, `function` with and without defaults, lazy and NSE calls, reserved names, `pairmat()`
   inside a model call. A snippet that runs code is checked against R's value; `unparse` is checked against R's `deparse()` of the formula; a
   construct the transpiler refuses expects `audit.transpile.UntranslatableError`. If D3 is no, each helper also gets a direct call on a tree
   from the fixture (`audit.transpile.tr(tree, audit.transpile.Context(set(), set()))`, then evaluated).
6. **Group C.** A direct test per open T1 candidate, on hand-built inputs, asserting the value its docstring and `src/audit/README.md` state
   (for example: two printed numbers differing in the last digit are `close_numbers`; `0` against `0.0000` is not). `universe` tests build a
   `Universe` from a small dict in the shape of `data/universe.json`; `apidiff` and `pycall` tests use small fits of bundled datasets. Each open
   T2 candidate gets a hypothesis test of the property its ledger `reason` states, using `rstrategies` where a strategy fits.
7. **Group E,** as D4 decides: scripts are loaded by path with `importlib.util.spec_from_file_location`, as `test_platform_probe.py` does.
8. **Close.** Re-judge T1 and T2 (`plan` re-plans every file whose tests changed), `check`, `merge`. Then, for each open group D
   candidate (and group E ones D4 dismisses), `phaseb.py dismiss . --question <Q> <id> --reason "<text>"` for each of T1 and T2 it is
   open on, with the reason (for group E, the reason D4 gives):
   "Group D of agents/specs/direct-tests-for-the-audit-harness/spec.md (decision D1): drives R, git, subprocesses, sockets, signals or the
   network; covered end to end by bin/metafor-audit.sh --recall, which runs the harness steps against R on the commits of earlier audits." Dismiss
   after the last merge: a merge keeps a dismissal only while the candidate's hash or findings are unchanged, and new tests change both.
   `phaseb.py report` for T1 and T2 then lists no open `src/audit` or `bin` candidate. Say that `--recall` is due after the `ast()` move
   and give its command; the operator starts it. Set this spec to implemented and move the `TODO.md` item to `DONE.md`.

## Settings

Test parameters, constants in `auditsnippets.py` or decorators, as in the property-tests spec:

| Setting | Type | Default | Description |
|---|---|---|---|
| `auditsnippets.SNIPPET_RTOL` | `float` | `1e-12` | relative tolerance of a numeric cell against R; base R and the shims compute the same arithmetic |
| `hypothesis.settings(max_examples=)` | `int` | `100` | examples per property test (hypothesis's default) |
| `hypothesis.settings(deadline=)` | `None` | `None` | no per-example deadline |

## External dependencies

- pytest and hypothesis (`hypothesis>=6.100`), already dev dependencies in `pyproject.toml`: no new dependency.
- R 4.6.2 with metafor 5.2-1, metadat and jsonlite, as pinned in `src/audit/data/r-versions.json`, to run the generator (and every test
  of groups A and B under D2's live option). The library may be the one `METAFOPY_AUDIT_RLIB` names.
- The auditme Phase B tooling under `~/.claude/skills/auditme` (`phaseb.py`, the T1 and T2 briefs), run and not changed. Check it with
  `python3 ~/.claude/skills/auditme/checks/phaseb.py --help`.

## Decisions

**D1. Which groups get direct tests, and which are dismissed.** The item's own question. Background: the recall check (`src/audit/README.md`,
"The harness is checked against the audits") reproduces the items earlier audits found by hand, and finds holes only where they hide one of
those items; it is run by the operator, not by CI. Audit 8 found holes in `valuetree` and over-wide `known.toml` globs that had hidden real
differences; the recall check found four items of audit 6 that the globs had swallowed. Affects: the size of steps 4 to 8.
- *Test A, B and C; dismiss D (recommended).* The shims, the translation and the comparison that decides what is a difference are each
  checked; the drivers stay with recall, where `test_harness_engine.py` and `test_harness_scripts.py` already test shard survival with fake
  processes.
- *Test A and B only; dismiss C and D.* About half the tests; the comparator and report helpers rest on recall alone.
- *Test everything, D with fakes* (monkeypatched `subprocess`, `socket`, `signal`). The most work; fakes check wiring, not R or git.

**D2. The oracle for the shims and the transpiler.**
- *Values recorded from R into a committed fixture (recommended).* The suite runs without R, in CI too; the oracle is frozen at the pin, as
  the audit's own oracle is. Cost: a generator to maintain and regenerate when `r-versions.json` moves.
- *Rscript run by the tests.* Always the installed R; every group A and B test skips where R is absent, so CI never runs them, and each
  module pays R's start-up.
- *Hand-written expected values.* No R, but values typed from knowledge of R, the source the project ranks below running R.

**D3. Declare `src/audit` modules' public names with `__all__` (step 1).** Background: no `src/audit` module but `rlib.py` has an
`__all__`, so the checker counts every helper as public, and a test of its caller does not cover it. The direct-tests spec for `src/metafopy`
(decision D1, answered 2026-11-21) made the same change there. Measured on 2026-11-22, 42 of the 71 public open T1 candidates already have a
caller test. Affects: every `src/audit` module, and how many group B and C candidates need a test of their own.
- *Yes (recommended).* Helpers that are only called inside their module become private; tests through the entry points (`translate_unit`,
  `classify`, `first_difference`) count, and the new tests target what no caller reaches.
- *No.* Every public helper gets a direct test, including the transpiler's internal `tr_*` functions, so the tests pin the internal layout.

**D4. The `bin` scripts.** `bin/logo.py` is a PEP 723 script whose dependencies (`uharfbuzz` among them) are not in the project
environment; it builds the committed logo files. `bin/metafor-audit.py::metafopy_source` is a pure argv reader. The `bin/platform-probe.py`
helpers run under the probe's existing tests' `main()`.
- *Test `metafor-audit.py` (direct and property test) and `platform-probe.py` (direct, and `main()` on the groups that reach `ls_runs` and
  `measure_selmodel_decreasing`); dismiss `logo.py` (recommended),* with the reason that its dependencies are outside the project
  environment and its output is the committed logo, checked by eye.
- *Test all three,* with `pytest.importorskip` for `logo.py`, which then skips in every project run.
- *Dismiss all three* as developer scripts.

Answered on 2026-11-22 in `agents/reviews/direct-tests-for-the-audit-harness-spec.md` (version 1.1 records them): the spec approved; D1
test A, B and C and dismiss D; D2 the committed fixture recorded from R; D3 yes, so step 1 runs; D4 test `metafor-audit.py` and
`platform-probe.py`, dismiss `logo.py`.
