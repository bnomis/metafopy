# Review: the direct-tests-for-the-audit-harness spec

**Review page:** https://claude.ai/artifact/Jq13nRr5LiB9QCYFkErDmc
**Date:** 2026-11-22

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/direct-tests-for-the-audit-harness/spec.md` (with
`measurements.md`), for the `TODO.md` item "Audit-harness and bin helpers with branches have no direct test". It tests, or dismisses in the
auditme ledger, every function under `src/audit` and `bin` that the T1 or T2 question finds untested. Approve or flag the spec, and answer the
four decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->
- Pass: the item is still open at `73ffe43`: `phaseb.py report` shows 117 open T1 and 12 open T2 candidates under `src/audit` and `bin`;
  none is under `src/tests`.
- Pass: every file and helper the spec names exists (`refutil.needs_rscript`, `refutil.posix_only`, `src/audit/data/r-versions.json`, the
  harness test files); the new names clash with nothing in `src`, `bin` or `agents/specs`.
- Pass: hypothesis (`>=6.100`) is already a dev dependency.
- Pass: the spec has 8 top-level sections; pre-commit passes on it.
<!-- doc-counts: accept-end -->

## The items

### The spec: direct tests for the audit harness

- **File:** agents/specs/direct-tests-for-the-audit-harness/spec.md (and measurements.md), version 1.0, proposed 2026-11-22
- **What it does:** For the TODO.md item "Audit-harness and bin helpers with branches have no direct test" (auditme review-1, question T1, 2026-11-21; plus the harness T2 property-test findings moved there on 2026-11-22). Measured 2026-11-22 at 73ffe43: 117 open T1 and 12 open T2 candidates under src/audit and bin, in five groups: A R-library shims (rlib.py, 37), B transpiler (transpile.py, 11), C pure helpers (47 T1, 10 T2), D drivers of R, git, subprocesses and sockets (13), E bin scripts (9 T1, 2 T2). Steps: (1) privatise with __all__ if D3 is yes; (2) re-judge T1 and T2, since every verdict predates the current briefs; (3) an R snippet fixture: a new generator src/tests/r/generate_audit_snippets.R records R's parse tree and value per snippet, and src/tests/auditsnippets.py runs each through the transpiler and shims and compares with R; (4-7) tests for groups A, B, C and E; (8) re-judge, dismiss group D (and any bin script D4 dismisses) in the ledger with the reason that bin/metafor-audit.sh --recall covers it, then close the spec and the item.
- **Other:** No harness behaviour change. R's ast() moves unchanged from src/audit/r/runscripts.R into src/audit/r/common.R so the generator writes the same trees the scripts step reads; after that move --recall is due and you start it. A test that finds a defect stops its task as blocked and files it with /todo-add and a strict xfail; it is never fixed here. Hypothesis tests run at max_examples=100, deadline=None; numeric cells agree with R to a relative 1e-12.
- **Recommendation: approve.**

## Decisions

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->

### D1. Which groups get direct tests, and which are dismissed in the ledger?

**Background:** This is the item's own question. The recall check (bin/metafor-audit.sh --recall, src/audit/README.md) reproduces what earlier audits found by hand, and so finds a hole only where it hides one of those items; only you run it, neither CI nor a hook does, since it needs the pinned R library. Audit 8 found holes in valuetree and over-wide known.toml globs that had hidden real differences; recall found four items of audit 6 that the globs had swallowed. Groups, measured 2026-11-22 at 73ffe43: A shims 37, B transpiler 11, C pure helpers 47 T1 + 10 T2 (printdiff, valuetree, report, apidiff, universe, ledger, ...), D drivers 13.

**What it affects:** The size of the task list's test steps (4 to 8), and which harness code rests on recall alone.

**Why the recommendation:** Group C holds the comparator (printdiff, valuetree) that decides what counts as a difference, which recall can only check against past findings; the drivers already have shard-survival tests with fake processes and fakes would test wiring, not R or git.

- **Test A, B and C; dismiss D (Recommended)**: Shims, translation and the comparison/report helpers each get direct tests; the 13 driver functions are dismissed as covered by --recall.
- **Test A and B only; dismiss C and D**: About half the tests; the comparator and report helpers rest on recall alone.
- **Test everything, D with fakes**: The most work: monkeypatched subprocess, socket and signal for the drivers; nothing dismissed except as D4 says.

### D2. The oracle for the R-library shims and the transpiler

**Background:** Groups A and B copy R behaviour, so R is the oracle (AGENTS.md: R metafor is the second numeric oracle after the publication). The pinned R is in src/audit/data/r-versions.json (R 4.6.2, metafor 5.2-1). CI installs no R, and tests that need it carry refutil.needs_rscript, which skips with 'Rscript is not on the path (CI installs no R)'.

**What it affects:** Whether these tests run in CI, the suite's run time, and a new generator to keep in step with r-versions.json.

**Why the recommendation:** A committed fixture runs everywhere, CI included, and freezes the oracle at the pin as the audit's own oracle is; a version check in load_snippet fails loudly when the pin moves.

- **Values recorded from R into a committed fixture (Recommended)**: New src/tests/r/generate_audit_snippets.R writes src/tests/reference/audit/snippets.json; tests need no R; regenerate when r-versions.json moves (a mismatch fails naming both versions).
- **Rscript run by the tests**: Always the installed R; every group A and B test skips where R is absent, so CI never runs them, and each module pays R's start-up.
- **Hand-written expected values**: No R at all, but values typed from knowledge of R, the source the project ranks below running R.

### D3. Declare each src/audit module's public names with __all__?

**Background:** Only src/audit/rlib.py has an __all__, so the audit checker counts every other helper as public, and a test of its caller does not cover it. The direct-tests spec for src/metafopy (its decision D1, answered 2026-11-21) made the same change there. Measured 2026-11-22: 42 of the 71 public open T1 candidates already have a test of a caller.

**What it affects:** Every src/audit module except rlib.py, and how many group B and C candidates need a test of their own.

**Why the recommendation:** It follows the decision already made for src/metafopy, and tests through the entry points (translate_unit, classify, first_difference) then count, instead of tests pinning the transpiler's internal tr_* layout.

- **Yes, add __all__ (Recommended)**: Helpers used only inside their module become private; new tests target what no caller reaches.
- **No**: Every public helper gets a direct test, including the internal tr_* functions, so tests pin the internal layout.

### D4. The bin scripts: test or dismiss?

**Background:** bin/logo.py is a PEP 723 script whose dependencies (uharfbuzz among them) are not in the project venv; it builds the committed logo files. bin/metafor-audit.py::metafopy_source is a pure argv reader. The bin/platform-probe.py helpers already run under src/tests/test_platform_probe.py's main().

**What it affects:** Group E: 9 T1 and 2 T2 candidates, and whether any test skips on every run.

**Why the recommendation:** The two scripts in the project environment are cheap to test; logo.py could only be tested by a test that always skips, which covers nothing.

- **Test metafor-audit.py and platform-probe.py; dismiss logo.py (Recommended)**: logo.py is dismissed with the reason that its dependencies are outside the project environment and its output is the committed logo, checked by eye.
- **Test all three, importorskip for logo.py**: The logo.py tests skip in every project run.
- **Dismiss all three as developer scripts**: No bin tests; the 11 group E candidates are dismissed.

<!-- doc-counts: accept-end -->

## Answers

Answered on 2026-11-22 on the review page (every answer the recommended option, no comments):

- The spec: approved.
- D1: test groups A, B and C; dismiss group D in the ledger as covered by `--recall`.
- D2: values recorded from R into a committed fixture (`src/tests/r/generate_audit_snippets.R`, `src/tests/reference/audit/snippets.json`).
- D3: yes, add `__all__` to each `src/audit` module.
- D4: test `bin/metafor-audit.py` and `bin/platform-probe.py`; dismiss `bin/logo.py`.
