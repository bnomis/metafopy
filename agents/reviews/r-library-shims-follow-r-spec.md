# Review: the r-library-shims-follow-r spec

**Review page:** https://claude.ai/artifact/T3BYEYJN9Y1Ta8psN2GX2S
**Date:** 2026-11-23

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/r-library-shims-follow-r/spec.md` (with `measurements.md`
and two measurement scripts), for the `TODO.md` item "R-library shims disagree with R". It makes each shim in `src/audit/rlib.py` that the
harness's direct tests found disagreeing with R give R's value, or refuse the call and say so. Approve or flag the spec, and answer the two
decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-23 - a dated record, not a claim about the present -->
- Pass: the item is still open at `3324124`: `test_harness_rlib.py` gives 93 passed and 13 strict xfails.
- Pass: the last recall run (`recall-20261123-061000`) reproduced 176 of 176 items, the baseline the closing task compares against.
- Pass: the spec has 7 top-level sections; pre-commit passes on it and its measurement files.
<!-- doc-counts: accept-end -->

## The items

### The spec: R-library shims follow R

- **File:** agents/specs/r-library-shims-follow-r/spec.md (with measurements.md and the two reach-*.R measurement scripts), version 1.0, proposed 2026-11-23
- **What it does:** For the TODO.md item "R-library shims disagree with R" (found 2026-11-22: 13 strict-xfail cases over 12 shims in src/audit/rlib.py). Every case is fixed to R's behaviour except the mixed list() call (decision D1). Tasks: (1) new snippets, including head_list and a matrix-times-vector case, and regenerate the R fixture; (2) class shims: r_class and r_inherits use R's class names instead of Python type names; (3) names: r_as_dataframe, r_cbind, r_isna and r_matrix keep R's row and column names; (4) vectors: r_length(NULL) is 0, r_as_numeric of one string, r_seq counting back from to, and recycled repeating operands as R does, including a vector against a matrix, where numpy today gives 10 20 60 80 for R's 10 40 30 80 without any error; (5) lists: r_tail and r_head by position (head has the same defect as tail), and r_list as D1 says; (6) close: the list stops for you to run bin/metafor-audit.sh --recall, then compares it with the 2026-11-23 run (176 of 176) before moving the item to DONE.md.
- **Other:** Measured 2026-11-23 at 3324124 on the 191 scripts the scripts step runs, counting both call sites and calls at run time in R: no audited script reaches any of the 13 cases on the Python side, so audit reports and the recall result should not change. A change in recall stops the task; known.toml and recall.toml are not edited to fit. No metafopy output changes. Choices made, each reversible: r_head is fixed with r_tail; r_class refuses an object it has no R class for; class(1:2) reads numeric because the transpiler writes integers as floats (recorded in the docstring).
- **Recommendation: approve.**

## Decisions

<!-- doc-counts: accept-start measured on 2026-11-23 - a dated record, not a claim about the present -->

### D1. list() mixing named and unnamed arguments: narrow the shim or fix it to R?

**Background:** R's list(1, b = 'x') is a list whose first element has no name. r_list returns a Python list for unnamed arguments and a dict for named ones, so it keys the unnamed element '1' (rlib.py:181). Measured 2026-11-23 at 3324124: the audited scripts call list() at 211 sites in 49 scripts, none mixing named and unnamed arguments, and none does so at run time.

**What it affects:** r_list only if narrowed; if fixed, a new list type plus the transpiler and the shims that read lists (r_dollar, r_index, r_index2, r_names, r_length, r_lapply, r_c).

**Why the recommendation:** No script makes this call, and a real fix touches the transpiler and seven shims; refusing it turns an invented difference into an 'unsupported' count, and stays reversible when a script needs it.

- **Narrow the contract (Recommended)**: r_list raises UnsupportedError for a mixed call and its docstring says so; a script that makes one is counted unsupported, not as a missing-field difference.
- **Fix to R**: A named-list type built from arguments in order, read by seven shims, with a transpiler change; the most work, for a call no script makes.

### D2. If D1 narrows: how does the list_mixed case leave the tests?

**Background:** The TODO item's fix direction (2026-11-22) says to drop a narrowed case. Keeping it instead as a test that the shim refuses the call means the refusal itself is tested and R's value stays recorded in the fixture.

**What it affects:** src/tests/test_harness_rlib.py, the generator src/tests/r/generate_audit_snippets.R and the fixture.

**Why the recommendation:** Dropping the case leaves nothing testing that the shim refuses rather than gives a wrong value, and needs a fixture regeneration for no gain.

- **Keep the snippet as a refusal test (Recommended)**: A REFUSED list in test_harness_rlib.py whose test expects audit.rlib.UnsupportedError; R's value stays recorded for a later fix.
- **Remove the snippet**: list_mixed leaves the generator and the fixture; only the docstring records the contract, and nothing tests the refusal.

<!-- doc-counts: accept-end -->

## Answers

Answered on 2026-11-23 on the review page (every answer the recommended option, no comments):

- The spec: approved.
- D1: narrow the contract: `r_list` raises `UnsupportedError` for a call mixing named and unnamed arguments.
- D2: keep the `list_mixed` snippet as a refusal test (a `REFUSED` list in `src/tests/test_harness_rlib.py`).
