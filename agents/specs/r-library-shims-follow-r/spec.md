# R-library shims follow R

**Version:** 1.2
**Status:** implemented
**Date:** 2026-11-23

The R-library shims of the audit harness (`src/audit/rlib.py`), through which the scripts step runs metafor's own examples and tests
translated to Python, give R's value for every case the direct tests of the harness found them to miss, or refuse the case with
`UnsupportedError` and say so in their docstring. A shim that gives a wrong value or raises makes the scripts step report a difference that
is the harness's, not metafopy's, and can hide a real one behind it.

## Motivation

<!-- doc-counts: accept-start found on 2026-11-22 and measured on 2026-11-23 at 3324124 - a dated record, not a claim about the present -->
The `TODO.md` item "R-library shims disagree with R" was found on 2026-11-22 by the direct tests of the audit harness
([Direct tests for the audit harness](../direct-tests-for-the-audit-harness/spec.md)): the strict `xfail` cases of `DISAGREE` in
`src/tests/test_harness_rlib.py`, 13 cases over 12 shims, against R values recorded in `src/tests/reference/audit/snippets.json`. Cause: each
shim was written for the calls the audited scripts make, not for R's full behaviour.

[measurements.md](measurements.md), measured on 2026-11-23 at `3324124`, gives each case's R and shim values, its root cause and its reach.
No audited script reaches any case on the Python side: the only R-side hits (one-string `as.numeric()` behind `R.Version()`, a named
`cbind()` through `do.call()`) sit behind R functions the harness refuses. So the fixes should leave every audit report and the recall
result unchanged, and they keep the shims from inventing or hiding differences as the scripts translated grow.
<!-- doc-counts: accept-end -->

## Module location

- `src/audit/rlib.py`: `r_as_dataframe`, `r_as_numeric`, `r_cbind`, `r_class`, `r_head`, `r_inherits`, `r_isna`, `r_length`, `r_list`,
  `r_matrix`, `recycled`, `r_seq`, `r_tail`
- `src/tests/test_harness_rlib.py`: `DISAGREE` emptied into `MATCHING` (and `REFUSED`, if decision D1 narrows `r_list`)
- `src/tests/r/generate_audit_snippets.R` and `src/tests/reference/audit/snippets.json`: the new snippets of Task 1
- `TODO.md`, `DONE.md`: the item, at the end

## API

No signature changes; each shim keeps its `BASE` name and arguments. The docstrings state the contract, for example:

```python
def r_class(x: Value) -> Value:
    """``class(x)``: R's class vector, as an object array. A data frame is ``data.frame`` (``escalc``, ``data.frame`` with a measure), a fitted
    model has its metafor classes (``metafopy.methods.fitview.r_class``), and a value without a class attribute its implicit class: ``NULL``,
    ``logical``, ``numeric`` (also for an R integer, which the transpiler writes as a float), ``character``, ``list``, ``function``, and
    ``matrix``, ``array`` for a two-dimensional array.

    Raises:
        UnsupportedError: any other object (a metafopy result without a metafor class in Python).
    """


def recycled(a: Value, b: Value) -> tuple[Value, Value]:
    """The operands of an arithmetic or comparison operator, recycled as R recycles them: two vectors of different lengths are both repeated
    from the start to the longer length, and a vector against a matrix is repeated down the columns to the matrix's shape. R's warning when the
    longer length is not a multiple of the shorter is not raised."""
```

## Error handling

- **A fix changes the recall result** (an item missed that the last run reproduced, a new oracle-moved item): the closing task stops as
  `blocked`, naming the item and the scripts unit whose findings changed (diff `audit*/findings.json` of the two recall runs). `known.toml` and
  `recall.toml` are not edited to fit; the cause is found and filed with `/todo-add`.
- **A fix would change a metafopy output**: out of scope; the task stops as `blocked`. Reading `metafopy.methods.fitview.r_class` changes
  nothing in metafopy.
- **The regenerated fixture changes an existing snippet** (Task 1): the generator ran under another R or metafor; `load_snippet` already
  fails on a version mismatch. The task stops; the fixture is not committed.
- **A new snippet passes before its fix** (`XPASS` of a strict `xfail`): it does not test the defect; rewrite it before going on.
- **An input outside a shim's contract** raises `audit.rlib.UnsupportedError` with the R call's form, which the scripts step counts as
  `unsupported`, not as a metafopy `crash`.

## Implementation

Each task moves its snippets from `DISAGREE` to `MATCHING` (sorted), runs `ruff check`, `ruff format --check`, `mypy src/audit/rlib.py` and
the tests that import `audit.rlib` directly: `src/tests/test_harness_rlib.py`, `test_harness_transpile.py`, `test_harness_scripts.py`,
`test_harness_compare.py`, `test_harness_errors.py`, `test_audit8_refusals.py`, `test_audit8_values_a.py` (find the set again with
`grep -l 'audit.rlib\|audit.scripts' src/tests/*.py`).

1. **Snippets.** Add to `generate_audit_snippets.R`, beside the existing cases: `head_list` = `head(list(a = 1, b = 2, c = 3), 2)`,
   `recycled_matrix_vector` = `matrix(1:4, 2) * c(10, 20)`, `seq_to_by_length_out` = `seq(to = 10, by = 2, length.out = 3)`, `class_null` =
   `class(NULL)`, `class_character` = `class('a')`, `inherits_fit` = `inherits(rma(yi, vi, data = escalc(measure = 'RR', ai = tpos, bi = tneg,
   ci = cpos, di = cneg, data = dat.bcg)), 'rma')`, `cbind_named_and_unnamed` = `cbind(1:2, b = 3:4)`, `matrix_rownames_only` =
   `matrix(1:4, 2, dimnames = list(c('r1', 'r2'), NULL))`. Regenerate with the command in External dependencies; check that every existing
   snippet's record is unchanged (compare the old and new JSON by id). Add each new id to `DISAGREE` with its reason, and run them: each must
   `XFAIL` today (probed on 2026-11-23: each gives a different value or raises).
2. **Class shims** (`r_class`, `r_inherits`: both use the Python type name for R's class). `r_class` as in API, reading
   `metafopy.methods.fitview.r_class` for fits; `r_inherits` becomes `bool(set(r_class(x)) & set(what))` for every input. Snippets
   `class_numeric`, `class_matrix`, `class_null`, `class_character`, `inherits_number`, `inherits_fit`; `class_data_frame` and
   `inherits_data_frame` keep passing.
3. **Names** (`r_as_dataframe`, `r_cbind`, `r_isna`, `r_matrix`: each drops R's row or column names). `r_as_dataframe` of a 2-D array names
   columns from a `Named` matrix's `colnames`, else `V1`, `V2`, ..., and rows from its `rownames`. `r_cbind` returns a `Named` matrix when any
   argument is named, `''` for an unnamed column (R's order of a named argument before an unnamed one is lost in translation, since the
   transpiler passes keywords after positionals: say so in the docstring). `r_isna` of a data frame is `r_as_matrix(x.isna())`, which already
   follows R's rule for row names. `r_matrix` takes `dimnames` (a list of two, either `None`) into a `Named`. Snippets `as_dataframe_matrix`,
   `cbind_named`, `cbind_named_and_unnamed`, `isna_data_frame`, `matrix_dimnames`, `matrix_rownames_only`.
4. **Vectors** (`r_length`, `r_as_numeric`, `r_seq`, `recycled`). `r_length(None)` is 0. `r_as_numeric` takes a str as a one-element
   vector (`numpy.atleast_1d`); its docstring says an unparsable string raises where R gives `NA` with a warning. `r_seq` with `to` and
   `length.out` and no `from` counts back from `to` by `by` (default 1). `recycled` as in API: 1-D arrays longer than one and of different
   lengths `numpy.resize` to the longer; a vector longer than one against a 2-D array is `numpy.resize(v, m.size).reshape(m.shape, order='F')`.
   The existing `test_recycled_turns_a_series_of_another_length_into_an_array` keeps passing. Snippets `length_null`, `as_numeric_text`,
   `seq_length_out_with_to`, `seq_to_by_length_out`, `recycled_operands`, `recycled_matrix_vector`.
5. **Lists** (`r_tail`, `r_head`, `r_list`). `r_tail` and `r_head` of a dict take items by position, as R does for a named list (negative
   `n` as R: all but the last or first `|n|`). `r_list` as decision D1 says. Snippets `tail_list`, `head_list`, `list_mixed`. If D1 narrows,
   `r_list` raises `UnsupportedError('list() mixing named and unnamed arguments')` and its docstring states the contract; the case leaves
   the tests as D2 says.
6. **Close.** `DISAGREE` is empty: remove it and its test. Run every test file of the set above, then stop the list as `blocked` with this
   note for the operator: "Run `bin/metafor-audit.sh --recall` from the list's worktree, whose `src` holds the changed shims (`source bin/activate.sh` first; the last run took 23
   minutes, 05:09-05:32 UTC on 2026-11-23), then resume this task." The task list stops rather than leaving a note, because the note of
   the last harness change (Task 10 of `agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md`, 2026-11-22) was not acted on until
   `/todo-review` filed it as a separate item the next day. On resume: read the new `recall.md` and compare it with
   `.audit/runs/recall-20261123-061000/recall.md` (176 of 176 reproduced, `a6-glmm-um-rs` apart as oracle moved). If they agree, move the
   `TODO.md` item to `DONE.md` naming the run, set this spec to implemented and update its row in `agents/specs/index.md`; if not, Error
   handling's first case.

## External dependencies

- R 4.6.2 with metafor 5.2-1, metadat 1.6-0 and jsonlite 2.0.0, as pinned in `src/audit/data/r-versions.json`, to regenerate the fixture in
  Task 1: `LC_ALL=C.UTF-8 R_LIBS=<pinned library> Rscript src/tests/r/generate_audit_snippets.R`, where the library is `.audit/Rlib` or the
  one `METAFOPY_AUDIT_RLIB` names. The tests themselves read the committed fixture and need no R.
- The measurement scripts [reach-static.R](reach-static.R) and [reach-dynamic.R](reach-dynamic.R) need the same R and the metafor source
  the scripts step fetched (`.audit/source/metafor`); they are records of how reach was measured, not part of the work.
- No new Python dependency.

## Decisions

Choices made, each reversible: every case but `list_mixed` is fixed to R (no script reaches them, but each fix is small), by the rule
that a case is narrowed only where no audited script reaches it and the fix is large; `r_head` is fixed with `r_tail` (the same defect);
`recycled` also recycles a vector against a matrix, since numpy gives a wrong value there without an error; `r_class` refuses an object it
has no R class for rather than return a Python type name.

**D1. `list()` mixing named and unnamed arguments (`list_mixed`).** Background: R's `list(1, b = 'x')` is a list whose first element has no
name. `r_list` returns a Python list for unnamed arguments and a dict for named ones, so it keys the unnamed element `'1'`
(`rlib.py:181`).
<!-- doc-counts: accept-start measured on 2026-11-23 at 3324124 - a dated record, not a claim about the present -->
Measured on 2026-11-23 at `3324124`: the audited scripts call `list()` at 211 sites in 49 scripts, none mixing named and unnamed
arguments, and no unit does so at run time.
<!-- doc-counts: accept-end -->
Affects: `r_list`; for the fix, also the transpiler and every shim that reads a list.
- *Narrow the contract (recommended).* `r_list` raises `UnsupportedError` for a mixed call and its docstring says so. A script that makes
  one is counted as `unsupported` instead of reporting an invented `missing-field`. Small; reversible when a script needs it.
- *Fix to R.* A list type holding names beside positions (a `list` subclass with a `names` tuple, which `audit.valuetree.lookup` already
  reads), built from the call's arguments in order (the transpiler passes keywords after positionals, so it needs an ordered form like
  `rc_pairs`), and read by `r_dollar`, `r_index`, `r_index2`, `r_names`, `r_length`, `r_lapply` and `r_c`. Several shims and the
  transpiler change for a call no script makes.

**D2. How a narrowed case leaves the tests** (only if D1 narrows). Background: the item's fix direction says to drop a narrowed case.
Affects: `test_harness_rlib.py`, the generator and the fixture.
- *Keep the snippet as a refusal test (recommended).* A `REFUSED` list in `test_harness_rlib.py`; its test runs the snippet and expects
  `audit.rlib.UnsupportedError`. R's value stays recorded for a later fix, and the refusal itself is tested. No regeneration for this case.
- *Remove the snippet.* `list_mixed` leaves the generator and the fixture (regenerated in Task 1); the docstring is the only record, and
  nothing tests that the shim refuses.

Answered on 2026-11-23 in `agents/reviews/r-library-shims-follow-r-spec.md` (version 1.1 records them): the spec approved; D1 narrow, so
`r_list` refuses a mixed call; D2 keep `list_mixed` as a refusal test in `REFUSED`.

Version 1.2, 2026-11-23: the recall run is made from the list's worktree, not the main checkout, since `bin/activate.sh` puts its own
checkout's `src` on `PYTHONPATH`, so a run from the main checkout would check the unchanged shims.
