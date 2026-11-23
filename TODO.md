# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Recall check due after the `ast()` move

<!-- doc-counts: accept-start found on 2026-11-23 - a dated record, not a claim about the present -->
Found on 2026-11-23 by `/todo-review`: Task 3 of `agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md` moved R's `ast()` unchanged from
`src/audit/r/runscripts.R` into `src/audit/r/common.R` (merged in `8a042d7`), and that list's Task 10 Notes say `bin/metafor-audit.sh
--recall` is due after the move and that the operator starts it. No run since the merge is recorded.
<!-- doc-counts: accept-end -->

Cause: the recall check runs only when the operator starts it; neither CI nor a hook runs it, since it needs the pinned R library.

Fix direction: run `bin/metafor-audit.sh --recall` from the main checkout. If it reproduces every earlier audit's findings, move this item to
`DONE.md` naming the run; if it misses one, fix the harness in `src/audit/data/` as `AGENTS.md` says, and file what it found with `/todo-add`.

## R-library shims disagree with R

Found on 2026-11-22 by the direct tests of the audit harness (`src/tests/test_harness_rlib.py`, strict `xfail` cases in `DISAGREE`), against R
values recorded in `src/tests/reference/audit/snippets.json`:

- `audit.rlib.r_as_dataframe` of an unnamed matrix has no `V1`, `V2` column names (snippet `as_dataframe_matrix`);
- `audit.rlib.r_as_numeric` of a single string fails with "iteration over a 0-d array" (`as_numeric_text`);
- `audit.rlib.r_cbind` with named arguments gives no column names (`cbind_named`);
- `audit.rlib.r_class` gives `float` for a number where R gives `numeric`, and one name for a matrix where R gives `matrix` and `array`
  (`class_numeric`, `class_matrix`);
- `audit.rlib.r_inherits` of a number against `numeric` is false (`inherits_number`);
- `audit.rlib.r_isna` of a data frame gives a frame with other column names (`isna_data_frame`);
- `audit.rlib.r_length` of `NULL` is 1 where R gives 0 (`length_null`);
- `audit.rlib.r_list` loses a positional argument that comes beside a named one (`list_mixed`);
- `audit.rlib.r_matrix` drops `dimnames` (`matrix_dimnames`);
- `audit.rlib.recycled` leaves two arrays of different length to numpy, which raises where R recycles (`recycled_operands`);
- `audit.rlib.r_seq` with `length.out` and `to` counts up from 1 where R counts back from `to` (`seq_length_out_with_to`);
- `audit.rlib.r_tail` of a dict is not indexed by position and fails (`tail_list`).

Cause: each shim was written for the calls the audited scripts make, not for R's full behaviour.

Fix direction: change the harness shim to R's behaviour (or document the narrower contract in its docstring and drop the case), then run
`bin/metafor-audit.sh --recall`, since a harness change can change what an audit reports. Remove the matching `DISAGREE` entry in the same change.
