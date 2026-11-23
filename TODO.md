# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

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

Spec: `agents/specs/r-library-shims-follow-r/spec.md`; tasks: `agents/tasks/r-library-shims-follow-r/tasks-1.md`.
