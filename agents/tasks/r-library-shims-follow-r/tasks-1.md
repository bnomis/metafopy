# Tasks: R-library shims follow R

**Spec:** `agents/specs/r-library-shims-follow-r/spec.md`
**Created:** 2026-11-23
**Version:** 1

The spec's decisions were answered on 2026-11-23 (`agents/reviews/r-library-shims-follow-r-spec.md`):
- D1: narrow `r_list`: a call mixing named and unnamed arguments raises `audit.rlib.UnsupportedError`, and its docstring says so;
- D2: keep the `list_mixed` snippet as a refusal test, in a `REFUSED` list in `src/tests/test_harness_rlib.py`.

`agents/specs/r-library-shims-follow-r/measurements.md` is a dated record of each case's R and shim values, cause and reach. Never edit it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `MYPYPATH=src mypy <files>`, with the same `PATH` (without `MYPYPATH`, `audit` resolves to the main checkout's installed
  package, which has no `py.typed`).
- **The harness test set** (spec, Implementation): `src/tests/test_harness_rlib.py`, `src/tests/test_harness_transpile.py`,
  `src/tests/test_harness_scripts.py`, `src/tests/test_harness_compare.py`, `src/tests/test_harness_errors.py`,
  `src/tests/test_audit8_refusals.py`, `src/tests/test_audit8_values_a.py`. Find it again with `grep -l 'audit.rlib\|audit.scripts' src/tests/*.py`
  and add any file that grep finds.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript src/tests/r/generate_audit_snippets.R`; the pin is
  `src/audit/data/r-versions.json`.

In every fix task: move the task's snippet ids from `DISAGREE` to `MATCHING` (sorted) in `src/tests/test_harness_rlib.py`, so each runs as a
passing test. Imports by full module path, no leading underscores (`claude/rules/`). Stop the task as `blocked` (spec, Error handling) if:
- a fix would change a metafopy output;
- a new snippet passes before its fix (`XPASS`);
- the regenerated fixture changes an existing snippet's record.

---

## Task 1: New snippets and the regenerated fixture

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 1; External dependencies

**Technical Goal:** the cases the spec adds beyond the 13 found ones are recorded from R and fail today as strict xfails, so each later fix
has a test.

**Subtasks:**

- [x] Add to `src/tests/r/generate_audit_snippets.R`, beside the existing cases, the snippets of spec step 1: `head_list`,
      `recycled_matrix_vector`, `seq_to_by_length_out`, `class_null`, `class_character`, `inherits_fit`, `cbind_named_and_unnamed`,
      `matrix_rownames_only`, with the R text the spec gives
- [x] Save the old fixture outside the repository, regenerate with the R command above, and compare old and new by snippet id: every
      existing record must be unchanged
- [x] Add each new id to `DISAGREE` with its reason; run `src/tests/test_harness_rlib.py`: each new id must `XFAIL`

**Acceptance Criteria:**

- [x] The fixture holds the new snippets, its existing records are unchanged, and each new case is a strict xfail

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_harness_rlib.py src/tests/test_auditsnippets.py` passes

**Notes:**

- `inherits_fit` uses `rma(c(...), c(...))` on literal vectors, not `data = escalc(...)`: the snippet runner passes an empty NSE set, so a column lookup raises `UnsupportedError` (not an xfail)
- The fixture's `meta` must match `r-versions.json`; `auditsnippets.load_snippet` fails otherwise

---

## Task 2: Class shims

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`r_class`); Implementation step 2

**Technical Goal:** `r_class` and `r_inherits` use R's class names, not Python type names.

**Subtasks:**

- [x] `r_class` as the spec's API docstring says, reading `metafopy.methods.fitview.r_class` for fits and raising `UnsupportedError` for any
      other object it has no R class for; the docstring says `numeric` also stands for an R integer
- [x] `r_inherits` becomes `bool(set(r_class(x)) & set(what))` for every input
- [x] Move `class_numeric`, `class_matrix`, `class_null`, `class_character`, `inherits_number`, `inherits_fit` to `MATCHING`;
      `class_data_frame` and `inherits_data_frame` keep passing

**Acceptance Criteria:**

- [x] Each of those snippets passes, and the harness test set passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `MYPYPATH=src mypy src/audit/rlib.py` passes
- [x] `pytest` on the harness test set passes

**Notes:**

- Depends on Task 1

---

## Task 3: Names kept as R keeps them

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 3

**Technical Goal:** `r_as_dataframe`, `r_cbind`, `r_isna` and `r_matrix` keep R's row and column names.

**Subtasks:**

- [x] `r_as_dataframe` of a 2-D array: columns from a `Named` matrix's `colnames`, else `V1`, `V2`, ...; rows from its `rownames`
- [x] `r_cbind` returns a `Named` matrix when any argument is named, `''` for an unnamed column; its docstring says R's order of a named
      argument before an unnamed one is lost, since the transpiler passes keywords after positionals
- [x] `r_isna` of a data frame is `r_as_matrix(x.isna())`
- [x] `r_matrix` takes `dimnames` (a list of two, either `None`) into a `Named`
- [x] Move `as_dataframe_matrix`, `cbind_named`, `cbind_named_and_unnamed`, `isna_data_frame`, `matrix_dimnames`, `matrix_rownames_only` to
      `MATCHING`

**Acceptance Criteria:**

- [x] Each of those snippets passes, and the harness test set passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `MYPYPATH=src mypy src/audit/rlib.py` passes
- [x] `pytest` on the harness test set passes

**Notes:**

- Depends on Task 1

---

## Task 4: Vectors and recycling

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`recycled`); Implementation step 4

**Technical Goal:** `r_length`, `r_as_numeric`, `r_seq` and `recycled` give R's values; `recycled` stops giving a wrong value without an
error for a vector against a matrix.

**Subtasks:**

- [x] `r_length(None)` is 0
- [x] `r_as_numeric` takes a str as a one-element vector (`numpy.atleast_1d`); its docstring says an unparsable string raises where R gives
      `NA` with a warning
- [x] `r_seq` with `to` and `length.out` and no `from` counts back from `to` by `by` (default 1)
- [x] `recycled` as the spec's API docstring says: 1-D arrays longer than one and of different lengths `numpy.resize` to the longer; a vector
      longer than one against a 2-D array is `numpy.resize(v, m.size).reshape(m.shape, order='F')`
- [x] Move `length_null`, `as_numeric_text`, `seq_length_out_with_to`, `seq_to_by_length_out`, `recycled_operands`,
      `recycled_matrix_vector` to `MATCHING`; `test_recycled_turns_a_series_of_another_length_into_an_array` keeps passing

**Acceptance Criteria:**

- [x] Each of those snippets passes, and the harness test set passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `MYPYPATH=src mypy src/audit/rlib.py` passes
- [x] `pytest` on the harness test set passes

**Notes:**

- Depends on Task 1

---

## Task 5: Lists

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5; decisions D1, D2

**Technical Goal:** `r_tail` and `r_head` take a named list's items by position, and `r_list` refuses a mixed call (D1), tested as a refusal
(D2).

**Subtasks:**

- [x] `r_tail` and `r_head` of a dict take items by position, as R does for a named list; a negative `n` as R (all but the last or first `|n|`)
- [x] `r_list` raises `UnsupportedError('list() mixing named and unnamed arguments')` for a call with both, and its docstring states the
      contract
- [x] Add `REFUSED = ['list_mixed']` and a test that runs each snippet in it and expects `audit.rlib.UnsupportedError`; remove `list_mixed`
      from `DISAGREE`. Keep its R record in the fixture
- [x] Move `tail_list`, `head_list` to `MATCHING`

**Acceptance Criteria:**

- [x] `tail_list` and `head_list` pass, `list_mixed` passes as a refusal, and the harness test set passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `MYPYPATH=src mypy src/audit/rlib.py src/tests/test_harness_rlib.py` passes
- [x] `pytest` on the harness test set passes

**Notes:**

- Depends on Task 1

---

## Task 6: The full check, then stop for the recall run

**Status:** in-progress
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 6; Error handling

**Technical Goal:** `DISAGREE` is empty and removed, the branch passes every check, and the list stops for the operator to run the recall
check on the branch's harness.

**Subtasks:**

- [x] `DISAGREE` is empty: remove it and its test
- [x] Run the final tests and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/r-library-shims-follow-r/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/r-library-shims-follow-r/tasks-1/final-tests.log`
- [x] Commit, then stop this task as `blocked` with this note for the operator: "Run the recall check on this branch's harness, from this
      worktree's root: `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH && METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib
      METAFOPY_AUDIT_WORK=/Users/simonb/src/metafopy/.audit bin/metafor-audit.sh --recall` (the last run took 23 minutes, 05:09-05:32 UTC on
      2026-11-23). Then set this task to done and restart the list." Do not run it yourself: the recall run is the operator's

**Acceptance Criteria:**

- [x] `DISAGREE` is gone, the final test run passes, and the task stops for the recall run

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `MYPYPATH=src mypy src/audit src/tests/test_harness_rlib.py` passes
- [x] The final test run above passes

**Notes:**

- Depends on Tasks 2-5
- It runs from this worktree, not the main checkout: `bin/metafor-audit.sh` puts its own checkout's `src` on `PYTHONPATH`
  (`bin/activate.sh`), so only a run from here checks the changed shims
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on

---

## Task 7: Compare the recall run and close

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 6; Error handling

**Technical Goal:** the recall result is unchanged by the shim fixes, the spec is marked implemented and the `TODO.md` item moves to
`DONE.md`.

**Subtasks:**

- [ ] Find the newest `/Users/simonb/src/metafopy/.audit/runs/recall-*/recall.md` written after Task 6's commit. If there is none, stop as
      `blocked`: "the recall run of Task 6 has not been made"
- [ ] Compare it with `.audit/runs/recall-20261123-061000/recall.md` (176 of 176 reproduced, `a6-glmm-um-rs` apart as oracle moved). If they
      differ, stop as `blocked`, naming each item that changed and the scripts unit whose findings changed (diff `audit*/findings.json` of the
      two runs); do not edit `known.toml` or `recall.toml` to fit
- [ ] Move the `TODO.md` item "R-library shims disagree with R" to the end of `DONE.md`: append its heading and body unchanged, then the line
      `Done: <date> - agents/specs/r-library-shims-follow-r/spec.md, agents/tasks/r-library-shims-follow-r/tasks-1.md; recall <run dir>:
      <its totals>; list_mixed refused (D1).`, and remove it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so

**Acceptance Criteria:**

- [ ] The new recall run matches the 2026-11-23 one, the spec says `implemented`, and the item is in `DONE.md` with its body unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/r-library-shims-follow-r/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 6 and the operator's recall run
- Only `TODO.md`, `DONE.md`, the spec and the index change here, so Task 6's full checks still stand
