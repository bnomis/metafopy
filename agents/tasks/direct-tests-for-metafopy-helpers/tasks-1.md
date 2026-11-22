# Tasks: Direct tests for metafopy helpers

**Spec:** `agents/specs/direct-tests-for-metafopy-helpers/spec.md`
**Created:** 2026-11-21
**Version:** 1

The spec's decisions were answered on 2026-11-21 (`agents/reviews/direct-tests-for-metafopy-helpers-spec.md`):
- D1: internal helpers leave their module's `__all__`;
- D2: no hypothesis tests here.

The candidates are the 162 listed in `agents/specs/direct-tests-for-metafopy-helpers/measurements.md`. That file is a dated record, so never edit it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **T1 rulings:** a test counts by the T1 brief (`~/.claude/skills/auditme/phaseb/briefs/T1.md`) plus the rulings in
  `agents/audit-reviews/review-1.md`, section "Thresholds settled during the run", bullet "T1, what counts as a direct test":
  - a call through a re-export of the same object counts, and so does one through a `singledispatch` generic or a dispatch table;
  - a wrapper that is a different function does not;
  - a dunder counts when the test runs it itself;
  - a `__post_init__` that only package code triggers does not;
  - calls on an instance, and from helpers in the test file, count.

A new test asserts a value metafopy already produces and R metafor agrees with: a parity-fixture value, or a field another test already
checks against R. A test that shows a defect stops the task as `blocked`, naming the function and the input. It is filed in `TODO.md`, not
fixed (`claude/rules/metafopy.md`).

---

## Task 1: Make the internal helpers in core, effects, bias and the top-level modules private

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 1; decision D1

**Technical Goal:** helpers that only the package calls stop being public, so a test of their caller covers them.

**Subtasks:**

- [x] For each name listed under "Named only in their own module's `__all__` (functions)" and under "Methods of classes named only in their own
      module's `__all__`" (use the class name) in `measurements.md`, whose module is under `core/`, `effects/` or `bias/`, or is a top-level
      module (`formatters.py`, `formula.py`, `mfopt.py`, `printing.py`): remove it from that module's `__all__`. Delete an `__all__` that is
      left empty
- [x] Before each removal, check that the name is in neither `metafopy.__all__` nor a subpackage `__init__.py`'s `__all__`. If it is, keep it
      and note it in the task's Notes as "stays public"
- [x] Run `python3 -c "import metafopy"`, then `mypy src/metafopy`, then the own tests of each changed module. If a removal breaks any of
      these, put the name back, keep it public, and note it

**Acceptance Criteria:**

- [x] Each listed name in these packages is in no `__all__`, or is noted as staying public with the reason
- [x] `import metafopy` works and no test fails

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy` passes
- [x] The own test files of every changed module pass, named in the commit body

**Notes:**

- No name stays public: none is in `metafopy.__all__` or a subpackage `__all__` (checked 2026-11-21)
- Needs `uv sync --extra interactive` for mypy and the docs-example tests
- No behaviour change: `__all__` lists only. No module does `import *` (checked 2026-11-21)

---

## Task 2: Make the internal helpers in methods, models and plot private

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 1; decision D1

**Technical Goal:** the same as Task 1, for the names whose module is under `methods/`, `models/` or `plot/`.

**Subtasks:**

- [x] As Task 1's subtasks, for the names in the same two groups of `measurements.md` whose module is under `methods/`, `models/` or `plot/`.
      These include the classes `ListAnovaResult`, `LSWaldResult`, `LSPredictResult`, `MVProfileResult`, `Problem` (`mv_lik`), `LSProblem`
      and `FunnelResult`
- [x] Run `python3 -c "import metafopy"`, `mypy src/metafopy` and the own tests of each changed module

**Acceptance Criteria:**

- [x] Each listed name in these packages is in no `__all__`, or is noted as staying public with the reason

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy` passes
- [x] The own test files of every changed module pass, named in the commit body

**Notes:**

- Depends on Task 1 (same method)
- `FunnelResult` stays public: it is in `metafopy.plot.__all__` (`src/metafopy/plot/__init__.py`), so it stays in `plot/funnel.py`'s `__all__` too (checked 2026-11-22)
- Every other listed name under `methods/`, `models/` and `plot/` is removed from its module's `__all__`; `models/mv_structs.py` keeps only its constants, `Group`, `Term` and `con_vcov_un_chol`
- The own tests of every changed module are covered by the whole `src/tests` suite, which passes

---

## Task 3: Re-judge the 162 candidates and record what is still open

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2

**Technical Goal:** after Tasks 1 and 2, find which of the 162 candidates still lack a test that counts, so the later tasks test exactly those.

**Subtasks:**

- [x] Run `python3 ~/.claude/skills/auditme/checks/phaseb.py plan . --question T1 --out <scratch>/t1`. Build `<scratch>/t1sel/` with one
      `batch-01.jsonl` (and more batches of up to 48 if needed) holding only the batch lines whose `id` is one of the 162 in `measurements.md`.
      Copy `plan.json` with its `batches` cut to those ids. `<scratch>` is a directory outside the repository
- [x] Judge each candidate by the T1 brief and the rulings above, by reading the tests (do not run them), and write `result-NN.jsonl` in the
      brief's format
- [x] `phaseb.py check <scratch>/t1sel` must report every batch complete. Then run `phaseb.py merge . --question T1 <scratch>/t1sel`
- [x] Write `agents/specs/direct-tests-for-metafopy-helpers/open-after-privatise.md`, a dated record ("measured on <date> at <commit>") of
      every candidate that `phaseb.py report . --question T1` still shows open under `src/metafopy`. Split it into "public" and "private" by the
      batch line's `private` flag
- [x] Stop as `blocked`, naming the candidate, if the brief and the rulings do not settle one

**Acceptance Criteria:**

- [x] All 162 have a fresh ledger verdict, and the open list is committed

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `python3 ~/.claude/skills/auditme/checks/phaseb.py check <scratch>/t1sel` reports every batch complete

**Notes:**

- Depends on Tasks 1 and 2
- `phaseb.py merge` keeps every verdict it is not given, so the other T1 candidates are untouched; they are re-judged at the next audit

---

## Task 4: Direct tests for the exported functions and the `metadata` and `methods` result classes

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 3

**User Story:** As a maintainer, I want each public function and result method a user can call to have a test that calls it, so that a
regression in it is caught where it is.

**Subtasks:**

- [x] For each public candidate in `open-after-privatise.md` from these places, write a direct test in the module's existing test file:
  - `effects/measures.py::replmiss` and `plot/rutils.py::fmtx`. These differ from the same-named `metafopy.replmiss` and
    `metafopy.fmtx`, so call them by their module path;
  - `metadata.py`;
  - `methods/` (`confint`, `leave1out`, `matreg`, `permutest`, `predict`, `rcalc`, `selmodel`).

  A `__post_init__` test builds the object itself. A property test reads the property on a fit
- [x] Run each changed test file

**Acceptance Criteria:**

- [x] Each of these candidates has a test that calls it and asserts a value

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] Each changed test file passes, named in the commit body

**Notes:**

- Depends on Task 3

---

## Task 5: Direct tests for the `models` result classes

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 3

**User Story:** As a maintainer, I want the methods of `RMAMVResult`, `GLMMResult`, `RMALSResult`, `MHResult`, `RMAResult`, `SelmodelResult` and
`tau2.Control` to have direct tests, so that a fit's printed and derived fields are checked where they are made.

**Subtasks:**

- [x] For each public candidate in `open-after-privatise.md` under `models/`, write a direct test in that model's test file. A `render`,
      `title_lines` or `heterogeneity_lines` test calls the method on a fit and compares it with the lines `str(fit)` already prints. A
      `hat_matrix` test checks the diagonal against `hatvalues()`. A `__post_init__` test builds the result through `dataclasses.replace`
      on a fit. A `Control.__getattr__` test reads an ignored control entry on an `rma_uni` fit
- [x] Run each changed test file

**Acceptance Criteria:**

- [x] Each of these candidates has a direct test that asserts a value

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] Each changed test file passes, named in the commit body

**Notes:**

- Depends on Task 3

---

## Task 6: Caller tests for private helpers in core, effects, bias and the top-level modules

**Status:** in-progress
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation step 4

**Technical Goal:** each private helper still open in these packages is reached by a test of a caller.

**Subtasks:**

- [ ] For each private candidate in `open-after-privatise.md` under `core/`, `effects/`, `bias/` or a top-level module, find a caller (grep
      its full module path) and an input that takes the helper's branch. Add the test to the caller's test file. Examples:
  - `glm_fit` through `rma_glmm`;
  - `clogit` through `rma_glmm(control={'optimizer': 'clogit'})`;
  - `farebrother` through `rma(method='GENQM')`.
- [ ] A helper that no public function can reach is dead code. Stop as `blocked` and name it; do not delete it
- [ ] Run each changed test file

**Acceptance Criteria:**

- [ ] Each of these candidates is reached by a test of a caller

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] Each changed test file passes, named in the commit body

**Notes:**

- Depends on Task 3

---

## Task 7: Caller tests for private helpers in methods, models and plot

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation step 4

**Technical Goal:** the same as Task 6, for `methods/`, `models/` and `plot/`, including the five helpers that were already private.

**Subtasks:**

- [ ] As Task 6's subtasks, for the private candidates under `methods/`, `models/` and `plot/`. These include:
  - `block_diagonal`: `rma_mv` with `V` given as a list of block matrices;
  - `distance_from_matrix`: `rma_mv` with `dist` given as a matrix;
  - `fitted_levels`: `predict` with no `newmods` on an `rma_mv` fit with an inner|outer term;
  - `fixed_btt` and `formula_text`.
- [ ] Run each changed test file

**Acceptance Criteria:**

- [ ] Each of these candidates is reached by a test of a caller

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] Each changed test file passes, named in the commit body

**Notes:**

- Depends on Task 3

---

## Task 8: Re-judge, the full check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 5

**Technical Goal:** none of the 162 is open, the branch passes every check once, the spec is marked implemented and the `TODO.md` item moves
to `DONE.md`.

**Subtasks:**

- [ ] Repeat Task 3's plan, filter, judge, check and merge for the 162. `phaseb.py report . --question T1` must show none of them open. A
      candidate still open goes back to the task that covers its group
- [ ] Move the `TODO.md` item "Public metafopy helpers with branches have no direct test" to the end of `DONE.md`: append its heading and body
      unchanged, then the line `Done: <date> - agents/specs/direct-tests-for-metafopy-helpers/spec.md,
      agents/tasks/direct-tests-for-metafopy-helpers/tasks-1.md.`, and remove it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-21)" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/direct-tests-for-metafopy-helpers/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/direct-tests-for-metafopy-helpers/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] None of the 162 is open in the T1 report, the spec says `implemented`, and the item is in `DONE.md` with its body unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/direct-tests-for-metafopy-helpers/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 4-7
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
- The venv needs the `interactive` extra for the docs examples (`uv sync --frozen --extra interactive`; installed 2026-11-21)
