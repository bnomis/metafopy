# Tasks: selmodel: a precision measure on a step-function selection model

**Spec:** `agents/specs/selmodel-prec-stepfun/spec.md` (version 1.1)
**Created:** 2026-11-15
**Version:** 1

**Waits for:** row 3 of the run order `agents/tasks/order-numerical-fixes.md`, the list `agents/tasks/truncest-piecewise-search/tasks-1.md`, which
changes the `truncest` path of `src/metafopy/models/selmodel.py` and the `truncest` case of `src/tests/correctness/cases/selmodel.toml`, both of
which this list also edits. Task 1 stops the list if that list is not merged on main.

The user answered the spec's open decision on 2026-11-15 in `agents/reviews/order-numerical-fixes-specs.md` (D3, what
`selmodel(type='stepfun', prec=...)` does): **(a) reject with `InputError`**. This list builds option (a) only, as the spec's "Tests and records"
section lists it, and changes metafopy's output without a new review. The decision covers `type='stepfun'` with `prec` and `decreasing=False` only:
`decreasing=True` and `type='stepcon'` keep metafor's warning "Argument 'prec' ignored" and fit without `prec`, as now.

Checked against the code on 2026-11-15 (metafopy at `89ec047`): the `'stepfun'` branch of `sel_spec.py` warns "Adding a precision measure to this
selection model is undocumented and experimental." where option (a) raises; `hack_stepfun_sei` is the only R parity fit of `stepfun` with `prec`
(`src/tests/r/generate_selmodel.R`), and the four `selmodel.prec.*.bangertdrowns2004` cases are the only cases covering the coverage items
`selmodel.prec.*` of `src/tests/correctness/test_coverage.py`. No `known.toml` entry names the combination, and the spec expects none.

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires.
Every R call and audit run sets `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (or `R_LIBS` to the same path): a worktree has no
`.audit/Rlib` of its own. No version bump: the change needs none (`pyproject.toml` version `0.0.1`, no release convention in the repository).

**Operator stop (conditional):** Task 1, "Operator stop: a new halfnorm precision case fails". It stops only if one of the new `halfnorm` cases
fails; otherwise the list runs to the end without the operator.

---

## Task 1: Re-check the starting point, and cover the precision items with halfnorm cases

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, item 2; Motivation; Open decision (answered)

**Technical Goal:** the coverage items `selmodel.prec.{sei,vi,ninv,sqrtninv}` are covered by the model `?selmodel` defines `prec` for (`halfnorm`,
Preston et al. 2004) instead of the step function that Task 2 rejects, so the correctness harness stays green across the change.

**Subtasks:**

- [x] Check that row 3 of the run order is merged on main: in the main checkout (the first entry of `git worktree list --porcelain`,
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')`), `git -C "$main" show main:DONE.md` holds the heading
      `## The truncest estimate stops at a local maximum of a step-function likelihood`, `git -C "$main" show main:TODO.md` does not, and
      `git merge-base --is-ancestor main HEAD` exits 0 (this branch is based on that main). Stop as `blocked`, naming what is missing, if not
- [x] Re-check the starting point: `grep -n 'undocumented and experimental' src/metafopy/models/sel_spec.py` finds the one warning in the
      `'stepfun'` branch; the four cases `selmodel.prec.{sei,vi,ninv,sqrtninv}.bangertdrowns2004` in `src/tests/correctness/cases/selmodel.toml`
      have `status = "mismatch"`; `grep -nwE 'DIFFERS|test_stepfun_prec_raises' src/tests/test_selmodel.py` finds nothing; and no case id
      `selmodel.prec.*.halfnorm.bangertdrowns2004` exists. Stop as `blocked`, naming what differs, if any is not so
- [x] In `src/tests/correctness/cases/selmodel.toml`, remove the four `selmodel.prec.*.bangertdrowns2004` cases
- [x] Add the cases `selmodel.prec.{sei,vi,ninv,sqrtninv}.halfnorm.bangertdrowns2004`, beside the other `halfnorm` cases: `kind = "derived"`,
      `function = "selmodel"`, `dataset = "dat.bangertdrowns2004"`, `setup = { rma_uni = { method = "ML" } }`,
      `args = { type = "halfnorm", prec = "<p>" }` (with the `control` of `selmodel.halfnorm.bangertdrowns2004` if it needs it to pass),
      `expect = {}`, `check = "test_derived.selmodel_fit"`, `tolerance = 2e-3` (as for the other `halfnorm` cases, whose likelihood is flat in
      `delta`), `covers = ["selmodel.prec.<p>"]`, and no `status`
- [x] Above the new cases, a comment naming the decision: `stepfun` with `prec` is rejected (option (a), answered 2026-11-15 in
      `agents/reviews/order-numerical-fixes-specs.md`, D3; `agents/specs/selmodel-prec-stepfun/spec.md`), so `prec` is covered by `halfnorm`, the
      model `?selmodel` defines it for (Preston et al. 2004)
- [x] Run `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py`. A probe of these cases on 2026-11-15 passed each
      `prec`
- [x] **Operator stop: a new halfnorm precision case fails.** Only if one of the new cases fails: do not set it to `mismatch`. Stop the task as
      `blocked` and report to the operator, as the correctness specs require for a new case that fails, each failing case id with metafopy's value,
      the derived value and the tolerance. The list resumes once the operator has said what to do (fix the case, or set it to `mismatch` with a
      `reason`) in an operator session, and that answer is committed. If every new case passes, there is no stop

**Acceptance Criteria:**

- [x] No case in `src/tests/correctness/cases/selmodel.toml` sets `type = "stepfun"` together with `prec`
- [x] Each of `selmodel.prec.sei`, `selmodel.prec.vi`, `selmodel.prec.ninv` and `selmodel.prec.sqrtninv` is covered by a passing `halfnorm` case
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py src/tests/correctness/test_properties.py` passes

**Notes:**

- Depends on `agents/tasks/truncest-piecewise-search/tasks-1.md` (row 3 of `agents/tasks/order-numerical-fixes.md`), merged on main
- Key files: `src/tests/correctness/cases/selmodel.toml`, `src/tests/correctness/test_derived.py` (`selmodel_fit`, unchanged),
  `src/tests/correctness/formulas.py` (`parametric_weight`, unchanged), `src/tests/correctness/test_coverage.py` (unchanged)
- The cases come first so that Task 2's `InputError` never meets a correctness case that still fits `stepfun` with `prec`

---

## Task 2: Reject a precision measure on a step-function selection model

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Error handling, option (a); Implementation, option (a); Tests and records, item 1

**User Story:** As a user of `selmodel`, I want `type='stepfun'` with `prec` to raise an error naming the reason, so that I do not get a fit whose
log-likelihood is not a density and whose LRT overstates selection.

**Subtasks:**

- [ ] In the `'stepfun'` branch of `src/metafopy/models/sel_spec.py`, replace
      `warnings.warn('Adding a precision measure to this selection model is undocumented and experimental.', stacklevel=3)` by
      `raise metafopy.errors.InputError("Argument 'prec' cannot be used with type='stepfun': a step-function selection model with a precision
      measure is not defined (see docs/differences.md).")`, raised before any fitting. Leave the `type_ == 'stepcon' or decreasing` path (warning
      "Argument 'prec' ignored", `preci = numpy.ones(k)`) as it is, and leave `ll_stepfun` in `src/metafopy/models/sel_lik.py` unchanged
- [ ] Update the `prec` description and `Raises` of `selmodel` in `src/metafopy/models/selmodel.py` as the spec's API section gives them: ignored
      with a warning for `'beta'`, `'trunc'`, `'truncest'`, `'stepcon'` and `'stepfun'` with `decreasing=True`; `InputError` for `'stepfun'`
      otherwise, because no publication defines a step function with a precision measure and metafor's version is not a density
      (`docs/differences.md`, "Where metafor departs from its reference", the entry Task 3 adds)
- [ ] In `src/tests/test_selmodel.py`, add a named mapping `DIFFERS = {'hack_stepfun_sei': '<reason>'}` (the reason: `prec` with `stepfun` raises
      `InputError` in metafopy, see `docs/differences.md`, "selmodel(type='stepfun') with a precision measure"), and drop its keys from `ALL_CASES`.
      The fixture `src/tests/reference/phase5/selmodel.json` and its generator `src/tests/r/generate_selmodel.R` stay unchanged, as the record of
      metafor
- [ ] Add `test_stepfun_prec_raises`, parametrised over `metafopy.models.selmodel.PRECISIONS`, asserting `metafopy.errors.InputError` with a
      `match` on "cannot be used with type='stepfun'" for the `hack_stepfun_sei` spec (its base model and `steps`, with that `prec`) and for
      `type='stepfun'`, `steps=[0.025, 0.5, 1]` on an ML fit of `dat.bangertdrowns2004`
- [ ] Add a test that `type='stepfun'` with `decreasing=True` and `prec='sei'` still warns "Argument 'prec' ignored" and gives the same estimates
      (`beta`, `tau2`, `delta`, `ll`) as the same call without `prec`, on the `hack_stepfun_decr` spec
- [ ] Check that no other test calls `stepfun` with `prec`: `grep -rn "prec=" src/tests --include=*.py | grep -v scaleprec` finds only `halfnorm`
      calls and the tests this task adds (on 2026-11-15 it found `src/tests/test_escalc_ni.py` and two invalid-`prec` tests in
      `src/tests/test_selmodel.py`)

**Acceptance Criteria:**

- [ ] `selmodel(res, type='stepfun', steps=..., prec=p)` raises `InputError` for each `p` in `PRECISIONS`, before any fitting
- [ ] `selmodel(..., type='stepfun', decreasing=True, prec=...)` and `type='stepcon'` with `prec` warn and fit as before
- [ ] `hack_stepfun_sei` is in `DIFFERS` with its reason, and no parity test runs on it

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `ruff format --check src/metafopy/models/sel_spec.py src/metafopy/models/selmodel.py src/tests/test_selmodel.py` and `ruff check` on the same
      files pass
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_selmodel.py src/tests/test_stepcon.py src/tests/test_selmodel_parallel.py src/tests/correctness` passes

**Notes:**

- Depends on Task 1
- Key files: `src/metafopy/models/sel_spec.py`, `src/metafopy/models/selmodel.py`, `src/tests/test_selmodel.py`,
  `src/tests/reference/phase5/selmodel.json` (unchanged), `src/tests/r/generate_selmodel.R` (unchanged)
- Raising rather than ignoring with a warning is settled in the spec (Implementation, option (a)): ignoring would return the plain step-function
  fit labelled as a precision model, and metafopy already raises where R passes on misuse (`docs/differences.md`, "Stricter than R")
- `sel_spec.py` and `selmodel.py` have no test files of their own beyond `src/tests/test_selmodel.py`; `test_stepcon.py` and
  `test_selmodel_parallel.py` reach the same `'stepfun'`/`'stepcon'` branch

---

## Task 3: Record the difference in the user guide and run the audit

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, items 3, 5 and 6; External dependencies

**Technical Goal:** `docs/differences.md` explains the rejection with R's output beside metafopy's, and the whole audit and its recall check run on
the change, adding a `known.toml` entry only if the audit flags `selmodel` with `prec` and `stepfun`.

**Subtasks:**

- [ ] Add a subsection `### selmodel(type="stepfun") with a precision measure` to the section `## Where metafor departs from its reference` of
      `docs/differences.md`, after the entries already there, in their format: a short summary (metafopy raises `InputError`; metafor warns that the
      combination is undocumented and experimental and fits it), the R call and the Python call side by side with the output of both, and `*Why:*`
      with the check date
- [ ] The R block: `selmodel(res, type="stepfun", steps=c(0.025, 0.5, 1), prec="sei")` on an ML fit of `dat.bangertdrowns2004`, showing its warning,
      ll and LRT. Produce its output with
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the R block>'` and paste it as printed (the spec measured ll
      -4.1029 and LRT 28.3186 on 2026-11-15; a different value stops the task as `blocked`, naming both)
- [ ] The Python block starts `# docs: skip` and shows the same call raising `InputError` with its message, run once with metafopy to paste the
      message as printed
- [ ] The `*Why:*`: no publication defines a step function with a precision measure (`?selmodel` names `prec` only for the continuous types,
      Preston et al. 2004); metafor's first interval has weight 1 in the normalising constant but `delta[0] / preci` for the observed p-value, so its
      likelihood is not a density, its estimates are the maximum of density C (weight 1 in the first interval, `delta[j] / preci` in the others) and
      its ll exceeds C's by `sum(-log(preci_i))` over the studies in the first interval, which inflates the LRT and shifts AIC, BIC and AICc; Vevea
      and Hedges (1995, Psychometrika 60:419-435, pp. 424-425) fix the first weight at 1. Name the decision and its date (2026-11-15, D3 of
      `agents/reviews/order-numerical-fixes-specs.md`) and the correctness cases `selmodel.prec.*.halfnorm.bangertdrowns2004` of Task 1
- [ ] Run `pytest src/tests/test_docs_examples.py`
- [ ] Run the whole audit from the worktree, in the background:
      `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh`. If `check-r` reports that the R library is missing or
      differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [ ] If its report flags `selmodel` with `prec` and `type = "stepfun"`, add a `[[known]]` entry to `src/audit/data/known.toml` matching exactly
      those case ids, with `kind`, `path`, a `reason` citing the decision and its date and `ref = "docs/differences.md"`, and rerun the same command
      to see it accepted. Otherwise add no entry: the spec expects none, since the audit's `options/selmodel/*/prec=` cases stop for want of `type`
      in both languages. Any other finding the newest earlier report under `/Users/simonb/src/metafopy/.audit/runs/` does not hold is new with this
      list: stop as `blocked`, naming it
- [ ] Run `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --recall`, in the background. A miss stops the task as
      `blocked`, naming the item (`AGENTS.md`: a miss is fixed in the harness, `src/audit/data/`, which is outside this list)
- [ ] Record in this task's Notes the run directories of both runs and whether a `known.toml` entry was added

**Acceptance Criteria:**

- [ ] `docs/differences.md` has the subsection under "Where metafor departs from its reference", with R's output and metafopy's error as printed
- [ ] The whole audit reports no finding new with this list, and every `known.toml` entry added matches at least one case of that run
- [ ] `bin/metafor-audit.sh --recall` reproduces every item

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_docs_examples.py src/tests/test_selmodel.py` passes

**Notes:**

- Depends on Task 2
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib`, the main
  checkout's R library (present on 2026-11-15)
- `src/tests/test_docs_examples.py` runs every Python block of `docs/` not marked `# docs: skip`, so the R and Python blocks of the new entry do not
  run there; their output comes from the commands above
- Key files: `docs/differences.md`, `src/audit/data/known.toml` (only if the audit flags the change)

---

## Task 4: File the fixed-delta[0] finding, the full test run and closing the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, items 4 and 5; "What the investigation found", "Related, outside this spec"; the spec as a whole

**Technical Goal:** the related fixed-`delta[0]` finding is filed as its own `TODO.md` item, the whole branch is checked once, the spec is marked
implemented and its item moves to `DONE.md`.

**Subtasks:**

- [ ] Re-check the related finding in both languages: `selmodel(type='stepfun', steps=[0.025, 0.5, 1], delta=[0.5, nan, nan])` on an ML fit of
      `dat.bangertdrowns2004` gives the estimates of `delta=[1, nan, nan]` (beta 0.10569, delta 0.40141, 0.27643 on 2026-11-15) and an ll lower by
      `n1 * log(0.5)` (-27.1257 against -17.4217), in metafopy and in R
      (`R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the same call>'`)
- [ ] If it holds, file it with the `todo-add` skill as its own item (not merged into another item, so no existing item's body changes): a
      user-fixed `delta[0]` other than 1 on a step-function selection model is a constant, because the first interval's weight is hard-coded to 1
      in the normalising constant; give the figures just measured, the date, the cause in `.selmodel.ll.stepfun` (`R/misc.func.hidden.selmodel.r`)
      and `ll_stepfun` (`src/metafopy/models/sel_lik.py`), and a fix direction that goes through a spec. If it no longer holds, file nothing and say
      so in this task's Notes with the values measured
- [ ] Set the spec's `**Status:**` in `agents/specs/selmodel-prec-stepfun/spec.md` to `implemented`
- [ ] Move the item "A precision measure on a step-function selection model makes the likelihood not a density" from `TODO.md` to `DONE.md`:
      append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/specs/selmodel-prec-stepfun/spec.md (option (a), reject with InputError, answered 2026-11-15 in agents/reviews/order-numerical-fixes-specs.md, D3), agents/tasks/selmodel-prec-stepfun/tasks-1.md`
      with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate, in the main checkout (the first entry of
      `git worktree list --porcelain`, here `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/selmodel-prec-stepfun/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/selmodel-prec-stepfun/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The fixed-`delta[0]` finding is a `TODO.md` item of its own, or this task's Notes say why it was not filed
- [ ] The spec's status is `implemented`
- [ ] The item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `ruff format --check`, `ruff check` and `mypy` pass
- [ ] The final test run above passes, `slow` tests included, and its log is saved at
      `/Users/simonb/src/metafopy/ralpher-logs/selmodel-prec-stepfun/tasks-1/final-tests.log`
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/selmodel-prec-stepfun/tasks-1.md):TODO.md > <scratch>/todo-before.md` (the
      `TODO.md` at the list's start) and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0
      (`<scratch>` is the session's scratch directory, outside the repository)

**Notes:**

- Depends on Task 3
- The fixed-`delta[0]` finding is outside the decision this list builds (D3 covers `prec` with `stepfun` only), so this list files it and does not
  fix it
- The baseline of `verify_bodies.py` is the commit that adds this list, so items that earlier rows of the run order moved to `DONE.md` before it are
  not in it, and the truncest item that row 3 moved is found in `DONE.md`

---
