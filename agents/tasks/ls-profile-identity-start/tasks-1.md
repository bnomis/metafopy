# Tasks: a feasible start for the reference ls_profile_ll under the identity link

**Source:** TODO.md, "The reference ls_profile_ll cannot fit the identity link at some feasible values"
**Created:** 2026-11-19
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

This list is row 2 of the run order `agents/tasks/order-citkowicz-cases-and-identity-profile.md` and waits for no other row. It changes only the
correctness harness's independent reference, `formulas.ls_profile_ll` in `src/tests/correctness/formulas.py`, so that its identity-link refits start
from a feasible point, then adds an identity-link `ls.confint.scale.*` case. No file under `src/metafopy/` changes; no version bump.

<!-- doc-counts: accept-start probed on 2026-11-19 - a dated record, not a claim about the present -->
Reproduced on 2026-11-19 (at `1842a93`): on the ML identity-link `dat.bcg` fit (`ls.fit.ML.identity.bcg`, `mods = scale = ~ ablat`, `escalc` RR),
the reference fit gives `alpha = (-0.02665867, 0.00205067)`, log-likelihood -7.13928882, and `ls_profile_ll(..., j=1, value=-0.00615, ...)` returns
-1e10, where the brute-force profile is -8.045852 (`agents/specs/ls-profile-starts/spec.md`, "Not in scope", 2026-11-18).
<!-- doc-counts: accept-end -->

Cause: every SLSQP start of `ls_profile_ll` (the line from the adjusted start to `start[free]`, `LS_PROFILE_LINE`) leaves `Z alpha < 0` there, and
`negative` returns `inf` at an infeasible point, so no refit leaves its start. The reference must stay independent of metafopy: do not import
`metafopy.methods.ls_profile.profile_starts`; follow its idea (keep only feasible starts, and add starts that are feasible) in the reference's own code.

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires.

**Operator stop (conditional):** Task 1, "stops for the operator: an existing reference value moved". It stops only if the change moves the expected
value of an existing case that calls `ls_profile_ll` or `ls_profile_bounds` (the `ls.confint.scale.*` cases and the profile property tests of
`src/tests/correctness/test_properties.py`) by more than 1e-8, since that would change what an existing case checks.

---

## Task 1: Give the identity-link refits a feasible start

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (a feasible start for the identity link)

**Technical Goal:** `ls_profile_ll` finds the profile log-likelihood at every feasible value of an identity-link scale coefficient, so the reference
can check identity-link profile bounds.

**Subtasks:**

- [x] Re-check the starting point: run the probe of the header (fit `ls_reference_fit` on the `ls.fit.ML.identity.bcg` data, then `ls_profile_ll` at
      `j=1, value=-0.00615`); it returns -1e10. Stop as `blocked`, naming what it returns, if not
- [x] Before changing code, record in this task's Notes the expected values of every existing case and test that calls `ls_profile_ll` or
      `ls_profile_bounds` (the four `ls.confint.scale.*` cases of `location_scale.toml` through `test_derived.ls_confint_scale`, and the calls in
      `test_properties.py`), printed to 12 digits
- [x] Write a failing test first in `src/tests/correctness/test_formulas_ls.py`: `ls_profile_ll` at `alpha.2 = -0.00615` of that fit is -8.045852 to 1e-6, and equals a
      one-dimensional brute-force maximum over the feasible interval of the free coefficient (the bounds where `Z alpha >= 0`) to 1e-6
- [x] In `ls_profile_ll`, for `link == 'identity'`: drop the starts with `min(Z @ full(s)) < 0`, and add feasible starts of the reference's own: the
      centre of the feasible interval of each free coefficient when there is one free coefficient, and otherwise, for example, a point that
      maximises `min(Z alpha)` with `alpha_j` at `value` within bounds (`scipy.optimize.linprog`), each moved onto the line start where that stays feasible. Keep the log-link
      path unchanged. Update the docstring to say how the identity-link starts are chosen
- [x] Rerun the probe: -8.045852 to 1e-6. Recompute the values recorded in the second subtask: if any moved by more than 1e-8, **stops for the
      operator: an existing reference value moved**: stop as `blocked` with both values in the Notes
- [x] Run `pytest src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py -k ls_`
      (name the new test so that it contains `ls_`)

**Acceptance Criteria:**

- [x] The new test fails before the change and passes after it
- [x] Every value recorded in the second subtask is unchanged to 1e-8
- [x] No file under `src/metafopy/` changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Key files: `src/tests/correctness/formulas.py` (`ls_profile_ll`, `ls_profile_bounds`, `ls_reference_fit`, `IDENTITY_SLACK`, `LS_PROFILE_LINE`),
  `src/tests/correctness/test_derived.py` (`ls_confint_scale`), `src/tests/correctness/cases/location_scale.toml`
- Baseline (2026-11-19, before the change; every existing caller uses the log link, whose path is unchanged), `ls_profile_bounds` at 0.95 and
  `ls_profile_ll` at the estimate and at the estimate plus 0.25 se, 12 digits; identical after the change (diffed):
  `ls.confint.scale.ML.bd` alpha.1 (-5.142535627868, -2.406912078227), alpha.2 (-2.459502669826, 4.793278917930);
  `ls.confint.scale.REML.bd` alpha.1 (-4.912020538027, -2.344536922771), alpha.2 (-0.272761583973, 5.755988651403);
  `ls.confint.scale.ML.bcg` alpha.1 (nan, 1.803008594649), alpha.2 (-0.127133605674, 1.381207533768);
  `ls.confint.scale.REML.bcg` alpha.1 (-16.010862366408, 3.402672532394), alpha.2 (-0.166862667671, 0.263879861326);
  peaks: ML.bd -17.058054672888, REML.bd -16.893376921444, ML.bcg -7.539290250092, REML.bcg -8.087277925239. The `property.ls.profile.*` cases
  call the same functions on the `bd` designs and give the same values.
- The header's brute-force value -8.045852 is not the profile maximum: a dense brute-force search over the feasible interval of alpha.1 (lower
  bound 0.33825 where `Z alpha >= 0`) gives -8.045643338 at that boundary, and the new `ls_profile_ll` gives -8.045643337. The test checks -8.045643
  and the brute-force maximum to 1e-6; the -8.045852 of the item was 2e-4 too low.
- New `ls_profile_identity_starts` in `formulas.py`: feasible line starts plus a `linprog` start maximising `min(Z alpha)` and its lines.
- `profile_starts` in `src/metafopy/methods/ls_profile.py` is the library's version of the same idea; read it, do not import it

---

## Task 2: An identity-link scale confint case

**Status:** in-progress
**Priority:** medium
**Complexity:** small
**Source Reference:** the item's fix direction (add an identity-link `ls.confint.scale.*` case checked against the brute-force profile)

**Technical Goal:** metafopy's identity-link profile bounds of a scale coefficient are checked against the independent reference.

**Subtasks:**

- [ ] Add `ls.confint.scale.ML.identity.bcg` to `src/tests/correctness/cases/location_scale.toml` after `ls.confint.scale.ML.bcg`: the same fields,
      with `link = "identity"`, `check = "test_derived.ls_confint_scale"`, `covers = ["ls.confint.scale"]`
- [ ] Run it. If metafopy's bounds and the reference's differ beyond the check's tolerance, record both in this task's Notes, check each against a
      brute-force profile in `<scratch>`, and **stop as `blocked` for the operator** (a change to metafopy's output is the user's to confirm,
      `claude/rules/metafopy.md`); change nothing under `src/metafopy/`
- [ ] Run `pytest src/tests/correctness/test_derived.py -k ls_confint_scale src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [ ] The new case passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 1
- **Blocked for the operator (2026-11-19):** the case `ls.confint.scale.ML.identity.bcg` is added to `location_scale.toml` (uncommitted, because it
  fails) and fails: for alpha.1 (the scale intercept) `confint(result, alpha=0, tol=1e-12)` gives a `nan` lower bound (warnings: "Error when trying to
  invert the Hessian", "Cannot obtain lower bound of profile likelihood CI due to convergence problems"), while the reference gives
  -0.214322988803. The reference is the one that checks out against a brute-force profile in `/tmp/scratch-ls`: at alpha.1 = -0.2143229888 a
  40001-point grid over the feasible alpha.2 interval (`Z alpha >= 0`, alpha.2 >= 0.0164863838) gives -9.060018239, the reference
  `ls_profile_ll` gives -9.060018228, and the crossing level `ll_max - chi2_{1,0.95}/2` is -9.060018228 (ll_max -7.139288818); the same
  agreement holds at alpha.1 = -0.25, -0.2, -0.4 and -1.0 (to 1e-8). So metafopy's identity-link lower bound of alpha.1 is missing where a crossing
  exists; the spec `agents/specs/ls-profile-starts/spec.md` expected this case to pass. Changing metafopy's output is the operator's to confirm
  (`claude/rules/metafopy.md`); nothing under `src/metafopy/` was changed. The other bounds (alpha.1 upper `nan` in both; alpha.2 both) were not
  reached because the check stops at the first mismatch.
- **Operator's answer (2026-11-19, the user, through the operator session):** "Mismatch case + TODO item". Checked by the operator session on
  2026-11-19 in this worktree: R metafor 5.2-1 also gives `NA` for the alpha.1 lower bound, with the same two warnings; the alpha.1 upper bound is
  not found in any of the three (metafopy `>0.2399`, R `>0.2401`, reference `nan`); alpha.2 is metafopy (-0.0136, 0.0165), reference
  (-0.013563, 0.016486), R `<-0.0082`, `>0.0082` (the approved `ls-profile-starts` change). Do this, then finish the task:
  - make `ls.confint.scale.ML.identity.bcg` check every bound but the alpha.1 lower one (if `test_derived.ls_confint_scale` cannot leave one
    bound out, add the smallest option it needs, for example a case field naming the bounds to skip, and test it), and pass;
  - add `ls.confint.scale.ML.identity.bcg.alpha1_lb`, the same fit checking only the alpha.1 lower bound, with `status = "mismatch"` and a
    `reason` giving metafopy's `NA`, R metafor's `NA` and the reference's -0.214322988803 (brute-force checked), "metafopy and R metafor agree;
    recorded on 2026-11-19, not yet reviewed";
  - add a `TODO.md` item after "The reference ls_profile_ll cannot fit the identity link at some feasible values", titled "metafopy's profile CI
    misses the identity-link lower bound of the ls scale intercept", with the figures above dated 2026-11-19, "Cause: not traced; the refits
    below the estimate fail ('Cannot obtain lower bound ... convergence problems'), as in R metafor", and "Fix direction: a spec (an output change,
    `claude/rules/metafopy.md`) for refits that find the crossing, then the mismatch case becomes a passing case";
  - then tick this task's subtasks and acceptance criterion as met by that answer.
- `agents/specs/ls-profile-starts/spec.md` (approved 2026-11-18) says metafopy's identity-link `<` bounds became crossings with the new starts, so the
  case is expected to pass

---

## Task 3: The full test run, and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md` with its outcome.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/ls-profile-identity-start/tasks-1.md; ls_profile_ll keeps only feasible identity-link starts and adds its own; case ls.confint.scale.ML.identity.bcg added`
      with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/ls-profile-identity-start/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/ls-profile-identity-start/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-profile-identity-start/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 2
- No version bump: no library output changes

---
