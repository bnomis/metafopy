# Tasks: a feasible start for identity-link profile refits that have none

**Spec:** `agents/specs/ls-profile-feasible-start/spec.md`
**Created:** 2026-11-19
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

The spec (version 1.1, approved by the operator on 2026-11-19 in `agents/reviews/ls-profile-feasible-start-spec.md`) adds `feasible_start` to
`src/metafopy/methods/ls_profile.py` and calls it from `profile_starts` only when an identity-link refit is left with no start but the default. It
changes one metafopy output, which the approval confirms: the ML identity-link `dat.bcg` lower bound of `alpha.1` becomes -0.2143 (it was NaN). No
other output is expected to move (`agents/specs/ls-profile-feasible-start/measurements.md`, measured 2026-11-19); a test that changes beyond the
correctness case named below is outside the approval, and the task that meets it stops for the operator. No version bump.

---

## Task 1: feasible_start and its use in profile_starts

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Error handling; Implementation steps 1 to 3

**Technical Goal:** identity-link profile refits that have no feasible start get one, so the ML `dat.bcg` profile of `alpha.1` is fitted below the
estimate and `confint` finds its lower crossing.

**Subtasks:**

- [ ] Starting-point check: `grep -rnwE 'feasible_start|FEASIBLE_WEIGHTS' src/metafopy` prints nothing; if it prints a match, stop the task as
      `blocked`, naming the match
- [ ] Add `FEASIBLE_WEIGHTS = (0.25, 0.5, 0.75)` beside `LINE_WEIGHTS` and `feasible_start(x, pos, val)` to `src/metafopy/methods/ls_profile.py`
      with the signature and docstring of the spec's API section: `scipy.optimize.linprog` with `c = [0, ..., 0, -1]`,
      `A_ub = [-Z[:, free], 1]`, `b_ub = Z[:, pos] * val`, bounds `(None, None)` per free coefficient and `(None, cap)` for `t`, `cap` the mean of
      `x.Z @ x.alpha` (1 where that is not positive); return `None` when `pos` has no other coefficient, the programme fails, or `t <= 0`
- [ ] In `profile_starts`, under the identity link and after the feasibility filter, when the list holds no start but `None`: append
      `feasible_start(x, pos, val)` if not `None`, and the points `start + w * (own - start)` for `w` in `FEASIBLE_WEIGHTS` whose `x.Z @ point` is
      nowhere negative; update its docstring (one sentence)
- [ ] Add tests to `src/tests/test_rma_ls.py` on the ML identity-link `dat.bcg` fit (`escalc` RR, `mods = scale = ~ ablat`):
      `feasible_start(x, 0, -0.2143)` has every `tau2_i > 0`; `profile_starts(x, 0, -0.2143)` contains it; `profile_starts(x, 0, 0.1)` equals the list
      it gives with `feasible_start` patched to raise (a feasible line start exists there, so it is not called); `confint(x, alpha=0, tol=1e-12)`
      gives a lower bound within 1e-8 of -0.214322988803 and no warning containing "Cannot obtain lower bound"
- [ ] Run `pytest src/tests/test_rma_ls.py`

**Acceptance Criteria:**

- [ ] The new tests pass, and every other test in `src/tests/test_rma_ls.py` passes unchanged
- [ ] `profile_starts` returns the same list as before wherever a start other than `None` survives the filter

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask passes

**Notes:**

- Key files: `src/metafopy/methods/ls_profile.py` (`LINE_WEIGHTS`, `profile_starts`, `profile_fit`), `src/metafopy/methods/ls_confint.py`
  (`confint_rma_ls`, `profile_lrt`), `src/tests/test_rma_ls.py`
- The reference's own version is `formulas.ls_profile_identity_starts` in `src/tests/correctness/formulas.py`; do not import it (the reference stays
  independent of metafopy, and metafopy does not import its tests)
- The fit itself warns "Error when trying to invert the Hessian"; that warning is the fit's, not `confint`'s, and stays

---

## Task 2: the correctness case passes

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 4

**Technical Goal:** the strict expected failure that recorded the missing bound becomes part of the passing case.

**Subtasks:**

- [ ] In `src/tests/correctness/cases/location_scale.toml`, delete the case `ls.confint.scale.ML.identity.bcg.alpha1_lb` and add `"alpha.1.lower"` to
      the `bounds` of `ls.confint.scale.ML.identity.bcg`
- [ ] Run `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [ ] `ls.confint.scale.ML.identity.bcg` checks the bounds `alpha.1.lower`, `alpha.1.upper`, `alpha.2.lower` and `alpha.2.upper` and
      passes, and no case id ends in `.alpha1_lb`
- [ ] The coverage test passes (the `ls.confint.scale` coverage item is still covered)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 1
- If any other correctness case changes result, stop the task as `blocked`, naming the case and its old and new values: the approval covers this one
  output only

---

## Task 3: records, the full test run, and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5; Module location

**Technical Goal:** the change is recorded where users read it, the whole branch is checked once, the spec is marked implemented, and the `TODO.md`
item moves to `DONE.md`.

**Subtasks:**

- [ ] Add a `CHANGELOG.md` "Unreleased (1.0 candidate)" entry: identity-link profile refits left with only the default start also start from the
      point of the largest minimum `tau2_i` (behaviour change); the ML `dat.bcg` (`mods = scale = ~ ablat`, RR) lower bound of `alpha.1` is the
      crossing -0.2143 where it was NaN with "Cannot obtain lower bound"
- [ ] Add a sentence to `docs/differences.md`, section "Profile-likelihood bounds of the scale coefficients": on the ML identity-link fit of `dat.bcg`
      log risk ratios with `mods = scale = ~ ablat`, metafor 5.2-1 gives NA for the lower bound of `alpha.1` because its refit from the default start
      fails, and metafopy gives the crossing -0.2143
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-19)" to "Implemented"
- [ ] Move the item "metafopy's profile CI misses the identity-link lower bound of the ls scale intercept" from `TODO.md` to `DONE.md`: append its
      heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/specs/ls-profile-feasible-start/spec.md and agents/tasks/ls-profile-feasible-start/tasks-1.md; profile_starts adds feasible_start when only the default start is left, and the lower bound is -0.2143`
      with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/ls-profile-feasible-start/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/ls-profile-feasible-start/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] `CHANGELOG.md` and `docs/differences.md` name the new bound -0.2143, and no other copy of the old NaN for this bound is left in `docs/`
      (`grep -rn 'alpha1_lb' docs src/tests` prints nothing)
- [ ] The spec says `implemented`; the item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-profile-feasible-start/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1 and 2
- If the final run shows a failure or changed result outside Tasks 1 and 2's cases, stop the task as `blocked` with the failing tests: the approval
  covers one output only
