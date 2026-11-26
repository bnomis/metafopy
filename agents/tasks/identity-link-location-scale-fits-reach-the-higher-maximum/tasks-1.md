# Tasks: Identity-link location-scale fits reach the higher maximum

**Spec:** `agents/specs/identity-link-location-scale-fits-reach-the-higher-maximum/spec.md`
**Created:** 2026-11-26
**Version:** 1

The spec was approved on 2026-11-26 with every choice under "Choices made, not asked"
(`agents/reviews/identity-link-location-scale-fits-reach-the-higher-maximum-spec.md`): the identity link follows the choice of 2026-11-25 and
shares `control['restarts']`, polish only, the reflection start kept, the test reference unchanged, the conditions of the log-link restart.
Approving it confirms the output changes the spec lists, and only those. It builds on the log-link restarts of
`agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md` (implemented 2026-11-25; `RESTART_GAIN`, `RESTART_RATIO`, `restart_starts`
and `is_regular_maximum` in `src/metafopy/models/rma_ls_fit.py`).

Dated figures go in `agents/specs/identity-link-location-scale-fits-reach-the-higher-maximum/measurements.md` (Task 1 creates it); tasks add to
it, never rewrite it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript`.
- **The bcglat permutations:** the data of `ls.permutest.location.exact.ML.z.bcglat` in `src/tests/correctness/cases/location_scale.toml`, fitted
  with `link='identity'`, the rows of `X` permuted in `correctness.formulas.distinct_label_orders` order, indices from 0. The spec names 18, 68,
  679 and 914 (R's default-start log-likelihoods on 2026-11-26: `-12.62494606`, `-12.58248234`, `-10.46818403`, `-12.64295234`).

Stop the task as `blocked` (spec, Output changes) when an output moves that the spec does not list: any fit of the test suite or a correctness
case other than the new one, an R parity fixture entry other than the one Task 4 adds, a printed example, or an audit difference that is not a
restarted identity fit. Stop it too when a restarted fit ends below its single-start fit. Widen no tolerance and regenerate no fixture except the
one Task 4 extends.

---

## Task 1: Starting checks and the baselines

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Motivation; Implementation, Tests and records (the scoped audit); Cost

**Technical Goal:** the work has not started, and the figures the later tasks compare against are recorded on the unchanged code.

**Subtasks:**

- [x] Check that no new name exists yet: `grep -rnwE 'identity_restart_starts|is_face_maximum|identity_restart_fit' src/metafopy` prints
      nothing; otherwise stop as `blocked`, naming the hit
- [x] Check that `LSControl.restarts`, `RESTART_GAIN`, `RESTART_RATIO` and `is_regular_maximum` exist in `src/metafopy/models/` (the log-link
      spec's work); otherwise stop as `blocked`
- [x] Create `measurements.md` beside the spec with a dated section for this task, inside `doc-counts` accept markers
- [x] Record, at the task's starting commit, with `link='identity'`: for permutations 18, 68, 679 and 914, metafopy's log-likelihood, the
      `tau2` of each `lat` group and the location z from `rma_uni`, and the log-likelihood from `formulas.ls_reference_fit`; the log-likelihood of
      every one of the 1287 permutation fits, saved as JSON outside the repository (in the session scratchpad), with its path in
      `measurements.md`; the median time of 20 identity fits of permutation 18 and the time of one loop over the 1287 permutation fits
- [x] Run the scoped audit on the unchanged code, `bin/metafor-audit.sh --only '*/ls_*'` (in the background; it is slow), and record the run
      directory and its count of unaccepted differences

**Acceptance Criteria:**

- [x] `measurements.md` holds every baseline above, dated, with the commit; the four permutations' log-likelihoods match R's above to 1e-6

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Measured during the spec on 2026-11-26 at `f0f6d44`: 39.4 ms per identity fit before the change (re-measure)
- Fit a permutation through `rma_uni` with the permuted `lat` as its own data column for `mods` and `scale = "~ lat"`, as the case
  `ls.fit.ML.log.bcglat.permuted433` does

---

## Task 2: The starts and the face guard

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`identity_restart_starts`, `is_face_maximum`); Implementation, The starts

**Technical Goal:** the derived identity-link starts and the regularity check on a face exist and are tested, with no fit changed yet.

**Subtasks:**

- [x] Add tests to `src/tests/test_rma_ls_restarts.py`: on a binary `Z` with an intercept, `identity_restart_starts` gives the two `tau2_g = 0`
      starts (the other group's `tau2` unchanged) and the reflection that swaps the two `tau2`; on a continuous `Z` every start returned has
      `Z a >= 0`; a start's near-zero entries are exactly `0.0` (no `-0.` residue that makes `Z a` negative); `is_face_maximum` is true at a face
      maximum (permutation 18's, reached by `rma_ls_opt.polish_identity` from its `tau2_{lat=0} = 0` start, ll `-11.42937466`) and false on a
      face whose projected information is singular. Run them on the code before this task: they must fail on the missing names
- [x] Add `identity_restart_starts` and `is_face_maximum` to `src/metafopy/models/rma_ls_fit.py` with the spec's signatures and docstrings: the
      projection `a = alpha - G r (r' alpha) / (r' G r)` with `G = pinv(Z'Z)` per distinct non-zero row, the zeroing below
      `1e-12 * max(1, max|a|)`, the intercept top-up within `ACTIVE_TOL * max(1, max|Z a|)`, the reflection about each non-constant column's
      midrange when `Z` has a constant column, and the active rows within `rma_ls_opt.ACTIVE_TOL` of zero relative to the largest `z_i' alpha`
- [x] Link the paper (Viechtbauer and Lopez-Lopez 2022, doi:10.1002/jrsm.1562) and the R source of the identity fit in the docstrings

**Acceptance Criteria:**

- [x] The new tests pass and failed before the change; nothing calls the new functions yet, so `pytest src/tests/test_rma_ls.py` is unchanged

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/models/rma_ls_fit.py src/tests/test_rma_ls_restarts.py` passes
- [x] `pytest src/tests/test_rma_ls_restarts.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/metafopy/models/rma_ls_fit.py` (the log-link `restart_starts` and `is_regular_maximum` to model on),
  `src/metafopy/models/rma_ls_opt.py` (`ACTIVE_TOL`, `polish_identity`)
- The test reference's `formulas.ls_face_basis` (`src/tests/correctness/formulas.py`) builds the same face basis; the library must not import it

---

## Task 3: The restart in `fit_rma_ls`

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`identity_restart_fit`, `LSControl.restarts`); Implementation, When it runs and The restart; Error handling; Settings;
Cost; Output changes

**Technical Goal:** an identity-link location-scale fit with every scale coefficient free polishes from the derived starts and reports the higher
maximum on its face.

**Subtasks:**

- [x] Add tests to `src/tests/test_rma_ls_restarts.py`: on permutation 18 with `link='identity'`, `rma_uni` reaches ll `-11.42937466` (to 1e-6)
      with the `tau2` of `lat = 0` equal to 0, and with `control={'restarts': False}` ll `-12.62494606` and location z `0.2490`; a
      `restarts=False` identity fit equals the fit of the code before this task on a case of `src/tests/test_rma_ls.py`. Run them before the change:
      the first must fail
- [x] Add `identity_restart_fit` to `src/metafopy/models/rma_ls_fit.py` as the spec gives it: each start through `polish_identity` with the fit's
      own `loglik`, `ls_score` and `ls_information` and every coefficient free, no optimiser; skip a start that raises `EstimationError` or
      `numpy.linalg.LinAlgError` or ends non-finite; keep the best end point only when it beats `par_opt` by more than `RESTART_GAIN` and
      `is_face_maximum` holds there
- [x] Call it in `fit_rma_ls` right after the existing `polish_identity` of the end point, under exactly the spec's conditions (identity link,
      `con.restarts`, no `optbeta`, no user weights, `alpha_arg` all NaN, a non-constant column of `Z`), before the Hessian
- [x] Update the docstring of `LSControl.restarts` in `src/metafopy/models/rma_ls_result.py` to say it covers both links
- [x] Record in `measurements.md`: the four permutations' new ll, `tau2` and z; over the 1287 permutation fits, the count whose ll changed by more
      than 1e-6 against Task 1's JSON (85 expected from the prototype), and that none went down; the median time of 20 identity fits of
      permutation 18 and of the 1287-fit loop, against Task 1's
- [x] Run `pytest src/tests/test_rma_ls.py src/tests/test_ls_vif.py src/tests/test_audit8_crashes.py` and
      `pytest src/tests/correctness/test_derived.py -k "ls."` (slow included): no test may change its result, or stop as `blocked` naming it

**Acceptance Criteria:**

- [x] The new tests pass, and the first failed before the change; no permutation fit ends below its Task 1 value; every listed test and
      location-scale correctness case keeps its result

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/models/rma_ls_fit.py src/metafopy/models/rma_ls_result.py src/tests/test_rma_ls_restarts.py` passes
- [x] `pytest src/tests/test_rma_ls_restarts.py src/tests/test_rma_ls.py` passes

**Notes:**

- Depends on Task 2
- Key file: `src/metafopy/models/rma_ls_fit.py` (the identity polish block and the log-link restart block after it, in `fit_rma_ls`)
- `make_ls_control` already refuses a non-bool `restarts` with `InputError`; nothing changes there

---

## Task 4: The correctness case and the R fixture

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation, Tests and records (the correctness case)

**Technical Goal:** the correctness suite checks the departure on permutation 18, and an R parity test checks that `restarts=False` keeps the
port of metafor's single start.

**Subtasks:**

- [ ] Add a correctness case `ls.fit.ML.identity.bcglat.permuted18` to `src/tests/correctness/cases/location_scale.toml`, beside
      `ls.fit.ML.log.bcglat.permuted433` and in its shape (the permuted `lat` as the data column `perm`, `mods = "~ perm"`, `scale = "~ lat"`,
      `link = "identity"`), with a comment naming this spec and R's ll at its default start, checked by `test_derived.ls_fit` against
      `ls_reference_fit`
- [ ] Add `out$bcg_perm18_identity` to `src/tests/r/generate_location_scale.R`, beside `out$bcg_perm433`, with a comment naming this spec;
      regenerate `src/tests/reference/phase5/location_scale.json`; check with `git diff` that no other entry of the fixture changed, or stop as
      `blocked`
- [ ] Add `'bcg_perm18_identity'` to the R-parity fits of `src/tests/test_rma_ls.py` with `link='identity'` and `control={'restarts': False}`
      (ll `-12.62494606`, z `0.2490` on 2026-11-26)

**Acceptance Criteria:**

- [ ] `pytest src/tests/test_rma_ls.py src/tests/correctness/test_derived.py -k "ls."` passes, the new case and the new parity test included

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_rma_ls.py` passes
- [ ] The pytest command of the acceptance criterion passes

**Notes:**

- Depends on Task 3
- Find the permuted vector of permutation 18 with `formulas.distinct_label_orders`, as Task 1 does, and write it into both the case and the R
  script literally

---

## Task 5: Documentation and the scoped audit

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation, Tests and records (docs and audit); Settings

**Technical Goal:** a user can find out that and why metafopy departs from R for the identity link and how to get R's fit, and the audit
accounts for every difference the restart causes.

**Subtasks:**

- [ ] Add to `docs/differences.md`, under "Location-scale fits with link="identity"": the restart, when it runs, the bcglat permutation 18
      example with R's and metafopy's log-likelihoods, `control={'restarts': False}`, and what is left out (fixed coefficients, `optbeta`,
      weights); update the log-link entry under "Optimiser stop points" where it says the identity link is left out
- [ ] Update every other mention that `restarts` covers the log link only (`grep -rn "restarts" docs src/metafopy`)
- [ ] Run `bin/metafor-audit.sh --only '*/ls_*'` again and compare with Task 1's run: each new difference is a restarted identity fit (check its
      log-likelihood is higher than R's); record each in `src/audit/data/known.toml` with this spec as its reason. A new difference of any other
      kind stops the task as `blocked`
- [ ] Record both runs' directories and counts in `measurements.md`

**Acceptance Criteria:**

- [ ] The scoped audit has no unaccepted difference; `pytest src/tests/test_docs_examples.py` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py src/tests/test_harness_accounts.py` passes

**Notes:**

- Depends on Task 4
- `bin/metafor-audit.sh --recall` is not run: it audits metafopy as of each earlier audit's commit, so a library change here cannot change it

---

## Task 6: The full local check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** the whole spec

**Technical Goal:** the branch passes every check before the operator merges it, the spec says it is implemented, and the `TODO.md` item moves
to `DONE.md`.

**Subtasks:**

- [ ] Run `pytest src/tests/correctness/test_derived.py -k "ls."` (slow cases included)
- [ ] Run the final tests and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/identity-link-location-scale-fits-reach-the-higher-maximum/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/identity-link-location-scale-fits-reach-the-higher-maximum/tasks-1/final-tests.log`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Move the `TODO.md` item "Identity-link location-scale fits can stop below a higher maximum on the face `tau2 = 0`" to the end of `DONE.md`:
      append its heading and body unchanged, then `Done: <date> - agents/specs/identity-link-location-scale-fits-reach-the-higher-maximum/spec.md,
      agents/tasks/identity-link-location-scale-fits-reach-the-higher-maximum/tasks-1.md.`, and remove it from `TODO.md`

**Acceptance Criteria:**

- [ ] The final test run passes, the spec says `implemented`, and the item is in `DONE.md` with its `Done:` line

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy src/metafopy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/identity-link-location-scale-fits-reach-the-higher-maximum/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1-5
- The list does not push
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
