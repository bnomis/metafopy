# Tasks: Location-scale fits reach the higher maximum

**Spec:** `agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md`
**Created:** 2026-11-25
**Version:** 1

The spec was approved on 2026-11-25 with every choice under its Decisions
(`agents/reviews/location-scale-fits-reach-the-higher-maximum-spec.md`): restarts on by default, R-parity tests pass `restarts=False`, the
identity link and fits with a fixed coefficient left out. Approving it confirms the output changes the spec lists, and only those.

Dated figures go in `agents/specs/location-scale-fits-reach-the-higher-maximum/measurements.md` (Task 1 creates it); tasks add to it, never
rewrite it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript`.
- **The bcglat permutations:** the data of `ls.permutest.location.exact.ML.z.bcglat` in `src/tests/correctness/cases/location_scale.toml`, the
  rows of `X` permuted in `correctness.formulas.distinct_label_orders` order, indices from 0; the eight with a higher interior maximum are 68, 264,
  433, 651, 805, 979, 1110 and 1278.

Stop the task as `blocked` (spec, Output changes) when an output moves that the spec does not list: a main `rma_uni` fit, a correctness case other
than the bcglat case and the new one, an R parity fixture other than those named in Task 5, a printed example, or an audit difference that is not
a restarted fit. Widen no tolerance and regenerate no fixture except the one Task 5 extends.

---

## Task 1: Starting checks and the baselines

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Motivation; Implementation, Tests and records (the scoped audit)

**Technical Goal:** the work has not started, and the figures the later tasks compare against are recorded on the unchanged code.

**Subtasks:**

- [x] Check that no new name exists yet: `grep -rnwE 'restarts|RESTART_STEP|RESTART_GAIN|RESTART_RATIO|restart_starts|is_regular_maximum' src/metafopy`
      prints nothing; otherwise stop as `blocked`, naming the hit
- [x] Check that `ls.permutest.location.exact.ML.z.bcglat` still has `status = "mismatch"` naming the `TODO.md` item; otherwise stop as `blocked`
- [x] Create `measurements.md` beside the spec with a dated section for this task, inside `doc-counts` accept markers
- [x] Record, at the task's starting commit: for the eight permutations, metafopy's log-likelihood and the `lat` z from `rma_uni` and from
      `formulas.ls_reference_fit`; the case's exact `permutest` hits (lat 308, intercept 794 of 1287 on 2026-11-25), the reference bounds and its
      count of permutations without a fit (68); the median time of 20 bcglat fits and of one exact permutest
- [x] Run the scoped audit on the unchanged code, `bin/metafor-audit.sh --only '*/ls_*'` (in the background; it is slow), and record the run
      directory and its count of unaccepted differences

**Acceptance Criteria:**

- [x] `measurements.md` holds every baseline above, dated, with the commit

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Measured during the spec on 2026-11-25 at `9bb0947`: one fit 4.7 ms, the exact permutest 7.5 s (the prototype's machine state; re-measure)

---

## Task 2: The test reference reaches the higher maximum

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation, The test reference

**Technical Goal:** `formulas.ls_reference_fit` reaches the highest regular interior maximum, so the correctness cases compare metafopy with the
estimate the paper defines and do not move when either side changes its starts.

**Subtasks:**

- [x] Write `src/tests/correctness/test_ls_reference_fit.py` (no test file of `formulas.ls_reference_fit` existed on 2026-11-25), a test that fits permutations 68, 264, 651, 805, 979 and 1110 and expects the higher log-likelihood
      (68: `-12.57323737`); run it on the unchanged code: it must fail
- [x] In `ls_reference_fit`, after the present search, refit from one start per non-constant column of `Z` (the slope moved by
      `-sign(alpha_j) * 4 / sd(Z_j)`, `sign(0) = 1`) in the reference's own code, and keep the end point with the highest log-likelihood that
      passes `ls_is_maximum`
- [x] Make `ls_is_maximum` take the information on the standardised `Z` (`ls_standardise`) with the threshold `1e-5` (a new constant beside
      `LS_DEGENERATE`, which stays for its other uses); update its docstring
- [x] Run every location-scale correctness case, `pytest src/tests/correctness/test_derived.py -k "ls."` (slow included), and the bcglat
      reference: record the new bounds and the count without a fit in `measurements.md`; the count must stay 68 and no other case may change
      its verdict, or stop as `blocked`

**Acceptance Criteria:**

- [x] The new test passes and failed on the unchanged code; the reference reaches the higher maximum on all eight permutations
- [x] Every other location-scale correctness case keeps its verdict; the bcglat case is still a strict xfail (metafopy is unchanged)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/tests/correctness/formulas.py src/tests/correctness/test_ls_reference_fit.py` passes
- [x] `pytest src/tests/correctness/test_ls_reference_fit.py` and `pytest src/tests/correctness/test_derived.py -k "ls."` pass (the bcglat
      case as xfail)

**Notes:**

- Depends on Task 1
- Key file: `src/tests/correctness/formulas.py` (`ls_reference_fit` near line 3067, `ls_is_maximum`, `ls_standardise`)
- The reference must not import metafopy's restart code: it is the independent side of the comparison

---

## Task 3: The `restarts` control setting

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`LSControl.restarts`); Error handling; Settings

**Technical Goal:** `rma_uni(..., scale=..., control={'restarts': False})` is accepted, stored on the fit and carried into refits; a value
that is not a bool is refused.

**Subtasks:**

- [ ] Write tests in `src/tests/test_rma_ls.py`: the default is `True`; `control={'restarts': False}` is stored in `res.control`; a refit
      through `metafopy.methods.ls_common.refit_ls` keeps it; `control={'restarts': 'FALSE'}` raises `metafopy.errors.InputError`
- [ ] Add `restarts: bool = True` to `LSControl` in `src/metafopy/models/rma_ls_result.py` with the spec's docstring
- [ ] In `make_ls_control` (`src/metafopy/models/rma_ls_fit.py`), raise `InputError` for a `restarts` that is not a `bool`; add it to the
      `Raises:` section

**Acceptance Criteria:**

- [ ] The tests pass; no fit changes yet (nothing reads the setting)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/models/rma_ls_result.py src/metafopy/models/rma_ls_fit.py` passes
- [ ] `pytest src/tests/test_rma_ls.py` passes

**Notes:**

- Depends on Task 1

---

## Task 4: The restart in `fit_rma_ls`

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`RESTART_*`, `restart_starts`, `is_regular_maximum`); Implementation, When it runs and The restart; Cost

**Technical Goal:** a log-link location-scale fit with every scale coefficient free restarts once per slope and reports the higher regular
maximum.

**Subtasks:**

- [ ] Write `src/tests/test_rma_ls_restarts.py`: on permutation 433, `rma_uni` reaches ll `-11.96789514` and `lat` z `1.9024`, and with
      `control={'restarts': False}` ll `-12.33206543` and z `-2.5047`; on a ridge permutation (one whose default fit has `|alpha| > 10`, from
      Task 1's records) the fit is unchanged; `restart_starts` gives one start per non-constant column; `is_regular_maximum` is true at the
      interior maximum and false at a ridge end point. Run it on the code before this task: the first test must fail
- [ ] Add `RESTART_STEP`, `RESTART_GAIN`, `RESTART_RATIO`, `restart_starts` and `is_regular_maximum` to `src/metafopy/models/rma_ls_fit.py`
      as the spec gives them; `is_regular_maximum` uses `LSProblem.ls_information` on the standardised `Z`
- [ ] In `fit_rma_ls`, after the end point (`polish_log_fit` included) and before the Hessian, run the restarts under exactly the spec's
      conditions (log link, `con.restarts`, no `optbeta`, every `alpha` free, infinite `alpha_min`/`alpha_max`, no user weights, a non-constant
      column), each through the same optimiser and polish without the Hessian; skip a restart that raises `EstimationError` or
      `numpy.linalg.LinAlgError` or gives a non-finite log-likelihood; replace the end point only when the gain exceeds `RESTART_GAIN` and
      `is_regular_maximum` holds
- [ ] Record in `measurements.md`: the eight permutations' new ll and z; the case's exact `permutest` hits (306 and 793 expected from the
      prototype); the median time of 20 bcglat fits and of one exact permutest, against Task 1's
- [ ] Run `pytest src/tests/test_rma_ls.py src/tests/test_ls_vif.py src/tests/test_audit8_crashes.py`: list each failing test in
      `measurements.md`; every failure must be one the spec names (Output changes), or stop as `blocked`. Task 5 makes them pass

**Acceptance Criteria:**

- [ ] The new tests pass, and the first failed before the change; exactly the eight bcglat permutations change in the exact permutest
- [ ] No main fit of the test suite changes: every location-scale correctness case keeps its verdict
      (`pytest src/tests/correctness/test_derived.py -k "ls."`), the bcglat case aside

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/models/rma_ls_fit.py src/tests/test_rma_ls_restarts.py` passes
- [ ] `pytest src/tests/test_rma_ls_restarts.py` passes

**Notes:**

- Depends on Tasks 2 and 3
- Prototype (2026-11-25, outside the repository): the same rule as a wrapper of `fit_rma_ls` changed 43 permuted refits and no main fit
- The bcglat case starts passing here, so its strict xfail fails until Task 5; say so in the commit body

---

## Task 5: Tests and records

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation, Tests and records; Decisions (R-parity tests pass `restarts=False`)

**Technical Goal:** the R-parity tests keep checking the port of metafor's single start, and the correctness suite checks the departure.

**Subtasks:**

- [ ] Remove `status = "mismatch"` and its `reason` from `ls.permutest.location.exact.ML.z.bcglat`
- [ ] Add a correctness case on permutation 433 (the permuted `lat` as its own data column for `mods`, `scale = "~ lat"`), checked by
      `test_derived.ls_fit` against `ls_reference_fit`
- [ ] Add the restarts=False fit of permutation 433 to `src/tests/r/generate_location_scale.R` (R's default fit: ll `-12.33206543`, z `-2.5047`
      on 2026-11-25), regenerate `src/tests/reference/phase5/location_scale.json` with it, and test metafopy's `restarts=False` fit against it in
      `src/tests/test_rma_ls.py`; no other entry of the fixture may change
- [ ] Pass `control={'restarts': False}` to the models of the R-parity tests whose fits changed in Task 4 (among them `test_permutest[...]`,
      `test_permutest_print[t]`, `test_permutest_exact_iter_and_exact`, the `test_ls_vif` permutation simulation), each with a comment naming
      this spec
- [ ] `test_audit8_crashes::test_ls_fit_on_a_ridge_is_fast`: if it fails on time, record the time before and after in `measurements.md` and stop
      as `blocked`; do not widen its limit

**Acceptance Criteria:**

- [ ] `pytest src/tests/test_rma_ls.py src/tests/test_ls_vif.py src/tests/test_audit8_crashes.py src/tests/correctness/test_derived.py -k "ls."`
      passes, the bcglat case included

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_rma_ls.py src/tests/test_ls_vif.py` passes
- [ ] The pytest command of the acceptance criterion passes

**Notes:**

- Depends on Task 4

---

## Task 6: Documentation and the scoped audit

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation, Tests and records (docs and audit); Settings

**Technical Goal:** a user can find out that and why metafopy departs from R here and how to get R's fit, and the audit accounts for every
difference the restarts cause.

**Subtasks:**

- [ ] Add an entry to `docs/differences.md` under "Optimiser stop points": when the restart runs, the bcglat permutation 433 example with R's and
      metafopy's log-likelihoods, `control={'restarts': False}`, and what is left out (identity link, fixed coefficients, bounds, `optbeta`, weights)
- [ ] Mention `restarts` wherever the location-scale `control` entries are documented (`grep -rn "polish_converged\|hessian_r" docs`)
- [ ] Run `bin/metafor-audit.sh --only '*/ls_*'` again and compare with Task 1's run: each new difference is a restarted fit (check its fit's
      log-likelihood is higher than R's); record each in `src/audit/data/known.toml` with this spec as its reason. A new difference of any other
      kind stops the task as `blocked`
- [ ] Record both runs' directories and counts in `measurements.md`

**Acceptance Criteria:**

- [ ] The scoped audit has no unaccepted difference; `pytest src/tests/test_docs_examples.py` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py src/tests/test_harness_accounts.py` passes

**Notes:**

- Depends on Task 5
- `bin/metafor-audit.sh --recall` is not run: it audits metafopy as of each earlier audit's commit, so a library change here cannot change it

---

## Task 7: The full local check and closing

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
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/location-scale-fits-reach-the-higher-maximum/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/location-scale-fits-reach-the-higher-maximum/tasks-1/final-tests.log`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Move the `TODO.md` item "Location-scale fits can stop at a lower local maximum than the test reference" to the end of `DONE.md`: append its
      heading and body unchanged, then `Done: <date> - agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md,
      agents/tasks/location-scale-fits-reach-the-higher-maximum/tasks-1.md.`, and remove it from `TODO.md`
- [ ] Add a dated note to the `TODO.md` item "Identity-link location-scale fits can stop below a higher maximum on the face `tau2 = 0`": the
      log-link restarts and `control['restarts']` are in place

**Acceptance Criteria:**

- [ ] The final test run passes, the spec says `implemented`, and the item is in `DONE.md` with its `Done:` line

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy src/metafopy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/location-scale-fits-reach-the-higher-maximum/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1-6
- The list does not push
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
