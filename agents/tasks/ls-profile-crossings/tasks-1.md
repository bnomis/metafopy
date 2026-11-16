# Tasks: location-scale profile-likelihood crossings

**Source:** TODO.md, "The location-scale profile-likelihood intervals of the scale coefficients are not crossings where the profile is flat"
**Created:** 2026-11-15
**Version:** 1

Row 6 of the run order `agents/tasks/order-numerical-fixes.md`. It waits for row 5, `agents/tasks/ls-identity-polish/tasks-1.md`, which polishes the
identity-link location-scale optimum in `src/metafopy/models/rma_ls_fit.py`; every refit this list changes goes through that module, so Task 1 checks
that row 5's change is on `main` before anything else.

The user answered *follow the reference* for both findings of the item on 2026-11-15 (`agents/specs/correctness-location-scale/mismatches.md`, items
`ls_plci_bcg` and `ls_plci_vl2022`), so this list changes metafopy's output without a new review, within what those two answers name: the
profile-likelihood bounds of the scale coefficients of a location-scale model (`confint(..., alpha=j)` of `rma_uni(..., scale=)`) and the profile
that `profile()` draws from the same refits.

**Measured on 2026-11-15** (metafopy at `89ec047`, R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`). The item names one suspected cause;
the measurements show two, one per finding:

- **The refit at a fixed `a` ends at a lower local maximum** (the `dat.bcg` cases). On the ML fit of `dat.bcg` log risk ratios with `mods = scale =
  ~ ablat` (scale intercept -4.957491, standard error 3.267588), `refit_ls(r, alpha=[a, nan])` gives `2 (ll_max - ll(a))` = 3.04141 at `a` = 1.0 with
  the slope at -0.0987, but 3.86864 at every `a` from 1.1 to 3.4 with the slope jumped to about -2.2 (towards `tau2 = 0`), just above the cut-off
  3.84146. The search therefore stops at about 1.02 where the reference profile, which keeps the slope on its continuous branch, crosses at 1.803009.
  On the REML fit the jump comes between 2.3 (2.49038) and 3.29 (6.67605), so the upper bound stops at 3.286 instead of 3.402673. R's refits jump in
  the same places (the same 2 (ll_max - ll(a)) at each of these `a`): metafor 5.2-1 gives ML `(NA, 1.019761)` with "Cannot obtain lower bound" and
  REML `(-16.01086, 2.311399)`, so metafopy will differ from metafor here.
- **The search interval is capped at ten times the estimate** (the Viechtbauer and Lopez-Lopez, 2022, cases). `search_limits` in
  `src/metafopy/methods/ls_confint.py` follows metafor's `confint.rma.ls`: the estimate plus and minus the smaller of `10 |vc|` and `10 crit se`. For a
  coefficient near zero the first term wins: metafor 5.2-1 prints the Table 1 sample size bound as `<-5.9300` (-0.5391 - 10 x 0.5391) and the Table 2
  social sciences bounds as `<-4.6807 >5.7208` (0.5201 -/+ 10 x 0.5201), which are the values metafopy gives. With `xlim = c(-20, 20)` metafor gives
  the paper's -7.1591 and 7.7000. The paper was computed with metafor, so its authors set a wider interval than the 5.2-1 default.

The fix follows the reference in both: Task 2 refits from more than one start and keeps the highest likelihood (as `formulas.ls_profile_bounds` does,
through `ls_profile_ll`), and Task 3 drops the `10 |vc|` cap, so that the interval searched is the estimate plus and minus `10 crit se` (19.6 standard
errors at 95%, against the reference's 20), limited by `alpha_min` and `alpha_max` as now. A bound the profile does not cross within that interval is
reported as metafor reports it: the end of the interval with a `<` or `>` sign. Both are departures from R metafor 5.2-1, recorded in Task 4.

Decisions made in writing this list (each reversible, none needing the user):

- The `10 |vc|` cap is dropped rather than the interval set to the reference's 20 standard errors: it keeps metafor's own `10 crit se` term, and the
  cap is what departs from the reference (it depends on how far the coefficient is from 0, which a recoding of the scale covariate changes).
- The multi-start refit lives in `src/metafopy/methods/ls_profile.py` (`profile_fit`, which `profile` already uses) and `ls_confint.profile_lrt`
  calls it, so `profile()` and `confint()` stay consistent. Starting values go through the existing `alpha_init` control setting of `refit_ls`, so
  `src/metafopy/methods/ls_common.py` and the public API do not change.
- metafopy's current results for `profile()` and `confint()` stay reachable: `confint(..., xlim=)` and `control={'vc.min': ..., 'vc.max': ...}`
  reproduce metafor's interval. No new argument is added.

No version bump: `pyproject.toml` is at `0.0.1` and nothing in this list requires one. No task stops for the operator.

The project has no affected-tests tool, so each task names its test files, and the last task runs the whole suite, `source bin/activate.sh && pytest
src/tests`, as the run order requires.

---

## Task 1: Confirm the two causes and the starting point

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's "Suspected cause" and both `###` findings

**Technical Goal:** the cause of each finding is confirmed in this worktree before any code changes, and the fix is shown to fit in
`ls_profile.py` and `ls_confint.py`.

**Subtasks:**

- [x] Check that row 5 is on `main`: `git log -n 1 --format=%H main -- agents/tasks/ls-identity-polish/tasks-1.md` prints a commit, `git show
      main:agents/tasks/ls-identity-polish/tasks-1.md | grep -cE '^\*\*Status:\*\* (todo|in-progress)'` prints 0, and `git merge-base --is-ancestor main
      HEAD` exits 0. Stop as `blocked`, naming which check failed, if not
- [x] Check the starting point: the four cases `ls.confint.scale.ML.bcg`, `ls.confint.scale.REML.bcg` and `published.vl2022.table1.plci.size`,
      `published.vl2022.table2.plci.soc` in `src/tests/correctness/cases/location_scale.toml` have `status = "mismatch"`, and `pytest
      src/tests/correctness/test_derived.py src/tests/correctness/test_published.py -rx` reports each of the four as an expected failure. Stop as
      `blocked`, naming it, for a case that passes or has another status
- [x] Confirm the refit cause: for the ML and REML fits of `dat.bcg` in the header, tabulate `2 (ll_max - ll(a))` from `refit_ls(r, alpha=[a, nan])`
      and from `formulas.ls_profile_ll` at `a` = 1.0, 1.1, 1.8, 2.3, 3.29, 3.4, with the slope each refit ends at, and record the table in this task's
      Notes. The cause is confirmed if metafopy's refit is below the reference's log-likelihood at the points where its slope has jumped
- [x] Confirm that a second start fixes it: at the same points, refit through `refit_ls(r, alpha=[a, nan], alpha_init=start)` from the fitted line
      start (the other coefficients set by least squares so that `Z alpha` stays as near `Z alpha_hat` as it can with `alpha_j = a`, as in
      `ls_profile_ll`) and keep the higher log-likelihood; record that it matches the reference to 1e-6. Check that `alpha_init` reaches the
      optimiser when `alpha` fixes a coefficient (`src/metafopy/models/rma_ls_fit.py`, the `alpha_init` block)
- [x] Confirm the search-interval cause: `search_limits` gives -5.93003 for the Table 1 sample size coefficient and 5.72067 for the Table 2 social
      sciences coefficient (the metafopy values in the item), and `confint(..., alpha=j, xlim=(-20, 20))` gives -7.159 and 7.700 to the paper's three
      decimals. Rerun the R check of the header in R to confirm the paper's Table 2 bound:
      `LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <scratch>/vl2022.R` with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib`, where
      the script builds the data of `published.vl2022.table2.plci.soc` and prints `confint(r2)` and `confint(r2, alpha = 3, xlim = c(-20, 20))` of
      `rma(yi, vi, mods = ~ ni100, scale = ~ sci + soc, test = "knha")`; record both printouts in the Notes
- [x] If either cause is not confirmed, or a fix needs a change outside `src/metafopy/methods/ls_profile.py` and `src/metafopy/methods/ls_confint.py`
      (other than tests, `src/audit/data/known.toml` and `docs/`) or to a public signature, stop as `blocked` and report what was found

**Acceptance Criteria:**

- [x] The Notes record row 5's commit on `main`, the refit table, the two-start check and both R printouts

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_published.py` passes (the four cases as strict expected failures)

**Notes:**

- Key files: `src/metafopy/methods/ls_profile.py` (`profile_fit`), `src/metafopy/methods/ls_confint.py` (`search_limits`, `profile_lrt`),
  `src/metafopy/methods/ls_common.py` (`refit_ls`, unchanged), `src/tests/correctness/formulas.py` (`ls_profile_ll`, `ls_profile_bounds`, unchanged)
- `<scratch>` is the session's scratch directory, outside the repository. Worktrees have no `.audit/Rlib`, hence `METAFOPY_AUDIT_RLIB`
- This task changes no code; its commit holds this list's Notes only

**Findings (2026-11-16):**

- Row 5 is on `main`: `agents/tasks/ls-identity-polish/tasks-1.md` was last changed in commit `6c8f896c365fdbfd4c1fc9a03f7e6671a4009217`, it has no
  `todo` or `in-progress` task, and `main` is an ancestor of `HEAD`.
- Refit table, `2 (ll_max - ll(a))` of `dat.bcg` with `mods = scale = ~ ablat`, `link = "log"`: `refit_ls(r, alpha=[a, nan])` (slope), the reference
  `formulas.ls_profile_ll`, and the best of the default start and the fitted line start (`alpha_init`):

  | method | a | refit (slope) | reference | best of two starts |
  |---|---|---|---|---|
  | ML | 1.0 | 3.04141 (-0.0987) | 3.04141 | 3.04141 |
  | ML | 1.1 | 3.86864 (-2.1891) | 3.13614 | 3.13614 |
  | ML | 1.8 | 3.86864 (-2.2503) | 3.83830 | 3.83830 |
  | ML | 2.3 | 3.86864 (-2.3022) | 3.86864 | 3.86864 |
  | ML | 3.29 | 3.86864 (-2.4103) | 3.86864 | 3.86864 |
  | ML | 3.4 | 3.86864 (-2.4356) | 3.86864 | 3.86864 |
  | REML | 1.0 | 1.27269 (-0.0882) | 1.27269 | 1.27269 |
  | REML | 1.1 | 1.35159 (-0.0909) | 1.35159 | 1.35159 |
  | REML | 1.8 | 1.97312 (-0.1106) | 1.97312 | 1.97312 |
  | REML | 2.3 | 2.49038 (-0.1250) | 2.49038 | 2.49038 |
  | REML | 3.29 | 6.67605 (-0.5969) | 3.69056 | 3.69056 |
  | REML | 3.4 | 6.67719 (-0.6056) | 3.83785 | 3.83785 |

  The cause is confirmed: the default-start refit is below the reference's log-likelihood where its slope has jumped (ML 1.1 and 1.8, REML 3.29 and
  3.4). The reference crosses the ML cut-off 3.84146 at 1.803009 (3.83830 at 1.8), where the single-start refit has already jumped to 3.86864.
- The two-start check: keeping the higher log-likelihood of the default start and the fitted line start matches the reference to at most 1.1e-10 at
  all twelve points. `alpha_init` reaches the optimiser when `alpha` fixes a coefficient (`rma_ls_fit.py` lines 515-529: `con.alpha_init` is taken
  as given, only mapped through `imZ` when the scale model has an aliased column).
- Search interval: `search_limits` gives -5.93003 for the Table 1 sample size coefficient (estimate -0.53909) and 5.72081 for the Table 2 social sciences
  coefficient (estimate 0.52007; the item's 5.72067 is the figure to the item's precision, the printed bound is `>5.7208`). `confint(..., alpha=1,
  xlim=(-20, 20))` on Table 1 gives `-7.1591`, and `confint(..., alpha=2, xlim=(-20, 20))` on Table 2 gives `7.7000` (lower `<-20.0000`).
- R metafor 5.2-1 on the Table 2 model (`rma(yi, vi, mods = ~ ni100, scale = ~ sci + soc, test = "knha")`):

  ```text
  confint(r2)                                   alpha.3   0.5201 <-4.6807 >5.7208
  confint(r2, alpha = 3, xlim = c(-20, 20))     alpha.3   0.5201 <-20.0000 7.7000
  ```

- Both fixes fit in `ls_profile.py` and `ls_confint.py`; no public signature changes.

---

## Task 2: Refit the profile from more than one start

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** "dat.bcg scale intercept, ML and REML"; the item's fix direction (refit from the fitted line and more than one start)

**Technical Goal:** each refit of `profile()` and `confint()` at a fixed scale coefficient reaches the maximum of the likelihood over the other
coefficients, so that the profile-likelihood bounds are crossings of the profile.

**Subtasks:**

- [ ] In `profile_fit` (`src/metafopy/methods/ls_profile.py`), refit from each of three starts and keep the fit with the highest log-likelihood:
      the default start (today's refit, so the result is never worse than now), the fit's own `alpha` with `alpha[pos] = val`, and the fitted line
      start of Task 1. Pass a start through `refit_ls(..., alpha_init=start)`; skip a start that leaves a `tau2_i` infeasible under the identity link.
      Return `None` only when every start fails
- [ ] Make `profile_lrt` in `src/metafopy/methods/ls_confint.py` call `metafopy.methods.ls_profile.profile_fit` instead of `refit_ls`, keeping its
      `EstimationError` when the refit fails
- [ ] Update the docstrings of `profile_rma_ls`, `confint_rma_ls` and the two helpers: each refit starts from more than one point and keeps the
      highest likelihood, as the profile likelihood of Viechtbauer and Lopez-Lopez (2022, section 2.5.3) requires, where metafor's `.profile.rma.ls`
      refits once from its default start; link the `docs/differences.md` subsection Task 4 adds
- [ ] In `ls_confint_scale` of `src/tests/correctness/test_derived.py`, read a bound that `confint` shows with a `<` or `>` sign
      (`ConfintResult.lb_sign`, `ub_sign`) as "not crossed" and compare it with the reference's `nan`, so that a bound with no crossing is checked as
      metafor reports it
- [ ] Set `ls.confint.scale.ML.bcg` and `ls.confint.scale.REML.bcg` to `status = "pass"` and drop their `reason`
- [ ] Run the R parity tests of `profile` and `confint` on location-scale fits (`pytest src/tests/test_rma_ls.py src/tests/test_audit5_numerics.py
      src/tests/test_audit8_args.py`). For each failure, check that metafopy's refit at that point has the higher log-likelihood, then mark that
      fixture value as differing as `agents/tasks/confint-ht-centre/tasks-1.md` Task 1 marks `ht_ci`: the JSON stays as R wrote it (never regenerate
      it), and the test compares the value with the reference (`formulas.ls_profile_bounds`, or the profile at that point from `ls_profile_ll`) and
      asserts that R's differs, with a comment naming the correctness case and the `docs/differences.md` subsection. A failure where metafopy's refit
      has the lower log-likelihood is a bug in this change: fix it, never mark it. List each marked value in this task's Notes
- [ ] Add a regression test in `src/tests/test_rma_ls.py`: on the ML fit of `dat.bcg` in the header, `profile_fit` at `a` = 1.1 gives `2 (ll_max -
      ll)` below 3.84 (3.86864 before this change) and `confint(..., alpha=0)` gives the upper bound 1.803009 to 1e-5

**Acceptance Criteria:**

- [ ] `ls.confint.scale.ML.bcg` and `ls.confint.scale.REML.bcg` pass with `status = "pass"`
- [ ] No refit of `profile_fit` ends below the log-likelihood of today's single-start refit at the same point (the default start is one of the three)
- [ ] Every R parity value that now differs is marked in its test, with the reference value checked instead

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_rma_ls.py src/tests/test_audit5_numerics.py src/tests/test_audit8_args.py src/tests/correctness/test_derived.py src/tests/correctness/test_published.py src/tests/correctness/test_properties.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/metafopy/methods/ls_profile.py`, `src/metafopy/methods/ls_confint.py`, `src/tests/correctness/test_derived.py`,
  `src/tests/correctness/cases/location_scale.toml`, `src/tests/test_rma_ls.py`, `src/tests/reference/` (unchanged)
- `ls_confint` importing `ls_profile` adds no cycle: `ls_profile` does not import `ls_confint`
- Three starts triple the refits of `profile()` and `confint()`; record the run time of `pytest src/tests/test_rma_ls.py` before and after in the
  Notes
- The test files are those that import `metafopy.methods.ls_profile` or call `confint`/`profile` on a location-scale fit (`grep -rlE
  'ls_profile|confint_extint_ls|profile_rma_ls|confint_rma_ls' src/tests` on 2026-11-15, plus `test_audit5_numerics.py`)

---

## Task 3: Search the bounds without the ten-times-the-estimate cap

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** "Viechtbauer and Lopez-Lopez (2022) Tables 1 and 2"; the item's fix direction (check metafor's `confint` on the Table 2 model)

**Technical Goal:** the default interval searched for a profile-likelihood bound no longer depends on how far the estimate is from zero, so that the
bounds of Viechtbauer and Lopez-Lopez (2022) are found.

**Subtasks:**

- [ ] In `search_limits` (`src/metafopy/methods/ls_confint.py`), take the estimate plus and minus `10 crit se` when the standard error is finite,
      without the `10 |vc|` term; keep `vc -/+ 10 |vc|` when it is `nan` (there is no other scale then), the `alpha_min` and `alpha_max` limits, and
      `vc_min`, `vc_max` and `xlim`, which still win
- [ ] Update the `control` description in `confint_rma_ls`'s docstring to the new default, say that metafor 5.2-1 also caps it at ten times the
      absolute estimate and that `xlim` reproduces metafor's interval, and link the `docs/differences.md` subsection Task 4 adds
- [ ] Set `published.vl2022.table1.plci.size` and `published.vl2022.table2.plci.soc` to `status = "pass"` and drop their `reason`; check that the
      passing `published.vl2022.table1.plci` and `published.vl2022.table2.plci` still pass
- [ ] Add a test in `src/tests/test_rma_ls.py`: on the Table 1 model (the data of `published.vl2022.table1.plci`), both bounds of the social sciences
      coefficient carry a `<` or `>` sign, at the ends of the new interval (the reference finds no crossing; the paper prints `< -10` and `> 10`), and
      `confint(..., xlim=(vc - 10 |vc|, vc + 10 |vc|))` gives metafor 5.2-1's `<-3.6100` and `>4.4122`
- [ ] Rerun the R parity tests of Task 2 and mark each value that now differs in the same way (a bound that R found at its capped search limit, or
      shows with a `<` or `>` sign at it), listing each in this task's Notes. A failure on a bound R found inside both intervals is a bug: fix it

**Acceptance Criteria:**

- [ ] `published.vl2022.table1.plci.size` and `published.vl2022.table2.plci.soc` pass with `status = "pass"`, and the two passing published cases
      still pass
- [ ] `confint(..., xlim=...)` with metafor's interval gives metafor's bounds on the Table 1 social sciences coefficient

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_rma_ls.py src/tests/test_audit5_numerics.py src/tests/test_audit8_args.py src/tests/correctness/test_derived.py src/tests/correctness/test_published.py src/tests/correctness/test_properties.py` passes

**Notes:**

- Depends on Task 2 (the Table 2 bound needs both the wider interval and refits that do not jump; Task 1 records whether the interval alone is
  enough)
- Key files: `src/metafopy/methods/ls_confint.py`, `src/tests/correctness/cases/location_scale.toml`, `src/tests/test_rma_ls.py`
- `test_ls_confint_control_changes_the_search_interval` in `src/tests/test_audit5_numerics.py` sets `vc.max` itself, so it does not depend on the
  default

---

## Task 4: Record the difference in the audit ledger and the user guide

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (`known.toml`, `docs/differences.md`)

**Technical Goal:** `bin/metafor-audit.sh` accepts the new differences with their reasons, its recall check still passes, and `docs/differences.md`
explains them under "Where metafor departs from its reference".

**Subtasks:**

- [ ] Run the location-scale `confint` and `profile` slice of the audit from the worktree, against the main checkout's R oracle, in the background:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib
      METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --only '*/confint/ls_*' --only '*/profile/ls_*'`. If `check-r` reports that the R
      library is missing or differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [ ] For each finding the run reports on a scale-coefficient bound or a profile log-likelihood, check that metafopy's refit has the higher
      log-likelihood (or that the bound lies outside metafor's capped interval), then add a `[[known]]` entry to `src/audit/data/known.toml` matching
      exactly those case ids, with `kind`, `path`, a `reason` (metafor refits once from its default start and caps the interval at ten times the
      estimate; metafopy keeps the highest of three refits and searches `10 crit se`) and `ref = "docs/differences.md"`. Update the reasons of the
      existing `*/confint/ls_na*` and `*/confint/ls_bcg*` entries if their figures change, and remove an entry the rerun reports as stale. Add no
      entry for a finding the run does not report
- [ ] Rerun the same command and check that its report lists no finding left from this list. If it lists any other finding, look for it in the
      newest earlier report under `$main/.audit/runs/`; one that is not there is new with this list, so stop as `blocked` naming it
- [ ] Run `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --recall` in the
      background and check that every item it reproduced before still reproduces; an item it now misses is fixed in `src/audit/data/`, not in a report
- [ ] Add a subsection `### Profile-likelihood bounds of the scale coefficients` to the section "Where metafor departs from its reference" of
      `docs/differences.md`, after the existing subsections: a short summary of both changes, the same calls in R and in Python with the output of
      both (the ML fit of `dat.bcg` with `mods = scale = ~ ablat`, and the Table 2 model of Viechtbauer and Lopez-Lopez, 2022), and *Why:* the profile
      likelihood maximises over the other coefficients (section 2.5.3 of the paper), and the paper's own bounds lie outside metafor's default
      interval. Name the four correctness cases. Produce the R output with `LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript -e '<the R block>'` and
      the Python output with metafopy, and paste both as printed
- [ ] Rerun the example of the existing subsection "Refits of a location-scale model that R's optimiser leaves short" (`confint(res, alpha = 1)` on
      `mods = ~ ablat, scale = ~ alloc`); if metafopy's printout or the -25.1411 in its text has changed, update both to what metafopy now prints,
      with the date

**Acceptance Criteria:**

- [ ] The audit slice reports no finding from this list, and every new `known.toml` entry matches at least one case of that run
- [ ] `--recall` reproduces every item it reproduced before this list
- [ ] `docs/differences.md` names the difference under "Where metafor departs from its reference", and no figure in it is stale

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on Task 3
- Prerequisite outside this repository: R metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib` (present on
  2026-11-15). Worktrees have no `.audit/`, hence `METAFOPY_AUDIT_RLIB` and `METAFOPY_AUDIT_WORK`
- `--only` is repeatable and matches case ids with `*` (`src/audit/cli.py`). A full audit run is the user's call; a difference the slice does not
  reach (a `scripts/` unit or help example that calls `confint` on a location-scale fit) surfaces in the next full run
- `src/tests/test_docs_examples.py` runs every Python block of `docs/`, so the new example is checked

---

## Task 5: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged, both `###` findings included, at the end of `DONE.md`, then a
      line `Done: 2026-MM-DD, agents/tasks/ls-profile-crossings/tasks-1.md` with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of `git worktree list
      --porcelain` (`/Users/simonb/src/metafopy`):
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/ls-profile-crossings/tasks-1" && source bin/activate.sh && pytest src/tests 2>&1 | tee "$main/ralpher-logs/ls-profile-crossings/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes, `slow` tests included, judged by its exit status
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-profile-crossings/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 4
- The full `pytest src/tests` run is required by the run order `agents/tasks/order-numerical-fixes.md`

---
