# Tasks: location-scale identity-link polish

**Source:** TODO.md, "The location-scale identity-link fit stops about 1e-5 short of the maximum"
**Created:** 2026-11-15
**Version:** 1

The user answered *follow the reference* on 2026-11-15 (`agents/specs/correctness-location-scale/mismatches.md`, item `ls_identity_opt`), so this
list changes metafopy's output without a new review, within what that answer names: the optimum of a location-scale fit with `link='identity'`.

Checked against the code on 2026-11-15 (metafopy at `89ec047`). With the identity link, `fit_rma_ls` in `src/metafopy/models/rma_ls_fit.py` calls
`constr_optim` in `src/metafopy/models/rma_ls_opt.py` (a port of R's `constrOptim`: an adaptive log barrier around Nelder-Mead, stopped when the
barrier objective changes by less than `outer_eps = 1e-5` relative) and takes `res.par` as the optimum, with no polish. The other identity-link
optimisers (`solnp`, `nloptr`, `constrOptim.nl`, `auglag`) go through `minimise_standin` and are not polished either.

**Measured on 2026-11-15** on a scratch run, with `formulas.ls_loglik` and `formulas.ls_score` of `src/tests/correctness/formulas.py`:

- `dat.bangertdrowns2004`, `mods = scale = ~ imag`, ML: metafopy's alpha `[0.03401541, 0.23040472]`, log-likelihood -17.058054743801; the analytic
  score there is `[0.0118, -0.00093]`, not zero. SLSQP with that score and `Z alpha >= 0`, started from metafopy's alpha, ends at
  `[0.03402053, 0.23031694]`, -17.058054672888, the reference (`formulas.ls_reference_fit`) to every printed digit. This optimum is interior
  (smallest `tau2_i` 0.034).
- `dat.bcg` log risk ratios, `mods = scale = ~ ablat`: the optimum is on the boundary. metafopy's smallest `tau2_i` is 2.3e-10 (ML) and 2.4e-9
  (REML); the reference's is -1e-10 (its `IDENTITY_SLACK`), and the score there is far from zero (`[-57.1, -742.2]` under ML), so the reference
  maximum is a constrained (KKT) point with the constraint of the lowest-`ablat` study active. A plain Newton step on the score does not reach it.
- R metafor 5.2-1 stops where metafopy does: the parity fixture `bcg_year_identity` of `src/tests/reference/phase5/location_scale.json` (ML,
  `mods = ~ ablat`, `scale = ~ I(ablat/10)`, the same model up to the scaling of the slope) holds alpha `[-0.026676029, 0.020520022]` and
  log-likelihood -7.13928898408783, which metafopy reproduces today. The reference intercept is -0.0266587, a relative change of 6.5e-4, beyond the
  1e-4 (`OPT`) that `src/tests/test_rma_ls.py` accepts. So the polish makes metafopy depart from metafor, and this list records that under
  "Where metafor departs from its reference" in `docs/differences.md`.

The R parity fixtures that hold identity-link fits (found by `grep -n 'link = "identity"' src/tests/r/*.R` on 2026-11-15): in
`src/tests/r/generate_location_scale.R`, `bd_int_identity`, `bd_identity`, `bd_identity_ml`, `bd_identity_ni` and `bcg_year_identity`; in
`src/tests/r/generate_audit4_numerics.R`, the `identity` fit; in `src/tests/r/generate_ls_vif_sim.R`, `res_ml`. The fixtures are never regenerated.

`agents/tasks/order-numerical-fixes.md` row 6 (the profile-likelihood crossings list) waits for this one: the profile refits the fit this list
polishes. The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order
requires.

---

## Task 1: Confirm the cause and the starting point

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's "Suspected cause" and its list of cases

**Technical Goal:** the cause is confirmed on all four reference fits before any code changes, and the ten cases are seen to fail today.

**Subtasks:**

- [x] Re-check the starting point: the ten cases `ls.fit.{ML,REML}.identity.{bd,bcg}`, `property.ls.fit.{ML,REML}.identity.{bd,bcg}` and
      `property.ls.links.{ML,REML}.bd` in `src/tests/correctness/cases/location_scale.toml` have `status = "mismatch"`, and
      `pytest src/tests/correctness -k 'identity or links'` reports each of them as a strict expected failure. Stop as `blocked`, naming the case,
      if one has another status or passes
- [x] Check that the names Task 2 adds are free: `grep -rnwE 'polish_identity|ls_score|ls_information' src/metafopy` finds nothing. Stop as
      `blocked`, naming the clash, if it finds one
- [x] In a scratch script outside the repository (`source bin/activate.sh`, run from `src/tests` so `correctness.formulas` imports), for each of
      the four fits of the ten cases: fit with `metafopy.rma`, then maximise `formulas.ls_loglik` from metafopy's alpha with `formulas.ls_score`
      under `Z alpha >= 0` (SLSQP), and compare the end point with `formulas.ls_reference_fit`. Record, per fit, metafopy's alpha and
      log-likelihood, the polished alpha and log-likelihood, the reference's, the smallest `Z alpha` at each and which rows are active (within
      1e-8 of zero) at the polished point, in this task's Notes
- [x] Record R metafor's alpha and log-likelihood for the same four fits, from
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the fits>'`, in this task's Notes. If the R library is missing, stop
      as `blocked` and say so
- [x] Decide which way Task 2 goes, by the rule below, and write the outcome in this task's Notes:
      - the polish from metafopy's point reaches the reference's log-likelihood to 1e-9 on all four fits: the cause is confirmed; Task 2 as written;
      - it does not on some fit, but a cause inside `rma_ls_fit.py` or `rma_ls_opt.py` explains it (a starting value, the barrier's `mu`, the
        stopping rule): Task 2 fixes that cause instead, or as well, and its subtasks are amended in this file before it starts;
      - the fix needs a change outside those two modules, or to a public signature (`rma_uni`, `rma`, `LSControl`, a result field): stop as
        `blocked` and report the cause and the change it needs

**Acceptance Criteria:**

- [x] The ten cases fail today, as strict expected failures
- [x] This task's Notes hold the per-fit figures and the decision for Task 2

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py` passes (the ten cases as expected failures)

**Notes:**

- Prerequisite outside the worktree: R metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib` (worktrees have no
  `.audit/Rlib`), present on 2026-11-15
- The header's figures for `dat.bangertdrowns2004` ML and `dat.bcg` ML and REML were measured on 2026-11-15; this task measures all four fits
  under ML and REML in one place, and Task 2 takes its numbers from here
- The four fits are the `args` of the cases: `dat.bangertdrowns2004` with `mods = scale = ~ imag`, `dat.bcg` (log risk ratio) with
  `mods = scale = ~ ablat`, each ML and REML, `link = "identity"`

**Findings (measured 2026-11-16, metafopy at `767acff`):**

The ten cases are `status = "mismatch"` and `pytest src/tests/correctness -k 'identity or links'` gives 10 xfailed. `polish_identity`, `ls_score` and
`ls_information` are absent from `src/metafopy`. The scratch scripts were outside the repository; the worktree has no `.venv`, so they ran with the
main checkout's interpreter and the worktree's `src` and `src/tests` on `sys.path`.

Per fit: metafopy's alpha and log-likelihood (R metafor 5.2-1 gives the same alpha and log-likelihood to every digit printed, for all four), the
SLSQP polish from metafopy's point, and `formulas.ls_reference_fit`. `min Z alpha` is the smallest `tau2_i`.

| Fit | Point | alpha | log-likelihood | min Z alpha | active rows (< 1e-8) |
|---|---|---|---|---|---|
| bd ML | metafopy and R | `[0.03401541, 0.23040472]` | -17.058054743801 | 0.034015 | none |
| bd ML | polished | `[0.03402053, 0.23031694]` | -17.058054672888 | 0.034021 | none |
| bd ML | reference | `[0.03402053, 0.23031694]` | -17.058054672888 | 0.034021 | none |
| bd REML | metafopy and R | `[0.03698802, 0.42979705]` | -16.893376988271 | 0.036988 | none |
| bd REML | polished | `[0.03699443, 0.42990539]` | -16.893376921444 | 0.036994 | none |
| bd REML | reference | `[0.03699443, 0.42990534]` | -16.893376921444 | 0.036994 | none |
| bcg ML | metafopy and R | `[-0.02667603, 0.00205200]` | -7.139288984088 | 2.3e-10 | none (rows 4, 7 sit at 2.3e-10 to 2.9e-9) |
| bcg ML | polished | `[-0.02665903, 0.00205069]` | -7.139288734422 | -1.6e-9 (infeasible, SLSQP `success False`) | 4, 7 |
| bcg ML | reference | `[-0.02665867, 0.00205067]` | -7.139288177296 | -1.0e-10 | 4, 7 |
| bcg REML | metafopy and R | `[0.33905883, -0.00616471]` | -7.988765308005 | 2.4e-9 | none |
| bcg REML | polished | `[0.33913247, -0.00616604]` | -7.988765271982 | 1.9e-9 | 1 |
| bcg REML | reference | `[0.33913546, -0.00616610]` | -7.988765268276 | -1.0e-10 | 1 |

- bd ML and REML: the polished log-likelihood is within 3e-14 and 3e-14 of the reference's; alpha agrees to 6e-8 or better.
- bcg: rows 4 and 7 of `Z` in the ML fit are two studies with the same `ablat` (13), so one constraint is active, not two. Maximising exactly on
  that line (`Z[4] alpha = 0`, zero slack) gives alpha `[-0.02665867, 0.00205067]`, the reference's, with log-likelihood -7.139288823439, which is
  5.7e-9 *below* the reference's -7.139288817730. Likewise REML on `Z[1] alpha = 0` (`ablat` 55): alpha `[0.33913556, -0.0061661]`, log-likelihood
  -7.988765268454, 1.8e-10 below the reference's. The reference's log-likelihood is higher only because `IDENTITY_SLACK` lets it end 1e-10 outside
  the feasible region; the maxima agree in alpha to the printed digits.
- The polish started from metafopy's point and run by SLSQP leaves the feasible region by up to 1.6e-9 on bcg ML. Task 2 must keep the result
  feasible (`Z alpha >= 0`) and must not compare log-likelihoods to the reference tighter than the reference's own slack: on bcg ML a feasible
  maximum is 5.7e-9 below the reference's log-likelihood, so the property cases' tolerance (see `src/tests/correctness/test_properties.py`) is
  checked in Task 2 before its test is made strict.

**Decision for Task 2:** the cause is confirmed. The polish from metafopy's point reaches the reference's alpha on all four fits and its
log-likelihood to 3e-14 on the interior fits; on the boundary fits the remaining gap (5.7e-9 and 1.8e-10 on an exactly feasible optimum) is the
reference's 1e-10 slack, not a cause in `rma_ls_fit.py` or `rma_ls_opt.py`. No change outside those two modules or to a public signature is needed.
Task 2 as written, with the feasibility point above.

---

## Task 2: Polish the identity-link optimum on the analytic score

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (the code change, the ten cases, the R parity fixtures)

**Technical Goal:** every location-scale fit with `link='identity'` ends at the maximum of its (restricted) log-likelihood over `Z alpha >= 0`, so
the ten cases pass, and the tests say where this departs from R metafor's fixtures.

**Subtasks:**

- [x] Add the analytic score and expected information of the scale coefficients with the identity link, as methods of `LSProblem` in
      `src/metafopy/models/rma_ls_fit.py` (`ls_score`, `ls_information`): with `tau2 = Z alpha`, `W = diag(1 / (v + tau2))`,
      `P = W - W X (X' W X)^-1 X' W` and `e = P y`, the score is `-1/2 Z' (d - e^2)` and the information `1/2 Z' (M o M) Z`, where `d` and `M`
      are `diag(W)` and `W` under ML and `diag(P)` and `P` under REML (`o` the elementwise product). Cite Viechtbauer and Lopez-Lopez (2022,
      equations 7 to 9) and Harville (1977) in the docstring. Do not import or copy `src/tests/correctness/formulas.py`: it is the independent
      reference the cases check against
- [x] Add `polish_identity` to `src/metafopy/models/rma_ls_opt.py`: from the optimiser's point, Fisher scoring (Newton with the expected
      information) on the free coefficients, with an active set for the rows with `z_i' alpha = 0`: each step is projected onto the null space of
      the active rows, cut at the first row it would take below zero (that row joins the set), halved while the log-likelihood does not rise, and
      a row leaves the set when its KKT multiplier has the wrong sign. Stop when the projected score is below 1e-10 relative or after a fixed cap
      of iterations (a module constant). Return the start unchanged if no step raises the log-likelihood. Every returned point satisfies
      `Z alpha >= 0` exactly (the likelihood is `-inf` below it) and has a log-likelihood no lower than the start's
- [x] In `fit_rma_ls`, call `polish_identity` after the identity-link optimiser, for every identity-link optimiser (`constrOptim` and the
      stand-ins), with `alpha` entries fixed by `alpha_arg` kept fixed. With `optbeta=True` the parameter vector holds `beta` as well; leave those
      fits unpolished and say so in the docstring (no case or fixture fits one)
- [x] Update the docstrings of `rma_ls_opt.py` and `fit_rma_ls` to say that the identity-link optimum is polished, why (R's `constrOptim` stops at
      its own tolerance, about 1e-5 short), and that metafor does not polish; link the `docs/differences.md` subsection Task 3 adds
- [x] Flip the ten cases to `status = "pass"` and drop their `reason`; update the comment above them in `location_scale.toml` (and the one before
      the profile cases that says "the identity link stops short of the maximum") to say the fit is now polished
- [x] Add regression tests to `src/tests/test_rma_ls.py`: on `dat.bangertdrowns2004` (`mods = scale = ~ imag`, ML) the score at the fit is zero to
      1e-6 and the log-likelihood is -17.058054672888 to 1e-9; on `dat.bcg` (`mods = scale = ~ ablat`, ML) the fit satisfies `Z alpha >= 0`, its
      log-likelihood is at least R's -7.139288984088 plus 1e-7, and the projected score at the active set is zero to 1e-6. Run each against the
      unpolished code first (comment out the call) and confirm it fails
- [x] Run the tests that hold the identity-link R fixtures: `pytest src/tests/test_rma_ls.py src/tests/test_audit4_numerics.py
      src/tests/test_ls_vif_sim.py src/tests/test_audit5_numerics.py src/tests/test_audit5_inputs.py`. Values still within the tolerances these
      tests use need no change. For each value outside them, check that the fit has `link='identity'` and that metafopy's log-likelihood is at
      least R's (the fixture's `fit_ml` or `fit_reml`); mark it as differing as `agents/tasks/confint-ht-centre/tasks-1.md` Task 1 marks `ht_ci`:
      the JSON stays as R wrote it, the test compares that value with `correctness.formulas.ls_reference_fit` (as `src/tests/test_robust.py`
      imports `correctness`) at the tolerance the correctness cases use, asserts that R's value differs, and carries a comment naming the cases and
      the `docs/differences.md` subsection. A value outside tolerance on a log-link fit, or an identity-link fit whose log-likelihood fell, is a bug
      in this change: fix it, never mark it
- [x] Record in this task's Notes each fixture value marked, with R's, metafopy's before and after, and the reference's

**Acceptance Criteria:**

- [x] The ten cases pass with `status = "pass"`
- [x] No identity-link fit in the test suite ends with a lower log-likelihood than before, or with `Z alpha < 0`
- [x] Every R fixture value that no longer matches is compared with the reference, and no log-link fixture value changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_rma_ls.py src/tests/test_audit4_numerics.py src/tests/test_ls_vif_sim.py src/tests/test_audit5_numerics.py src/tests/test_audit5_inputs.py src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1 (its Notes hold the per-fit figures and may amend this task)
- Key files: `src/metafopy/models/rma_ls_fit.py`, `src/metafopy/models/rma_ls_opt.py`, `src/tests/test_rma_ls.py`,
  `src/tests/correctness/cases/location_scale.toml`, `src/tests/reference/phase5/location_scale.json` (unchanged)
- The polish is analytic Fisher scoring, not a second SciPy run: SLSQP, which the reference uses, ends slightly outside the region (the reference's
  smallest `tau2_i` is -1e-10), where `LSProblem.loglik` is `-inf`
- The polish runs whatever `control.polish_converged` says: that flag skips the Nelder-Mead polish of the BFGS path, and the identity polish costs a
  few analytic steps, so `permutest` refits and the observed fit reach the same kind of optimum
- The numerical Hessian is unchanged. At a boundary optimum its central differences step below `tau2_i = 0` and it fails as it does today on
  `dat.bcg` (R's fixture `bcg_year_identity` has `se_alpha` `null` too); `test_audit4_numerics.py` checks the Hessian of an interior fit whose
  scale slope is near zero, whose values move with the optimum
- `metafopy.methods.ls_profile`, `ls_confint`, `ls_permutest` and `ls_vif` refit through `fit_rma_ls`, so their identity-link refits are polished
  too; row 6 of the run order depends on this

**Findings (2026-11-16):**

- `polish_identity` also snaps active rows to the boundary (`snap_to_boundary`, slack 1e-13 relative): a row that starts 2e-10 inside the region otherwise
  costs the score (about 57 on bcg ML) times the slack, 1.3e-8 of log-likelihood. On bcg the polished ML log-likelihood is -7.139288823 (the feasible
  maximum) and the reference's is 5.7e-9 above it, from `IDENTITY_SLACK`; `ls_loglik_not_below_reference` in `correctness/test_properties.py` now
  allows 1e-9 of the log-likelihood's magnitude, the scale `ls_agree` uses.
- The R fixtures that differ, each from a fit with `link='identity'` (largest relative difference from R in the quantities of that fit, measured
  after the change; the fixtures are unchanged): `bd_int_identity` 6e-4, `bd_identity` 8e-4, `bd_identity_ml` 3e-4, `bd_identity_ni` 2e-2 (alpha, tau2
  and the I2 of studies at the boundary: R's barrier value 2e-8 against 0 here), `bcg_year_identity` 9e-4 (alpha 6.5e-4). `test_fit` allows
  `IDENTITY_FIT` (5e-2) for them, and the new `test_identity_fit_is_the_maximum` compares alpha with `correctness.formulas.ls_reference_fit` to 1e-6,
  asserts R's alpha differs, and that the log-likelihood is not below R's. `test_audit4_numerics.py` compares alpha with the reference and the
  standard errors with the analytic ones (1e-5), which now agree better than R's (R's slope was 9% off). `test_ls_vif_sim.py::test_sim_ml_identity`
  allows 1e-2 for the simulated VIFs of the permuted identity refits (measured 8e-3 at most). No log-link value changed.
- Tests run: the Definition of Done set, 1315 passed, 1 skipped, 6 xfailed (the six are selection-model expected failures, unrelated).

---

## Task 3: Record the difference in the audit ledger and the user guide

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (`known.toml`, `docs/differences.md`)

**Technical Goal:** `bin/metafor-audit.sh` accepts the new difference with its reason, the recall check still passes, and `docs/differences.md`
explains the difference under "Where metafor departs from its reference".

**Subtasks:**

- [x] Run the location-scale identity-link slice of the audit from the worktree, against the main checkout's R oracle, in the background:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --only 'construct/ls_*/link=*identity*' --only 'scripts/misc_rma_ls*'`
      (`$main` is the main checkout, the first entry of `git worktree list --porcelain`). If `check-r` reports that the R library is missing or
      differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [x] For each finding it reports on an identity-link fit (`alpha`, `tau2`, `beta` or a value that follows from them, `ll` and the fit statistics),
      add a `[[known]]` entry to `src/audit/data/known.toml` that matches exactly those case ids, kinds and paths, with a `reason` (R's
      `constrOptim` stops about 1e-5 short of the identity-link maximum; metafopy polishes it on the analytic score within `Z alpha >= 0`) and
      `ref = "docs/differences.md"`. Add no entry for a finding the run does not report, and no glob wider than the findings (the README warns
      that `*/ls_*` swallowed recall items before). Stop as `blocked`, naming it, for a finding on a log-link fit
- [x] Rerun the same command and check that its report lists no finding left from this change. A finding not in the newest earlier report under
      `$main/.audit/runs/` and not from this change stops the task as `blocked`, naming it
- [x] Run `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --recall` in the
      background and check that every item is still found, in particular `a4-ls-identity-se` (`src/audit/data/recall.toml`), whose finds are
      `construct/ls_bcg/link=*identity*` and `construct/ls_bangert/link=*identity*`. If a new entry hides it, narrow the entry (the harness, per
      `AGENTS.md`), never the recall item
- [x] Add a subsection `### Location-scale fits with link="identity"` to the section "Where metafor departs from its reference" of
      `docs/differences.md`, after the existing subsections: a short summary (R's `constrOptim` stops at its tolerance; metafopy ends at the
      maximum within `tau2_i >= 0`, which can lie on the boundary), the same call in R and in Python with the output of both (the ML fit of
      `dat.bcg` log risk ratios with `mods = scale = ~ ablat`: alpha and log-likelihood), and *Why:* naming Viechtbauer and Lopez-Lopez (2022) and
      the cases. Produce the R output with `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the R block>'` and the
      Python output with metafopy, and paste both as printed
- [x] Re-run every Python example of `docs/` that fits `link='identity'` (`grep -rn "link='identity'" docs/`; on 2026-11-15 the subsection
      "Standard errors of a location-scale model on a flat scale" of `docs/differences.md`) and update each printed output that changed, with a
      sentence where the explanation no longer holds

**Acceptance Criteria:**

- [x] The slice reports no finding from this change, and every new `known.toml` entry matches at least one case of that run
- [x] `--recall` finds every item
- [x] `docs/differences.md` names the difference under "Where metafor departs from its reference", and every printed identity-link output in
      `docs/` is what metafopy prints now

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_docs_examples.py src/tests/test_harness_accounts.py` passes

**Notes:**

- Depends on Task 2
- Prerequisite outside the worktree: R metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib`. The audit's work
  directory `.audit/` is ignored by git and exists only in the main checkout, hence both variables
- The audit has no identity-link fixture: the link reaches the location-scale fixtures through the `construct` step (`link` in
  `src/audit/data/domains.toml`), as `construct/ls_<fixture>/link=...`, and metafor's own tests through `scripts/misc_rma_ls*`. A full audit run
  is the user's call; a difference the slice does not reach surfaces in the next full run
- `src/tests/test_docs_examples.py` runs the Python blocks of `docs/` but does not compare their printed output, hence the re-run subtask

**Findings (2026-11-16):**

- The slice reported 31 findings, all on identity-link fits: 18 on `construct/ls_bangert/link="identity"` and 13 in nine units of `scripts/misc_rma_ls`
  (confint, profile and predict of identity-link fits: units 21, 42, 43, 65, 66, 67, 80, 82, 175). `ls_bcg` identity gave none. Two `known.toml` entries
  cover them; the rerun of the slice lists none.
- `--recall` reproduced 176 of 176 items, `a4-ls-identity-se` included; one item (`a6-glmm-um-rs`) is reported as "oracle moved", unrelated.
- `se_alpha` of the flat-scale example in `docs/differences.md` moved from 0.53962141 to 0.53968806 with the polish; that section and its copy in
  `docs/differences-notes.md` are updated.

---

## Task 4: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/ls-identity-polish/tasks-1.md` with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (`$main` is the main checkout,
      `/Users/simonb/src/metafopy`, the first entry of `git worktree list --porcelain`):
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/ls-identity-polish/tasks-1" && source bin/activate.sh && pytest src/tests 2>&1 | tee "$main/ralpher-logs/ls-identity-polish/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes, `slow` tests included, judged by its exit status
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-identity-polish/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository; the baseline is `TODO.md` as it stood at the commit that adds this list, before any of its tasks)

**Notes:**

- Depends on Task 3
- `pytest src/tests` is the run order's requirement for this list's last task, wider than the project's usual scoped run
- No version bump: the change is a numerical fix with no new or changed public API

---
