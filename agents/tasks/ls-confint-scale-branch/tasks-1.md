# Tasks: which branch is the profile maximum for ls.confint.scale.ML.bcg

**Source:** TODO.md, "ls.confint.scale.ML.bcg stays an expected failure with no decision"
**Created:** 2026-11-17
**Version:** 1

The item comes from the closing notes of `agents/tasks/ls-profile-crossings/tasks-1.md` (Task 3), which left the case `ls.confint.scale.ML.bcg` in
`src/tests/correctness/cases/location_scale.toml` as `status = "mismatch"`: for the upper bound of `alpha.2` (the `ablat` coefficient of the scale model
of the ML fit of `dat.bcg` with `mods = scale = ~ ablat`, `measure = "RR"`, log link) the reference `formulas.ls_profile_ll` in
`src/tests/correctness/formulas.py` and metafopy's `profile_fit` in `src/metafopy/methods/ls_profile.py` end on different branches of the profile. On
one branch `tau2_i = exp(Z alpha)` goes to zero (the intercept `alpha.1` runs to minus infinity and `2 (ll_max - ll)` flattens at a plateau); on the
other `alpha.1` stays finite (the continuous branch). The profile is the maximum over the other coefficients (section 2.5.3 of Viechtbauer and
Lopez-Lopez, 2022, doi:10.1002/jrsm.1562), so the branch with the higher log-likelihood is the right one.

The approved plan (2026-11-17): find which branch is the maximum at a = 1.3169 by refitting over a grid of starts in metafopy, in the reference and in
R metafor 5.2-1, and compare log-likelihoods. If the continuous branch is higher, the reference refit is wrong: fix `formulas.ls_profile_ll` and set the
case to `pass` (or record its new status). If the `tau2 = 0` branch is higher, metafopy's profile misses it, which would change metafopy's output: the
list stops for the operator with the figures, changes nothing under `src/metafopy/`, and files nothing else.

<!-- doc-counts: accept-start probed on 2026-11-17 - a dated record, not a claim about the present -->
Probed on 2026-11-17 (metafopy at `c75f3d0`, R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`), `2 (ll_max - ll)` with `ll_max`
-7.539290250091879, cut-off 3.841458820694124 (chi-square 1 df, 95%):

| a (`alpha.2`) | grid of starts (best) | R from `alpha.init = c(-72, a)` | R default start | reference `ls_profile_ll` | metafopy `profile_fit` |
|---|---|---|---|---|---|
| 1.3169166758 | 3.685006 (`alpha.1` -72.019) | 3.685006 | 3.868641 (plateau) | 3.685006 | 3.685006 |
| 1.3170 | | | | 3.868641 (plateau) | 3.685207 |
| 1.3598 / 1.36 | 3.788971 | 3.789459 (1.36) | 3.868641 | 3.868641 | 3.788971 |
| 1.38 | 3.838488 | 3.838488 | 3.868641 | 3.868641 | 3.868641 (plateau) |
| 1.382 | 3.843409 | | | 3.868641 | 3.868641 |
| 1.40 | 3.868641 (continuous 3.887835 is lower) | 3.887835 | 3.868641 | 3.868641 | 3.868641 |

So the continuous branch is the maximum at a = 1.3169 (higher log-likelihood than the plateau's 3.868641), and the reference is wrong: its two starts
jump to the plateau just above 1.31692, and its `brentq` converges on that jump (1.316916702213052), not on a crossing. The probe also found a second
difference the item did not name: metafopy's own refit jumps to the plateau between 1.36 and 1.38, while the continuous branch stays below the cut-off
until it crosses it at about 1.3812 (between 1.380 and 1.382). metafopy's `confint(res, alpha=1, tol=1e-12)` now gives the upper bound 1.3598 with no
`>` sign (the case's `reason` still says `>`), which is the jump in its own profile, not a crossing. R gives `>0.4245`, its interval capped at ten times
the estimate (the accepted difference in `docs/differences.md`, "Profile-likelihood bounds of the scale coefficients"), and from its default start its
refits sit on the plateau.

The expected path is therefore Task 2's first branch (fix the reference), after which the reference's bound is about 1.3812 and metafopy's 1.3598, so
the case cannot become `pass` without changing metafopy's output: Task 2 then records the new status and stops for the operator (see Task 2).
<!-- doc-counts: accept-end -->

The project has no affected-tests tool, so each task names its test files. Probe scripts stay in the session's scratch directory, outside the repository.

---

## Task 1: Find which branch is the profile maximum

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (refit over a grid of starts, compare both log-likelihoods)

**Technical Goal:** which branch has the higher log-likelihood at a = 1.3169166758 is known from three programs, and so is where the maximum profile
crosses the cut-off, recorded in this task's Notes with the figures.

**Subtasks:**

- [ ] Write a Python probe in the session's scratch directory that, with `src/tests` on `sys.path`, loads the case from
      `correctness.cases.load_cases()`, fits it with `correctness.cases.run_case`, builds `yi, vi, X, Z` with `correctness.test_derived.ls_inputs`,
      and for `a` in 1.3169166758, 1.3170, 1.36, 1.37, 1.38, 1.39, 1.40 prints `2 (ll_max - ll)` from: (a) a grid of starts, `alpha.1` from -130 to 10
      in steps of 5, each minimised by Nelder-Mead over `alpha.1` with `alpha.2 = a` on `formulas.ls_loglik`, keeping the best; (b)
      `formulas.ls_profile_ll(..., j=1, value=a, start=fit['alpha'])` with `fit = formulas.ls_reference_fit(...)`; (c)
      `metafopy.methods.ls_profile.profile_fit(res, 1, a)`. Print the `alpha.1` of each grid optimum, so the branch is visible
- [ ] Find the crossing of the grid-of-starts profile with the cut-off `scipy.stats.chi2.ppf(0.95, 1)` by `brentq` on a bracket where it changes sign
      (the probe found 1.380 below and 1.382 above), and print `formulas.ls_profile_bounds(..., 1, 0.95)` and `metafopy.confint(res, alpha=1, tol=1e-12)`
      with its `ub_sign`
- [ ] Write an R script in the scratch directory that loads metafor, prints `packageVersion("metafor")`, fits
      `rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg),
      method = "ML")` and, for the same values of `a`, refits with `alpha = c(NA, a)` from the default start and from
      `control = list(alpha.init = c(a0, a))` for `a0` in -100, -72, -60, printing `2 (logLik(res) - logLik(fit))` and `fit$alpha[1]` with
      `sprintf("%.6f", ...)`. Run it with the main checkout's R library (the main checkout is the first entry of `git worktree list --porcelain`):
      `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`. Stop as
      `blocked` if metafor does not load or its version is not 5.2.1
- [ ] Record in this task's Notes, dated, with the commit: the table of the three programs' values, the crossing found in the second subtask, both
      programs' bounds, and one verdict at a = 1.3169166758 - "the continuous branch is higher (the reference is wrong)" or "the `tau2 = 0` branch is
      higher (metafopy misses it)". Judge it by the grid of starts, confirmed by R's best refit (agreement within 1e-5 in `2 (ll_max - ll)`); if the two
      disagree on which branch is higher, stop as `blocked` with both figures. Also record whether metafopy's own profile leaves the maximum branch
      below the crossing (the header's probe says it does between 1.36 and 1.38)

**Acceptance Criteria:**

- [ ] This task's Notes record the values of the grid of starts, the reference, metafopy and R at each `a`, the crossing of the maximum profile, and one
      of the two verdicts

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes; no code or test changes
- The header's 2026-11-17 probe expects "the continuous branch is higher (the reference is wrong)" and a crossing at about 1.3812
- Key files: `src/tests/correctness/formulas.py` (`ls_loglik`, `ls_profile_ll`, `ls_profile_bounds`, `ls_reference_fit`),
  `src/metafopy/methods/ls_profile.py` (`profile_starts`, `profile_fit`), `src/metafopy/methods/ls_confint.py` (`search_limits`, `profile_lrt`)

---

## Task 2: Fix the reference refit, or stop with the figures

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (fix `formulas.ls_profile_ll` and set the case to `pass`, or metafopy misses the branch)

**Technical Goal:** the reference profile follows the branch of highest log-likelihood, and the case's `status` and `reason` say what the two programs
now give; or, if metafopy is the one that misses the maximum at a = 1.3169, the list stops for the operator with Task 1's figures.

**Subtasks:**

- [ ] Read Task 1's verdict. If it is missing, stop as `blocked`
- [ ] **If the `tau2 = 0` branch is higher (metafopy misses it):** stops for the operator: metafopy's profile of `alpha.2` misses the higher
      `tau2 = 0` branch at a = 1.3169166758 of `ls.confint.scale.ML.bcg`, and fixing it would change metafopy's `confint` and `profile` output. Put Task
      1's figures in the stop message. Change nothing under `src/metafopy/` or `src/tests/`, file nothing in `TODO.md`, and mark the other branch's
      subtasks ("In that branch") not applicable
- [ ] **If the continuous branch is higher (the reference is wrong):** change `formulas.ls_profile_ll` in `src/tests/correctness/formulas.py` so it
      keeps the best of a wider set of starts: besides `adjusted` and `start[free]`, starts along the line from `start[free]` to `adjusted` and a grid
      over the free coefficients wide enough to reach the continuous branch (Task 1's `alpha.1` optima give the range), keeping the minimum of all
      refits as now. Update its docstring to say why (the profile of `alpha.2` has a `tau2 = 0` plateau that two starts can fall onto while the
      continuous branch is higher). Keep the function's signature
- [ ] In that branch, check the fixed reference against Task 1's grid of starts: at each `a` of Task 1 its `2 (ll_max - ll)` agrees within 1e-5, and
      `formulas.ls_profile_bounds(..., 1, 0.95)` gives an upper bound within 1e-5 of Task 1's crossing. If it does not, widen the starts; if it still
      does not, stop as `blocked` with the figures
- [ ] In that branch, run `pytest src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_derived.py -k "ls"` and check that no other
      location-scale case changes outcome; a case that changes stops the task as `blocked`, naming it and its old and new values
- [ ] In that branch, run the case (`pytest src/tests/correctness/test_derived.py -k "ls.confint.scale.ML.bcg" -rX -rx`). If metafopy's bound now
      agrees with the reference within `LS_PROFILE` (`src/tests/correctness/test_derived.py`), remove the case's `status` and `reason` so it runs as a
      passing case, and mark the next subtask not applicable
- [ ] In that branch, if metafopy's bound still differs (the header's probe expects metafopy's 1.3598 against a reference near 1.3812): keep
      `status = "mismatch"` and rewrite the case's `reason` with the new figures (the reference's bound, metafopy's bound and sign, and that metafopy's
      `profile_fit` leaves the continuous branch for the plateau between the values Task 1 found, so its root search converges on that jump). Commit
      that with the reference fix, then stops for the operator: metafopy's profile of `alpha.2` for `ls.confint.scale.ML.bcg` leaves the branch of
      highest log-likelihood before the crossing (Task 1's figures), so its upper bound is a jump, not a crossing; fixing `profile_starts` or
      `profile_fit` in `src/metafopy/methods/ls_profile.py` would change metafopy's `confint` and `profile` output. Change nothing under `src/metafopy/`
      and file nothing in `TODO.md`; the operator decides whether that fix joins this list or becomes its own item

**Acceptance Criteria:**

- [ ] The `tau2 = 0` branch is higher: the list has stopped for the operator with Task 1's figures, and no file under `src/` changed
- [ ] The continuous branch is higher: `formulas.ls_profile_ll` matches Task 1's grid of starts within 1e-5 at each `a` and its bound is Task 1's
      crossing; no other location-scale case changes outcome; and the case either passes or is a `mismatch` whose `reason` gives the new figures, with
      the list stopped for the operator over metafopy's own jump

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_derived.py` passes (expected failures as recorded)

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/formulas.py` (`ls_profile_ll`, `ls_profile_bounds`), `src/tests/correctness/cases/location_scale.toml`
  (`ls.confint.scale.ML.bcg`), `src/tests/correctness/test_derived.py` (`ls_confint_scale`, `LS_PROFILE`), `src/tests/correctness/test_formulas_ls.py`
- `ls_profile_ll` is also used by `ls_profile_bounds` for every `ls.confint.scale.*` case and by the profile checks, hence the run over every
  location-scale case
- Nothing under `src/metafopy/` changes in either branch; the stops name what would

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/ls-confint-scale-branch/tasks-1.md; <outcome>` with the date of the commit and the outcome in a few words (which
      branch was the maximum, and the case's new status)
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of `git worktree list
      --porcelain` (`/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/ls-confint-scale-branch/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/ls-confint-scale-branch/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and its outcome, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes, judged by its exit status
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-confint-scale-branch/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2, and runs only after any stop of Task 2 has been cleared by the operator
- No version bump
- The full `pytest src/tests` run is the run order's required final check

---
