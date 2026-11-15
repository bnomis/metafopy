# Tasks: truncest piecewise search

**Source:** TODO.md, "The truncest estimate stops at a local maximum of a step-function likelihood"
**Created:** 2026-11-15
**Version:** 1

The user answered *follow the reference* on 2026-11-15 (the case `selmodel.truncest.bangertdrowns2004` in
`src/tests/correctness/cases/selmodel.toml`, and `agents/specs/correctness-publication-bias/mismatches.md`), so this list changes metafopy's
output from R metafor 5.2-1 without a new review, within what that decision names. It is row 3 of `agents/tasks/order-numerical-fixes.md`; the
selmodel `stepfun` with `prec` list (row 7) waits for it because both edit selmodel code.

**Measured on 2026-11-15** (metafopy at `89ec047`, `rma(yi, vi, method='ML')` on `dat.bangertdrowns2004`, 48 studies), on a scratch probe that
maximises `formulas.truncation_loglik` over each stretch between observed effects (Nelder-Mead, the cut mapped logistically into the open stretch)
and with `selmodel(res, type='truncest', delta=[nan, cut])` at fixed cuts:

| Fit | truncation point | delta | beta | tau2 | ll |
|-----|------------------|-------|------|------|----|
| metafopy and R today | 0.27073361 | 1.42227 | 0.26491 | 0.054216 | -17.7909 |
| metafopy, cut fixed at 0.27073361 (delta refitted) | 0.27073361 | 1.69309 | 0.29265 | 0.057497 | -17.7302 |
| the item's "maximum" (SLSQP from the start values) | 0.20000565 | 1.48969 | 0.27598 | 0.054737 | -17.7406 |
| metafopy, cut fixed at 0.20000565 | 0.20000565 | 1.65221 | 0.28570 | 0.051674 | -17.7012 |
| best stretch, (0.42, 0.49) | 0.49 from below | 0.12748 | 0.09610 | 0.0096689 | -10.6519 |
| metafopy, cut fixed at 0.48999999 | 0.48999999 | 0.12748 | 0.09610 | 0.0096689 | -10.6519 |
| metafopy, cut fixed at 0.49 (the study at 0.49 is no longer above the cut) | 0.49 | 0.15988 | 0.10576 | 0.011325 | -12.5993 |

So the item's reference values are not the maximum either: the likelihood keeps rising towards the right-hand end of the stretch (0.42, 0.49)
and jumps down at 0.49, because a study counts as selected only when `y > cut` (metafor's `.selmodel.ll.trunc`, and `formulas.truncation_loglik`).
The supremum is not attained: there is no maximum-likelihood estimate of the truncation point on these data, only a limit at an observed effect,
and the estimates it gives (beta 0.096) are far from those the user saw when answering (beta 0.276). How to report an estimate that is not
attained is outside the decision, so Task 1 stops for the user (see the skill rule that a decision covers only what it names). If the user
answers before the run starts, Task 1 finds the answers and goes straight on.

**How the stretch search sits beside the existing optimiser.** The joint fit with `optimizer` / `optmethod` and `optcontrol` (default `optim`
BFGS with a finite-difference gradient, `src/metafopy/models/sel_opt.py`) is kept and runs first, as metafor's. The stretch search runs after it,
only for `type='truncest'` with the truncation point free (`delta[1]` NaN): for each stretch between consecutive distinct observed effects that
meets `[delta.min[1], delta.max[1]]` (clipped to them), it runs the same optimiser with the same `optcontrol`, with the cut mapped into that
stretch by metafor's default logistic `.mapfun` (`sel_lik.map_delta` with `fun=None` and the stretch ends as bounds), starting from the joint
fit's beta, tau2 and `delta[0]` and from the stretch midpoint (or `delta.init[1]` in the stretch that holds it). The fit with the highest ll
wins, the joint fit included, so a fit never gets worse than today's. A fixed truncation point (`delta=[nan, c]`, as in the parity fixture
`hack_truncest`) runs no search, so its output does not change. The fit under `tau2 = 0` for `LRT_tau2` runs the same search, so the test
compares two maxima. `retopt=True` returns the winning parameters on the joint fit's scale (the cut as itself, `mapfun[1] = 'I'`). The Hessian
is taken at the winner as today, on the piece of the likelihood that holds it. `profile()` and `confint()` refit through `selmodel()`, so they
inherit the search; Task 2 measures the run time.

The project has no affected-tests tool, so each task names its test files, and the last task runs the full suite as the run order requires.

---

## Task 1: Record the stretch maxima and stop for the user

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction ("search each stretch between observed effects and keep the best"), checked against the data

**Technical Goal:** the user decides how metafopy reports a truncation point whose likelihood has a supremum at an observed effect and no
maximum, before any code changes.

**Subtasks:**

- [x] Re-check the starting point: `selmodel.truncest.bangertdrowns2004` in `src/tests/correctness/cases/selmodel.toml` has `status = "mismatch"`,
      `pytest src/tests/correctness/test_derived.py -k truncest` passes with it as a strict expected failure, and the `type_ == 'truncest'` branch
      of `src/metafopy/models/sel_spec.py` sets `delta_min[1]` and `delta_max[1]` from `yi.min() - sd` and `yi.max()`. Stop as `blocked`, naming
      what differs, if not
- [x] If `agents/tasks/truncest-piecewise-search/review-1.md` exists and its Answers section names an option with a date, mark this task done
      and go to Task 2
- [x] Repeat the header's measurement with a scratch script outside the repository (session scratch directory): for each stretch, the best ll
      of `formulas.truncation_loglik` with the cut in the open stretch, and metafopy's `selmodel(res, type='truncest', delta=[nan, c])` at the
      winning stretch's right end minus 1e-8 and at the end itself. Stop as `blocked` if the winner differs from the header's table by more than
      1e-4 in ll, naming both
- [x] Write `agents/tasks/truncest-piecewise-search/review-1.md` in the layout of the shared `reviews.md` rule: what is reviewed (the table above,
      remeasured); the automatic check (the measurement reproduced); the decision "how to report a truncation point whose supremum is not
      attained", with its background (metafor and the reference count a study as selected only when `y > cut`; the item's reference values are
      a local maximum; beta moves from 0.265 to 0.096 on this dataset), what it affects (`selmodel(type='truncest')` estimates, the correctness
      case, the audit's `type="truncest"` findings, `docs/differences.md`), and these options with their consequences:
      - **B (recommended): the supremum, one step inside the stretch.** When a stretch's best cut lies within `1e-6 * max(1, |end|)` of its
        open end, refit with the cut fixed at `end - 1e-8 * max(1, |end|)`, report that value, warn that the truncation point sits at an
        observed effect and its likelihood has no maximum there, and leave the cut out of the Hessian (its SE, z, p and Wald interval NaN).
        The reported ll matches `formulas.truncation_loglik` at the reported estimates, and the reference is followed as answered.
        Recommended because it is the answer the user gave applied to the real maximum, and the warning says what the number is
      - **A: count a study at the cut as selected (`y >= cut`) for `truncest`.** The supremum becomes a maximum at the observed effect, with no
        warning. Departs from metafor's likelihood and from `formulas.truncation_loglik` (which the case would then follow), and makes
        `truncest` and `trunc` disagree at a cut equal to an observed effect
      - **C: keep metafor's search.** No output change; the case is rewritten to record that the maximum-likelihood truncation point need not
        exist, stays a documented mismatch, and the item moves to `DONE.md` as not fixed. Tasks 2 to 4 then reduce to that record
- [x] Publish the review as an interactive page as `reviews.md` says, if this session has the Artifact tool; put `**Review page:** <URL>` in the
      document's header. Without the tool, write `**Review page:** to be published by the operator session` there
- [x] Stop the list as `blocked: awaiting answers in agents/tasks/truncest-piecewise-search/review-1.md`. Once the decision has an answer, an
      operator session publishes the page if it is missing, reads the page's answers, copies each into the review document's Answers section
      with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [x] `review-1.md` holds the remeasured table, the decision with options A, B and C and their consequences, and an empty Answers section
- [x] No file under `src/` has changed

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness/test_derived.py -k truncest` passes

**Notes:**

- Closed without a stop: `review-1.md` already holds the answer, option B, dated 2026-11-15, so the remeasurement, the review document and the page were not repeated

- Operator stop: this task is the list's one stop for the operator, for the decision above. It is a no-op if the answer is already in
  `review-1.md` when the list starts
- Key files: `src/metafopy/models/selmodel.py` (the fit, `type_ == 'truncest'`), `src/metafopy/models/sel_spec.py` (bounds and start values),
  `src/metafopy/models/sel_lik.py` (`ll_trunc`, `map_delta`), `src/tests/correctness/formulas.py` (`truncation_loglik`)

---

## Task 2: Search each stretch of the truncation point

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (the code change, the correctness case), as answered in `review-1.md`

**Technical Goal:** `selmodel(type='truncest')` returns the best fit over every stretch between observed effects, reported as the answered
option says, and the correctness case checks it against a reference that searches the stretches too.

**Subtasks:**

- [x] Read the answer from the Answers section of `agents/tasks/truncest-piecewise-search/review-1.md`, never from the page. Stop as `blocked`,
      naming what is missing, if the section is empty or names no option
- [x] Confirm the case fails today: set `selmodel.truncest.bangertdrowns2004` to `status = "pass"` on a scratch edit and run
      `pytest src/tests/correctness/test_derived.py -k truncest`; it must fail on beta, delta or ll. Restore the file
- [x] Make the reference search the stretches: in `selmodel_maximum` of `src/tests/correctness/test_derived.py`, for `kind == 'truncest'` with
      the cut free, run the bounded SLSQP once per stretch between consecutive distinct `yi` inside `[delta_min[1], delta_max[1]]` (bounds the
      stretch ends, the right end less `1e-8 * max(1, |end|)` under option B, the closed stretch under option A, using the `y >= cut` likelihood
      that option A adds to `formulas.truncation_loglik` as an argument) and keep the best. Under option C skip this subtask
- [x] In `src/metafopy/models/selmodel.py`, add the stretch search after the joint fit as the header describes (a module-level function, the
      joint fit kept as a candidate, the same `choice` and `optcontrol`, `delta.min[1]`/`delta.max[1]` clipping the stretches, `delta.init[1]`
      as the start in its stretch), apply it to the `tau2 = 0` refit for `LRT_tau2` as well, return the winner from `retopt=True` on the joint
      fit's scale, and report the cut as the answered option says (option B: the step inside, the warning, the cut left out of the Hessian;
      option A: `y >= cut` in `sel_lik.ll_trunc` for `truncest` only). Under option C make no code change
- [x] Update `selmodel`'s docstring for `type='truncest'`: the stretch search, how a supremum at an observed effect is reported, that metafor
      stops at a local maximum, and a link to the `docs/differences.md` subsection Task 3 adds. Link the R source of `.selmodel.ll.trunc`
- [x] Set `selmodel.truncest.bangertdrowns2004` to `status = "pass"` and drop its `reason` (option C: rewrite the `reason` as the answer says and
      keep `status = "mismatch"`)
- [x] Add regression tests to `src/tests/test_selmodel.py`: on `dat.bangertdrowns2004` the `truncest` fit's ll is the best stretch's (-10.6519 in
      the header, to 1e-4) and no fixed cut on a grid of each stretch's midpoints gives a higher ll; a fixed truncation point
      (`delta=[nan, 0.1]` on the `hack` data, the fixture `hack_truncest`) gives the same fit as before; `delta.max=[100, 0.3]` restricts the
      search to stretches below 0.3; the warning of option B is issued. Time one `truncest` fit and one `confint()` of it; mark a test that takes
      more than 10 seconds `@pytest.mark.slow`, and record both times in this task's Notes
- [x] Run `pytest src/tests/test_selmodel.py`. The parity fixture `src/tests/reference/phase5/selmodel.json` holds one `truncest` fit,
      `hack_truncest`, with the truncation point fixed (`delta_fix` `[false, true]`, checked 2026-11-15), so it should pass unchanged. If a
      `truncest` value of a fit with a free cut fails, keep the JSON as R wrote it (never regenerate it) and mark that value as differing in
      the way `test_confint_types` of `src/tests/test_rma_uni_help.py` marks `ht_ci`: compare it with the reference and assert that R's
      differs, with a comment naming the correctness case and the `docs/differences.md` subsection. A failure of a fit with a fixed cut, or
      of any other type, is a bug in this change: fix it, never mark it

**Acceptance Criteria:**

- [x] `selmodel.truncest.bangertdrowns2004` passes with `status = "pass"` (option C: stays a strict expected failure with the new reason)
- [x] Every `selmodel` parity test of a type other than `truncest`, and `hack_truncest`, passes unchanged

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_selmodel.py src/tests/test_stepcon.py src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py`
      passes

**Notes:**

- Depends on Task 1
- Option B (answered 2026-11-15). Timings on 2026-11-15: one `truncest` fit of `dat.bangertdrowns2004` takes 0.75 s, one `confint()` of it 8.1 s (under 10 s, so no test is marked `slow`). The `hack_truncest` parity fixture passes unchanged.
- The reference refits the near-end point in each stretch whose interior cut lies within 1% of an observed open end, instead of testing a 1e-6 threshold, because the optimiser stops short of the end
- Keep the change local: the stretch search lives in `src/metafopy/models/selmodel.py`, and `sel_lik.py` changes only under option A. Check
  the new function's name against the package first (`grep -rnw '<name>' src/metafopy`)
- The existing BFGS joint fit stops short even inside its own piece (cut fixed at its own 0.27073361 gives ll -17.7302 against -17.7909), so the
  refit inside each stretch is needed even where the joint fit's piece wins
- `src/metafopy/models/selmodel.py` has `src/tests/test_selmodel.py` as its own tests

---

## Task 3: Record the difference in the audit ledger and the user guide

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (`known.toml`, `docs/differences.md`)

**Technical Goal:** `bin/metafor-audit.sh` accepts the new difference with its reason, `--recall` still reproduces the earlier audits, and
`docs/differences.md` explains the difference under "Where metafor departs from its reference".

**Subtasks:**

- [x] Under option C, add no `known.toml` entry and no `docs/differences.md` subsection; run only the `--recall` subtask below
- [x] Run the audit slice for `truncest` from the worktree, against the main checkout's R library, in the background:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); METAFOPY_AUDIT_WORK="$main/.audit" METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --only 'options/selmodel/*/type="truncest"' --steps options`
      and, the same way, `--only scripts/selmodel/ --only scripts/misc_selmodel/ --only scripts/plot.rma.uni.selmodel/ --steps scripts` (the main
      checkout is the first entry of `git worktree list --porcelain`). If `check-r` reports that the R library is missing or differs from
      `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [x] For each finding on a `truncest` fit with a free truncation point (the `options/selmodel/<fixture>/type="truncest"` cases; the script
      units of metafor's tests and examples that call `selmodel(..., type = "truncest")` without fixing `delta[2]`), add a `[[known]]` entry to
      `src/audit/data/known.toml` that matches exactly those case ids, with `kind`, `path`, a `reason` (metafor's optimiser stops inside one
      piece of a likelihood that is a step function of the truncation point; metafopy searches each stretch between observed effects) and
      `ref = "docs/differences.md"`. Add no entry for a finding the run does not report: an entry that matches nothing is reported as stale.
      Check the existing `truncest` entries (the `optimizer = "mads"` entry on `scripts/misc_selmodel/*` and the flat-likelihood entry naming
      `truncest`); narrow or remove one the new entries make stale, and say so in this task's Notes
- [x] Rerun the same commands and check that the reports list no `truncest` finding. A finding on another type, or one that is not in the
      newest earlier report under `$main/.audit/runs/`, is new with this list: stop as `blocked`, naming it
- [x] Run `main=...; METAFOPY_AUDIT_WORK="$main/.audit" METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --recall`
      in the background and check that it reproduces every item it reproduced before. A miss is fixed in the harness (`src/audit/data/`), not
      in a report; stop as `blocked` if the fix is not a ledger entry this task added
- [x] Add a subsection `### selmodel(type="truncest"): the truncation point` to the section "Where metafor departs from its reference" of
      `docs/differences.md`, in the page's layout: a short summary, the same call in R and in Python with the output of both (the `ML` fit of
      `dat.bangertdrowns2004`), the name of the correctness case, and *Why:* the likelihood is a step function of the truncation point with a
      jump at each observed effect, and how metafopy reports the answered option. Produce the R output with
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the R block>'` and the Python output with metafopy, and paste
      both as printed

**Acceptance Criteria:**

- [x] The audit slices report no `truncest` finding, and every new `known.toml` entry matches at least one case of those runs
- [x] `--recall` reproduces every item it reproduced before this list
- [x] `docs/differences.md` names the difference under "Where metafor departs from its reference" (not under option C)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_docs_examples.py src/tests/test_harness_accounts.py` passes

**Notes:**

- Depends on Task 2
- Done 2026-11-15. The first options slice showed one new warning on `options/selmodel/sel_bcg/type="truncest"` ("invalid value encountered in
  reduce"): the stretch search tries extreme cuts, where `ll_trunc` sums a NaN term outside its `errstate`. Fixed in the search's objective
  in `src/metafopy/models/selmodel.py` (`numpy.errstate(all='ignore')` around the likelihood in `refit`), so `sel_lik.py` is unchanged; the
  joint fit is as before. Both slices were rerun after it.
- Ledger: four `[[known]]` entries at the end of `src/audit/data/known.toml` for the nine `options/selmodel/*/type="truncest"` cases (value-diff,
  print-diff, and two warning-extra: delta at a bound, and the truncation point at an observed effect) and two for
  `scripts/misc_selmodel/test/127` (value-diff, warning-extra). No existing `truncest` entry became stale (the slices and
  `test_harness_accounts.py` report none). The script slice needs `--only scripts/selmodel` (a substring: the `scripts/` prefix is stripped and
  `*` removed), not the trailing-slash form of the subtask, which matched no unit.
- Worktrees have no `.venv`; a symlink `.venv` to the main checkout's venv (untracked, not committed) let `bin/metafor-audit.sh` run.
- `--recall` reproduces 176 of 176 items, with the same one item the pinned R does not show (`a6-glmm-um-rs`) as in the earlier recall runs.
- Prerequisite outside this repository: R metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib` (worktrees have no
  `.audit/Rlib`, hence `METAFOPY_AUDIT_RLIB` on every call)
- The audit's `truncest` cases, seen in `.audit/runs/20261102-223017/findings.json`: `options/selmodel/<fixture>/type="truncest"` on `sel_bcg`,
  `trimfill_bcg`, `uni_counts`, `uni_factor`, `uni_mods`, `uni_mods_na`, `uni_mods_year`, `uni_na` and `uni_reml`, and units of
  `scripts/misc_selmodel/test/*`. A full audit run is the user's call; a difference the slices do not reach surfaces in the next full run
- `src/tests/test_docs_examples.py` runs every Python block of `docs/`, so the new example is checked

---

## Task 4: The full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/truncest-piecewise-search/tasks-1.md` with the date of the commit (under option C, add `(not fixed: the
      answer in agents/tasks/truncest-piecewise-search/review-1.md)` to that line)
- [x] Run the full test suite from the worktree and save its whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`):
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/truncest-piecewise-search/tasks-1" && source bin/activate.sh && pytest src/tests 2>&1 | tee "$main/ralpher-logs/truncest-piecewise-search/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [x] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The full test run above passes, `slow` tests included, and its log is saved
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/truncest-piecewise-search/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the
      session's scratch directory, outside the repository; the baseline is `TODO.md` as it stood when the list started)

**Notes:**

- Depends on Task 3
- No version bump: the project's rules (`AGENTS.md`, `claude/rules/metafopy.md`) require none for a fix

---
