# Tasks: Location-scale fits stable across platforms

**Spec:** `agents/specs/location-scale-fits-stable-across-platforms/spec.md`
**Created:** 2026-11-24
**Version:** 1

The spec's decisions were answered on 2026-11-24 (`agents/reviews/location-scale-fits-stable-across-platforms-spec.md`): the spec approved,
which confirms the library output changes; D1 refuse a leverage-1 study with R's error; D2 the capped Newton polish `polish_log`; D3 the exact
permutation cases check the widened bounds only.

`agents/specs/location-scale-fits-stable-across-platforms/measurements.md` holds the dated CI, local, R and probe figures; tasks add to it, never
rewrite it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript`.
- **Reproduce before fixing:** each case's local reproduction (spec, Implementation) is written as a test first and run against the code before
  the change; it must fail there and pass after. CI is the only place the original failures showed, so this is the evidence the fix works.

Stop the task as `blocked` (spec, Error handling) when any output outside the case's own moves past its committed tolerance (an R parity fixture,
a correctness case other than the four, a printed example), or when the case 4 counts differ from `measurements.md`. Widen no tolerance.

The recall check (`bin/metafor-audit.sh --recall`) is not run: it audits metafopy as of each earlier audit's commit (`--metafopy-src`), so a
library change here cannot change its result.

---

## Task 1: File the local-maximum finding

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Decisions, D3 (the finding filed separately)

**Technical Goal:** the finding that on some permutations of `ls.permutest.location.exact.ML.z.bcglat` metafopy and R stop at a lower local
maximum than the test's reference is recorded before the list changes the code that shows it.

**Subtasks:**

- [ ] Read the finding in `measurements.md` (permutation 68: metafopy and R log-likelihood `-12.58248229`, the reference `-12.57323737`; 8
      regular permutations in all)
- [ ] File it in `TODO.md` with the `/todo-add` shape: dated finding, cause marked unconfirmed, fix direction (decide which maximum metafopy
      reports; compare with R), placed by the file's order

**Acceptance Criteria:**

- [ ] `TODO.md` has the item, dated 2026-11-24, naming the case, the permutations and the values

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

---

## Task 2: Case 1, a study of leverage 1

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`LEVERAGE_ONE`, `marginal_residuals`); Implementation, Case 1; decision D1

**Technical Goal:** a study with `1 - h_ii < LEVERAGE_ONE` has a zero residual in the scale start, so the `differences.md` example raises R's
error on every platform.

**Subtasks:**

- [ ] Write the test in `src/tests/test_rma_ls.py` as the spec gives it (unperturbed and nine perturbed inputs, `numpy.random.default_rng(1)`),
      run it on the unchanged code: the unperturbed case must fit (so the test fails), as on the Mac
- [ ] Add `LEVERAGE_ONE = 1e-10` and the leverage check to `marginal_residuals` in `src/metafopy/models/rma_ls_fit.py`, as the API docstring says
- [ ] Remove the `docs/differences.md` section "A location-scale fit with an interaction whose column R drops" (with its compare block) and the
      `src/audit/data/known.toml` entry `construct/ls_na/mods={"$formula":"~ ablat * alloc"}`; grep `docs/` for any link to the section and fix it
- [ ] Run `bin/metafor-audit.sh --only 'construct/ls_na*'` (with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib`
      `METAFOPY_AUDIT_WORK=/Users/simonb/src/metafopy/.audit`): it must report that both sides refuse; record the run directory in the commit body

**Acceptance Criteria:**

- [ ] The new test passes; it failed on the unchanged code; `test_docs_examples.py` passes on `differences.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/models/rma_ls_fit.py` passes
- [ ] `pytest src/tests/test_rma_ls.py src/tests/test_docs_examples.py` passes

**Notes:**

- `marginal_residuals` is called only by `start_alpha` (spec, Case 1)

---

## Task 3: Case 2, the test's reference fit in standardised coordinates

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`LS_DEGENERATE`, `ls_is_maximum`); Implementation, Case 2

**Technical Goal:** `ls_reference_fit` finds the optimum of a log-link model on every platform, without keeping underflowed points.

**Subtasks:**

- [ ] Write the reproduction in `src/tests/correctness/test_formulas_ls.py` (`LS_SCORE_TOLERANCE` monkeypatched to `5e-7` and to `1e-9` on the
      `bcg2` data, ML and REML; `ls_is_maximum` false at the degenerate REML point the spec names, true at the optimum); run it on the unchanged
      code: it must raise `LinAlgError` there
- [ ] Add `LS_DEGENERATE` and `ls_is_maximum` to `src/tests/correctness/formulas.py`, and change `ls_reference_fit` for the log link as the spec
      says: centre and scale the non-constant columns of `Z` (`ddof=1`), fit on `Zs`, map `alpha` and `va` back, keep a start only when
      `ls_is_maximum` holds, skip the final Nelder-Mead; the identity link unchanged
- [ ] Run every non-slow `ls.` case of `src/tests/correctness/test_derived.py`: all pass

**Acceptance Criteria:**

- [ ] The reproduction passes and failed on the unchanged code; `ls.vif.reestimate.ML.bcg2` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/correctness/formulas.py src/tests/correctness/test_formulas_ls.py` passes
- [ ] `pytest src/tests/correctness/test_formulas_ls.py` and `pytest src/tests/correctness/test_derived.py -k "ls." -m "not slow"` pass

---

## Task 4: Case 3, the log-link Newton polish

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`LSProblem.log_score`, `log_information`, `polish_log` and its constants); Implementation, Case 3; decision D2

**Technical Goal:** metafopy's log-link location-scale fit ends at the same point on every platform.

**Subtasks:**

- [ ] Write `src/tests/test_rma_ls_opt.py` with the spec's reproduction (five draws of `yi * (1 + 1e-15 N(0, 1))` on `ls.vif.REML.bcg2`'s model:
      the location vif within `1e-8` relative of the unperturbed fit's and of the reference's) and the degenerate-fit test; run it on the
      unchanged code: the draws must spread (the test fails)
- [ ] Add `log_score` and `log_information` to `LSProblem` in `src/metafopy/models/rma_ls_fit.py`; test both against finite differences of the
      log-likelihood in `test_rma_ls_opt.py`
- [ ] Add `polish_log` and its constants to `src/metafopy/models/rma_ls_opt.py`, and call it in `fit_rma_ls` after `rma_ls_opt.minimise` as the
      spec says (log link, default optimiser, not with `optbeta=True`, through `mapfun` where bounds are set; not logged by `verbose`, not
      counted in `maxiter`)
- [ ] Update the `rma_ls_opt` module docstring and the location-scale optimiser line of `docs/differences-notes.md`
- [ ] Run `src/tests/test_rma_ls.py` and every other R parity test of location-scale fits at their committed tolerances; record the largest change
      per fixture, and the exact p of `ls.permutest.location.exact.ML.z.bcglat` before and after (794/1287 before), in `measurements.md`

**Acceptance Criteria:**

- [ ] The reproduction passes and failed on the unchanged code; `ls.vif.REML.bcg2` passes; no fixture is regenerated

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/models/rma_ls_fit.py src/metafopy/models/rma_ls_opt.py src/tests/test_rma_ls_opt.py` passes
- [ ] `pytest src/tests/test_rma_ls_opt.py src/tests/test_rma_ls.py` passes

**Notes:**

- Precedent: the capped Newton polish of `agents/specs/rma-mv-reml-stop/spec.md`

---

## Task 5: Case 4, the exact permutation bounds

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Implementation, Case 4; decision D3

**Technical Goal:** the exact permutation cases pass on the widened bounds that Task 3's reference gives, with no change of their own.

**Subtasks:**

- [ ] Re-measure the classification of `ls.permutest.location.exact.ML.z.bcglat` with Tasks 3 and 4 in: the number of permutations without a
      reference fit and the bounds (expected 77, 774 to 851 of 1287). A different count stops the task as `blocked`, recorded in
      `measurements.md` with the reason
- [ ] Run both exact permutation cases (`ls.permutest.location.exact.ML.z.bcglat`, `ls.permutest.scale.exact.ML.z.bcglat`): both pass, the
      scale case with its verdict unchanged; add the counts to `measurements.md`

**Acceptance Criteria:**

- [ ] Both exact permutation cases pass, and the counts are recorded

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness/test_derived.py -k "permutest and exact"` passes

**Notes:**

- Depends on Tasks 3 and 4

---

## Task 6: The full local check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation, Closing

**Technical Goal:** the branch passes every check on the development Mac before the operator merges and pushes it, and the CI check that
follows is recorded on the `TODO.md` item.

**Subtasks:**

- [ ] Run `pytest src/tests/correctness/test_derived.py -k "ls."` (slow cases included) and `src/tests/test_docs_examples.py`
- [ ] Run the final tests and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/location-scale-fits-stable-across-platforms/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/location-scale-fits-stable-across-platforms/tasks-1/final-tests.log`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Add a dated note at the end of the `TODO.md` item "CI fails three numerical tests that pass on a local macOS run": the list is done and
      the item moves to `DONE.md` once the CI run of the merge commit is green on every runner (spec, Closing steps 3 and 4); the operator
      session checks it with `gh run list --branch main --limit 3` and `gh run view <id> --log-failed`

**Acceptance Criteria:**

- [ ] The final test run passes, the spec says `implemented`, and the `TODO.md` item carries the note

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy src/metafopy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/location-scale-fits-stable-across-platforms/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1-5
- The list does not push; the CI check happens after the operator's `/ship`, outside the list, since the merge gate needs every task done
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
