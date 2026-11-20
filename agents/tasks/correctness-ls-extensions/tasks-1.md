# Tasks: Correctness tests: location-scale extensions

**Spec:** `agents/specs/correctness-ls-extensions/spec.md`
**Created:** 2026-11-20
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

The spec (version 1.1, approved by the operator on 2026-11-20 in `agents/reviews/correctness-ls-extensions-spec.md`, every decision answered) adds
derived and property correctness cases for location-scale `profile`, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha` and
`beta`, the `alpha_min`/`alpha_max` box, REML `optbeta` and the `vb` of ML `optbeta` fits (against the observed information, decision D1), with
four exemptions. No file under `src/metafopy/` changes; no version bump. The list follows the sibling lists `correctness-uni-diagnostics` and
`correctness-mv-diagnostics` (references in `formulas.py` that never import metafopy, checks in `test_derived.py` and `test_properties.py`, items in
`test_coverage.py`); Task 7 hoists anything the new code duplicates from them.

**Operator stops:** Task 5 stops only if Fox and Monette (1992), JASA 87:178-183, doi:10.1080/01621459.1992.10475190, is not in the papers folder
`/Users/simonb/src/metafopy papers/` (placing it there before the run lets the task go straight through). Task 6 stops for the mismatch review: the
spec expects two mismatch cases, `ls.alpha.bounds.ML.bcg.wide` and `ls.profile.ML.identity.bcg`.

---

## Task 1: The references

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Definitions and references; API (formulas.py); Error handling; Implementation step 1

**Technical Goal:** independent references for constrained, bounded and joint location-scale fits, the joint information, BLUPs, standardised
residuals and the GVIF exist in `formulas.py`, with their own harness tests.

**Subtasks:**

- [ ] Starting-point check: `grep -nE 'def (ls_constrained_fit|ls_joint_information|ls_joint_fit|ls_blups|ls_rstandard|gvif|ls_profile|ls_vif|ls_vif_reestimate|ls_blup|ls_ranef|ls_unweighted|ls_alpha_fixed|ls_alpha_bounds|ls_beta_fixed|ls_optbeta_vb|ls_optbeta_va|ls_rstandard_types_agree)\b|^LS_VIF ' src/tests/correctness/*.py`
      prints nothing; stop as `blocked`, naming the match, if it does (the module `src/metafopy/methods/ls_profile.py` and the existing check
      `ls_optbeta` are not clashes)
- [ ] Add to `src/tests/correctness/formulas.py`, with the signatures and docstrings of the spec's API section and coded from the equations its
      "Definitions and references" section names: `ls_constrained_fit` (eq 7 or 9 over the free alpha, the rest held, eq 17; L-BFGS-B within
      `[lower, upper]` when given; starts include `ls_profile_ll`'s fitted-line start), `ls_joint_information` (the analytic observed information
      of eq 5), `ls_joint_fit` (eq 5 over the free beta and every alpha, eq 15), `ls_blups` (Henderson's equations with `G = diag(tau2)`),
      `ls_rstandard` (section 2.8, marginal and conditional) and `gvif` (Fox and Monette's ratio of determinants of the correlation matrix without
      the intercept). Reuse `ls_reference_fit`, `ls_profile_ll`, `model_matrix` and `uni_blups` by calling them; `numpy` and `scipy` only, no
      metafopy import
- [ ] The errors of the spec's Error handling section: each reference raises `RuntimeError` naming the optimiser's message rather than return a
      non-optimum
- [ ] Harness tests in `src/tests/correctness/test_formulas_ls.py`: `ls_joint_information` against central finite differences of eq 5; `gvif` of a
      two-moderator OLS against `Var[b_j] / Var[b'_j]`; `ls_blups` with constant `tau2` against `uni_blups`; `ls_constrained_fit` with nothing held
      equal to `ls_reference_fit`
- [ ] Run `pytest src/tests/correctness/test_formulas_ls.py`

**Acceptance Criteria:**

- [ ] Each reference reproduces its closed form or finite-difference check to the tolerance its harness test names
- [ ] `formulas.py` still imports no metafopy module (the harness's import-ban test passes)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask passes

**Notes:**

- Key files: `src/tests/correctness/formulas.py`, `src/tests/correctness/test_formulas_ls.py`
- Measured values to expect are in `agents/specs/correctness-ls-extensions/measurements.md`; the probe scripts of 2026-11-20 were outside the
  repository and are not needed

---

## Task 2: The checks, the coverage items and the exemptions

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (the checks); Coverage list; Error handling; Implementation step 2

**Technical Goal:** the case checks exist, and the coverage gate names every new item, failing until Tasks 3 and 4 add the cases.

**Subtasks:**

- [ ] In `src/tests/correctness/test_derived.py`, add the checks the spec's API section lists, each at the tolerance it names: `ls_profile` (`ll`
      at each grid value of `profile(result, alpha=j, steps=7)` against `ls_profile_ll`, a reference -1e10 and metafopy NaN agreeing, NaN on one
      side failing naming the grid value, and `maxll`), `ls_vif`, `ls_vif_reestimate` (skipping, and saying so, a coefficient whose reference
      refit raises), `ls_blup`, `ls_ranef`, `ls_rstandard`, `ls_unweighted`, `ls_alpha_fixed`, `ls_alpha_bounds`, `ls_beta_fixed` and
      `ls_optbeta_vb` (against the inverse of `ls_joint_information` over the free coefficients, decision D1)
- [ ] Generalise the existing `ls_optbeta` check to take the method from the case, so it serves REML as well as ML; keep its docstring's account of
      `vb` and point it at `ls_optbeta_vb`
- [ ] In `src/tests/correctness/test_properties.py`, add `ls_optbeta_va` (an `optbeta` fit's `va` equals the ordinary fit's) and
      `ls_rstandard_types_agree` (marginal and conditional `z` are equal)
- [ ] In `src/tests/correctness/test_coverage.py`, add `LS_VIF = ('location', 'reestimate', 'scale')` and extend `ls_items()` with the spec's
      Coverage list: `ls.profile.<link>` (links read from the code as for `ls.fit`), `ls.vif.<kind>`, `ls.blup`, `ls.ranef`, `ls.rstandard.<type>`
      (types from `typing.get_args(inspect.signature(metafopy.methods.ls_resid.rstandard_rma_ls).parameters['type'].annotation)`),
      `ls.unweighted`, `ls.alpha.fixed`, `ls.alpha.bounds`, `ls.beta.fixed.ML`, `ls.optbeta.REML`, `ls.optbeta.vb.ML`, and the exempt items
      `ls.fitstats.deviance`, `ls.unweighted.fitstats`, `ls.beta.fixed.REML`, `ls.optbeta.vb.REML`
- [ ] In `src/tests/correctness/cases/location_scale.toml`, a new section with the `[[exempt]]` entries for the exempt items, each with reason
      "defined only in metafor's code; R parity covers it"
- [ ] Run `pytest src/tests/correctness/test_coverage.py src/tests/correctness/test_formulas_ls.py`: the gate fails listing exactly the new
      non-exempt items, and every other test passes

**Acceptance Criteria:**

- [ ] The gate names every item of the spec's Coverage list; the only failures are the new items with no case yet
- [ ] The existing `ls.optbeta.*` ML cases still pass with the generalised check

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask behaves as stated there
- [ ] `pytest src/tests/correctness -m 'not slow' -k optbeta` passes

**Notes:**

- Depends on Task 1
- The commit leaves the coverage gate failing by design until Task 4; say so in the commit body

---

## Task 3: The profile, vif, blup, ranef and rstandard cases

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Cases (the `ls.profile`, `ls.vif`, `ls.blup`, `ls.ranef`, `ls.rstandard` rows); Implementation step 3

**Technical Goal:** the cases of the first group of the spec's Cases table exist with their sources.

**Subtasks:**

- [ ] In the new section of `src/tests/correctness/cases/location_scale.toml`, add the cases `ls.profile.ML.log.bd`, `ls.profile.REML.log.bcg`,
      `ls.profile.REML.identity.bd`, `ls.profile.ML.identity.bcg`, `ls.vif.ML.bd2`, `ls.vif.REML.bcg2`, `ls.vif.reestimate.ML.bd2`,
      `ls.vif.reestimate.ML.bcg2`, `ls.blup.ML.z.bd`, `ls.blup.REML.t.bcg`, `ls.ranef.REML.z.bd`, `ls.ranef.ML.t.bcg`,
      `ls.rstandard.marginal.REML.bd` and `ls.rstandard.conditional.ML.bcg`, on the datasets `measurements.md` defines (`bd`, `bcg`, `bd2`,
      `bcg2`), each with its check and `covers`
- [ ] Each case's `source` has `citation`, `doi` where one exists, a `locator` from the spec's sources table and `read`; the `vif` cases cite Fox
      and Monette (1992) with locator "the GVIF definition", which Task 5 completes
- [ ] A case whose check disagrees with metafopy beyond its tolerance is committed with `status = "mismatch"` and a `reason` naming both values,
      the grid value or coefficient, whether R metafor 5.2-1 agrees with metafopy, and "Recorded on <date>, not yet reviewed"; the spec expects this
      for `ls.profile.ML.identity.bcg` only. Never loosen a tolerance or change a case's data to make it pass
- [ ] Run `pytest src/tests/correctness/test_derived.py -m 'not slow' -k 'ls_'`

**Acceptance Criteria:**

- [ ] Every case of this group passes or is a strict expected failure with a reason
- [ ] The coverage gate lists only the items of Task 4's group as missing

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 2
- The profile cases take 2 to 7 s each with their reference (measurements.md); none is `slow`

---

## Task 4: The option cases and the property cases

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Cases (the `ls.unweighted`, `ls.alpha`, `ls.beta`, `ls.optbeta` rows and the property cases); Implementation step 3

**Technical Goal:** the remaining cases of the spec's Cases table exist with their sources, and the coverage gate passes.

**Subtasks:**

- [ ] Add the cases `ls.unweighted.ML.bd`, `ls.unweighted.REML.bcg`, `ls.alpha.fixed.ML.bd` (`alpha = [nan, 2.7]`), `ls.alpha.fixed.REML.bcg`
      (`[-3.5, nan]`), `ls.beta.fixed.ML.bd` (`optbeta`, `beta = [nan, 0.05]`), `ls.beta.fixed.ML.bcg` (`[nan, 0.02]`), `ls.alpha.bounds.ML.bd`,
      `ls.alpha.bounds.REML.bd`, `ls.alpha.bounds.ML.bcg` and `ls.alpha.bounds.ML.bcg.wide` (the boxes the spec's table gives),
      `ls.optbeta.REML.bd`, `ls.optbeta.REML.bcg`, `ls.optbeta.vb.ML.bd` and `ls.optbeta.vb.ML.bcg`, each with its check, `covers` and `source`
- [ ] The property cases `property.ls.optbeta.va.ML.bd` and `property.ls.rstandard.types.REML.bcg` in
      `src/tests/correctness/cases/properties.toml`, checks `test_properties.ls_optbeta_va` and `test_properties.ls_rstandard_types_agree`
- [ ] A disagreement beyond tolerance is a `mismatch` case as in Task 3; the spec expects this for `ls.alpha.bounds.ML.bcg.wide` only (metafopy
      alpha 0, -0.0795806, ll -8.6762, against the bounded maximum -4.95749, 0.03859, ll -7.5393)
- [ ] Run `pytest src/tests/correctness -m 'not slow'`

**Acceptance Criteria:**

- [ ] The coverage gate passes: every new item has a case or an exemption
- [ ] Every case passes or is a strict expected failure with a reason

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 3

---

## Task 5: Sources

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Definitions and references (sources table); Implementation step 4

**Technical Goal:** every locator is checked against its paper; the list stops only if Fox and Monette (1992) is missing.

**Subtasks:**

- [ ] If no PDF of Fox and Monette (1992), "Generalized collinearity diagnostics", is in `/Users/simonb/src/metafopy papers/` (search the file
      names and, with `pdftotext`, the first page of each PDF for "Generalized Collinearity"), write `agents/specs/correctness-ls-extensions/mismatches.md`
      with a "Sources needed" section naming it (JASA 87:178-183, doi:10.1080/01621459.1992.10475190) and what is wanted (the GVIF definition and
      its section), set this task `blocked` with "waiting for Fox and Monette (1992) in the papers folder", and stop. **Stops for the operator.**
      The operator places the paper in the folder and restarts the list
- [ ] Otherwise read its GVIF definition (text extracted into a scratch directory outside the repository), check the `gvif` reference against it,
      and set the `vif` cases' locators to its equation or section
- [ ] Check each other case's `source.locator` against the paper it names in the folder; correct any that names the wrong equation or section
- [ ] Run `pytest src/tests/correctness -m 'not slow'`

**Acceptance Criteria:**

- [ ] Every locator names the equation or section its value comes from
- [ ] A definition that differs from the reference changes the reference, and any resulting disagreement is a `mismatch` case as in Task 3

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 4

---

## Task 6: The mismatch review

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5

**Technical Goal:** each mismatch case has a review item, and the list stops until each has an answer.

**Subtasks:**

- [ ] Write (or complete) `agents/specs/correctness-ls-extensions/mismatches.md` in the shape of `agents/specs/correctness-mv-diagnostics/mismatches.md`:
      header (task list, spec), "What is reviewed and how", "Automatic checks" (dated: the `not slow` run, and that every `status = "mismatch"`
      case of this list has an item), one item per mismatch case (case, source, reference, metafopy and R metafor values, cause if found), a
      "Decisions" entry per item with the options *follow the reference*, *keep metafor* and *investigate*, each with its consequence and a
      recommendation, and an empty `## Answers` section. For `ls.profile.ML.identity.bcg`, restate the answer of 2026-11-15 in
      `agents/specs/correctness-location-scale/mismatches.md` (item `ls_identity_opt`, *follow the reference*, implemented as the identity-link
      polish) and ask only whether the remaining shortfall needs a further fix. If there is no mismatch case, say so under Automatic checks, write
      "None: no answer needed" under Answers, mark the remaining subtasks done with the note "no mismatch case" and set this task `done`
- [ ] Run `pytest src/tests/correctness -m 'not slow'` (the quick form before a human review)
- [ ] **Stops for the operator: the mismatch review.** Set this task `blocked` with "waiting for the mismatch review in
      agents/specs/correctness-ls-extensions/mismatches.md". An operator session publishes the review page, and once every item has an answer it
      reads the page's answers, copies each into the Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case has an item with its figures, options and a recommendation, or the document says there is none

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 5

---

## Task 7: Applying the answers, the full test run, and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 6

**Technical Goal:** the answers are acted on, duplicated code is hoisted, the whole branch is checked once, the spec is marked implemented and the
`TODO.md` item, whose last part this closes, moves to `DONE.md`.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-ls-extensions/mismatches.md`, never from a page; if a mismatch item has
      no answer there, stop as `blocked`, naming what is missing
- [ ] For each *follow the reference* answer, file a `TODO.md` item (the fix, flipping the case, marking the parity fixtures, `known.toml`,
      `docs/differences.md`) at the top of `TODO.md`; for *keep metafor*, add "accepted as a difference from the reference on <date>" to the case's
      `reason`; for *investigate*, file a `TODO.md` item with the figures
- [ ] Hoist anything this list duplicates from the rma_uni and rma_mv diagnostics work to a single owner: grep `src/tests/correctness` for each
      literal collection, constant and one-line helper the new code defines (the interval quantile, BLUP solves, NaN-naming loops), and import
      rather than copy. Choose the owner from the existing import graph; prefer a module the consumer already imports over adding a new edge
- [ ] Move the `TODO.md` item "Correctness coverage the extension specs left for later" to the end of `DONE.md`: append its heading and body
      unchanged, then the line `Done: <date> - agents/specs/correctness-mv-diagnostics/spec.md (the rma_mv part) and agents/specs/correctness-ls-extensions/spec.md (the location-scale part), agents/tasks/correctness-ls-extensions/tasks-1.md.`,
      and remove it from `TODO.md`; the item "No correctness case covers user weights with a scale model" stays
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-20)" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/correctness-ls-extensions/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/correctness-ls-extensions/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] Every answer is acted on as above
- [ ] The spec says `implemented`; the item is in `DONE.md` with its body unchanged and gone from `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-ls-extensions/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is a scratch directory
      outside the repository)

**Notes:**

- Depends on Task 6
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its exit
  status before going on
