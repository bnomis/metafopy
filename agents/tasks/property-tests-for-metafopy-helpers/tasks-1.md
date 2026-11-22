# Tasks: Property tests for metafopy helpers

**Spec:** `agents/specs/property-tests-for-metafopy-helpers/spec.md`
**Created:** 2026-11-22
**Version:** 1

The spec's decisions were answered on 2026-11-22 (`agents/reviews/property-tests-for-metafopy-helpers-spec.md`):
- D1: all three groups get property tests (A, the numerical kernels; B, number formatting; C, names, formulas and strings);
- D2: the T2 findings under `src/audit` and `bin` belong to the harness item, so they are judged here but never tested.

`agents/specs/property-tests-for-metafopy-helpers/measurements.md` lists the candidates by group. It is a dated record, so never edit it.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **T2 rule:** a candidate is covered when a test decorated with hypothesis `given` calls it directly, by its full module path
  (`~/.claude/skills/auditme/phaseb/briefs/T2.md`, "Thresholds").

How to write each property test, in every test task:

- **Property:** check the property that the candidate's `reason` in `agents/audit-reviews/ledger/T2.jsonl` states. Read it with
  `grep -F '"id": "<candidate id>"' agents/audit-reviews/ledger/T2.jsonl`.
- **File:** put the test in the existing test file that already imports the candidate's module. Where none does, use a new
  `src/tests/test_<module>.py`.
- **Docstring:** name the candidate id.
- **Settings:** decorate every test with `@hypothesis.settings(max_examples=100, deadline=None)`.
- **Imports:** use full module paths (`import hypothesis.strategies`, `import metafopy.core.tukey`) and no leading underscores
  (`claude/rules/`).
- **Shared strategies:** take them from `src/tests/rstrategies.py` (Task 2) where one fits.
- **Tolerances** (spec, Implementation step 3):
  - a scipy oracle at a relative tolerance of 1e-8;
  - 1e-6 for the `farebrother` and `tukey` series;
  - a slack of 4 ulps (`math.ulp`) for a bound or a monotonicity check;
  - the tolerance the reason gives, where it gives one.

When a property fails (spec, Error handling):

- **Inside the domain the R original documents:** stop the task as `blocked`, naming the candidate and hypothesis's shrunk example. File the
  defect with `/todo-add` and never fix it: an output change needs the user's confirmation (`claude/rules/metafopy.md`).
- **Only outside that domain:** narrow the strategy, and cite the R documentation or source line in the test's docstring.
- **R breaks the property the same way:** check it in `Rscript` (`LC_ALL=C.UTF-8 Rscript`, R library at
  `/Users/simonb/src/metafopy/.audit/Rlib`). Record R's output in a comment, test the property R does keep, and note the change in the
  candidate's ledger `reason` at Task 9's re-judge.
- **Slow test:** a test that takes more than 10 seconds gets `@pytest.mark.slow`.

---

## Task 1: Re-judge T2 at HEAD and record what is open

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 1

**Technical Goal:** the T2 ledger was judged at `31d809f`, and `phaseb.py plan` re-plans 121 of 193 candidates at `7c8bfeb`. A fresh verdict
for each tells the later tasks exactly which candidates to test.

**Subtasks:**

- [x] Run `python3 ~/.claude/skills/auditme/checks/phaseb.py plan . --question T2 --out <scratch>/t2`. `<scratch>` is a directory outside
      the repository
- [x] Judge every candidate in every batch by the T2 brief, by reading the candidate's source and the tests it lists (do not run them).
      Write `result-NN.jsonl` beside each `batch-NN.jsonl` in the brief's format. Candidates under `src/audit` and `bin` are judged the same
      way
- [x] `phaseb.py check <scratch>/t2` must report every batch complete. Then run `phaseb.py merge . --question T2 <scratch>/t2`
- [x] Write `agents/specs/property-tests-for-metafopy-helpers/open-after-rejudge.md`, a dated record ("measured on <date> at <commit>") of
      every open `src/metafopy` candidate in `phaseb.py report . --question T2`. Give each one a row with its id, its group (A, B or C, as in
      `measurements.md`) and the task that tests it, from the module lists in Tasks 3 to 8. Put a candidate that is not in `measurements.md`
      into the group the spec's definitions give: A computes a probability, density, weight or statistic; B turns numbers into text or lays
      text out; C matches names, parses formulas or rewrites strings. Give it the last task of that group (4, 6 or 8)
- [x] Stop as `blocked`, naming the candidate, if the brief does not settle one

**Acceptance Criteria:**

- [x] Every planned candidate has a fresh ledger verdict, and the open list with its task column is committed

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `python3 ~/.claude/skills/auditme/checks/phaseb.py check <scratch>/t2` reports every batch complete

**Notes:**

- `phaseb.py merge` keeps every verdict it is not given, so the 72 reused verdicts stay as they are
- The T2 findings under `src/audit` and `bin` stay open in the ledger for the harness item (decision D2)

---

## Task 2: Shared hypothesis strategies

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Implementation step 2

**Technical Goal:** one place for the input domains the property tests share, so each test does not rebuild them.

**Subtasks:**

- [x] Write `src/tests/rstrategies.py` with `probabilities`, `df_values`, `finite_floats` and `r_identifiers`, with the signatures and
      docstrings of the spec's API section. `r_identifiers` draws names matching R's syntactic-name rule (a letter, or a dot not followed by a
      digit, then letters, digits, `.` and `_`) and excludes R's reserved words (`if`, `else`, `repeat`, `while`, `function`, `for`, `next`,
      `break`, `TRUE`, `FALSE`, `NULL`, `Inf`, `NaN`, `NA`, `NA_integer_`, `NA_real_`, `NA_character_`, `NA_complex_`, `in`)
- [x] Write `src/tests/test_rstrategies.py`: one `given` test per strategy, checking its bounds (open interval, finiteness, positivity,
      the name rule)

**Acceptance Criteria:**

- [x] Each strategy draws only values inside its documented domain

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/tests/rstrategies.py src/tests/test_rstrategies.py` passes
- [x] `pytest src/tests/test_rstrategies.py` passes

**Notes:**

- Helper modules in `src/tests` are imported by bare name (`import refutil`), so the tests write `import rstrategies`
- `degrees_of_freedom` is already a package function name; the strategy is `df_values` (2026-11-22)

---

## Task 3: Property tests for the special-function kernels in `core` and the small statistics

**Status:** in-progress
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 3 (group A)

**Technical Goal:** the kernels ported from R's C code get tested over their domains, where a property can find a bug that fixed points miss.

**Subtasks:**

- [ ] One property test per open candidate in `open-after-rejudge.md` with Task 3. These are the candidates in `core/bvn.py`,
      `core/dbinom.py`, `core/farebrother.py`, `core/tukey.py`, `core/nchg.py`, `bias/ranktest.py`, `bias/fsn.py`, `effects/rtet.py`,
      `effects/fivenum_estimators.py::weibull_shape_start`, `models/tau2.py` and `methods/ls_anova.py`
- [x] Use the oracle the reason names: `scipy.stats.multivariate_normal` for `bvn`; `scipy.stats.binom` and `math.lgamma` for `dbinom`;
      `scipy.stats.chi2` for `farebrother`; `scipy.stats.hypergeom` and `scipy.stats.nchypergeom_fisher` for `nchg`; brute-force permutation
      counts for small `n` for `pkendall`

**Acceptance Criteria:**

- [ ] Every Task 3 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` on every test file the task writes or edits passes
- [ ] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2
- 2026-11-22: tests are written for every candidate in the Task 3 list and pass, except `weibull_shape_start`. Its property fails inside the
  documented domain: `weibull_shape_start(1.0, 0.14836888830451764)` gives 7.94188553535 where R's `nlminb` gives 8. The test is a strict
  `xfail` in `src/tests/test_conv_fivenum_methods.py`, and the defect is filed in `TODO.md` ("`weibull_shape_start` stops at its start value
  for large shapes"). The task stays `in-progress` until the operator confirms the output change and it is fixed.
- 2026-11-22: `qtukey` raises `ZeroDivisionError` where R returns NaN for 17 or more means and a small p (filed in `TODO.md`); the test
  narrows `nmeans` to 2 to 10 with R's output in a comment. `farebrother`'s `ruben` test accepts `ifault` 5 (R gives the same, recorded in a
  comment), and the `nchg` scipy comparison is made only above log probability -30 (scipy is inaccurate in the far tail).
- `bias/fsn.py` and `bias/ranktest.py` are reached through `importlib.import_module(...)` because `metafopy.bias.fsn` and
  `metafopy.bias.ranktest` name the re-exported functions, not the modules. Task 9's re-judge should confirm the T2 check counts these calls.

---

## Task 4: Property tests for the distribution, weight and map functions of `emm_adjust`, `sel_spec` and `sel_lik`

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 3 (group A)

**Technical Goal:** the remaining numerical kernels: the F and t distribution functions of the emmeans adjustments, and the selection-model
weight functions and parameter maps.

**Subtasks:**

- [x] One property test per open candidate in `open-after-rejudge.md` with Task 4. These are the group A candidates in
      `methods/emm_adjust.py` (`pf_cdf`, `pf_sf`, `pt_sf`, `qdunnx`, `qf`, `qt`), `models/sel_spec.py` (the `s_*` weight functions) and
      `models/sel_lik.py`, and any group A candidate the record adds
- [x] Use `scipy.stats.f` and `scipy.stats.t` as oracles for the distribution functions, and `scipy.special.expit` for `apply_named`'s
      `plogis`. One parametrised test may cover the `s_*` weight functions that share the property "1 for p <= s[0], in (0, 1], non-increasing
      in p", provided each function is called directly

**Acceptance Criteria:**

- [x] Every Task 4 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` on every test file the task writes or edits passes
- [x] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2

---

## Task 5: Property tests for number-to-text formatting

**Status:** done
**Priority:** medium
**Complexity:** large
**Spec Reference:** Implementation step 4 (group B)

**Technical Goal:** the functions that print numbers as R does keep their round trips, rounding bounds and widths over every finite input.

**Subtasks:**

- [x] One property test per open candidate in `open-after-rejudge.md` with Task 5. These are the candidates in `printing.py`,
      `core/labelled.py`, `core/timing.py`, `formatters.py`, `models/selmodel.py::r_as_character`, `formula.py::r_number` and
      `models/mv_factor.py::format_number`

**Acceptance Criteria:**

- [x] Every Task 5 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` on every test file the task writes or edits passes
- [x] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2
- 2026-11-22: the `r_as_character` tests (`formatters.py`, `models/selmodel.py`) cover magnitudes below 1e15: R 4.6.2 keeps every digit of larger
  whole numbers (`as.character(1000000000000006)` is `"1000000000000006"`), so the 15-digit round trip does not hold there in R either.
  `fmt_df_scalar` is tested below 1e4 (its `:g` keeps six significant digits). `r_as_character(123456789012345678.0)` gives `...680` where R
  gives `...696` (outside the tested domain, not filed).

---

## Task 6: Property tests for layout and the small numeric helpers of printing and plotting

**Status:** done
**Priority:** medium
**Complexity:** large
**Spec Reference:** Implementation step 4 (group B)

**Technical Goal:** the rest of group B: plot helpers, table layout and level conversion.

**Subtasks:**

- [x] One property test per open candidate in `open-after-rejudge.md` with Task 6. These are the group B candidates in `plot/rutils.py`,
      `plot/forest.py` (`anno_symbol`, `finish_annotext`, `parse_predtail`), `plot/gosh.py`, `plot/infl.py::n2mfrow`, `plot/profile.py`,
      `methods/vif.py`, `methods/blup.py`, `methods/ls_common.py`, `methods/deltamethod.py`, `bias/inputs.py`, `effects/summary.py`,
      `models/rma_uni.py` (`coef_span`, `level_to_alpha`), and any group B candidate the record adds
- [x] Use `matplotlib.colors.to_rgba` as the oracle for `hex_color`, as its reason names

**Acceptance Criteria:**

- [x] Every Task 6 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` on every test file the task writes or edits passes
- [x] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2
- 2026-11-22: tests are written for every Task 6 candidate and pass. `plot/forest.py` is reached through `importlib.import_module(...)`:
  `metafopy.plot.forest` names the re-exported function for mypy, as with `bias/fsn.py` in Task 3. Only the summary tests keep the
  "percent matches proportion" check of `level_to_alpha`; the piecewise test in each file covers the same behaviour.

---

## Task 7: Property tests for formula parsing and model-call rewriting

**Status:** done
**Priority:** medium
**Complexity:** large
**Spec Reference:** Implementation step 5 (group C)

**Technical Goal:** formulas built from generated R names parse back to their parts, and rewriting functions are idempotent where the
ledger says so.

**Subtasks:**

- [x] One property test per open candidate in `open-after-rejudge.md` with Task 7. These are the group C candidates in `formula.py`,
      `models/call.py`, `models/mv_factor.py::split_top_level`, `models/rma_mv.py`, `models/rma_uni.py::abbreviate_names`,
      `models/mv_print.py`, `models/mv_hessian.py`, `plot/infl.py::abbreviate`, `methods/rcalc.py` and `methods/matreg.py::parse_formula`
- [x] Build the formulas from `rstrategies.r_identifiers()`

**Acceptance Criteria:**

- [x] Every Task 7 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` on every test file the task writes or edits passes
- [x] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2

---

## Task 8: Property tests for name matching and uniqueness

**Status:** todo
**Priority:** medium
**Complexity:** large
**Spec Reference:** Implementation step 5 (group C)

**Technical Goal:** the copies of R's `pmatch`, `charmatch`, `match.arg`, `make.unique` and `make.names` keep R's rules for every input.

**Subtasks:**

- [ ] One property test per open candidate in `open-after-rejudge.md` with Task 8. These are the group C candidates in `methods/matreg.py`
      (other than `parse_formula`), `methods/predict.py`, `methods/common.py`, `methods/emm_adjust.py` (`match_method`, `parse_side`),
      `methods/emm_grid.py`, `methods/emm_nesting.py`, `methods/hettest.py`, `effects/datatools.py`, `effects/measures.py`,
      `effects/fivenum_estimators.py::sum_stat_index`, `metadata.py`, `interactive/forest.py`, `core/padjust.py`, `models/sel_opt.py` and
      `models/sel_spec.py::match_type`, and any group C candidate the record adds
- [ ] Use the oracles the reasons name: `matreg.match_columns` and `predict.match_names` against each other, and a brute-force deletion check
      for `is_subsequence`

**Acceptance Criteria:**

- [ ] Every Task 8 candidate in the record has a `given` test that calls it directly, and the tests pass

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` on every test file the task writes or edits passes
- [ ] `pytest` on every test file the task writes or edits passes, named in the commit body

**Notes:**

- Depends on Tasks 1 and 2
- `interactive/forest.py` needs the `interactive` extra (`uv sync --frozen --extra interactive`; installed 2026-11-21)

---

## Task 9: Re-judge, the full check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 6; decision D2

**Technical Goal:** no `src/metafopy` T2 candidate is open, the branch passes every check once, the spec is marked implemented and the
`TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Repeat Task 1's plan, judge, check and merge. A test file that Tasks 3 to 8 wrote or edited re-plans the candidates its new tests
      reach. `phaseb.py report . --question T2` must show no open `src/metafopy` candidate. A candidate still open goes back to the task its
      row in `open-after-rejudge.md` names
- [ ] Move the `TODO.md` item "Numerical cores have no property-based tests" to the end of `DONE.md`: append its heading and body unchanged,
      then the line `Done: <date> - agents/specs/property-tests-for-metafopy-helpers/spec.md,
      agents/tasks/property-tests-for-metafopy-helpers/tasks-1.md; the src/audit and bin findings moved to the harness item (D2).`, and remove
      it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved 2026-11-22" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/property-tests-for-metafopy-helpers/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/property-tests-for-metafopy-helpers/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] No `src/metafopy` candidate is open in the T2 report, the spec says `implemented`, and the item is in `DONE.md` with its body unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/property-tests-for-metafopy-helpers/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 3-8
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
