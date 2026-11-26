# Tasks: Audit review 2

**Source:** `agents/audit-reviews/review-2.md`
**Created:** 2026-11-26
**Version:** 2

Fixes approved on the review page (answered 2026-11-26; see the review's Answers section) for finding groups F1-F13 and decisions D1-D2:

- D1 (answered "wrap"): the audit harness's file and subprocess calls raise `audit.errors.HarnessError`, at the 20 open A1 sites and the 34
  whose dismissals carried over (Tasks 10 and 11).
- D2 (answered "trim all", internal modules only): every flagged name leaves the `__all__` of the internal modules; the `metafopy` and
  `metafopy.plot` findings are dismissed as public API, and the `level_to_alpha` read is fixed (Task 12).

How to run things, in every task (from the worktree root):

- Tests: `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`.
- Type check: `mypy <files>` with the same `PATH`.
- The audit tools need `AUDITOR_RULES_PATH`, which `/Users/simonb/src/metafopy/.envrc` sets. Run them, and `git commit`, through direnv:
  `direnv exec /Users/simonb/src/metafopy <command>`. A commit run without it fails its `auditme-changed` hook with
  "AUDITOR_RULES_PATH is not set".
- Phase B ledger findings: `direnv exec /Users/simonb/src/metafopy auditor phaseb report . --question <Q>`. It prints every open finding with
  its candidate id, so a task works from the report rather than from the review's lists.

When fixing a defect, write the failing test first. A change to what metafopy prints or returns stops the task as `blocked` for the operator
(`claude/rules/metafopy.md`: output changes only after the user confirms them); none of these tasks is meant to make one. A finding a task
cannot fix as the review recommends stops that task as `blocked` and names the finding; it is never widened into a decision the review did
not make.

---

## Task 1: Starting checks

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** all

**Technical Goal:** confirm the branch starts where review-2 measured it, so later tasks work on the findings the review approved.

**Subtasks:**

- [x] Run `direnv exec /Users/simonb/src/metafopy auditor check . > <scratch>/audit-start.jsonl` and confirm every hit is accepted
- [x] For each of A1, A4, A17, T1, Doc4 and Doc5 run the ledger report and record its open count in this task's Notes; review-2 measured
      20, 10, 61, 6, 39 and 20 on 2026-11-26. A different count stops the task as `blocked`, naming the question and both counts
- [x] Check the new module name is free: `grep -rnw 'harness_io' src/audit` prints nothing (Task 10 adds `src/audit/harness_io.py`);
      otherwise stop as `blocked`

**Acceptance Criteria:**

- [x] The sweep is clean and the six counts match review-2's

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Key files: `agents/audit-reviews/review-2.md`, `agents/audit-reviews/ledger/`
- Checked 2026-11-26: the sweep has only accepted hits; open counts A1 20, A4 10, A17 61, T1 6, Doc4 39, Doc5 20, all matching review-2;
  `harness_io` is unused in `src/audit`.

---

## Task 2: `InternalError` in ten `Raises:` sections

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** F3 (A4)

**User Story:** As a user catching metafopy's errors, I want each function's `Raises:` section to name the `InternalError` it can let
through, so that I know what to handle.

**Subtasks:**

- [x] Add `metafopy.errors.InternalError`, with the condition the finding names, to the `Raises:` section of each of: `confint`
      (`src/metafopy/methods/confint.py`), `profile` and `profile_rma_uni` (`methods/profile.py`), `residuals` (`methods/residuals.py`),
      `vif` and `vif_rma_uni` (`methods/vif.py`), `evaluate` (`models/sel_lik.py`), `prepare_tables` (`models/tables.py`), `funnel`
      (`plot/funnel.py`) and `labbe` (`plot/labbe.py`)
- [x] Check each against its A4 finding text in the ledger report, so the wording names where the error comes from

**Acceptance Criteria:**

- [x] Each of the ten docstrings declares `InternalError`; no code changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` on the eight changed files passes

**Notes:**

- Generics are judged by dispatch: review-1's decision D1 (answered 2026-11-21)
- Done 2026-11-26: the drift rule then flagged three callers (`rma_mh`, `rma_peto`, `selmodel`); each now declares `InternalError` too.

---

## Task 3: User-guide example output

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** F1 (Doc5): the example-output findings

**User Story:** As a reader of the user guides, I want each example's printed output to be what the code prints, so that I can trust the
pages.

**Subtasks:**

- [ ] For each Doc5 finding about printed output (`docs/differences.md:587`, `docs/guide/binary.md:109`, `docs/guide/diagnostics.md`
      residuals, influence, heteroscedasticity and VIF blocks, `docs/guide/effect-sizes.md:103` and `:118`), run the page's example and
      paste exactly what it prints, blank lines included
- [ ] Leave the code alone: where the printed output looks wrong rather than the page, stop as `blocked` and name the example

**Acceptance Criteria:**

- [ ] Every output block named by these findings matches a fresh run character for character

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Key files: `docs/guide/diagnostics.md`, `docs/guide/effect-sizes.md`, `docs/guide/binary.md`, `docs/differences.md`

---

## Task 4: User-guide claims and lists, and the README

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** F1 (Doc5): the claims and lists; F7 (Doc3)

**User Story:** As a reader, I want what the guides say about metafopy to be true, and the README example to follow the project's
conventions, so that I am not misled.

**Subtasks:**

- [ ] Correct each Doc5 claim against the code, changing the page, never the code:
      `docs/guide/binary.md:115` (`blup` and `ranef` raise on an `rma_glmm` result), `docs/differences-notes.md:61` (`emmprep` of
      `rma_mh`/`rma_peto` fits), `:67` (`rma_mv`'s ported optimisers), `:161` (`forest(clim=)` on cumulative results), `:191`
      (`rma(cutoff=)`), `docs/from-r.md:103` (`escalc` returns an `EscalcFrame`), `docs/reference/methods.md:3` (`hettest`, `vcov` and
      `coef_summary` are plain functions)
- [ ] For each list a finding says reads as complete but is not (`README.md:39`, `docs/guide/advanced.md:49`, `docs/guide/emmeans.md:61`,
      `docs/guide/plots.md:28`, `:37`, `:122`, `:139`, `docs/index.md:12`), complete it or mark it as examples ("for example", "such as")
- [ ] Rewrite the example at `README.md:13` with full module paths (`import numpy`, `import metafopy`), as the guides write it

**Acceptance Criteria:**

- [ ] Each claim matches the code, each partial list says it is partial, and the README example uses full module paths

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py` passes
- [ ] `/Users/simonb/src/metafopy/.venv/bin/mkdocs build --strict` passes

**Notes:**

- Depends on Task 3 (both edit `docs/guide/`)

---

## Task 5: Specs that no longer match the code

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Source Reference:** F2 (Doc4)

**Technical Goal:** each implemented spec says what the code does, so the next reader of a spec is not misled.

**Subtasks:**

- [ ] For each Doc4 finding in the ledger report, correct the statement in the spec, or turn a "today" sentence into a dated record
      ("before this spec, on <date>, ...") where it describes the state the spec started from
- [ ] Change specs only: where a finding says the code, not the spec, is wrong, stop as `blocked` and name it

**Acceptance Criteria:**

- [ ] Each of the 39 findings' statements is corrected or dated

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Key files: the 16 specs named in review-2's F2

---

## Task 6: Direct tests for six functions

**Status:** todo
**Priority:** medium
**Complexity:** small
**Source Reference:** F4 (T1)

**Technical Goal:** each of the six functions T1 flagged has a test that calls it directly, covering the case its finding names.

**Subtasks:**

- [ ] `src/audit/pycall.py::shift_index`: a `resolve_value`/`run_call` test with a `'$idx'` tag (a list, a bool, an int or float, another
      value), in `src/tests/test_harness_pycall.py` or the file that already tests `pycall`
- [ ] `src/audit/rlib.py::accepts` and `::shift_index`: tests through `rcall` with `options(na.action=...)` set, and of shifting an R index
      from 1 to 0 (a scalar, a vector, a list, a name, `None`), in the file that tests `rlib`
- [ ] `ConfintList.__init__` and `ConfintResult.summary` (`src/metafopy/methods/confint.py`), and `RMAResult.hat_matrix`
      (`src/metafopy/models/rma_uni.py`, weighted and `weighted=False`), in their modules' own test files, each value checked against R or
      an existing reference fixture

**Acceptance Criteria:**

- [ ] Each test calls its function directly and passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest` on each test file the task writes or edits passes

**Notes:**

- Find the existing test file for each module with `grep -rln '<module>' src/tests`

---

## Task 7: A quadratic lookup and two repeated computations

**Status:** todo
**Priority:** medium
**Complexity:** small
**Source Reference:** F10 (P1), F11 (P3)

**Technical Goal:** remove work that grows with the data for no gain, without changing any result.

**Subtasks:**

- [ ] `src/metafopy/effects/reshape.py:734` and `:738` (`to_wide`): build `{study: rank}` once from the sorted unique studies and look each
      row up in it, instead of `ustudy.index(s)`
- [ ] `src/metafopy/methods/emm_adjust.py:239` (`critical_values`): compute the `mvt` critical value once per distinct `(level, df,
      correlation block)` within a family. Each call seeds its integrator with `MVT_SEED`, so reusing a value cannot change it; confirm
      with the existing `mvt` tests
- [ ] `src/audit/generate.py:205` (`generate`): compute `universe.draws(generic)` once per generic, outside the loop over subjects

**Acceptance Criteria:**

- [ ] The existing tests of `to_wide`, the emmeans adjustments and `generate` pass unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_reshape.py src/tests/test_emmeans_inference.py src/tests/test_emmeans.py` and the test file of
      `audit.generate` pass (find it with `grep -rln 'audit.generate' src/tests`)

**Notes:**

- If a test file named above does not exist, use the one `grep -rln '<module>' src/tests` finds and name it in the commit

---

## Task 8: One owner for fit statistics and the test `LSProblem`

**Status:** todo
**Priority:** low
**Complexity:** medium
**Source Reference:** F12 (A15)

**Technical Goal:** the ML/REML fit statistics are built in one place, and the location-scale tests build their problem in one fixture.

**Subtasks:**

- [ ] Grep `src/metafopy/models` for every hand-written `FitStats(` built from `ll`, `parms` and `n` (at least `rma_mv.py:1253`,
      `rma_ls_fit.py:1073` and `binary.py:59`); add one helper beside `FitStats` that returns it from `(ll, parms, n)` with the same
      arithmetic, and call it at each site
- [ ] Keep the arithmetic's order exactly, so each value is bit-identical; any changed printed value stops the task as `blocked`
- [ ] In `src/tests/test_rma_ls_restarts.py`, build the 12-field `LSProblem` (`:80`, `:98` and any other copy) through one fixture or
      helper in that file

**Acceptance Criteria:**

- [ ] No hand-written ML/REML `FitStats` assembly is left; the fit tests pass unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` on the changed files passes
- [ ] `pytest src/tests/test_rma_mv.py src/tests/test_rma_ls.py src/tests/test_rma_ls_restarts.py` and the own test files of every other
      module the task changes pass

---

## Task 9: Names, an alias and three small fixes

**Status:** todo
**Priority:** low
**Complexity:** small
**Source Reference:** F8 (S2), F9 (S4), F13 (A8, A9, Sec2)

**Technical Goal:** clear the small sampled findings.

**Subtasks:**

- [ ] Rename the lower-snake module-level hypothesis strategies to UPPER_SNAKE in `test_name_matching_properties.py`,
      `test_formula_rewriting_properties.py`, `test_labelled_print.py`, `test_print.py`, `test_tukey.py`, `test_optim.py`,
      `test_audit3_misc.py`, `test_farebrother.py`, `test_rtet.py` and `test_sel_lik.py` (all in `src/tests/`), with every use
- [ ] In `src/tests/test_name_matching_properties.py:40`, drop `st = hypothesis.strategies` and write `hypothesis.strategies.` in full
- [ ] In `src/tests/test_harness_compare.py:372`, pass `universe=` to `audit.steps.Context(...)` instead of setting it afterwards
- [ ] In `src/tests/auditsnippets.py`, make `load_snippet` load only: move the `require_pinned_versions` call to its callers, or rename it to
      say it checks the pin, whichever touches fewer files
- [ ] In `src/audit/cli.py`, reject a `--timeout` that is not positive: write the failing test first (in the test file of `audit.cli`), then
      raise `audit.errors.HarnessError` naming the value

**Acceptance Criteria:**

- [ ] Every renamed test still runs; a zero or negative `--timeout` fails with `HarnessError`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest` on each test file the task edits, and on `src/tests/test_auditsnippets.py` and the test file of `audit.cli`, passes

---

## Task 10: Harness I/O helpers that raise `HarnessError`

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Source Reference:** D1 (answered "wrap"), F5 (A1)

**Technical Goal:** one module holds the audit harness's file and process envelope, so each call site translates `OSError` and
`subprocess` failures to `audit.errors.HarnessError` by calling it.

**Subtasks:**

- [ ] Write `src/audit/harness_io.py` with small typed helpers, each catching `OSError` (and, for processes, `subprocess.SubprocessError`)
      and raising `audit.errors.HarnessError` from it, naming the path or the command and what the harness was doing: `read_text(path,
      what)`, `write_text(path, text, what)`, `write_bytes(path, data, what)`, `make_dir(path, what)`, `run_command(args, what, **kwargs)`
      returning the `subprocess.run` result, and `start_process(args, what, **kwargs)` returning the `Popen`. Docstrings with `Raises:`
- [ ] Tests in `src/tests/test_harness_io.py`: each helper on a good path, and on a failing one (a missing directory, a missing executable)
      raising `HarnessError` whose message names the path or command

**Acceptance Criteria:**

- [ ] Each helper translates its failures and passes its tests

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/audit/harness_io.py` passes
- [ ] `pytest src/tests/test_harness_io.py` passes

**Notes:**

- Depends on Task 1 (the name check)

---

## Task 11: Route the harness's 54 call sites through the helpers

**Status:** todo
**Priority:** medium
**Complexity:** large
**Source Reference:** D1 (answered "wrap"), F5 (A1)

**Technical Goal:** every A1 site in `src/audit`, open or carried as dismissed, translates its failure to `HarnessError`.

**Subtasks:**

- [ ] List the sites: the open ones from `direnv exec /Users/simonb/src/metafopy auditor phaseb report . --question A1` (20 on 2026-11-26),
      and the dismissed ones that still carry a finding:
      `python3 -c "import json; [print(d['id']) for d in map(json.loads, open('agents/audit-reviews/ledger/A1.jsonl')) if d.get('dismissed') and any(d['findings'].values())]"`
      (34 on 2026-11-26)
- [ ] Replace each call with the matching `audit.harness_io` helper, passing a `what` that names the step; where a site already catches
      `OSError` and returns a default (`engine.resident_bytes`, `report.git_head`), raise through the helper instead only if the default is
      not a deliberate fallback the code documents; otherwise stop as `blocked` and name the site
- [ ] Remove each `# auditme: accept[filesystem-op-without-envelope]` marker the change makes stale
- [ ] Add each function's `HarnessError` to its `Raises:` section where it has one

**Acceptance Criteria:**

- [ ] No call in the two lists above can raise `OSError` or a `subprocess` error untranslated
- [ ] The harness tests pass unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/audit` passes
- [ ] `pytest` on the test files of every changed `src/audit` module passes (find them with `grep -rln 'audit.<module>' src/tests`)
- [ ] `direnv exec /Users/simonb/src/metafopy auditor check .` reports no unaccepted hit

**Notes:**

- Depends on Task 10
- `src/audit/pyrun.py` and `src/audit/scripts.py` also run as scripts (`<module>` sites); both already import `audit.*`, so they can use
  the helpers

---

## Task 12: Trim internal `__all__` lists

**Status:** todo
**Priority:** low
**Complexity:** medium
**Source Reference:** D2 (answered "trim all", internal modules only), F6 (A17)

**Technical Goal:** each internal module's `__all__` lists only names another module reads, while metafopy's public packages keep their
exports.

**Subtasks:**

- [ ] For every A17 finding except `src/metafopy/__init__.py` and `src/metafopy/plot/__init__.py`, drop the names it lists from that
      module's `__all__`, documented ones included (as answered); keep each name defined in its module
- [ ] Dismiss the two public-package findings:
      `direnv exec /Users/simonb/src/metafopy auditor phaseb dismiss . --question A17 <id> --reason "2026-11-26: decision D2 of agents/audit-reviews/review-2.md (answered 2026-11-26): metafopy's public API, read by users, as review-1 D3 kept it"`
      for the ids `src/metafopy/__init__.py` and `src/metafopy/plot/__init__.py` as the report prints them
- [ ] Fix the read in `src/metafopy/interactive/data.py:20` and `:50`: read `level_to_alpha` through a module that exports it (export it
      from `metafopy.plot.rutils`'s package or move it to a public module), not through a submodule `metafopy.plot` leaves out
- [ ] If `mkdocs build --strict` then fails because a reference page renders a name a module no longer exports, stop as `blocked` naming
      the page and the name: the answer covered dropping documented names, not changing the reference pages

**Acceptance Criteria:**

- [ ] No name an A17 finding lists for an internal module is left in that module's `__all__`; the two public-package findings are
      dismissed in the ledger; `level_to_alpha` is read through an exporting module

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_signatures.py src/tests/test_transf.py src/tests/test_vcalc.py` and the own test files of every changed
      module pass
- [ ] `/Users/simonb/src/metafopy/.venv/bin/mkdocs build --strict` passes

**Notes:**

- The ledger file `agents/audit-reviews/ledger/A17.jsonl` changes through the dismiss command; commit it with the task

---

## Task 13: The full check, the summary and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** all; auditme step 8b

**Technical Goal:** the whole branch passes every check once, and the audit summary records what was fixed.

**Subtasks:**

- [ ] Run `direnv exec /Users/simonb/src/metafopy auditor check . > <scratch>/audit.jsonl`. Every hit must be accepted; an unaccepted hit
      this list introduced is fixed here
- [ ] In `agents/audit-reviews/summary.md`, move every group this list fixed (F1-F13) from "Unresolved findings" to "Recently resolved" with
      "Resolved in: tasks-2 (date)"
- [ ] Mark each task above `done` only if its commit is on the branch
- [ ] Run the final tests from the worktree and save their whole output for the merge gate:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/audit-reviews/tasks-2 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/audit-reviews/tasks-2/final-tests.log`

**Acceptance Criteria:**

- [ ] `summary.md` lists each fixed group as resolved, and the final run passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `direnv exec /Users/simonb/src/metafopy pre-commit run --all-files` passes
- [ ] `direnv exec /Users/simonb/src/metafopy pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `/Users/simonb/src/metafopy/.venv/bin/mkdocs build --strict` passes
- [ ] The final test run above passes

**Notes:**

- Depends on Tasks 1-12
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
- No `TODO.md` item moves in this list
