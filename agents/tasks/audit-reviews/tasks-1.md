# Tasks: Audit review 1

**Source:** `agents/audit-reviews/review-1.md`
**Created:** 2026-11-21
**Version:** 1

Fixes approved on the review page (answered 2026-11-21; see the review's Answers section) for finding groups F1-F20 and decisions D1-D3.
Decisions D4 and D5 (T1 and T2 test catch-up) are `TODO.md` items, not tasks here. The D2 and D3 dismissals are already in the ledger.

How to run things, in every task (from the worktree root):

- Tests: `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`.
- Type check: `mypy <files>` with the same `PATH`.
- Phase B ledger findings: `python3 ~/.claude/skills/auditme/checks/phaseb.py report . --question <Q>`. It prints every open finding with
  its candidate id, so a task works from the report rather than from the review's lists.

When fixing a defect, write the failing test first. A change to what metafopy prints or returns, other than the bootstrap bands in Task 12,
which the review approved, stops the task as `blocked` for the operator (`claude/rules/metafopy.md`: output changes only after the user
confirms them). A finding a task cannot fix as the review recommends stops that task as `blocked` and names the finding; it is never widened
into a decision the review did not make.

---

## Task 1: Asserts that validate user input

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** F3 (A12)

**User Story:** As a user running with `python -O`, I want `qqnorm` to reject an unrecognised `label` and `profile` to pick its range the same
way it does without `-O`, so that optimised runs behave like normal ones.

**Subtasks:**

- [x] Write a failing test in `src/tests/test_audit4_plots.py` (the qqnorm test file): `qqnorm` on an `rma_mh` fit with an unrecognised
      string `label` raises `metafopy.errors.InputError`
- [x] In `src/metafopy/plot/qqnorm.py:242-243`, replace the two `assert ... is not None` with an explicit check that raises `InputError`,
      naming the accepted `label` values; add `InputError` to `qqnorm`'s `Raises:` section if it is not there
- [x] In `src/metafopy/methods/profile.py:309-312`, replace the `assert ci.random is not None` and its `except ... AssertionError` fallback
      with an explicit `if ci.random is None:` branch that takes the same SE-based range
- [x] Run the two test files

**Acceptance Criteria:**

- [x] `grep -n "assert" src/metafopy/plot/qqnorm.py src/metafopy/methods/profile.py` shows neither site
- [x] The new test passes, and the profile plot tests are unchanged

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/plot/qqnorm.py src/metafopy/methods/profile.py` passes
- [x] `pytest src/tests/test_plots.py src/tests/test_profile_plots.py` passes

**Notes:**

- Key files: `src/metafopy/plot/qqnorm.py`, `src/metafopy/methods/profile.py`

---

## Task 2: Declarations, small doc fixes and the dismissals that need a marker

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** F4 (Doc3), F5 (D1), F19 (Doc1, spec review), F20 (T3, D2)

**Technical Goal:** make the declared dependencies and the short docs match the code, and record the approved "no change" findings in the
files Phase B reads.

**Subtasks:**

- [x] Run `uv add --dev 'nbformat>=5.10'` (`uv.lock` already holds 5.11.1). Stop as `blocked` if the lock changes any other package's
      version
- [x] `bin/logo.py:28`, `:31`, `:32`: add a scoped `# auditme: accept[D1]` comment above the imports, saying the script's PEP 723 block
      declares them
- [x] `src/audit/cli.py:118` and `bin/logo.py:265`: add `# auditme: accept[T3]` comments. The reason for `cli.main` is that
      `bin/metafor-audit.sh --recall` is its end-to-end check; for the logo script, that it is a one-off build tool
- [x] `pyproject.toml:45`, `:48`, `:54` (`ipykernel`, `pandas-stubs`, `scipy-stubs`): add a TOML comment
      `# auditme: accept[D2] <reason>` beside each. The reasons: ipykernel is the kernel nbclient starts; mypy reads the two stub packages
- [x] `README.md:52-58`: say the defining publication is the first oracle and R metafor the second; run fixtures with
      `LC_ALL=C.UTF-8 Rscript`; say the fixtures come from the scripts in `src/tests/r/`, with `generate_fixtures.R` as one of them
- [x] `AGENTS.md`, "Audit" section: say that `bin/metafor-audit.sh` writes `wrong-or-missing.md` under the ignored
      `.audit/runs/<timestamp>/`
- [x] Fix the five spec-template gaps from the review's "Spec review" section:
  - annotate the `fsn` signature in `agents/specs/auditme-clean/spec.md:65`;
  - turn the spec names in `correctness-ls-extensions/spec.md:27` and `:54`, `correctness-publication-bias/spec.md:14` and
    `correctness-uni-diagnostics/spec.md:65` into relative links

**Acceptance Criteria:**

- [x] `uv lock --check` passes and `pyproject.toml` declares `nbformat`
- [x] Each marker names its question and a reason
- [x] `README.md` agrees with `AGENTS.md` on the oracle order and the `Rscript` command

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_showcase_notebook.py` passes

**Notes:**

- `AGENTS.md` forbids editing `uv.lock` by hand or without instruction. `uv add --dev` is the instruction, since the review approved declaring
  `nbformat`

---

## Task 3: User guides whose examples or claims no longer match the code

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** F1 (Doc5), the guide pages

**User Story:** As a reader of the user guide, I want every example's output and every statement to match what metafopy does, so that I can
trust the page.

**Subtasks:**

- [x] List the open Doc5 findings with the report command, and take those on these pages:
  - `docs/getting-started.md`
  - `docs/guide/advanced.md`
  - `docs/guide/datasets.md`
  - `docs/guide/diagnostics.md`
  - `docs/guide/effect-sizes.md`
  - `docs/guide/emmeans.md`
  - `docs/guide/models.md`
- [x] For each, run the page's example in the worktree's environment and replace the shown output with what the code prints now. Change
      the shown output only: if the output itself looks wrong (not merely moved), stop as `blocked` and name the example
- [x] Correct each contradicted statement:
  - complete the `method` table (`'CO'`, `'VC'`) and the `selmodel` `type` list (`'trunc'`, `'truncest'`, the `'*1'`/`'*2'` variants);
  - add the second `hettest` block's output
- [x] Run `pytest src/tests/test_docs_examples.py`

**Acceptance Criteria:**

- [x] Every Doc5 finding on these pages is fixed; running each changed example prints exactly what the page shows

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- The ledger verdict for a changed page is re-judged at the next audit, when its hash changes

---

## Task 4: The differences pages

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** F1 (Doc5), `docs/differences.md` and `docs/differences-notes.md`

**User Story:** As a reader comparing metafopy with metafor, I want the differences pages to describe today's behaviour, so that I do not
plan around a difference that has gone.

**Subtasks:**

- [x] Take the open Doc5 findings on `docs/differences.md` and `docs/differences-notes.md` from the report command
- [x] In `docs/differences-notes.md`, fix these statements:
  - the `control=` entries that are ignored, not raised, by `confint`, `hettest` and `permutest` (line 219);
  - `ni`, `ddf`, `progbar`, `res.control` and `p` for `rma_mv` and `rma_glmm` (lines 19-35 and 73);
  - the "Not ported" list (lines 229-231), keeping only what is still not ported;
  - the `vcalc(sparse=True)` printing (line 53).
- [x] In `docs/differences.md`, re-capture these undated output blocks:
  - the `selmodel` flat-likelihood numbers;
  - the `rma_mv` `GEN` `tau2`;
  - the missing header lines in the `conv_fivenum`, `escalc` selection and `digits=` blocks;
  - the "Optimiser stop points" list of where the ported optimisers are used.

  Leave every section that states its own date as a record
- [x] Run `pytest src/tests/test_docs_examples.py`

**Acceptance Criteria:**

- [x] Every Doc5 finding on the two pages is fixed, and no dated section is rewritten

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on nothing; can run beside Task 3
- Key files: `docs/differences.md`, `docs/differences-notes.md`

---

## Task 5: `Raises:` sections that miss or over-declare domain errors

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** F2 (A4), decision D1

**Technical Goal:** every open A4 finding fixed, so that each function's `Raises:` section names exactly the domain errors that reach its
caller.

**Subtasks:**

- [x] List the open A4 findings with the report command (40 when measured on 2026-11-21). Each gives the names missing or extra
- [x] Add each missing name with a one-line reason, and remove each extra one. For the 16 `singledispatch` generics, list the errors their
      registered implementations raise, as decision D1 ruled. That includes implementations registered by call (`generic.register(Type, fn)`)
- [x] Run `python3 ~/.claude/skills/auditme/checks/audit.py .` and check that no `raises-section-drift` or `missing-raises-section` hit
      appears that is not accepted
- [x] Run the own tests of the changed modules (the docstring tests read the sections)

**Acceptance Criteria:**

- [x] Each A4 finding in the report is fixed in the docstring it names; no code behaviour changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_docstring_links.py src/tests/test_harness_errors.py` passes

**Notes:**

- Docstrings only. A finding that can be fixed only by changing what the code raises stops as `blocked`

---

## Task 6: Specs that no longer match the code

**Status:** done
**Priority:** medium
**Complexity:** medium
**Source Reference:** F6 (Doc4)

**Technical Goal:** each implemented spec says what the code does, so a later reader is not misled.

**Subtasks:**

- [x] List the open Doc4 findings with the report command (30, in 11 specs)
- [x] For each, correct the statement to what the code does. Bump the spec's `**Version:**` minor number, and add a dated line under the
      header saying what was corrected and that it came from audit review-1. Never edit a section that states its own date
- [x] For `auditme-clean`, the hooks now come from `bnomis/auditor` with `AUDITOR_DEPLOY_KEY` (changed in `31d809f`)

**Acceptance Criteria:**

- [x] Every Doc4 finding is corrected; each touched spec has a new version and a dated change line

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Specs only; no code

---

## Task 7: Standard-library errors where a domain error exists

**Status:** done
**Priority:** medium
**Complexity:** small
**Source Reference:** F7 (A3)

**Technical Goal:** the library raises `metafopy.errors`, and the audit harness raises its own error class.

**Subtasks:**

- [x] `src/metafopy/methods/predict.py:245`: raise `metafopy.errors.InputError` instead of `ValueError`, with a test
- [x] Add `src/audit/errors.py` with `class HarnessError(Exception)` (a docstring saying it is a step or run failure of the harness). Raise
      it at `pycall.py:99`, `rlib.py:801`, `rlib.py:807`, `recall.py:59`, `steps.py:84`, `:87`, `:105`, `:110`, `rdriver.py:48`,
      `scripts.py:67` and `:209`
- [x] Grep `src/audit`, `bin` and `src/tests` for every `except ValueError` and `except RuntimeError` that catches one of these, and every
      `pytest.raises(ValueError|RuntimeError)` that expects one, and update each to `HarnessError`. Before that, check whether any of
      them is R's `stop()` mapped deliberately to `ValueError`; if one is, stop as `blocked` and name it
- [x] Run the harness tests

**Acceptance Criteria:**

- [x] Each of the 12 sites raises a domain error, and every harness test passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/audit src/metafopy/methods/predict.py` passes
- [x] `pytest src/tests/test_harness_errors.py src/tests/test_harness_pycall.py src/tests/test_harness_compare.py src/tests/test_harness_engine.py src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py src/tests/test_predict_args.py`
      passes

**Notes:**

- Check for a name clash first: `grep -rnwE 'HarnessError' src/audit` must find nothing

---

## Task 8: Literal chains without `assert_never`

**Status:** done
**Priority:** medium
**Complexity:** medium
**Source Reference:** F8 (A11)

**Technical Goal:** the type checker proves each chain over a `Literal` or an enum covers every value.

**Subtasks:**

- [x] List the open A11 findings with the report command (36)
- [x] For each chain, give every value its own arm and end with `else: typing.assert_never(x)`. For a two-way split on a bool computed from
      a Literal, test the Literal values instead
- [x] Run `mypy` on each changed module and the changed modules' own tests

**Acceptance Criteria:**

- [x] Every A11 finding has an explicit arm per value and an `assert_never`; no behaviour changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes on every changed module
- [x] The own test files of every changed module pass, named in the commit body

**Notes:**

- `src/tests/test_matreg.py::check_fit` is a test-side chain; fix it the same way

---

## Task 9: Bundled-dataset reads and unchecked inputs

**Status:** done
**Priority:** medium
**Complexity:** small
**Source Reference:** F9 (A1, `datasets/base.py::read`), F17 (Sec2, Sec4)

**User Story:** As a user, I want `datasets.load` to refuse an unknown name and report a damaged bundled file as a domain error, so that I
never read a file outside the dataset folder.

**Subtasks:**

- [x] Write failing tests in `src/tests/test_datasets.py`:
  - `load('dat./../x')` and `load_tree` with a name outside the list raise `metafopy.errors.NotFoundError` (or the error `load` already
    raises for an unknown name);
  - a stubbed `pandas.read_csv` raising `OSError` surfaces as `metafopy.errors.InternalError`
- [x] In `src/metafopy/datasets/base.py`, check the name against the known datasets before joining the path, in `load` and in `load_tree`
      (`src/metafopy/datasets/trees.py`). Wrap both `pandas.read_csv` calls in `read()` in one handler for
      `(OSError, pandas.errors.ParserError, ValueError)` that raises `InternalError`, and update the `Raises:` sections
- [x] `src/audit/cli.py:103`: make a non-number or an unknown `--recall` value a usage error (`parser.error(...)`) naming the known audits,
      with a test

**Acceptance Criteria:**

- [x] The new tests pass, and every existing dataset test passes unchanged

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/datasets src/audit/cli.py` passes
- [x] `pytest src/tests/test_datasets.py src/tests/test_audit5_datasets.py src/tests/test_audit7_datasets.py src/tests/test_harness_compare.py`
      passes

**Notes:**

- Key files: `src/metafopy/datasets/base.py`, `src/metafopy/datasets/trees.py`, `src/audit/cli.py`

---

## Task 10: Reads that bypass a package's surface

**Status:** done
**Priority:** medium
**Complexity:** small
**Source Reference:** F10 (A17)

**Technical Goal:** names that a package exports are read through the package.

**Subtasks:**

- [x] List the open A17 findings with the report command (the `metafopy`, `effects` and `plot` candidates). Take only their
      submodule-bypass parts; the unconsumed `__all__` names were dismissed by decision D3
- [x] Change these reads to go through the package:
  - `src/audit/valuetree.py`: `metafopy.metadata.MetaData` and `metafopy.effects.reshape.TableArray`;
  - `src/metafopy/models/rma_uni.py`: `metafopy.effects.measures.MEASURES`;
  - `binary.py`, `rma_uni.py` and `sel_result.py`: `metafopy.plot.rma` and `metafopy.plot.selmodel`;
  - `src/metafopy/methods/profile.py:179`.

  Where the deferred import exists to avoid an import cycle, import the package inside the function instead, and keep any `AGENTS.md`
  row or marker the deferred import needs
- [x] Run `python3 -c "import metafopy"` and the changed modules' own tests

**Acceptance Criteria:**

- [x] No read of a name from a submodule remains where the package exports it, and `import metafopy` has no cycle

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes on every changed module
- [x] `pytest src/tests/test_plots.py src/tests/test_selmodel.py src/tests/test_profile_plots.py src/tests/test_harness_compare.py` passes

---

## Task 11: Small design defects

**Status:** done
**Priority:** low
**Complexity:** medium
**Source Reference:** F13 (A5, A7, A8, A9, A10, A14, A22, A25, P3), F19 (S1, A20)

**Technical Goal:** fix the small defects the sampled questions found, each at its named site.

**Subtasks:**

- [x] A5, `forest.py:2020` (`addpoly` with a predict result): document in its docstring that an explicit `pi_se`, `pi_dist` or `pi_ddf`
      wins over the stored value, and make `pi_dist` follow that rule. Use `None` as its default, meaning "from the model"; this changes
      no output for a caller who passes nothing
- [x] A8: build each dataclass with its final fields instead of mutating it after construction, at:
  - `src/audit/generate.py:351`
  - `src/audit/report.py:212`
  - `src/metafopy/models/rma_mv.py:826` and `:894`
- [x] A10: rename `selmodel.py:57` `has_tau2` to `estimates_tau2`, and `rlib.py:1238` `is_text` to `is_nonempty_text`, updating every
      caller
- [x] A14: build the per-generic `by_class` index once per `Universe` (`src/audit/universe.py:55`)
- [x] A22: make the `transf.` branch at `src/audit/rlib.py:73` re-raise `MissingFunctionError` instead of falling through
- [x] A25: make `Step` (`src/audit/steps.py:61`) `frozen=True`
- [x] P3: compute `invcalc(X, w)` once in `rma_ls_fit.py:288-289`, and reuse `sat` at `glmm.py:622`
- [x] A9:
  - rename `src/tests/correctness/cases.py:509` `build_tree_r` to `load_tree_r`;
  - add scoped `# auditme: accept[A9]` markers, with their reasons, for `src/audit/steps.py:74` (an audit step whose job is running R)
    and `src/metafopy/core/rng.py:263` (R seeds from the clock on first use too);
  - for `src/audit/pycall.py:65`, add a marker saying it reads a bundled dataset only for a `$data` tag
- [x] A7: add a scoped `# auditme: accept[A7]` marker at `mv_lik.py:128`. `Factorization` exists to factorise once
- [x] S1: annotate the return of `src/tests/test_harness_scripts.py:40` `stand_in`
- [x] A20: wrap the 11 test-module dicts in `types.MappingProxyType`:
  - `glmm_cases.py:44`, `:46`, `:48`, `:50`, `:58`
  - `test_rma_mv.py:80`
  - `glmm_extras_cases.py:31`, `:103`
  - `test_rma_ls.py:99`
  - `test_nchg.py:18`
  - `correctness/test_properties.py:203`

  Copy where a reader mutates one
- [x] Run the own tests of every changed module

**Acceptance Criteria:**

- [x] Each named site is fixed or carries the scoped marker above, and no output changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes on every changed module
- [x] The own test files of every changed module pass, named in the commit body

**Notes:**

- A rename that reaches a public name in `metafopy.__all__` stops as `blocked`. These are not public, so none should

---

## Task 12: Project rules: RNG, links, protocol names and the R-list results

**Status:** done
**Priority:** medium
**Complexity:** medium
**Source Reference:** F18 (S4)

**Technical Goal:** the project rules in `claude/rules/metafopy.md` and `AGENTS.md` hold at every site S4 found.

**Subtasks:**

- [x] `src/metafopy/plot/selmodel.py:179`: draw the `ci='boot'` bands with the R-exact `metafopy.methods.simulate.mvrnorm` (which takes a `metafopy.core.rng.RRandom`) instead of
      `numpy.random.default_rng().multivariate_normal`, seeded as the other R-exact procedures are. The review approved this output change
      (F18, 2026-11-21); compare against an R `plot(selmodel, ci="boot")` fixture if one exists, and otherwise test that a fixed seed
      reproduces
- [x] `src/metafopy/methods/emm_adjust.py:143-159`: add a comment and a line in `docs/differences.md` saying that the `mvt` adjustment
      seeds SciPy's integrator with `MVT_SEED`, which cannot match R's `mvtnorm` draw for draw
- [x] Add scoped `# auditme: accept[S4]` markers at:
  - the protocol-named methods: `_repr_html_` at `printing.py:229` and `rma_uni.py:313` (IPython), `_constructor` at `labelled.py:328`
    and `:388` (pandas);
  - the `list`-subclass results (`ListAnovaResult`, `ConfintList`, `ProfileList`): R returns lists here and the result must index like one
- [x] Add the R source or paper link to the module docstring of each of these modules (each is named in the S4 block of the review):
  - `core/bfgs.py`, `core/bvn.py`, `core/farebrother.py`, `core/optim.py`, `core/rng.py`, `core/zeroin.py`
  - `effects/rtet.py`, `metadata.py`
  - `models/rma_ls.py`, `models/rma_ls_opt.py`, `models/rma_ls_result.py`, `models/sel_result.py`, `models/tau2.py`,
    `models/robust_print.py`
  - `plot/histplot.py`, `plot/setlab_table.py`

  Use the URL forms the other modules use
- [x] Run the tests below

**Acceptance Criteria:**

- [x] No `numpy.random` remains in a ported procedure apart from the documented `mvt` seeding, and every listed module links its source

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/plot/selmodel.py src/metafopy/methods/emm_adjust.py` passes
- [x] `pytest src/tests/test_selmodel.py src/tests/test_docstring_links.py src/tests/test_emmeans_inference.py` passes

---

## Task 13: Rescanning loops

**Status:** done
**Priority:** low
**Complexity:** small
**Source Reference:** F16 (P1)

**Technical Goal:** group rows once instead of masking the whole array for every level.

**Subtasks:**

- [x] Replace the per-level full-array masks with one grouping (`numpy.unique(..., return_inverse=True)` or a `groupby`), keeping the order
      of the results, at:
  - `src/metafopy/methods/robust.py:400`
  - `src/metafopy/methods/rcalc.py:272`
  - `src/metafopy/effects/reshape.py:752` and `:772`
- [x] Run the parity tests for the three modules; every value must stay identical to the fixtures

**Acceptance Criteria:**

- [x] The parity tests pass with no tolerance change

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/methods/robust.py src/metafopy/methods/rcalc.py src/metafopy/effects/reshape.py` passes
- [x] `pytest src/tests/test_robust.py src/tests/test_robust_extras.py src/tests/test_audit3_robust.py src/tests/test_audit4_robust.py src/tests/test_reshape.py`
      and the `rcalc` tests (`grep -l rcalc src/tests/test_*.py`) pass

---

## Task 14: Duplicated construction of result objects, and same-value slots

**Status:** done
**Priority:** low
**Complexity:** large
**Source Reference:** F14 (A15), F15 (A13)

**Technical Goal:** each result class with duplicated assembly gets one helper, so a new field cannot leave a copy behind.

**Subtasks:**

- [x] For each of the 17 A15 sites in the review's A15 block, factor the shared assembly into one helper beside the class, or a
      `classmethod`, and call it from every site. The sites:
  - `glmm_fit.ModelFit`
  - `binary.CumulResult` and `Leave1outResult` (with `leave1out.py`)
  - `predict.LocationModel` and `PredictResult`
  - `selmodel.LRTResult` (with `anova.py:909`)
  - `ls_anova.MVLRTResult`
  - `ls_resid.RanefResult`
  - `vif_mv.VIFEntry`
  - `robust.RobustInfo`
  - `anova.WaldResult`
  - `forest.PredGeometry`
  - `trimfill`/`regtest` `MetaData`
  - `glmm_fit.GLMMProblem`
  - `glmm.GLMMFit`
  - the all-NaN `FitStats`
  - `formulas.MVTests`
- [x] F15 (A13): at each of the 12 A13 sites, add a docstring line to the field saying when it equals its sibling, for example "`parms`
      equals `p` when no coefficient is fixed". No code change
- [x] Run the own tests of every changed module; no value may change

**Acceptance Criteria:**

- [x] Each duplicated assembly has one owner, every caller uses it, and every test passes unchanged

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes on every changed module
- [x] The own test files of every changed module pass, named in the commit body

**Notes:**

- Large, but mechanical. If it outgrows one session, commit each class's helper separately within the task

---

## Task 15: The full check, the summary and closing

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** all; auditme step 8b

**Technical Goal:** the whole branch passes every check once, and the audit summary records what was fixed.

**Subtasks:**

- [x] Run `python3 ~/.claude/skills/auditme/checks/audit.py . > <scratch>/audit.jsonl`. Every hit must be accepted; an unaccepted hit
      that this list introduced is fixed here
- [x] In `agents/audit-reviews/summary.md`, move every group this list fixed (F1-F10, F13-F20) from "Unresolved findings" to "Recently
      resolved" with "Resolved in: tasks-1 (date)"; F11 and F12 stay, pointing at their `TODO.md` items
- [x] Mark each task above `done` only if its commit is on the branch
- [x] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/audit-reviews/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/audit-reviews/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] `summary.md` lists each fixed group as resolved, and the final run passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes

**Notes:**

- Depends on Tasks 1-14
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
- No `TODO.md` item moves in this list: D4 and D5's items are open work for later specs
