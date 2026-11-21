# Audit Review 1

**Date:** 2026-11-21
**Scope:** full (`src/metafopy`, `src/audit`, `src/tests`, `bin`, `agents/specs`, `docs`, `AGENTS.md`, `README.md`)
**Commit:** `31d809f`
**Script output:** /tmp/audit-metafopy.jsonl (1265 hits, 1265 accepted, 0 unaccepted)
**Convergence note:** first auditme review of this project: there is no review-0 to compare with. Every listed question (A1, A4, A11, A17, A19, T1, T2, Doc1, Doc2, Doc4, Doc5) judged every candidate as new, and its findings in unchanged code are catch-up, not regressions. Every sampled question ran on the whole tree. Checker digest `d7c6456b5bcd470af77894a99cae6cbcd5f5aa5e4e7e86ad0db82d6c6493de5b`; `requests.md` had no open entry at the start.
**Review page:** https://claude.ai/artifact/XWmcd8XMA6kdWdv8F4Wtd8

**What is being reviewed and how.** The first full audit of metafopy with the auditme checklist, run at `31d809f`. The mechanical sweep is clean: every one of its hits already carries an accept marker. Phase B judged 3518 listed candidates through 109 reviewer batches, and ran each sampled question on the whole tree. For each group of findings below, approve the recommended action or flag it, and answer the decisions. Once answered, the approved groups become a task list (`/taskdev agents/audit-reviews/review-1.md`) or `TODO.md` items as each group says.

## Summary

The sweep reports 1265 hits, all accepted, so there are no mechanical findings. Phase B leaves 611 open findings on the listed questions and 134 on the sampled ones, plus 5 spec-template gaps. Most of the volume sits in four places:
- T1: 279 functions with branches and no direct test.
- T2: 144 pure functions with no property test.
- A1: 61 file and subprocess calls in the audit harness (`src/audit`, `bin`) that let `OSError` escape untranslated.
- Doc4 and Doc5: 59 places where specs or user docs no longer match the code.

The findings most worth fixing first are these:
- Example output in the user guides that the code no longer prints (Doc5).
- Domain errors that reach callers without being declared in `Raises:` (A4).
- Two `assert` statements that are the only check on a user argument (A12).
- README instructions that contradict `AGENTS.md` (Doc3).
- `nbformat` imported by a test but not declared (D1).

## Automatic checks

- Pass: `requests.md` `## Open` was empty when the audit started.
- Pass: `checks/audit.py` ran on every root (`src/audit`, `src/metafopy`, `src/tests`, `bin`): 1265 hits, 0 unaccepted.
- Pass: `phaseb.py check` reports "all batches complete" for every listed question; every candidate has a verdict in `agents/audit-reviews/ledger/`.
- Pass: no check was added to `checks/auditme/rules/`, and no predicate's enumeration was widened, during the audit.
- Note: there was no `summary.md` and no `agents/tasks/audit-reviews/` before this review, so there were no prior findings to re-verify.

## Thresholds settled during the run

Several reviewers reported thresholds that their brief left open, and batches judged them differently. To keep a single threshold per question, I settled each one across all batches as follows. Every ruling is reversible; each verdict it changed says so in the ledger's `reason`.

- **T1, what counts as a direct test.**
  - A call through a package re-export of the same object counts (`metafopy.emmeans` is `metafopy.methods.emm_grid.emmeans`), and importing the top-level package counts as importing the module.
  - So does a call through a `functools.singledispatch` generic, a dispatch table or a parameter that always reaches the candidate.
  - A wrapper that is a different function does not count.
  - A dunder counts when the test runs it itself, by constructing, printing, indexing or reading an attribute. A one-line `__str__` returning `self.summary()` makes `str(x)` a call of `summary`. A `__post_init__` that only package code triggers does not count.
  - Calls on an instance, and calls from helpers inside the test file, count.
  - Tests that expect an error count; skipped and xfail tests do not.
  - Every flagged T1 verdict (359) was judged again under these rulings: 84 became covered. Batch 12 was already judged under them.
- **A4, invariant guards.** A declared `InternalError` that a same-package callee raises as an invariant guard counts as raised. The project's accept markers at `src/metafopy/models/mv_lik.py:312`, `mv_opt.py:46` and `rma_mv.py:442` already treat these guards that way. This cleared three "extra `InternalError`" verdicts: `glm_fit`, `glmm_fit.single_level` and `glmm_fit.fit_model`.
- **A4, `singledispatch` generics.** Batches disagreed on this, and it is a real choice, so it is decision D1 below. Until you answer, each generic is judged on its own body only, which is how most batches judged it.
- **Doc4 and Doc5, dated sentences and duplicates.**
  - A dated sentence inside an undated section counts as a dated record.
  - A statement that falls under two candidates is reported under both. The consolidated findings below count it once.
- **S4, plot result classes.** Metafor prints nothing for its plot results. The `__str__` rule asks for a *metafor-style* `__str__`, so the 15 plot result classes without one are not findings and are not listed below.

## Mechanical findings (from checks/audit.py)

None: all 1265 hits are accepted, by inline `# auditme: accept[...]` markers or `AGENTS.md` rows, each with its reason in the JSONL.

## Phase B per-question results

Listed questions: every candidate judged. Sampled questions: mode **full** (no previous `**Commit:**`).

### A1 (listed): External call envelope

- **Candidates judged:** 63 (every candidate; none left unjudged)
- **Open findings:** 61

- **Hits:**
  - `src/audit/cases.py::load_fixtures::call:(audit.config.DATA_DIR / 'fixtures.json').read_text#1` - call (audit.config.DATA_DIR / 'fixtures.json').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/cases.py::write_cases::call:path.write_text#1` - call path.write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/cli.py::main::call:out.mkdir#1` - call out.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/cli.py::recall::call:out.mkdir#1` - call out.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/domains.py::Domains.load::call:(audit.config.DATA_DIR / 'domains.toml').read_text#1` - call (audit.config.DATA_DIR / 'domains.toml').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/domains.py::Recipes.load::call:(audit.config.DATA_DIR / 'recipes.toml').read_text#1` - call (audit.config.DATA_DIR / 'recipes.toml').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::PyLauncher.start::call:self.job_path.write_text#1` - call self.job_path.write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::PyLauncher.start::call:subprocess.Popen#1` - call subprocess.Popen can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::read_outcomes::call:path.read_text#1` - call path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::resident_bytes::call:subprocess.run#1` - call subprocess.run (ps) is caught as OSError and returns None, which is a default, not translation; raise a domain *Error or accept the documented fallback.
  - `src/audit/engine.py::run_cases::call:(work / f'py-{i}.json').read_text#1` - call (work / f'py-{i}.json').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::run_cases::call:fixtures_path.write_text#1` - call fixtures_path.write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::run_cases::call:work.mkdir#1` - call work.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::run_py_shard::call:(work / f'py-{i}.json').write_text#1` - call (work / f'py-{i}.json').write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::run_py_shard::call:r_path.read_text#1` - call r_path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/engine.py::run_r_shard::call:(work / f'r-{i}.json').write_text#1` - call (work / f'r-{i}.json').write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/ledger.py::Ledger.load::call:path.read_text#1` - call path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pycall.py::aliases::call:(audit.config.DATA_DIR / 'domains.toml').read_text#1` - call (audit.config.DATA_DIR / 'domains.toml').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pyrun.py::<module>::call:pathlib.Path(sys.argv[1]).read_text#1` - call pathlib.Path(sys.argv[1]).read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pyrun.py::load_shard::call:cases_path.read_text#1` - call cases_path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pyrun.py::load_shard::call:fixtures_path.read_text#1` - call fixtures_path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pyrun.py::load_shard::call:r_path.read_text#1` - call r_path.read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/pyrun.py::run_guarded::call:open#1` - call open() of the trace file sits in the except BaseException handler with no try of its own, so an OSError escapes untranslated; wrap it and raise a domain *Error (or catch OSError).
  - `src/audit/rdriver.py::run::call:subprocess.run#1` - call subprocess.run can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/rdriver.py::start::call:subprocess.Popen#1` - call subprocess.Popen can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::Corpus.load::call:(audit.config.DATA_DIR / 'recall.toml').read_text#1` - call (audit.config.DATA_DIR / 'recall.toml').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::checkout::call:path.parent.mkdir#1` - call path.parent.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::checkout::call:subprocess.run#1` - call subprocess.run can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::run_audit::call:(run_dir / 'findings.json').read_text#1` - call (run_dir / 'findings.json').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::run_audit::call:subprocess.run#1` - call subprocess.run can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/recall.py::write::call:path.write_text#1` - call path.write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/report.py::git_head::call:subprocess.run#1` - call subprocess.run (git rev-parse) is caught as OSError/CalledProcessError and returns 'unknown', a default, not translation; raise a domain *Error or accept the fallback.
  - `src/audit/report.py::rerender::call:(directory / 'findings.json').read_text#1` - call (directory / 'findings.json').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/report.py::write::call:(out / 'findings.json').write_text#1` - call (out / 'findings.json').write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/report.py::write::call:(out / 'wrong-or-missing-full.md').write_text#1` - call (out / 'wrong-or-missing-full.md').write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/report.py::write::call:path.write_text#1` - call path.write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::<module>::call:pathlib.Path(sys.argv[1]).read_text#1` - call pathlib.Path(sys.argv[1]).read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::fetch_source::call:(audit.config.DATA_DIR / 'metafor-source.sha256').read_text#1` - call (audit.config.DATA_DIR / 'metafor-source.sha256').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::fetch_source::call:archive.write_bytes#1` - call archive.write_bytes can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::fetch_source::call:target.mkdir#1` - call target.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::fetch_source::call:urllib.request.urlopen#1` - call urlopen OSError is collected and, after every URL fails, re-raised as RuntimeError, which is not one of the domain errors; raise a domain *Error (e.g. a new fetch error in the harness) instead.
  - `src/audit/scripts.py::list_scripts::call:out_path.read_text#1` - call read_text OSError is translated to RuntimeError, which is not one of the domain errors; raise a domain *Error instead.
  - `src/audit/scripts.py::list_scripts::call:out_path.unlink#1` - call out_path.unlink can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::list_scripts::call:subprocess.run#1` - call subprocess.run can raise OSError or subprocess.SubprocessError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::run::call:(out / f'py-{i}.json').read_text#1` - call (out / f'py-{i}.json').read_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::run::call:out.mkdir#1` - call out.mkdir can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::run_py_scripts::call:(work / f'py-{i}.json').write_text#1` - call (work / f'py-{i}.json').write_text can raise OSError with no try or envelope translating it; wrap the call and raise one of the domain *Error classes.
  - `src/audit/scripts.py::run_py_scripts::call:r_path.read_text#1` - call r_path.read_text in run_py_scripts (src/audit/scripts.py:241) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/scripts.py::run_r_scripts::call:(work / f'r-{i}.json').write_text#1` - call (work / f'r-{i}.json').write_text in run_r_scripts (src/audit/scripts.py:234) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/scripts.py::stream_scripts::call:r_path.read_text#1` - call r_path.read_text in stream_scripts (src/audit/scripts.py:167) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/scripts.py::stream_scripts::call:scripts_path.read_text#1` - call scripts_path.read_text in stream_scripts (src/audit/scripts.py:168) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/scripts.py::write_scripts::call:path.write_text#1` - call path.write_text in write_scripts (src/audit/scripts.py:213) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::build_universe::call:path.read_text#1` - call path.read_text in build_universe (src/audit/steps.py:128) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::build_universe::call:path.read_text#2` - call path.read_text in build_universe (src/audit/steps.py:129) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::build_universe::call:snapshot.read_text#1` - call snapshot.read_text in build_universe (src/audit/steps.py:129) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::build_universe::call:snapshot.write_text#1` - call snapshot.write_text in build_universe (src/audit/steps.py:128) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::check_r::call:(ctx.work / 'r-versions.json').read_text#1` - call (ctx.work / 'r-versions.json').read_text in check_r (src/audit/steps.py:81) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/steps.py::check_r::call:lock_path.read_text#1` - call lock_path.read_text in check_r (src/audit/steps.py:91) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/audit/universe.py::Universe.load::call:path.read_text#1` - call path.read_text in Universe.load (src/audit/universe.py:19) has no try/except, so an OSError escapes untranslated; wrap it in an OSError handler that raises a domain error naming the file and the step that wrote or reads it
  - `src/metafopy/datasets/base.py::read::call:pandas.read_csv#1` - call pandas.read_csv on the bundled CSV in read() sits in no try, and neither caller (load at src/metafopy/datasets/base.py:91, load_tree at src/metafopy/datasets/trees.py:89) has an envelope, so OSError or pandas.errors.ParserError escapes; wrap the reads in a handler for (OSError, pandas.errors.ParserError, ValueError) that raises metafopy.errors.InternalError (a corrupt bundled file is a packaging fault)
  - `src/metafopy/datasets/base.py::read::call:pandas.read_csv#2` - call pandas.read_csv on the bundled CSV in read() sits in no try, and neither caller (load at src/metafopy/datasets/base.py:91, load_tree at src/metafopy/datasets/trees.py:89) has an envelope, so OSError or pandas.errors.ParserError escapes; wrap the reads in a handler for (OSError, pandas.errors.ParserError, ValueError) that raises metafopy.errors.InternalError (a corrupt bundled file is a packaging fault)

### A4 (listed): Docstring `Raises` accuracy

- **Candidates judged:** 475 (every candidate; none left unjudged)
- **Open findings:** 26

- **Hits:**
  - `src/metafopy/effects/conv.py::conv_2x2` - missing ArgumentError
  - `src/metafopy/effects/conv.py::conv_delta` - missing ArgumentError
  - `src/metafopy/effects/conv.py::conv_wald` - missing ArgumentError
  - `src/metafopy/effects/fivenum.py::conv_fivenum` - missing ArgumentError, NotFoundError
  - `src/metafopy/effects/measures.py::escalc` - missing NotFoundError
  - `src/metafopy/effects/reshape.py::collect` - missing ArgumentError, NotFoundError
  - `src/metafopy/effects/reshape.py::to_long` - missing ArgumentError, NotFoundError
  - `src/metafopy/effects/reshape.py::to_table` - missing ArgumentError, NotFoundError
  - `src/metafopy/methods/anova.py::wald_between` - missing InternalError
  - `src/metafopy/methods/fitted.py::fitted_values` - missing ArgumentError
  - `src/metafopy/methods/hettest.py::hettest` - extra ArgumentError
  - `src/metafopy/methods/ls_confint.py::confint_rma_ls` - missing InternalError
  - `src/metafopy/methods/ls_profile.py::profile_rma_ls` - missing InternalError
  - `src/metafopy/methods/ls_vif.py::part_vif` - missing InternalError
  - `src/metafopy/methods/profile.py::profile` - extra InputError
  - `src/metafopy/methods/rcalc.py::rcalc_list` - missing ArgumentError
  - `src/metafopy/methods/robust.py::robust` - extra InputError
  - `src/metafopy/models/call.py::records_call` - extra InternalError
  - `src/metafopy/models/rma_glmm.py::build_design` - extra InputError
  - `src/metafopy/models/rma_ls_fit.py::minimise_standin` - extra InputError
  - `src/metafopy/models/rma_mh.py::rma_mh` - missing NotFoundError
  - `src/metafopy/models/rma_peto.py::rma_peto` - missing NotFoundError
  - `src/metafopy/models/tables.py::prepare_tables` - missing NotFoundError
  - `src/metafopy/plot/forest.py::forest` - missing ArgumentError
  - `src/metafopy/plot/funnel.py::funnel` - missing ArgumentError
  - `src/metafopy/plot/labbe.py::labbe` - missing ArgumentError

### A11 (listed): Exhaustion via assert_never

- **Candidates judged:** 178 (every candidate; none left unjudged)
- **Open findings:** 36

- **Hits:**
  - `src/metafopy/bias/regtest.py::regtest::if:model#1` - unexhausted lm
  - `src/metafopy/bias/regtest.py::regtest::if:predictor#1` - unexhausted ni, ninv, sqrtni, sqrtninv
  - `src/metafopy/bias/tes.py::tes::if:alternative#1` - unexhausted less
  - `src/metafopy/bias/tes.py::tes::if:alternative#2` - unexhausted less
  - `src/metafopy/bias/tes.py::tes::if:tes_alternative#1` - unexhausted less
  - `src/metafopy/bias/tes.py::tes::if:tes_alternative#2` - unexhausted less
  - `src/metafopy/bias/tes.py::tes::if:tes_alternative#3` - unexhausted greater, less
  - `src/metafopy/bias/tes.py::tes::if:tes_alternative#4` - unexhausted two.sided
  - `src/metafopy/bias/tes.py::tes::if:test#1` - unexhausted exact
  - `src/metafopy/bias/trimfill.py::trimfill::if:estimator#1` - unexhausted L0, Q0
  - `src/metafopy/bias/trimfill.py::trimfill::if:estimator#2` - unexhausted Q0
  - `src/metafopy/bias/trimfill.py::trimfill::if:estimator#3` - unexhausted Q0
  - `src/metafopy/core/glm.py::glm_fit::if:family#1` - unexhausted poisson
  - `src/metafopy/core/glmm.py::general_row_terms::if:prob.family#1` - unexhausted poisson
  - `src/metafopy/core/glmm.py::normalise_theta::if:structure#1` - unexhausted independent, scalar
  - `src/metafopy/interactive/forest.py::forest_chart::if:order#1` - unexhausted input
  - `src/metafopy/methods/influence.py::influence_fit::if:na_action#1` - unexhausted 'fail', 'omit'
  - `src/metafopy/methods/influence.py::rstandard_rma_uni::if:type#1` - unexhausted 'conditional'
  - `src/metafopy/methods/influence.py::rstudent_rma_uni::if:na_action#1` - unexhausted 'fail', 'omit'
  - `src/metafopy/methods/ls_confint.py::confint_rma_ls::if:x.test#1` - unexhausted 'z'
  - `src/metafopy/methods/ls_predict.py::predict_rma_ls::if:x.test#1` - unexhausted 'z'
  - `src/metafopy/methods/ls_predict.py::predict_rma_ls::if:x.test#2` - unexhausted 'z'
  - `src/metafopy/methods/ls_resid.py::rstandard_rma_ls::if:type#1` - unexhausted 'conditional'
  - `src/metafopy/methods/matreg.py::MatregResult.summary::if:self.test#1` - unexhausted 'z'
  - `src/metafopy/methods/matreg.py::confint_matreg::if:model.test#1` - unexhausted 'z'
  - `src/metafopy/methods/matreg.py::predict_matreg::if:x.test#1` - unexhausted 'z'
  - `src/metafopy/models/call.py::records_call.wrapper::if:kind#1` - unexhausted inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
  - `src/metafopy/models/rma_ls_fit.py::fit_rma_ls::if:test#1` - unexhausted 'z'
  - `src/metafopy/models/rma_ls_fit.py::fit_rma_ls::if:test#2` - unexhausted 'z'
  - `src/metafopy/models/sel_lik.py::evaluate::if:spec.kind#1` - unexhausted trunc
  - `src/metafopy/plot/forest.py::addpoly::if:ps#1` - unexhausted bar, dist, polygon, shade
  - `src/metafopy/plot/forest.py::addpoly::if:ps#2` - unexhausted bar, dist, polygon, shade
  - `src/metafopy/plot/forest.py::addpoly::if:ps#3` - unexhausted bar, dist, polygon, shade
  - `src/metafopy/plot/forest.py::arrow::if:side#1` - unexhausted r
  - `src/metafopy/plot/rma.py::plot_rma::if:na_action#1` - unexhausted fail, omit
  - `src/tests/test_matreg.py::check_fit::if:res.test#1` - unexhausted 't' and 'z' as explicit arms, with assert_never(res.test) in the final else

### A17 (listed): Public-surface drift

- **Candidates judged:** 8 (every candidate; none left unjudged)
- **Open findings:** 5

- **Hits:**
  - `src/metafopy/__init__.py` - surface Names in __all__ with no consumer outside the package and the test tree, and no literal getattr, importlib.import_module or entry-point use (the audit harness getattr(metafopy, spelling) in src/audit/pycall.py:299 and src/audit/apidiff.py:57 is non-literal): BlupResult, ConfintResult, Control, CumulResult, DeltaMethodResult, EmmGrid, EmmSummary, FitStats, FsnResult, GLMMControl, GLMMResult, GoshResult, HcResult, HettestResult, InfluenceResult, LRTResult, LSControl, Leave1outResult, MHResult, MVControl, MatregConfintResult, MatregPredictResult, MatregResult, PermutestResult, PetoResult, PredictResult, ProfileResult, RCalcResult, RMALSResult, RMAMVResult, RMAResult, RandomTerm, RanefResult, RanktestResult, RegtestResult, ResidualResult, SelProfileResult, SelmodelResult, TesResult, TrimfillResult, VIFEntry, VIFResult, WaldResult, __version__, addpoly, aggregate_escalc, anova, baujat, bldiag, blsplit, blup, cbind_escalc, cmicalc, confint, contrast, contrmat, conv_2x2, conv_delta, conv_fivenum, conv_wald, cooks_distance, cumul, deltamethod, df_residual, dfbetas, dfround, emmeans, emmprep, escalc, fmtp, fmtp2, fmtt, fmtx, forest, forest_cumul, fsn, funnel, getmfopt, gosh, hc, hettest, influence, labbe, leave1out, llplot, matreg, model_matrix, nobs, pairmat, pairs, parse_random, plot_cumul, plot_gosh, plot_infl, plot_permutest, plot_profile, plot_rma, plot_selmodel, plot_vif, points_regplot, predict, profile, qqnorm, radial, ranef, ranktest, rbind_escalc, rcalc, regplot, regrid, regtest, replmiss, residuals, rma_mh, rma_mv, rma_peto, robust, rstandard, rstudent, se, selmodel, setmfopt, simulate, tes, to_long, to_table, to_wide, transf_abt, transf_ahw, transf_arcsin, transf_clestod, transf_dtobesd, transf_dtocles, transf_dtocles_int, transf_dtocliffd, transf_dtolnor_logis, transf_dtolnor_norm, transf_dtomd, transf_dtoovl, transf_dtorbis, transf_dtorpb, transf_dtou1, transf_dtou2, transf_dtou3, transf_exp_int, transf_exp_mode, transf_iabt, transf_iabt_int, transf_iabt_mode, transf_iahw, transf_iahw_int, transf_iahw_mode, transf_iarcsin, transf_iarcsin_int, transf_iarcsin_mode, transf_iirft, transf_ilogit, transf_ilogit_int, transf_ilogit_mode, transf_ipft, transf_ipft_hm, transf_iprobit, transf_iprobit_int, transf_iprobit_mode, transf_irft, transf_isqrt, transf_lnortod_logis, transf_lnortod_norm, transf_lnortord, transf_lnortorr, transf_lnortortet_digby, transf_lnortortet_pearson, transf_logit, transf_pft, transf_probit, transf_r2toz, transf_rpbtod, transf_rpbtorbis, transf_rtod, transf_rtorpb, transf_rtoz, transf_ztor, transf_ztor2, transf_ztor_int, transf_ztor_mode, trimfill, update, vcalc, vcov, vec2mat. Either record that the top-level __all__ is the public API for end users (an accept marker) or drop the names nothing needs., src/audit/valuetree.py:127 reads metafopy.metadata.MetaData from its submodule although metafopy/__init__.py exports it; use metafopy.MetaData., src/audit/valuetree.py:207 reads metafopy.effects.reshape.TableArray from its submodule although metafopy/__init__.py (and metafopy.effects) export it; use metafopy.TableArray.
  - `src/metafopy/datasets/__init__.py` - surface PhyloTree and load_tree are in __all__ but nothing outside metafopy.datasets and the test tree reads them, and no literal getattr, import_module or entry point does (only the load docstring in src/metafopy/datasets/base.py:44 mentions load_tree); re-export them from metafopy/__init__.py if they are public, or drop them from __all__.
  - `src/metafopy/effects/__init__.py` - surface src/audit/valuetree.py:207 reads metafopy.effects.reshape.TableArray from its submodule although metafopy.effects exports it; use metafopy.TableArray or metafopy.effects.TableArray., src/metafopy/models/rma_uni.py reads metafopy.effects.measures.MEASURES (import metafopy.effects.measures at line 27) although metafopy.effects exports MEASURES; use metafopy.effects.MEASURES, or mark the submodule import with an accept marker if it avoids an import cycle.
  - `src/metafopy/interactive/__init__.py` - surface ChartData, chart_data, forest_chart and funnel_chart have no consumer outside metafopy.interactive and the test tree, and no literal getattr, import_module or entry point reads them; the package is an opt-in public surface (docstring: import metafopy.interactive explicitly), so either record that with an accept marker or drop the names nothing needs.
  - `src/metafopy/plot/__init__.py` - surface The result classes BaujatResult, ForestResult, FunnelResult, LabbeResult, LlplotResult, PlotCumulResult, PlotGoshResult, PlotInflResult, PlotPermutestResult, PlotRmaResult, PlotVifResult, QQNormResult, RadialResult, RegplotResult, SelPlotResult and the alias galbraith are in __all__ with no consumer outside the package and the test tree and no literal dynamic use; galbraith is also not re-exported by metafopy/__init__.py. Re-export or drop them., src/metafopy/methods/profile.py:179 calls importlib.import_module('metafopy.plot.profile').plot_profile although metafopy.plot exports plot_profile (line 405 of the same file already uses importlib.import_module('metafopy.plot').plot_profile); use the package., src/metafopy/models/binary.py:222-224 and src/metafopy/models/rma_uni.py read metafopy.plot.rma.plot_rma from its submodule although metafopy.plot exports it; the deferred import can be import metafopy.plot and metafopy.plot.plot_rma., src/metafopy/models/sel_result.py reads metafopy.plot.selmodel.plot_selmodel from its submodule although metafopy.plot exports it; use metafopy.plot.plot_selmodel.

### A19 (listed): Wrong logger level for broad-except

- **Candidates judged:** 6 (every candidate; none left unjudged)
- **Open findings:** 0

- **Hits:** none

### T1 (listed): Test coverage of branchy logic

- **Candidates judged:** 1227 (every candidate; none left unjudged)
- **Open findings:** 279

- **Hits:**
  - `bin/logo.py::animated_gif` - missing a direct test of animated_gif that exercises its branches (if)
  - `bin/logo.py::simulate` - missing a direct test of simulate that exercises its branches (if)
  - `bin/logo.py::smoothstep` - missing a direct test of smoothstep that exercises its branches (if)
  - `bin/metafor-audit.py::metafopy_source` - missing a direct test of metafopy_source that exercises its branches (if)
  - `bin/platform-probe.py::case_by_id` - missing a direct test of case_by_id that exercises its branches (if)
  - `bin/platform-probe.py::environment` - missing a direct test of environment that exercises its branches (if)
  - `bin/platform-probe.py::ls_runs` - missing a direct test of ls_runs that exercises its branches (if, try)
  - `bin/platform-probe.py::measure_selmodel_decreasing` - missing a direct test of measure_selmodel_decreasing that exercises its branches (try)
  - `bin/platform-probe.py::score_norm` - missing a direct test of score_norm that exercises its branches (try)
  - `src/audit/apidiff.py::callables_for` - missing a direct test of callables_for that exercises its branches (if)
  - `src/audit/apidiff.py::comparable_default` - missing a direct test of comparable_default that exercises its branches (if, try)
  - `src/audit/apidiff.py::compare_function` - missing a direct test of compare_function that exercises its branches (if)
  - `src/audit/apidiff.py::parameters` - missing a direct test of parameters that exercises its branches (if, try)
  - `src/audit/apidiff.py::python_defaults` - missing a direct test of python_defaults that exercises its branches (try)
  - `src/audit/apidiff.py::run` - missing a direct test of run that exercises its branches (if)
  - `src/audit/apidiff.py::source_mentions` - missing a direct test of source_mentions that exercises its branches (if, try)
  - `src/audit/apidiff.py::spellings_for` - missing a direct test of spellings_for that exercises its branches (if)
  - `src/audit/cli.py::main` - missing a direct test of main that exercises its branches (if)
  - `src/audit/domains.py::refs_in` - missing a direct test of refs_in that exercises its branches (if)
  - `src/audit/engine.py::py_stop` - missing a direct test of py_stop that exercises its branches (if)
  - `src/audit/engine.py::r_stop_reason` - missing a direct test of r_stop_reason that exercises its branches (if)
  - `src/audit/generate.py::args_of` - missing a direct test of args_of that exercises its branches (if)
  - `src/audit/generate.py::default_string` - missing a direct test of default_string that exercises its branches (if)
  - `src/audit/generate.py::option_cases` - missing a direct test of option_cases that exercises its branches (if)
  - `src/audit/generate.py::option_values` - missing a direct test of option_values that exercises its branches (if)
  - `src/audit/generate.py::sweep` - missing a direct test of sweep that exercises its branches (if)
  - `src/audit/generate.py::unaccounted_functions` - missing a direct test of unaccounted_functions that exercises its branches (if)
  - `src/audit/ledger.py::Ledger.classify` - missing a direct test of Ledger.classify that exercises its branches (if)
  - `src/audit/printdiff.py::close_numbers` - missing a direct test of close_numbers that exercises its branches (if)
  - `src/audit/printdiff.py::same_token` - missing a direct test of same_token that exercises its branches (if)
  - `src/audit/pycall.py::check_arguments` - missing a direct test of check_arguments that exercises its branches (if, try)
  - `src/audit/pycall.py::coef_for` - missing a direct test of coef_for that exercises its branches (if)
  - `src/audit/pycall.py::getitem` - missing a direct test of getitem that exercises its branches (if, try)
  - `src/audit/pycall.py::head` - missing a direct test of head that exercises its branches (if)
  - `src/audit/pycall.py::integral_arguments` - missing a direct test of integral_arguments that exercises its branches (if, try)
  - `src/audit/pycall.py::match_positional` - missing a direct test of match_positional that exercises its branches (if, try)
  - `src/audit/pycall.py::option_value` - missing a direct test of option_value that exercises its branches (if)
  - `src/audit/pycall.py::print_call` - missing print() with arguments: passed to summary(), the digits branches, and MissingArgumentError for an argument summary() lacks
  - `src/audit/pycall.py::shift_index` - missing 1-based to 0-based shift of an int, a list, and a bool left unchanged
  - `src/audit/pycall.py::summary_for` - missing a data frame giving summary_escalc, a result without summary raising MissingFunctionError, and the SummaryText wrapping
  - `src/audit/pycall.py::tail` - missing a result with __getitem__ and to_frame cut with x[-n:], and a plain frame through frame_of
  - `src/audit/pyrun.py::case_result` - missing R giving no result (runner-error), and classify raising (runner-error with its message)
  - `src/audit/pyrun.py::drawn_text` - missing a case that draws text and one that draws none
  - `src/audit/pyrun.py::execute` - missing a case whose needed fixture failed giving a FixtureError record, and a case run under run_guarded
  - `src/audit/pyrun.py::readable` - missing a plotmath p-value range, a plain() expression, and other text returned unchanged
  - `src/audit/pyrun.py::run_guarded` - missing a guarded call that returns, one that raises, and one that times out
  - `src/audit/rdriver.py::cluster_port` - missing a port already given being skipped for a fresh one
  - `src/audit/rdriver.py::run` - missing an Rscript run exiting non-zero raising RuntimeError with its stderr
  - `src/audit/recall.py::checkout` - missing an existing worktree reused and a missing one created with git worktree add
  - `src/audit/recall.py::run_audit` - missing running the audit for an audit number with and without hints
  - `src/audit/report.py::cause_key` - missing missing-arg/missing-function with a path, a finding with a path, a print-diff without a path, and another kind without a path
  - `src/audit/report.py::function_of` - missing a construct finding mapped through fixture_function, and an id without a slash
  - `src/audit/report.py::git_head` - missing git rev-parse failing (OSError or CalledProcessError) giving unknown
  - `src/audit/report.py::line_of` - missing a missing-arg/missing-function key, a labelled value-diff key, and more than four functions shown as "and N more"
  - `src/audit/report.py::result_class` - missing matrix/options ids, a construct id, a scripts finding, and the function_of fallback
  - `src/audit/rlib.py::Named.__array_wrap__` - missing a ufunc on a Named matrix giving a scalar, a non-2-D result, and a 2-D result keeping its names
  - `src/audit/rlib.py::accepts` - missing a plain function that takes the argument, a singledispatch function dispatched on the first argument, and a callable without a signature
  - `src/audit/rlib.py::as_array` - missing a Series or DataFrame converted with to_numpy and a plain value through numpy.asarray
  - `src/audit/rlib.py::format_cell` - missing a whole float printed without decimals and any other value
  - `src/audit/rlib.py::r_as_dataframe` - missing a data frame, a result with to_frame, an emmeans summary frame, an EmmGrid, and a plain value
  - `src/audit/rlib.py::r_as_matrix` - missing a list.rma result, a data frame with automatic and with named rows, and a vector
  - `src/audit/rlib.py::r_as_numeric` - missing a factor giving its codes from 1, a string array, and a numeric array
  - `src/audit/rlib.py::r_assign_names` - missing names<- on a data frame, a Series, a dict, and another value
  - `src/audit/rlib.py::r_attr` - missing a value without attrs, an attribute found by its R name, one found by its Python name, and one not found
  - `src/audit/rlib.py::r_capture_output` - missing capture.output of a thunk that prints returns its lines
  - `src/audit/rlib.py::r_cbind` - missing cbind of vectors, a matrix and a data frame, with and without deparsed names
  - `src/audit/rlib.py::r_class` - missing class of an escalc data frame, a plain data frame and another value
  - `src/audit/rlib.py::r_cor` - missing cor of a matrix and of two vectors
  - `src/audit/rlib.py::r_dataframe` - missing data.frame from positional vectors, a data frame and named columns, with recycling
  - `src/audit/rlib.py::r_diag` - missing diag of a matrix, a scalar and a vector
  - `src/audit/rlib.py::r_equal` - missing == between a number and text compares them as text
  - `src/audit/rlib.py::r_head` - missing head of a list.rma result, a data frame and a vector
  - `src/audit/rlib.py::r_index` - missing [ on a data frame (one and two subscripts, names, logical, drop), a Series, a dict, a list.rma and a matrix
  - `src/audit/rlib.py::r_index2` - missing [[ on a dict, a data frame, a list and a result object by name and position, with a missing name raising
  - `src/audit/rlib.py::r_inherits` - missing inherits of an escalc data frame and of another object
  - `src/audit/rlib.py::r_isna` - missing is.na of a data frame, a list and an array
  - `src/audit/rlib.py::r_lapply` - missing lapply over a dict, a data frame and a vector
  - `src/audit/rlib.py::r_length` - missing length of a data frame, a list or dict and an array
  - `src/audit/rlib.py::r_list` - missing list with only named, mixed and only positional arguments
  - `src/audit/rlib.py::r_matrix` - missing matrix with neither, one or both of nrow and ncol, and byrow
  - `src/audit/rlib.py::r_names` - missing names of a data frame, a Series, a dict, a fit with coef_names and a value without names
  - `src/audit/rlib.py::r_par` - missing par('usr'), par(name = value) returning the old values, and par() with no settings
  - `src/audit/rlib.py::r_paste` - missing paste with sep and collapse, recycling the shorter argument
  - `src/audit/rlib.py::r_rbind` - missing rbind of data frames, of vectors, and of named vectors giving row names
  - `src/audit/rlib.py::r_rep` - missing rep with each, a vector times, a scalar times and length.out
  - `src/audit/rlib.py::r_seq` - missing seq with along.with, one argument, length.out with and without to, and by (ascending and descending)
  - `src/audit/rlib.py::r_split` - missing split of a vector, a data frame and a matrix by a grouping
  - `src/audit/rlib.py::r_stopifnot` - missing stopifnot with a true and a false condition
  - `src/audit/rlib.py::r_tail` - missing tail of a list.rma result, a data frame and a vector
  - `src/audit/rlib.py::r_var` - missing var of a vector, a matrix and two vectors
  - `src/audit/rlib.py::recycled` - missing a binary operator on a Series and an operand of another length
  - `src/audit/rlib.py::truth` - missing an if condition of length zero raises UnsupportedError
  - `src/audit/scripts.py::fetch_source` - missing an existing source tree, a checksum mismatch, a download error on every URL and a good download
  - `src/audit/scripts.py::run_py_scripts` - missing run_py_scripts on a shard whose Python process dies partway, checking the dead-script record and the py-{i}.json totals
  - `src/audit/steps.py::build_universe` - missing build_universe with and without a fresh R universe, checking the snapshot it loads or writes
  - `src/audit/steps.py::generate_cases` - missing generate_cases with quick set and unset
  - `src/audit/steps.py::scripts_step` - missing scripts_step with scripts skipped and run, checking the outcome it records
  - `src/audit/transpile.py::args_code` - missing args_code on named and unnamed arguments
  - `src/audit/transpile.py::callee_name` - missing callee_name on a symbol callee and a non-symbol callee
  - `src/audit/transpile.py::replacement` - missing replacement on each replacement form (`$<-`, `[<-`, a function replacement)
  - `src/audit/transpile.py::tr` - missing tr on each node type (sym, num, string, NA, NULL, call)
  - `src/audit/transpile.py::tr_assign` - missing tr_assign on a plain, a nested and a superassignment target
  - `src/audit/transpile.py::tr_call` - missing tr_call on each special form it branches on (operators, if, for, function, NSE calls)
  - `src/audit/transpile.py::tr_function` - missing tr_function with and without default arguments
  - `src/audit/transpile.py::tr_sym` - missing tr_sym on a reserved name and an ordinary symbol
  - `src/audit/transpile.py::unparse` - missing unparse on each node type, checking the R text it gives
  - `src/audit/transpile.py::walk` - missing walk on nested call and leaf nodes
  - `src/audit/transpile.py::with_model` - missing with_model on a model call and a non-model call
  - `src/audit/universe.py::Universe.draws` - missing Universe.draws for a function with and without recorded draws
  - `src/audit/universe.py::Universe.method_for` - missing Universe.method_for on a class with a method and one without
  - `src/audit/universe.py::Universe.refuses` - missing Universe.refuses on a case R refuses and one it accepts
  - `src/audit/valuetree.py::compare_vector` - missing compare_vector on vectors that match, differ in length, differ in a value and are index-shifted
  - `src/audit/valuetree.py::index_shifted` - missing index_shifted on a vector shifted by one and one that is not
  - `src/audit/valuetree.py::label_fields` - missing label_fields on a record with labels and one without
  - `src/audit/valuetree.py::named_vector` - missing named_vector on a named and an unnamed vector
  - `src/audit/valuetree.py::num_cell` - missing num_cell on a number, NA, NaN and infinite value
  - `src/audit/valuetree.py::vector_node` - missing vector_node on a numeric, logical, character and failing conversion
  - `src/metafopy/bias/ranktest.py::kendall_test` - missing kendall_test with exact and normal-approximation p-values, with and without ties
  - `src/metafopy/core/bvn.py::bvnu` - missing a direct test of bvnu (upper-orthant probability, including its degenerate-correlation branches)
  - `src/metafopy/core/calculus.py::derivative_mixed` - missing a direct test of derivative_mixed (empty-weight and multi-order branches)
  - `src/metafopy/core/calculus.py::hessian` - missing a direct test of hessian (p == 1 and p > 1 branches)
  - `src/metafopy/core/clogit.py::clogistic` - missing a direct test of clogistic
  - `src/metafopy/core/clogit.py::clogit` - missing a direct test of clogit (exact and ties dispatch)
  - `src/metafopy/core/clogit.py::clogit_settings` - missing a direct test of clogit_settings (unknown keys, bad method, bad init)
  - `src/metafopy/core/clogit.py::coxph_ties` - missing a direct test of coxph_ties (efron, breslow, approximate)
  - `src/metafopy/core/clubsandwich.py::contrast_df` - missing a direct test of contrast_df
  - `src/metafopy/core/clubsandwich.py::vcov_cr` - missing a direct test of vcov_cr (CR0-CR3 adjustment branches)
  - `src/metafopy/core/clubsandwich.py::wald_test` - missing a direct test of wald_test (HTA/HTB/EDF/EDT and the singular-matrix except branch)
  - `src/metafopy/core/dbinom.py::dbinom_raw` - missing a direct test of dbinom_raw (p == 0, p == 1, x == 0, x == n branches)
  - `src/metafopy/core/farebrother.py::farebrother` - missing a direct test of farebrother
  - `src/metafopy/core/glm.py::aliased_columns` - missing a direct test of aliased_columns
  - `src/metafopy/core/glm.py::check_link` - missing a direct test of check_link (valid and invalid link)
  - `src/metafopy/core/glm.py::deviance_residuals` - missing a direct test of deviance_residuals
  - `src/metafopy/core/glm.py::family_constant` - missing a direct test of family_constant (binomial and poisson branches)
  - `src/metafopy/core/glm.py::glm_fit` - missing direct test of glm_fit
  - `src/metafopy/core/glm.py::saturated_loglik` - missing direct test of saturated_loglik
  - `src/metafopy/core/glm.py::valid_mu` - missing direct test of valid_mu
  - `src/metafopy/core/glmm.py::fit_glmm` - missing direct test of fit_glmm
  - `src/metafopy/core/glmmadaptive.py::fit_adaptive` - missing direct test of fit_adaptive
  - `src/metafopy/core/labelled.py::EscalcFrame.digit_for` - missing direct test of EscalcFrame.digit_for
  - `src/metafopy/core/labelled.py::EscalcFrame.text` - missing direct test of EscalcFrame.text
  - `src/metafopy/core/labelled.py::LabelledMatrix.__str__` - missing test of str() on a LabelledMatrix (2-D and non-2-D)
  - `src/metafopy/core/labelled.py::LabelledMatrix.transpose` - missing direct test of LabelledMatrix.transpose swapping names
  - `src/metafopy/core/labelled.py::VCovMat.text` - missing direct test of VCovMat.text
  - `src/metafopy/core/labelled.py::design_matrix` - missing a direct test of design_matrix exercising its branches
  - `src/metafopy/core/labelled.py::matrix_text` - missing a direct test of matrix_text exercising its branches
  - `src/metafopy/core/linalg.py::drop_redundant` - missing a direct test of drop_redundant exercising its branches
  - `src/metafopy/core/linalg.py::is_symmetric` - missing a direct test of is_symmetric exercising its branches
  - `src/metafopy/core/nchg.py::study_loglik` - missing a direct test of study_loglik exercising its branches
  - `src/metafopy/core/numderiv.py::grad_scalar` - missing a direct test of grad_scalar exercising its branches
  - `src/metafopy/core/parallel.py::map_sims` - missing a direct test of map_sims exercising its branches
  - `src/metafopy/core/parallel.py::tick` - missing a direct test of tick exercising its branches
  - `src/metafopy/core/rng.py::RRandom.prob_sample_no_replace` - missing a direct test of RRandom.prob_sample_no_replace exercising its branches
  - `src/metafopy/core/rng.py::RRandom.prob_sample_replace` - missing a direct test of RRandom.prob_sample_replace exercising its branches
  - `src/metafopy/core/rng.py::RRandom.rbits` - missing a direct test of RRandom.rbits exercising its branches
  - `src/metafopy/core/rng.py::RRandom.sample_int` - missing a direct test of RRandom.sample_int exercising its branches
  - `src/metafopy/core/rng.py::RRandom.unif_index` - missing a direct test of RRandom.unif_index exercising its branches
  - `src/metafopy/core/rng.py::RRandom.unif_rand` - missing a direct test of RRandom.unif_rand exercising its branches
  - `src/metafopy/core/rng.py::RRandom.walker` - missing a direct test of RRandom.walker exercising its branches
  - `src/metafopy/core/rng.py::get_rng` - missing a direct test of get_rng exercising its branches
  - `src/metafopy/core/stationarity.py::newton_decrement` - missing a direct test of newton_decrement exercising its branches
  - `src/metafopy/core/timing.py::timed` - missing a direct test of timed exercising its branches
  - `src/metafopy/core/zeroin.py::zeroin2` - missing a direct test of zeroin2 (public in metafopy.core.zeroin) covering its branches, e.g. an infinite function value replaced by the largest finite value and the maxiter exit; it is reached only through uniroot tests
  - `src/metafopy/effects/fivenum_estimators.py::estimate` - missing a test calling fivenum_estimators.estimate directly for each method (qe, bc, mln, blue)
  - `src/metafopy/effects/measures.py::replmiss` - missing a direct test of measures.replmiss (None/empty y, None x, length-1 y recycling, length mismatch error)
  - `src/metafopy/effects/rtet.py::rtet` - missing a direct test of rtet: the degenerate tables (empty margin, bi=ci=0, ai=di=0), a zero cell, and the optimiser/Hessian failure warnings
  - `src/metafopy/formatters.py::format_c` - missing a direct test of format_c: width None/0, NA/NaN/Inf with and without the '-' flag, and drop0
  - `src/metafopy/formula.py::formula_context` - missing a direct test of formula_context: a caller variable that shadows an R formula function, callable versus non-callable
  - `src/metafopy/formula.py::r_factor` - missing a direct test of metafopy.formula.r_factor: given levels, float levels renamed as R prints them, and labels of the wrong length raising InputError
  - `src/metafopy/metadata.py::MetaData.design` - missing a direct test of MetaData.design: a formula, an array, a data frame and None for mods, with and without intercept
  - `src/metafopy/metadata.py::MetaData.names_all` - missing a test that reads MetaData.names_all with and without row_names and with integer and non-integer data index
  - `src/metafopy/metadata.py::MetaData.ni_f` - missing a test that reads MetaData.ni_f with ni_all None and set
  - `src/metafopy/metadata.py::MetaData.weights_f` - missing a test that reads MetaData.weights_f with weights_all None and set
  - `src/metafopy/methods/anova.py::ListAnovaResult.__init__` - missing a test that constructs ListAnovaResult directly
  - `src/metafopy/methods/binary.py::refit_tables` - missing direct test of refit_tables, including the failed refit that returns None
  - `src/metafopy/methods/confint.py::ConfintResult.ci_null` - missing test that reads ci_null, for an interval at or below zero and for one that is not
  - `src/metafopy/methods/emm_adjust.py::critical_values` - missing direct test of critical_values, for each adjustment and one-sided tail
  - `src/metafopy/methods/emm_adjust.py::p_values` - missing direct test of p_values, for each adjustment and tail
  - `src/metafopy/methods/emm_contrasts.py::build_contrasts` - missing a direct test of build_contrasts for each contrast method and the custom-coefficient branch
  - `src/metafopy/methods/emm_links.py::measure_link` - missing a direct test of measure_link for a known measure and an unknown one
  - `src/metafopy/methods/emm_nesting.py::nonempty_display` - missing a direct test of nonempty_display (it is public, so caller tests do not count)
  - `src/metafopy/methods/emm_summary.py::summary` - missing a direct test of emm_summary.summary (it is public; tests reach it only through EmmGrid.summary)
  - `src/metafopy/methods/fitted.py::hat_values` - missing a direct test of fitted.hat_values for vector and matrix hat input (it is public; tests reach it only through hatvalues())
  - `src/metafopy/methods/fitted.py::rows_of` - missing a direct test of fitted.rows_of for each model class and the ArgumentError branch (it is public; tests reach it only through callers)
  - `src/metafopy/methods/fitview.py::check_method` - missing a direct test of check_method: a fit of a class not in classes raises "no applicable method", and a fit of an allowed class passes
  - `src/metafopy/methods/leave1out.py::Leave1outResult.to_frame` - missing a test calling to_frame() on a leave1out result (k None) and a cumul result (k column, tau2 present and absent)
  - `src/metafopy/methods/listrma.py::rows` - missing a direct test of rows: an index that subsets the arrays, frames and slab while order and ids stay, and an index that raises IndexError
  - `src/metafopy/methods/ls_anova.py::LSWaldResult.stat_line` - missing a direct test of stat_line for an F-test (df[1] set) and a chi-square test, location (QM) and scale (QS)
  - `src/metafopy/methods/ls_predict.py::LSPredictResult.__post_init__` - missing construction of LSPredictResult with a Z matrix and z_names
  - `src/metafopy/methods/matreg.py::MatregResult.stat` - missing test statistic of a matreg fit
  - `src/metafopy/methods/matreg.py::MatregResult.table_lines` - missing coefficient table of a matreg fit
  - `src/metafopy/methods/mv_diagnostics.py::refit_mv` - missing a direct test of refit_mv (drop, fixed and init branches, and a failing refit returning None)
  - `src/metafopy/methods/mv_profile.py::MVProfileResult.component_grid` - missing a test reading .sigma2/.rho/.gamma2/.phi of an MVProfileResult, including the AttributeError for the wrong component
  - `src/metafopy/methods/pairmat.py::resolve` - missing a direct test of resolve with a PairmatSpec and with a plain value
  - `src/metafopy/methods/permutest.py::PermutestResult.to_frame` - missing a direct test of PermutestResult.to_frame for a z-test and a t-test (ddf) result
  - `src/metafopy/methods/permutest.py::fixed_btt` - missing a test of permutest(btt='<string>', fixed=True), matching and not matching a coefficient
  - `src/metafopy/methods/predict.py::PredictResult.__post_init__` - missing a test constructing PredictResult directly (with x_names/slab that do and do not match X/vcov)
  - `src/metafopy/methods/predict.py::fitted_levels` - missing a test of predict() without newmods on an rma_mv fit with moderators and an inner|outer random term
  - `src/metafopy/methods/rcalc.py::RCalcResult.__post_init__` - missing a test that constructs RCalcResult directly with a dense V and a dat holding var1.var2, checking V becomes a VCovMat named by var1.var2
  - `src/metafopy/methods/selmodel.py::SelProfileResult.delta` - missing a test reading .delta of a selection-parameter profile (metafopy.profile(sel, delta=1)) and expecting AttributeError from .delta of a tau2 profile
  - `src/metafopy/methods/vcov.py::vcov_frame` - missing a direct call of vcov_frame checking its single-frame result and the InternalError when vcov returns several matrices
  - `src/metafopy/methods/weights.py::weights_mv` - missing a direct call of weights_mv on an rma_mv fit for each type and na_action
  - `src/metafopy/methods/weights.py::weights_uni` - missing a direct call of weights_uni on rma_uni and rma_ls fits for each type and na_action
  - `src/metafopy/mfopt.py::option_digits` - missing a direct call of option_digits with and without digits given and with the global digits option set
  - `src/metafopy/mfopt.py::option_space` - missing a direct call of option_space with the space option set to True and False
  - `src/metafopy/mfopt.py::reduce_digits` - missing a direct call of reduce_digits on scalar, sequence and named digits
  - `src/metafopy/models/binary.py::fit_statistics` - missing fit_statistics called directly (k_yi < 1 returns NaN stats; vi <= 0 deviance branch)
  - `src/metafopy/models/binary.py::format_header` - missing format_header called directly (showfit on/off; NaN I2/H2/QE lines left out)
  - `src/metafopy/models/glmm_fit.py::fit_model` - missing fit_model called directly on FitSpecs for a one-group measure, UM.FS/UM.RS, CM.EL OR and CM.AL
  - `src/metafopy/models/glmm_result.py::GLMMResult.heterogeneity_lines` - missing GLMMResult.heterogeneity_lines called directly (its branches per model and test availability)
  - `src/metafopy/models/glmm_result.py::GLMMResult.render` - missing GLMMResult.render called directly (showfit on/off)
  - `src/metafopy/models/glmm_result.py::GLMMResult.title_lines` - missing GLMMResult.title_lines called directly (FE vs ML titles per model)
  - `src/metafopy/models/mv_factor.py::evaluate_variable` - missing evaluate_variable called directly (column name, expression, and the not-found error)
  - `src/metafopy/models/mv_factor.py::indicator` - missing indicator called directly
  - `src/metafopy/models/mv_factor.py::interaction` - missing interaction called directly (an interaction(a, b) term of a random formula)
  - `src/metafopy/models/mv_factor.py::split_top_level` - missing split_top_level called directly (nested parentheses and quoted commas)
  - `src/metafopy/models/mv_factor.py::to_factor` - missing to_factor called directly (numeric, string, categorical and missing values)
  - `src/metafopy/models/mv_hessian.py::variance_hessian` - missing variance_hessian called directly (cvvc modes and the no-component-present return)
  - `src/metafopy/models/mv_inputs.py::block_diagonal` - missing rma_mv with V given as a list of block matrices (and a non-square block raising InputError)
  - `src/metafopy/models/mv_inputs.py::build_x` - missing build_x called directly (no mods, formula mods, matrix mods)
  - `src/metafopy/models/mv_inputs.py::extract_random` - missing extract_random with each random-term form
  - `src/metafopy/models/mv_inputs.py::prepare_v` - missing prepare_v with list, sparse, column-name, scalar and wrong-length V
  - `src/metafopy/models/mv_inputs.py::prepare_w` - missing prepare_w with vector, matrix, sparse and non-square or asymmetric W
  - `src/metafopy/models/mv_inputs.py::prepare_yi` - missing prepare_yi with each yi input form
  - `src/metafopy/models/mv_inputs.py::take_square` - missing take_square with a dense and a sparse matrix
  - `src/metafopy/models/mv_lik.py::Problem.penalty` - missing Problem.penalty with and without lambda1/lambda2 and an intercept
  - `src/metafopy/models/mv_lik.py::evaluate` - missing evaluate on a problem with each structure
  - `src/metafopy/models/mv_lik.py::final_fit` - missing final_fit for REML and ML fits
  - `src/metafopy/models/mv_lik.py::quad_form` - missing quad_form with a dense and a sparse matrix
  - `src/metafopy/models/mv_print.py::abbreviate` - missing abbreviate with duplicate abbreviations that must be lengthened
  - `src/metafopy/models/mv_print.py::format_result` - missing format_result with showfit on and off
  - `src/metafopy/models/mv_result.py::RMAMVResult.R` - missing RMAMVResult.R with and without known correlation matrices
  - `src/metafopy/models/mv_result.py::RMAMVResult.__post_init__` - missing RMAMVResult constructed directly, with and without Z_S levels and G/H
  - `src/metafopy/models/mv_result.py::RMAMVResult.dimnames_of` - missing dimnames_of for no group, spatial, general and factor structures
  - `src/metafopy/models/mv_result.py::RMAMVResult.named_inner` - missing named_inner for factor, general and other structures
  - `src/metafopy/models/mv_result.py::RMAMVResult.random` - missing RMAMVResult.random with a string, a list and no random formula
  - `src/metafopy/models/mv_result.py::RMAMVResult.weight_matrix` - missing weight_matrix with a user W and with the inverse of M
  - `src/metafopy/models/mv_result.py::formula_text` - missing RMAMVResult.formulas with and without G/H groups
  - `src/metafopy/models/mv_structs.py::con_e` - missing con_e for each structure
  - `src/metafopy/models/mv_structs.py::con_vcov_un` - missing con_vcov_un with and without the Cholesky parametrisation
  - `src/metafopy/models/mv_structs.py::distance_from_matrix` - missing rma_mv with dist given as a distance matrix (valid, missing values, non-square, asymmetric, missing levels)
  - `src/metafopy/models/mv_structs.py::process_after_rmna` - missing process_after_rmna for factor, spatial and general structures
  - `src/metafopy/models/mv_structs.py::process_after_sub` - missing process_after_sub for each structure
  - `src/metafopy/models/rma_ls.py::fit_scale` - missing a test calling metafopy.models.rma_ls.fit_scale directly (e.g. its unknown-method and na_action=fail branches)
  - `src/metafopy/models/rma_ls_fit.py::LSProblem.score_parts` - missing a test calling LSProblem.score_parts directly
  - `src/metafopy/models/rma_ls_fit.py::fit_rma_ls` - missing a test calling metafopy.models.rma_ls_fit.fit_rma_ls directly
  - `src/metafopy/models/rma_ls_fit.py::make_ls_control` - missing a test calling make_ls_control directly (mapping, LSControl and unknown-name inputs)
  - `src/metafopy/models/rma_ls_opt.py::constr_optim` - missing a test calling metafopy.models.rma_ls_opt.constr_optim directly
  - `src/metafopy/models/rma_ls_opt.py::minimise` - missing a test calling metafopy.models.rma_ls_opt.minimise directly
  - `src/metafopy/models/rma_ls_opt.py::polish_identity` - missing a test calling metafopy.models.rma_ls_opt.polish_identity directly
  - `src/metafopy/models/rma_ls_print.py::coef_table` - missing a test calling metafopy.models.rma_ls_print.coef_table directly
  - `src/metafopy/models/rma_ls_print.py::format_result` - missing a test calling metafopy.models.rma_ls_print.format_result directly
  - `src/metafopy/models/rma_ls_result.py::RMALSResult.__post_init__` - missing a test constructing an RMALSResult (or dataclasses.replace of one) and checking the labelled X/Z/H matrices
  - `src/metafopy/models/rma_ls_result.py::RMALSResult.hat_matrix` - missing a test calling RMALSResult.hat_matrix directly (weighted and unweighted fits)
  - `src/metafopy/models/rma_mh.py::MHResult.render` - missing a test calling MHResult.render directly (OR, IRR and non-computable CMH/Tarone branches)
  - `src/metafopy/models/rma_mh.py::MHResult.weight_values` - missing a test calling MHResult.weight_values directly for each measure
  - `src/metafopy/models/rma_uni.py::RMAResult.hat_matrix` - missing a test that calls RMAResult.hat_matrix() directly on an unweighted and a weighted fit (the self.weighted branch and the user-weights branch)
  - `src/metafopy/models/robust_print.py::counts_lines` - missing a test that calls counts_lines directly with equal and with unequal cluster sizes
  - `src/metafopy/models/robust_print.py::legend_lines` - missing a test that calls legend_lines directly for robumethod 'default', for the Satterthwaite/HTZ clubSandwich case and for any other combination
  - `src/metafopy/models/sel_lik.py::evaluate` - missing a test that calls sel_lik.evaluate directly for a continuous, a step-function and a truncation Problem
  - `src/metafopy/models/sel_lik.py::map_delta` - missing a test that calls map_delta directly with fun=None on (0, 1) and on other bounds, and with a named function that is clipped
  - `src/metafopy/models/sel_lik.py::map_inverse` - missing a test that calls map_inverse directly with fun=None on (0, 1) and on other bounds, and with a named function
  - `src/metafopy/models/sel_opt.py::minimize` - missing a test that calls sel_opt.minimize directly for each optimizer branch it dispatches to
  - `src/metafopy/models/sel_result.py::SelmodelResult.__post_init__` - missing a test that constructs SelmodelResult directly with and without a hessian
  - `src/metafopy/models/sel_result.py::format_selmodel` - missing a test that calls format_selmodel directly (fixed-effects, moderators, showfit and trim-and-fill branches)
  - `src/metafopy/models/sel_spec.py::build_spec` - missing a test that calls build_spec directly for each selection type
  - `src/metafopy/models/tables.py::pair` - missing a test that calls pair directly with a scalar, a one-element, a two-element and a three-element value
  - `src/metafopy/models/tables.py::prepare_tables` - missing a test that calls prepare_tables directly (one-group, rate and 2x2 measures, missing values, drop00)
  - `src/metafopy/models/tau2.py::Control.__getattr__` - missing a test that reads an ignored control entry (e.g. res.control.optimizer) from an rma_uni fit, and one for a missing name raising AttributeError
  - `src/metafopy/plot/funnel.py::FunnelResult.to_frame` - missing FunnelResult.to_frame() with and without fill (trim-and-fill result adds the fill column)
  - `src/metafopy/plot/histplot.py::draw_histogram` - missing draw_histogram() called directly, with freq True/False, given and default xlim/ylim, xaxis False and empty main
  - `src/metafopy/plot/histplot.py::draw_tail` - missing draw_tail() called directly for tail='lower' and 'upper', including val outside all breaks
  - `src/metafopy/plot/histplot.py::panels` - missing panels() called directly for n == 1 and n > 1 (unused grid cells switched off)
  - `src/metafopy/plot/rutils.py::fmtx` - missing a direct test of metafopy.plot.rutils.fmtx, including the InputError for several digits with a vector
  - `src/metafopy/printing.py::drop_stars` - missing a direct test of drop_stars with signif_stars True and False
  - `src/metafopy/printing.py::fmt_df` - missing a direct test of fmt_df with finite, NaN and infinite values
  - `src/metafopy/printing.py::fmt_df_scalar` - missing a direct test of fmt_df_scalar with None/NaN, infinite, integral and fractional values
  - `src/metafopy/printing.py::fmt_wrapped_table` - missing a direct test of fmt_wrapped_table with a table wide enough to wrap
  - `src/metafopy/printing.py::signif_footer` - missing a direct test of signif_footer with and without the legend and extra lines

### T2 (listed): Property-based tests

- **Candidates judged:** 193 (every candidate; none left unjudged)
- **Open findings:** 144

- **Hits:**
  - `bin/logo.py::smoothstep` - missing property test
  - `bin/metafor-audit.py::metafopy_source` - missing property test
  - `src/audit/generate.py::default_string` - missing property test
  - `src/audit/ledger.py::bracket_sets` - missing property test
  - `src/audit/printdiff.py::close_numbers` - missing property test
  - `src/audit/printdiff.py::decimals` - missing property test
  - `src/audit/printdiff.py::first_difference` - missing property test
  - `src/audit/printdiff.py::same_line` - missing property test
  - `src/audit/printdiff.py::same_token` - missing property test
  - `src/audit/pyrun.py::plain_text` - missing property test
  - `src/audit/report.py::normal_path` - missing property test
  - `src/audit/steps.py::selected` - missing property test
  - `src/metafopy/bias/fsn.py::rnd_fsn` - missing property test
  - `src/metafopy/bias/inputs.py::print_vector` - missing property test
  - `src/metafopy/bias/ranktest.py::pkendall` - missing property test
  - `src/metafopy/core/bvn.py::bvn_rect` - missing property test
  - `src/metafopy/core/bvn.py::bvnu` - missing property test
  - `src/metafopy/core/dbinom.py::bd0` - missing property test
  - `src/metafopy/core/dbinom.py::dbinom_raw` - missing property test
  - `src/metafopy/core/dbinom.py::pow1p` - missing property test
  - `src/metafopy/core/dbinom.py::stirlerr` - missing property test
  - `src/metafopy/core/farebrother.py::farebrother` - missing property test
  - `src/metafopy/core/farebrother.py::ruben` - missing property test
  - `src/metafopy/core/labelled.py::decimals` - missing property test
  - `src/metafopy/core/labelled.py::format_real` - missing property test
  - `src/metafopy/core/labelled.py::scientific_places` - missing property test
  - `src/metafopy/core/labelled.py::trim_zeros` - missing property test
  - `src/metafopy/core/nchg.py::fisher_log_probability` - missing property test
  - `src/metafopy/core/nchg.py::ln_fac` - missing property test
  - `src/metafopy/core/nchg.py::tabulated_log_probability` - missing property test
  - `src/metafopy/core/padjust.py::match_adjust` - missing property test
  - `src/metafopy/core/timing.py::format_time` - missing property test
  - `src/metafopy/core/timing.py::number` - missing property test
  - `src/metafopy/core/tukey.py::ptukey` - missing property test
  - `src/metafopy/core/tukey.py::qtukey` - missing property test
  - `src/metafopy/core/tukey.py::wprob` - missing property test
  - `src/metafopy/effects/datatools.py::charmatch` - missing property test
  - `src/metafopy/effects/datatools.py::make_names` - missing property test
  - `src/metafopy/effects/datatools.py::shorten_names` - missing property test
  - `src/metafopy/effects/fivenum_estimators.py::sum_stat_index` - missing property test
  - `src/metafopy/effects/fivenum_estimators.py::weibull_shape_start` - missing property test
  - `src/metafopy/effects/measures.py::make_unique_dot1` - missing property test
  - `src/metafopy/effects/rtet.py::rtet` - missing property test
  - `src/metafopy/effects/summary.py::level_to_alpha` - missing property test
  - `src/metafopy/formatters.py::r_as_character` - missing property test
  - `src/metafopy/formula.py::call_spans` - missing property test
  - `src/metafopy/formula.py::parse_random` - missing property test
  - `src/metafopy/formula.py::r_deparse` - missing property test
  - `src/metafopy/formula.py::r_logical_calls` - missing property test
  - `src/metafopy/formula.py::r_logical_formula` - missing property test
  - `src/metafopy/formula.py::r_number` - missing property test
  - `src/metafopy/formula.py::r_power` - missing property test
  - `src/metafopy/formula.py::r_tokens` - missing property test
  - `src/metafopy/interactive/forest.py::unique_labels` - missing property test
  - `src/metafopy/metadata.py::make_unique` - missing property test
  - `src/metafopy/methods/blup.py::format_r_frame` - missing property test
  - `src/metafopy/methods/common.py::match_arg` - missing property test
  - `src/metafopy/methods/deltamethod.py::get_digits` - missing property test
  - `src/metafopy/methods/emm_adjust.py::match_method` - missing property test
  - `src/metafopy/methods/emm_adjust.py::parse_side` - missing property test
  - `src/metafopy/methods/emm_adjust.py::pf_cdf` - missing property test
  - `src/metafopy/methods/emm_adjust.py::pf_sf` - missing property test
  - `src/metafopy/methods/emm_adjust.py::pt_sf` - missing property test
  - `src/metafopy/methods/emm_adjust.py::qdunnx` - missing property test
  - `src/metafopy/methods/emm_adjust.py::qf` - missing property test
  - `src/metafopy/methods/emm_adjust.py::qt` - missing property test
  - `src/metafopy/methods/emm_grid.py::appearance` - missing property test
  - `src/metafopy/methods/emm_nesting.py::factor_id` - missing property test
  - `src/metafopy/methods/emm_nesting.py::strip_supersets` - missing property test
  - `src/metafopy/methods/emm_nesting.py::term_info` - missing property test
  - `src/metafopy/methods/hettest.py::expand_boot` - missing property test
  - `src/metafopy/methods/ls_anova.py::omnibus` - missing property test
  - `src/metafopy/methods/ls_common.py::center_stars` - missing property test
  - `src/metafopy/methods/matreg.py::match_columns` - missing property test
  - `src/metafopy/methods/matreg.py::parse_formula` - missing property test
  - `src/metafopy/methods/matreg.py::pmatch` - missing property test
  - `src/metafopy/methods/matreg.py::resolve_x` - missing property test
  - `src/metafopy/methods/matreg.py::resolve_y` - missing property test
  - `src/metafopy/methods/matreg.py::subsequence_distance` - missing property test
  - `src/metafopy/methods/predict.py::is_subsequence` - missing property test
  - `src/metafopy/methods/predict.py::level_indices` - missing property test
  - `src/metafopy/methods/predict.py::match_names` - missing property test
  - `src/metafopy/methods/rcalc.py::formula_variables` - missing property test
  - `src/metafopy/methods/rcalc.py::match_nfun` - missing property test
  - `src/metafopy/methods/vif.py::format_btt` - missing property test
  - `src/metafopy/methods/vif.py::format_vector` - missing property test
  - `src/metafopy/models/call.py::split_terms` - missing property test
  - `src/metafopy/models/call.py::update_formula` - missing property test
  - `src/metafopy/models/call.py::update_side` - missing property test
  - `src/metafopy/models/mv_factor.py::format_number` - missing property test
  - `src/metafopy/models/mv_factor.py::split_top_level` - missing property test
  - `src/metafopy/models/mv_hessian.py::inner_names` - missing property test
  - `src/metafopy/models/mv_hessian.py::lower_triangle_order` - missing property test
  - `src/metafopy/models/mv_print.py::abbreviate` - missing property test
  - `src/metafopy/models/mv_print.py::abbreviate_once` - missing property test
  - `src/metafopy/models/rma_mv.py::replace_call_notation` - missing property test
  - `src/metafopy/models/rma_uni.py::abbreviate_names` - missing property test
  - `src/metafopy/models/rma_uni.py::coef_span` - missing property test
  - `src/metafopy/models/rma_uni.py::level_to_alpha` - missing property test
  - `src/metafopy/models/sel_lik.py::apply_named` - missing property test
  - `src/metafopy/models/sel_lik.py::exp_safe` - missing property test
  - `src/metafopy/models/sel_lik.py::map_delta` - missing property test
  - `src/metafopy/models/sel_lik.py::map_inverse` - missing property test
  - `src/metafopy/models/sel_opt.py::choose_optimizer` - missing property test
  - `src/metafopy/models/sel_opt.py::partial_match` - missing property test
  - `src/metafopy/models/sel_spec.py::match_type` - missing property test
  - `src/metafopy/models/sel_spec.py::s_halfnorm` - missing property test
  - `src/metafopy/models/sel_spec.py::s_halfnorm1` - missing property test
  - `src/metafopy/models/sel_spec.py::s_halfnorm2` - missing property test
  - `src/metafopy/models/sel_spec.py::s_logistic` - missing property test
  - `src/metafopy/models/sel_spec.py::s_logistic1` - missing property test
  - `src/metafopy/models/sel_spec.py::s_logistic2` - missing property test
  - `src/metafopy/models/sel_spec.py::s_negexp` - missing property test
  - `src/metafopy/models/sel_spec.py::s_negexp1` - missing property test
  - `src/metafopy/models/sel_spec.py::s_negexp2` - missing property test
  - `src/metafopy/models/sel_spec.py::s_negexppow` - missing property test
  - `src/metafopy/models/sel_spec.py::s_power` - missing property test
  - `src/metafopy/models/sel_spec.py::s_power1` - missing property test
  - `src/metafopy/models/sel_spec.py::s_power2` - missing property test
  - `src/metafopy/models/selmodel.py::r_as_character` - missing property test
  - `src/metafopy/models/tau2.py::i2_from_q` - missing property test
  - `src/metafopy/plot/forest.py::anno_symbol` - missing property test
  - `src/metafopy/plot/forest.py::finish_annotext` - missing property test
  - `src/metafopy/plot/forest.py::parse_predtail` - missing property test
  - `src/metafopy/plot/gosh.py::hex_color` - missing property test
  - `src/metafopy/plot/infl.py::abbreviate` - missing property test
  - `src/metafopy/plot/infl.py::n2mfrow` - missing property test
  - `src/metafopy/plot/profile.py::per_panel` - missing property test
  - `src/metafopy/plot/rutils.py::color_ramp` - missing property test
  - `src/metafopy/plot/rutils.py::fmtx1` - missing property test
  - `src/metafopy/plot/rutils.py::level_to_alpha` - missing property test
  - `src/metafopy/plot/rutils.py::r_pretty` - missing property test
  - `src/metafopy/plot/rutils.py::rround` - missing property test
  - `src/metafopy/plot/rutils.py::usr_range` - missing property test
  - `src/metafopy/printing.py::fmt_df_scalar` - missing property test
  - `src/metafopy/printing.py::fmt_named_vector` - missing property test
  - `src/metafopy/printing.py::fmt_num` - missing property test
  - `src/metafopy/printing.py::fmt_pval` - missing property test
  - `src/metafopy/printing.py::fmt_table` - missing property test
  - `src/metafopy/printing.py::fmt_wrapped_table` - missing property test
  - `src/metafopy/printing.py::number_rows` - missing property test
  - `src/metafopy/printing.py::signif_footer` - missing property test
  - `src/metafopy/printing.py::signif_stars` - missing property test
  - `src/metafopy/printing.py::zero_below` - missing property test

### Doc1 (listed): AGENTS.md project structure

- **Candidates judged:** 3 (every candidate; none left unjudged)
- **Open findings:** 1

- **Hits:**
  - `AGENTS.md::metafopy > Audit` - stale wrong-or-missing.md

### Doc2 (listed): AGENTS.md tools and libraries list

- **Candidates judged:** 3 (every candidate; none left unjudged)
- **Open findings:** 0

- **Hits:** none

### Doc4 (listed): Specs match implementation

- **Candidates judged:** 1091 (every candidate; none left unjudged)
- **Open findings:** 30

- **Hits:**
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Error handling::(unresolved)` - drift agents/specs/audit-shard-survival/spec.md:180 says a non-zero exit with no case running ends the shard with the exit status and log path in each case's detail; engine.py:122 ends the shard only when the attempt also finished no case (a process that dies between cases is restarted on the cases left), and R's dead detail (engine.py:137) gives neither exit status nor log path
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Implementation > One shard loop for both languages::src/audit/engine.py::run_shard` - drift agents/specs/audit-shard-survival/spec.md:122 says a process that stopped with no case running gives each remaining case dead() and ends the loop; engine.py:122 does so only when that attempt also finished no case, and otherwise restarts on the cases left
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Implementation > The Python shard process::src/audit/engine.py::run_shard` - drift agents/specs/audit-shard-survival/spec.md:146 says the process seeds 1234 before each case 'as `run_shard` does today'; `audit.engine.run_shard` (src/audit/engine.py:91) is the generic restart loop and seeds nothing, the pyrun `run_shard` the sentence meant is removed, and seeding is in `audit.pyrun.load_shard`/`case_result` (src/audit/pyrun.py:234, 262)
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Implementation > The memory watch::(unresolved)` - drift agents/specs/audit-shard-survival/spec.md:156 '`watch` replaces `wait_or_kill`' names `wait_or_kill`, which the code once had and no longer defines anywhere in src/ (literal reading of the unresolved-name threshold; the sentence records the rename)
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Out of scope::src/audit/scripts.py::run` - drift agents/specs/audit-shard-survival/spec.md:215-216 says `audit.scripts.run` uses a `ProcessPoolExecutor` for its Python side and R processes with no restart; it now runs both sides in a `ThreadPoolExecutor` (src/audit/scripts.py:286-291) and each shard goes through `audit.engine.run_shard` with restart (scripts.py:229, 264)
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Settings::(unresolved)` - drift agents/specs/audit-shard-survival/spec.md:190 calls 2.0 "today's literal in `wait_or_kill`"; `wait_or_kill` no longer exists and `watch` reads `audit.config.POLL_SECONDS` (src/audit/engine.py:73)
  - `agents/specs/audit-shard-survival/spec.md::Audit shard survival > Tests::(unresolved)` - drift agents/specs/audit-shard-survival/spec.md:197 says the stand-ins follow 'the existing `wait_or_kill` tests'; there are no such tests, the tests are of `watch` (src/tests/test_harness_engine.py:22-45)
  - `agents/specs/auditme-clean/spec.md::Auditme clean > API::(unresolved)` - drift agents/specs/auditme-clean/spec.md:77-78 says `check_class` raises `ArgumentError` only when given `must`; `must` defaults to 'rma' and is checked on every call (src/metafopy/methods/fitview.py:77-86), so it raises whenever `x` is not of class 'rma', given `must` or not
  - `agents/specs/auditme-clean/spec.md::Auditme clean > API::src/metafopy/methods/fitview.py::check_class` - drift agents/specs/auditme-clean/spec.md:77 says check_class raises ArgumentError only when given `must`; src/metafopy/methods/fitview.py:77-85 defaults must='rma' and raises ArgumentError on every call whose x is not of class must, whether or not the caller passes it (unchanged since 58e5d76, before the spec).
  - `agents/specs/auditme-clean/spec.md::Auditme clean > Settings::(unresolved)` - drift agents/specs/auditme-clean/spec.md:190 names the CI secret `AGENTS_DEPLOY_KEY`; since 31d809f `.github/workflows/ci.yml:32` reads `secrets.AUDITOR_DEPLOY_KEY` (the old name was removed in that commit)
  - `agents/specs/auditme-clean/spec.md::Auditme clean > Settings::.pre-commit-config.yaml` - drift agents/specs/auditme-clean/spec.md:188 describes the setting as the agents `rev` (shared `main`); `.pre-commit-config.yaml:9-10` pins `git@github.com:bnomis/auditor.git` at rev 6a65d14, not the agents repository
  - `agents/specs/correctness-glmm/spec.md::Correctness tests for rma_glmm > API::(unresolved)` - drift agents/specs/correctness-glmm/spec.md:177 lists a `conditional_` variant of `glmm_fit`; `src/tests/correctness/glmm_formulas.py` has only `poisson_glmm_fit` (861) and `one_group_glmm_fit` (880), and the conditional models are fitted by `glmm_fit` itself via its `model` argument
  - `agents/specs/correctness-mv-robust/spec.md::Correctness tests` - rma_mv and robust > API::src/tests/correctness/formulas.py::hotelling_wald: drift agents/specs/correctness-mv-robust/spec.md:96 says hotelling_wald returns tuple[float, float, float]; src/tests/correctness/formulas.py:2745 returns a WaldTest dataclass (statistic, df1, df2, pval)
  - `agents/specs/correctness-mv-robust/spec.md::Correctness tests` - rma_mv and robust > API::src/tests/correctness/formulas.py::saddlepoint_pvalue: drift agents/specs/correctness-mv-robust/spec.md:97 says saddlepoint_pvalue returns float; src/tests/correctness/formulas.py:2796 returns a FloatArray, one p-value per coefficient
  - `agents/specs/correctness-mv-robust/spec.md::Correctness tests` - rma_mv and robust > Error handling::(unresolved): drift agents/specs/correctness-mv-robust/spec.md:166 says mv_fit fails the case when scipy's success is false or the gradient norm is above 1e-6; src/tests/correctness/formulas.py:2121 sets MVFit.success to norm <= MV_GRADIENT_LIMIT (1e-6) alone, and src/tests/correctness/test_derived.py:532 fails only on that, so a run whose scipy result reports failure but whose gradient norm is small is compared
  - `agents/specs/correctness-publication-bias/spec.md::Correctness tests` - publication bias > Error handling::(unresolved): drift agents/specs/correctness-publication-bias/spec.md:132 says a setup key other than where, drop, escalc and rma_uni raises ValueError; src/tests/correctness/cases.py:75 SETUP_KEYS also accepts subset, tree_r, vblocks and vconst, so those keys do not raise
  - `agents/specs/correctness-publication-bias/spec.md::Correctness tests` - publication bias::(unresolved): drift agents/specs/correctness-publication-bias/spec.md:22: says `grep -n 'def ' src/tests/correctness/test_derived.py` lists a back-transform check; the back-transform check `back_transformed` is defined in src/tests/correctness/test_published.py:31, and test_derived.py has none
  - `agents/specs/correctness-publication-bias/spec.md::Correctness tests` - publication bias::TODO.md: drift agents/specs/correctness-publication-bias/spec.md:20: names the `TODO.md` item "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals"; TODO.md no longer holds it
  - `agents/specs/correctness-tests/spec.md::Correctness tests > API::(unresolved)` - drift agents/specs/correctness-tests/spec.md:134-135: says derived and property cases carry `function = "formulas.<name>"` or a check and the loader refuses anything else; src/tests/correctness/cases.py:50 sets FORMULA_MODULES = ('formulas', 'glmm_formulas'), so `glmm_formulas.<name>` is also accepted
  - `agents/specs/correctness-uni-diagnostics/spec.md::Correctness tests` - rma_uni diagnostics > Cases::(unresolved): drift agents/specs/correctness-uni-diagnostics/spec.md:222 says the published case's `expect` holds the printed values; `uni.diag.published.bcg` holds them under `printed` (src/tests/correctness/cases/rma_uni_diagnostics.toml:48), a separate key that cases.py parse_printed reads, and has no `expect`, agents/specs/correctness-uni-diagnostics/spec.md:228 names the case `property.mv.diagnostics-equal-uni`, which no case file defines and git history never had outside the spec; the case that ties the rma_mv and rma_uni diagnostics is `property.mv.diagnostics-one-row-per-id.bcg` (src/tests/correctness/cases/properties.toml:488, check `test_properties.mv_diagnostics_equal_uni`)
  - `agents/specs/correctness-uni-diagnostics/spec.md::Correctness tests` - rma_uni diagnostics > Cases::src/metafopy/models/rma_mv.py::rma_mv: drift agents/specs/correctness-uni-diagnostics/spec.md:228 names the case `property.mv.diagnostics-equal-uni`, which no case file defines and git history never had outside the spec; the case that ties the rma_mv and rma_uni diagnostics is `property.mv.diagnostics-one-row-per-id.bcg` (src/tests/correctness/cases/properties.toml:488, check `test_properties.mv_diagnostics_equal_uni`)
  - `agents/specs/correctness-uni-diagnostics/spec.md::Correctness tests` - rma_uni diagnostics > Coverage list::src/tests/correctness/test_coverage.py::uni_diagnostics_items: drift agents/specs/correctness-uni-diagnostics/spec.md:178 says `uni_diagnostics_items()` comes with one new constant in test_coverage.py, `UNI_FITS`; test_coverage.py:48-50 adds three, `UNI_FITS`, `UNI_INFLUENCE` and `UNI_EXEMPT`
  - `agents/specs/cross-platform-numerics/spec.md::Cross-platform numerics > Error handling::(unresolved)` - drift agents/specs/cross-platform-numerics/spec.md:244 says linux-tests.sh exits 2 "with the message above" (Docker is not running: start Docker Desktop) also when the image does not build; bin/linux-tests.sh prints "The image $IMAGE did not build" in that case
  - `agents/specs/cross-platform-numerics/spec.md::Cross-platform numerics > Error handling::src/tests/correctness/formulas.py::ls_reference_fit` - drift agents/specs/cross-platform-numerics/spec.md:249-250 says ls_reference_fit raises RuntimeError naming the case and every start's message and score; its RuntimeError names only the link and method and the messages (a score only for log-link starts rejected by the score check), and the case is named by test_derived.ls_reference's AssertionError
  - `agents/specs/cross-platform-numerics/spec.md::Cross-platform numerics > Implementation > Tests of the new code::bin/platform-probe.py::measure_groups` - drift agents/specs/cross-platform-numerics/spec.md:235 says measure_groups(['cm-el-tight-se']) returns a GroupMeasurement; bin/platform-probe.py:432 returns dict[str, GroupMeasurement] keyed by group name (the test indexes measured[GROUP])
  - `agents/specs/r-oracle-pin/spec.md::R oracle pin > Module location::src/tests/test_harness_accounts.py` - drift agents/specs/r-oracle-pin/spec.md:66 names src/tests/test_harness_errors.py and src/tests/test_harness_accounts.py as the tests of the pin; neither holds any test of check_r, library_drift, allow_r_drift or oracle_moved
  - `agents/specs/r-oracle-pin/spec.md::R oracle pin > Module location::src/tests/test_harness_errors.py` - drift agents/specs/r-oracle-pin/spec.md:66 names src/tests/test_harness_errors.py and src/tests/test_harness_accounts.py as the tests of the pin; neither holds any test of check_r, library_drift, allow_r_drift or oracle_moved
  - `agents/specs/rma-mv-reml-stop/spec.md::rma_mv REML stop > Implementation > Tests::bin/platform-probe.py` - drift agents/specs/rma-mv-reml-stop/spec.md:148 says the mv-reml-score group records the polish's step count; bin/platform-probe.py records polish.hessians, the number of score_jacobian calls, which its own note calls an upper bound on the Newton steps taken, not the step count
  - `agents/specs/scripts-shard-survival/spec.md::Scripts shard survival > Implementation > The R side::src/audit/scripts.py::run_r_scripts` - drift agents/specs/scripts-shard-survival/spec.md:148 gives the died reason as `R process died on this case (killed)`, but run_r_scripts's dead calls audit.engine.r_stop_reason(stop, timeout, 'unit') (src/audit/scripts.py:226), so the scripts' dead unit reads `R process died on this unit (killed)`
  - `agents/specs/scripts-shard-survival/spec.md::Scripts shard survival > Tests::src/audit/engine.py::run_shard` - drift agents/specs/scripts-shard-survival/spec.md:215 says run_shard is tested 'with plain-dict items', but run_shard reads each item's `.id` attribute (Item protocol, src/audit/engine.py:119), so a dict cannot be an item; the test drives it with a frozen dataclass `Plain(id: str)` (src/tests/test_harness_engine.py:255-274)

### Doc5 (listed): User docs match implementation

- **Candidates judged:** 271 (every candidate; none left unjudged)
- **Open findings:** 29

- **Hits:**
  - `docs/differences-notes.md::Detailed notes on the differences > Arguments, defaults and errors::(unresolved)` - drift docs/differences-notes.md:219 says control entries a function does not use (`optimizer` for `confint`, `hettest`, `permutest`) raise; the code ignores them (confint(res, control={'optimizer': 'nlminb'}), hettest(..., control={'optimizer': ...}) and permutest(..., control={'optimizer': ...}) all return results, via control_names.read_elsewhere), as line 217 itself says
  - `docs/differences-notes.md::Detailed notes on the differences > Arguments, defaults and errors::bin/metafor-audit.sh` - drift docs/differences-notes.md:219 says control entries a function does not use (`optimizer` for `confint`, `hettest`, `permutest`) raise; the code ignores them (confint(res, control={'optimizer': 'nlminb'}), hettest(..., control={'optimizer': ...}) and permutest(..., control={'optimizer': ...}) all return results, via control_names.read_elsewhere), as line 217 itself says
  - `docs/differences-notes.md::Detailed notes on the differences > Names, results and printing::(unresolved)` - drift docs/differences-notes.md:53 says a sparse vcalc(sparse=True) 'stays a scipy.sparse.csr_array and prints as SciPy does, where R prints a dgCMatrix'; vcalc returns metafopy.core.labelled.SparseVCovMat (src/metafopy/methods/vcalc.py:551), whose __str__/__repr__ print R's dgCMatrix layout ('n x m sparse Matrix of class "dgCMatrix"', zeros as '.')
  - `docs/differences-notes.md::Detailed notes on the differences > Names, results and printing::src/metafopy/core/labelled.py::VCovMat` - drift docs/differences-notes.md:53 says a sparse vcalc(sparse=True) 'stays a scipy.sparse.csr_array and prints as SciPy does, where R prints a dgCMatrix'; vcalc returns metafopy.core.labelled.SparseVCovMat (src/metafopy/methods/vcalc.py:551), whose __str__/__repr__ print R's dgCMatrix layout ('n x m sparse Matrix of class "dgCMatrix"', zeros as '.')
  - `docs/differences-notes.md::Detailed notes on the differences > Names, results and printing::src/metafopy/models/rma_glmm.py::rma_glmm` - drift docs/differences-notes.md:33 says `res.control` of `rma_glmm` is the settings dict as passed; rma_glmm stores a `GLMMControl` dataclass built by make_control (src/metafopy/models/rma_glmm.py:78-, 539, 721)
  - `docs/differences-notes.md::Detailed notes on the differences > Names, results and printing::src/metafopy/models/rma_mv.py::rma_mv` - drift docs/differences-notes.md:19 says `rma_mv` fits have `k_eff` but no `ni`; rma_mv sets `ni=ni_f[not_na]` from the escalc frame's ni (src/metafopy/models/rma_mv.py:1369, field at src/metafopy/models/mv_result.py:177), docs/differences-notes.md:27 says `res.ddf` of a `robust(club_sandwich=True)` fit stays `None`; for an rma_mv fit robust() sets `ddf=pieces['dfs']`, the per-coefficient vector (src/metafopy/methods/robust.py:517), only rma_uni gets None (robust.py:366), docs/differences-notes.md:29 says `progbar=` defaults to `False` everywhere; `vif()` defaults to `progbar=True` (src/metafopy/methods/vif_mv.py:203 for rma_mv, src/metafopy/methods/vif.py:272, src/metafopy/methods/ls_vif.py:257), docs/differences-notes.md:33 says `res.control` of `rma_mv` is the settings dict as passed; rma_mv stores an `MVControl` dataclass built by make_control (src/metafopy/models/rma_mv.py:120-147, 683, 1372), docs/differences-notes.md:35 repeats that `ddf` of `robust(club_sandwich=True)` stays `None`; for rma_mv it is the vector of degrees of freedom (src/metafopy/methods/robust.py:517)
  - `docs/differences-notes.md::Detailed notes on the differences > Not ported::(unresolved)` - drift docs/differences-notes.md:229 lists `abbrev` as not ported; rma_uni and rma_ls take `abbrev` (src/metafopy/models/rma_uni.py:517, src/metafopy/models/rma_ls.py:118), docs/differences-notes.md:229 lists `retdat` as not ported; vcalc takes `retdat` (src/metafopy/methods/vcalc.py:303), docs/differences-notes.md:229 lists `skiphet` and `retopt` as not ported; selmodel accepts both as extra arguments (src/metafopy/models/selmodel.py:60, used at 932), docs/differences-notes.md:229 says `print(x, num=, legend=)` of `rma.uni` fits is not ported; RMAResult.summary takes `num` and `signif_legend` and documents `num` as metafor's `print(x, num = TRUE)` (src/metafopy/models/rma_uni.py:284-298), docs/differences-notes.md:231 says the column names metafor gives `Z.S`, `Z.G1` and `Z.G2` are not implemented; RMAMVResult names their columns as R does (`mf.s[[j]]`, `mf.g[[1]]`, `mf.g[[nvars]]` plus level) (src/metafopy/models/mv_result.py:193-195, 381-392)
  - `docs/differences-notes.md::Detailed notes on the differences > Optimisers and flat likelihoods::src/metafopy/models/mv_result.py::RMAMVResult.p` - drift docs/differences-notes.md:73 says `RMAMVResult.p` is the number of columns of `X`; rma_mv sets `p=p_eff`, the number of estimated coefficients (`beta_est.sum()`), so p equals p_eff when `beta` fixes coefficients (src/metafopy/models/rma_mv.py:1226, 1304)
  - `docs/differences-notes.md::Detailed notes on the differences > Optimisers and flat likelihoods::src/metafopy/models/rma_mv.py::rma_mv` - drift docs/differences-notes.md:73 says `RMAMVResult.p` is the number of columns of `X`; rma_mv passes `p=p_eff` (src/metafopy/models/rma_mv.py:1226, 1304)
  - `docs/differences.md::Differences from metafor > By design > Frames from escalc() print like R, selections do not::(unresolved)` - drift docs/differences.md:219 the output of print(dat.iloc[:3]) leaves out the column-header line pandas prints first ('   trial            author  year  tpos  ...  ablat   alloc        yi        vi')
  - `docs/differences.md::Differences from metafor > By design > digits= where R passes it to print()::(unresolved)` - drift docs/differences.md:302 the print(res) output leaves out the coefficient row the code prints after the header ('   -0.71  0.18  -3.97  <.01  -1.07  -0.36  ***'), docs/differences.md:308 the with_digits() output leaves out the header line the code prints before the row ('    pred     se  ci.lb  ci.ub  pi.lb pi.ub')
  - `docs/differences.md::Differences from metafor > Optimiser stop points > rma_mv with struct GEN stops at another maximum::(unresolved)` - drift docs/differences.md:398 shows res.tau2 as [3.13185220e+03 7.94227959e-04]; the code prints [3.13185214e+03 7.94227943e-04] (logLik and rho lines match)
  - `docs/differences.md::Differences from metafor > Optimiser stop points > selmodel on a flat likelihood::(unresolved)` - drift docs/differences.md:630 says delta is 1.022e-05 here; the code gives 1.013e-05 (1.01337247e-05), docs/differences.md:680 shows sel.tau2 as 0.057367159330093655; the code prints 0.057367183357824106, docs/differences.md:681 shows sel.delta as [1.02207630e-05 1.74123073e+00]; the code prints [1.01337247e-05 1.74123884e+00]
  - `docs/differences.md::Differences from metafor > Optimiser stop points::(unresolved)` - drift docs/differences.md:348 says R's optim BFGS and Nelder-Mead are ported only for rma_glmm (CM.EL) and selmodel; the code also uses the ported Nelder-Mead for location-scale fits with an identity link (constrOptim in src/metafopy/models/rma_ls_opt.py), for escalc(measure='RTET') (src/metafopy/effects/rtet.py) and for conv_fivenum's MLE (src/metafopy/effects/fivenum_estimators.py), and the ported BFGS in conv_2x2's diagnostic-cell solver (diag_cells in src/metafopy/effects/conv.py)
  - `docs/differences.md::Differences from metafor > Optimiser stop points::src/metafopy/models/rma_glmm.py::rma_glmm` - drift docs/differences.md:348 says R's optim BFGS and Nelder-Mead are ported only for rma_glmm (CM.EL) and selmodel; the code also uses the ported Nelder-Mead for location-scale fits with an identity link (constrOptim in src/metafopy/models/rma_ls_opt.py), for escalc(measure='RTET') (src/metafopy/effects/rtet.py) and for conv_fivenum's MLE (src/metafopy/effects/fivenum_estimators.py), and the ported BFGS in conv_2x2's diagnostic-cell solver (diag_cells in src/metafopy/effects/conv.py)
  - `docs/differences.md::Differences from metafor > Random numbers > Seeded draws reproduce R::(unresolved)` - drift docs/differences.md:953 the metafopy output of conv_fivenum omits the header line '    n  mn  q1  md  q3  mx      mean        sd' that the code prints before the rows
  - `docs/differences.md::Differences from metafor > Random numbers > Seeded draws reproduce R::src/metafopy/effects/fivenum.py::conv_fivenum` - drift docs/differences.md:953 the metafopy output of conv_fivenum omits the header line '    n  mn  q1  md  q3  mx      mean        sd' that the code prints before the rows
  - `docs/getting-started.md::Getting started > Back-transform and predict::(unresolved)` - drift docs/getting-started.md:96-97 shows the output of print(predict(res, transf=numpy.exp)) with a row label 0 and two-space column gaps (pi.ub 1.549); the code prints a blank first line, no row label (the slab is ''), single-space gaps and pi.ub 1.5490 ('   pred  ci.lb  ci.ub  pi.lb  pi.ub' / ' 0.4894 0.3441 0.6962 0.1546 1.5490')
  - `docs/guide/advanced.md::Selection models and location-scale models > Location-scale models::(unresolved)` - drift docs/guide/advanced.md:152-154 shows print(predict(res, newmods=[5, 25], newscale=[5, 25])) with two-space column gaps ('     pred      se   ci.lb ...' / '1  0.1466  0.0516 ...'); the code prints single-space gaps ('    pred     se  ci.lb  ci.ub   pi.lb  pi.ub' / '1 0.1466 0.0516 0.0455 0.2477 -0.2315 0.5247')
  - `docs/guide/advanced.md::Selection models and location-scale models > Selection models::(unresolved)` - drift docs/guide/advanced.md:48 lists the values of type as the parametric functions 'beta', 'halfnorm', 'negexp', 'logistic', 'power', 'negexppow' or a step function, without 'for example'; selmodel (src/metafopy/models/selmodel.py:386-387) also accepts 'trunc' and 'truncest' and the variants 'halfnorm1', 'negexp1', ... 
  - `docs/guide/datasets.md::Datasets::(unresolved)` - drift docs/guide/datasets.md:16 shows the first five datasets as ['dat.assink2016', 'dat.bangertdrowns2004', 'dat.bcg', 'dat.berkey1998', 'dat.cohen1981'], but list_datasets()[:5] now returns ['dat.aloe2013', 'dat.anand1999', 'dat.anker2025', 'dat.assink2016', 'dat.axfors2021']
  - `docs/guide/datasets.md::Datasets::src/metafopy/datasets/base.py::list_datasets` - drift docs/guide/datasets.md:16 shows the first five datasets as ['dat.assink2016', 'dat.bangertdrowns2004', 'dat.bcg', 'dat.berkey1998', 'dat.cohen1981'], but list_datasets()[:5] now returns ['dat.aloe2013', 'dat.anand1999', 'dat.anker2025', 'dat.assink2016', 'dat.axfors2021']
  - `docs/guide/diagnostics.md::Diagnostics > Heteroscedastic heterogeneity::(unresolved)` - drift docs/guide/diagnostics.md:165-175 shows only the output of the first print; the example's second print, hettest(res, test=['lrt', 'ksn'], boot=200, seed=1234), also prints two result blocks (Likelihood ratio test X^2(df = 12) = 6.0136, p = 0.9095 and Kolmogorov-Smirnov test (normal) statistic = 0.1624, p = 0.4472, each with Bootstrapping: Yes (199/200 iterations)) that the output block leaves out
  - `docs/guide/effect-sizes.md::Effect sizes > Correlations and proportions::(unresolved)` - drift docs/guide/effect-sizes.md:104-106: print(predict(res, transf=numpy.tanh)) prints a blank line, then a header and row with no row index `0` and different column widths (`   pred  ci.lb  ci.ub   pi.lb  pi.ub` / ` 0.1488 0.0878 0.2087 -0.0371 0.3248`), then a blank line, docs/guide/effect-sizes.md:119-121: print(predict(..., transf=transf.ilogit)) likewise prints a leading blank line, no row index `0` and other column widths (` 0.7575 0.6605 0.8337 0.4661 0.9179`), then a blank line
  - `docs/guide/emmeans.md::Estimated marginal means > Contrasts and multiplicity adjustment::(unresolved)` - drift docs/guide/emmeans.md:73-85: the pairs/contrast summaries print left-aligned contrast labels and three significant digits (` alternate / random     1.307 0.458 Inf    1   0.763  0.7255`), not `    alternate / random  1.307 0.4579 Inf    1  0.7633  0.7255`; the same holds for every row of both tables
  - `docs/guide/emmeans.md::Estimated marginal means > Reference grid and marginal means::(unresolved)` - drift docs/guide/emmeans.md:29-35: print(emmprep(res)) prints only `alloc ablat prediction SE df` columns (e.g. ` alternate   33.5     -0.619 0.294 Inf`), left-aligned levels, no asymp.LCL/asymp.UCL columns and no `Confidence level used: 0.95` line, docs/guide/emmeans.md:47-53: means.summary(type='response') prints three significant digits and left-aligned levels (` alternate     0.538 0.1580 Inf     0.303     0.957`), not ` alternate   0.5382   0.158 Inf    0.3028    0.9568`
  - `docs/guide/models.md::Fitting models > Methods you call on a fit > anova::(unresolved)` - drift docs/guide/models.md:266-268: the printed anova(full, reduced) table shows NaN in empty cells, R^2 as 87.7302 and all columns on one line; the code prints blank cells, R^2 as 87.7302% and, at the default width of 80, wraps R^2 onto a second block
  - `docs/guide/models.md::Fitting models > Methods you call on a fit > predict::(unresolved)` - drift docs/guide/models.md:205-210: the printed predict output is shown pandas-style (rows 0, 1, 2; second table with row label 0 and pi.ub 1.549); the code prints R-style rows 1, 2, 3, the transformed table without a row label and with pi.ub 1.5490, preceded by a blank line
  - `docs/guide/models.md::Fitting models > The model and the estimator::(unresolved)` - drift docs/guide/models.md:58-69: the table of `method` values (it lists aliases such as 'FE', 'CE' and 'MP') omits 'CO' and 'VC', which src/metafopy/models/tau2.py ESTIMATORS accepts as aliases of 'HE' (rma_uni docstring: "'HE' (also 'CO' and 'VC')")


### A2 (sampled, full tree): Except tuple coverage

No hits

Confirmation: 328 .py files inspected (src/metafopy, src/audit, src/tests, bin); 207 try blocks listed, those whose body touches disk or a json/pandas parser judged: src/audit/engine.py:39, src/audit/engine.py:57, src/audit/report.py:88, src/audit/scripts.py:56, src/audit/scripts.py:206 (ValueError covers UnicodeDecodeError and JSONDecodeError), src/audit/apidiff.py:73 (inspect.getsource; linecache swallows UnicodeDecodeError), src/tests/correctness/cases.py:376; src/tests/test_docs_examples.py:31 left to the sweep (filesystem-envelope-gap); no translating decorator or contextmanager over I/O found.

### A3 (sampled, full tree): Domain-error layering

- src/metafopy/methods/predict.py:245 - raises stdlib ValueError('unsupported expression') in a package defining metafopy.errors
- src/audit/pycall.py:99 - raises stdlib ValueError for an unknown value tag; src/audit defines its own *Error classes
- src/audit/rlib.py:801 - r_stop raises stdlib ValueError; src/audit defines its own *Error classes
- src/audit/rlib.py:807 - r_stopifnot raises stdlib ValueError; src/audit defines its own *Error classes
- src/audit/recall.py:59 - raises stdlib ValueError for duplicate recall items
- src/audit/steps.py:84 - raises stdlib RuntimeError for a metafor version mismatch
- src/audit/steps.py:87 - raises stdlib RuntimeError
- src/audit/steps.py:105 - raises stdlib RuntimeError
- src/audit/steps.py:110 - raises stdlib RuntimeError
- src/audit/rdriver.py:48 - raises stdlib RuntimeError when an R script fails
- src/audit/scripts.py:67 - raises stdlib RuntimeError when the metafor source cannot be fetched
- src/audit/scripts.py:209 - raises stdlib RuntimeError when the script list cannot be read

Confirmation: 328 .py files inspected; 161 files contain raise statements; src/tests and bin define no *Error class of their own (bin/platform-probe.py:109 and :442 KeyError therefore out of the predicate); every metafopy raise outside the row above uses metafopy.errors.*, NonFiniteError or a type outside the predicate's ValueError/RuntimeError/KeyError/TypeError set.

### A12 (sampled, full tree): assert misuse

- src/metafopy/methods/profile.py:309 - `assert ci.random is not None` is caught by `except ... AssertionError` at line 312 to fall back to the SE-based xlim; under python -O the AttributeError escapes instead
- src/metafopy/plot/qqnorm.py:242 - `assert temp_lb is not None` is the only rejection of an unrecognised string `label` when no envelope is drawn (user input)
- src/metafopy/plot/qqnorm.py:243 - `assert temp_ub is not None`, same user-input path as line 242

Confirmation: asserts outside test files found in 17 files under src/metafopy, src/audit and bin (src/tests modules, including helpers, treated as test files); every other assert judged a type-narrowing invariant guarded by earlier validation or control flow; the `# auditme: accept[assert-in-product-code]` markers at src/metafopy/models/rma_ls_fit.py:709 and :736 do not name A12 and were judged (invariants, not hits).

### A5 (sampled, full): Two sources of truth

- src/metafopy/plot/forest.py:2020 - addpoly on a predict result merges `pi_se`/`pi_dist`/`pi_ddf` arguments with the stored `x.pi_se`/`x.pi_dist`/`x.pi_ddf` by undocumented precedence: an explicit `pi_se`/`pi_ddf` beats the stored value though the docstring says "taken from the model", and `pi_dist` is replaced by the stored value only when it equals its default `'norm'` (an explicit `'norm'` silently loses, an explicit `'t'` wins)

Confirmation: 328 files inspected; matching constructs (a parameter and a stored attribute or config field for the same value) in 49 functions/methods across 36 files; every other site resolves `None` to the stored value, as its docstring states.

### A6 (sampled, full): Stage method dual-path

No hits

Confirmation: 328 files inspected; 648 non-`__init__` methods; methods that both assign `self.<attr>` and return a value: 1 (src/audit/engine.py:171 `PyLauncher.start`, not a pipeline-stage class and not tuple-returning).

### A7 (sampled, full): Constructor purity

- src/metafopy/models/mv_lik.py:128 - `Factorization.__init__` performs expensive compute: a dense Cholesky (`scipy.linalg.cho_factor`) or sparse LU (`scipy.sparse.linalg.splu`) of the matrix argument

Confirmation: 328 files inspected; explicit `__init__` in 9 classes across 8 files (formatters.py, methods/confint.py, methods/anova.py, core/rng.py, models/mv_lik.py, effects/measures.py x2, audit/rlib.py, audit/engine.py); none mutates an argument (`Inputs` copies `values`); no sweep hit under this predicate.

### A8 (sampled, full): Late dataclass mutation

- src/audit/generate.py:351 - `gen = Generated(...)` (line 176) then `gen.uncovered = sorted(set(gen.uncovered))`
- src/audit/report.py:212 - `ctx = audit.steps.Context(...)` (line 209) then `ctx.coverage = doc.get('coverage', {})`
- src/metafopy/models/rma_mv.py:826 - `sinfo = SInfo(...)` (line 780, or from `process_s_before`) then `sinfo.factors = [f.take(keep) ...]`
- src/metafopy/models/rma_mv.py:894 - same `sinfo` then `sinfo.factors = [f.drop_unused() ...]`

Confirmation: 328 files inspected; 174 `@dataclass` classes defined (156 frozen); every function scanned for a name bound to a project dataclass constructor call and later assigned through `name.attr =`, `setattr` or `object.__setattr__`; 4 sites in 3 files.

### A9. Pure-named function side effects

- src/audit/steps.py:74 - `check_r` runs R through `audit.rdriver.run` (subprocess) and reads/tests files (`read_text`, `lock_path.exists`) under a pure `check_` name
- src/metafopy/core/rng.py:263 - `get_rng` reads the clock (`time.time()`) to seed and assigns the module global on first use
- src/audit/pycall.py:65 - `resolve_value` reads a bundled dataset from disk (`metafopy.datasets.load`) for a `$data` tag under a pure `resolve_` name
- src/tests/correctness/cases.py:509 - `build_tree_r` reads a bundled tree file (`metafopy.datasets.load_tree`) under a pure `build_` name

Confirmation: 328 files inspected; 184 pure-prefixed functions examined (direct and same-package transitive I/O). `src/audit/steps.py:120 build_universe` is already reported by the sweep (`build-name-has-side-effect`). `load_*` functions reading the file/directory they name (refutil.load_phase*, reference_data.load_case, cases.load_*, audit.cases.load_fixtures, pyrun.load_shard, datasets.load_tree) are carved out; `warnings.warn` and `setrlimit` are not I/O under this predicate.

### A10. Predicate semantics

- src/metafopy/methods/selmodel.py:57 - `has_tau2` returns `method not in FIXED_EFFECTS and not x.tau2_fix`: answers "has an estimated tau2" (the hidden `not tau2_fix` conjunct), not "has tau2"
- src/audit/rlib.py:1238 - `is_text` has a hidden `arr.size > 0` conjunct: an empty string array is not text

Confirmation: 328 files inspected; 28 `is_`/`has_`/`should_` functions in 22 files read end to end.

### A13. Same value to multiple structural slots

- src/metafopy/methods/matreg.py:752 - `MatregResult(p=m, parms=m)` passes `m` to two slots
- src/metafopy/methods/accessors.py:151 - `NobsAll(studies=obj.k, not_na=obj.k)` passes `obj.k` to two slots
- src/metafopy/methods/selmodel.py:607 - `SelProfileResult(I2=empty, H2=empty)` aliases one NaN array into two slots
- src/metafopy/methods/mv_profile.py:237 - `MVProfileResult(I2=empty, H2=empty)` aliases one NaN array into two slots
- src/metafopy/methods/emm_links.py:98 - `Link('identity', metafopy.transf.arr, metafopy.transf.arr, ...)` passes the same function as link and inverse link
- src/metafopy/core/glmm.py:179 - `LinkTerms(mu, mu, mu, ...)` passes one array to three derivative slots for the log link
- src/metafopy/models/rma_mv.py:1304 - `RMAMVResult(p=p_eff, p_eff=p_eff)` passes `p_eff` to two slots
- src/metafopy/models/glmm_fit.py:692 - `ModelFit(..., p, p, k, 'FE')` passes `p` to `parms` and `p_eff`
- src/metafopy/models/glmm_fit.py:804 - `ModelFit(..., p, p, ...)` passes `p` to `parms` and `p_eff`
- src/metafopy/models/mv_structs.py:232 - `Group(nlevels=nlevels, nlevels_f=nlevels)` passes one tuple to two slots
- src/metafopy/models/rma_glmm.py:698 - `GLMMResult(int_incl=int_incl, intercept=int_incl)` passes one value to two slots
- src/tests/correctness/glmm_formulas.py:562 - `Problem(..., ni, ni)` aliases one array into `n1` and `n2`

Confirmation: 328 files inspected; 2794 class-constructor calls in 245 files checked by AST for a repeated non-literal argument. Literal NaN fill values and identical row/column labels of a square matrix (`index`/`columns`, `LabelledMatrix`/`VCovMat`/`RFrame` dimnames) and `facecolors`/`edgecolors` in a matplotlib call are not counted.

### A14. Repeated computation per request

- src/audit/generate.py:197 - `universe.method_for(generic, classes)` is called for every fixture x generic (again at generate.py:236, apidiff.py:160, universe.py:72), and each call rebuilds the per-generic `by_class` index over all of `Universe.methods` (universe.py:55); the index is a property of the loaded universe and could be built once per run

Confirmation: 328 files inspected (src/metafopy, src/audit, src/tests, bin); state-valued helpers checked: `audit.report.git_head` (one call per report), `audit.universe.snapshot_path`, `audit.rdriver.environment` (fresh port per R process by design), `metafopy.mfopt.*` option resolvers (no function calls one more than twice), `audit.ledger.Ledger.load`, `Universe` lookups.

### A15. Duplicate constructor / serialisation logic

- src/metafopy/models/glmm_fit.py:402 - `ModelFit(...)` assembled field by field (19 fields, the same NAN placeholders and `beta=...[:p]`, `vb=...[:p, :p]` slicing) at glmm_fit.py:402, 458, 489, 568, 692, 804, 820, 835 with no shared helper
- src/metafopy/methods/binary.py:442 - `CumulResult(...)` built from the same `col[...]` field set (21 fields) as leave1out.py:484; the two assemblies are copied
- src/metafopy/methods/binary.py:344 - `Leave1outResult(...)` built from the same `col[...]` field set as leave1out.py:315
- src/metafopy/methods/predict.py:719 - `LocationModel(...)` assembled inline (17 of 20 fields identical) instead of calling `uni_location_model` (predict.py:455); also predict.py:797
- src/metafopy/methods/predict.py:1363 - `PredictResult(...)` assembled with the same 15 of 18 fields as predict.py:1055
- src/metafopy/methods/selmodel.py:691 - `LRTResult(...)` (and the R2 computation before it) repeats anova.py:909
- src/metafopy/methods/ls_anova.py:334 - `MVLRTResult(...)` repeats the assembly at anova.py:1164
- src/metafopy/methods/ls_resid.py:104 - `RanefResult(...)` copies every field from `out` exactly as ranef.py:94 does
- src/metafopy/methods/vif_mv.py:178 - `VIFEntry(...)` assembled with the same six field expressions as vif.py:376 and ls_vif.py:135
- src/metafopy/methods/robust.py:506 - `RobustInfo(...)` assembled with the same six field expressions as robust.py:357
- src/metafopy/methods/anova.py:1204 - `WaldResult(...)` assembled with the same field set as anova.py:533 and anova.py:703
- src/metafopy/plot/forest.py:2315 - `PredGeometry(...)` built with the same eight positional arguments as forest.py:1541
- src/metafopy/bias/trimfill.py:96 - `MetaData(...)` assembled with the same nine fields (six identical values) as regtest.py:381 and trimfill.py:253
- src/metafopy/models/glmm_fit.py:441 - `GLMMProblem(...)` built positionally with the same nine-argument layout (`y, size, offset, study, ..., family, ..., link`) at glmm_fit.py:399, 441, 449, 480, 563
- src/metafopy/core/glmm.py:645 - `GLMMFit(...)` in `stage_one_result` repeats the eight-field assembly of glmm.py:630
- src/metafopy/models/rma_glmm.py:663 - all-NaN `FitStats(math.nan, ...)` duplicated from binary.py:41
- src/tests/correctness/formulas.py:2486 - `MVTests(...)` built from the same eight positional expressions as formulas.py:2481 and formulas.py:2656

Confirmation: 328 files inspected; every call of a class defined in the repository with five or more arguments compared across call sites by field set and field expression; sites that only return same-named locals (`Out` in effects/measures.py, `Modes`, `Deleted`, `ConfintResult`, clubsandwich `WaldResult`) and the per-source adapters in methods/fitview.py not counted.

### A24. Memoisation of subprocess / shell-out calls

No hits

Confirmation: 328 files inspected; 11 files with `subprocess` calls (src/audit/{recall,rdriver,engine,report,scripts}.py, bin/logo.py, src/tests/test_{harness_compare,signatures,harness_scripts,harness_engine,interactive}.py); `git_head` runs once per report, `check_r`/`build_universe`/`list_scripts` once per run, `recall.checkout`/`run_audit` once per commit, `resident_bytes` polls a changing value, `logo.render` renders distinct outputs.

### P3. Repeated expensive computation per request

- src/metafopy/models/rma_ls_fit.py:289 - `metafopy.core.linalg.invcalc(X, w)` computed twice with the same arguments on consecutive lines (288 and 289) in `marginal_residuals`
- src/metafopy/core/glmm.py:622 - `saturated_loglik(prob.family, prob.y, prob.size)` recomputed though the same value is already held in `sat` (glmm.py:588) in `fit_glmm`

Confirmation: 328 files inspected; every function scanned for identical non-trivial call expressions repeated within its body (194 non-test files); repeats in exclusive branches, loop iterations with changed state (clogit Newton steps, glmm `information`/`row_terms`, vcalc `is_positive_definite` after `near_pd`, rma_mv `make_term` after mutation) and cheap scalar quantiles not counted.

### Phase B (sampled, full tree): A16, A18, A20, A21, A22, A23, A25

### A16 (sampled): CLI entry envelope

No hits

Confirmation: 328 .py files inspected under src/metafopy, src/audit, src/tests, bin; 0 files with matching constructs (no `cli/` directory; `src/audit/cli.py` has no `run_*` function and no cappa-typed parameter).

### A18 (sampled): Hot-path setup

No hits

Confirmation: 328 files inspected; 0 files with matching constructs (no `on_step`/`on_log`/`on_evaluate`/`on_batch_*` callback method; `src/tests/correctness/glmm_formulas.py:1371` `on_log_tau2` is a nested formula function, not a callback).

### A20 (sampled): Mutable module-level singleton

- src/tests/glmm_cases.py:44 - `BCG` mutable dict literal read as `glmm_cases.BCG` by src/tests/glmm_package_cases.py, src/tests/glmm_extras_cases.py, src/tests/test_glmm_extras.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_cases.py:46 - `COL` mutable dict literal read as `glmm_cases.COL` by src/tests/glmm_package_cases.py, src/tests/glmm_extras_cases.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_cases.py:48 - `NW` mutable dict literal read as `glmm_cases.NW` by src/tests/glmm_package_cases.py, src/tests/glmm_extras_cases.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_cases.py:50 - `IRR` mutable dict literal read as `glmm_cases.IRR` by src/tests/glmm_package_cases.py, src/tests/glmm_extras_cases.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_cases.py:58 - `CASES` mutable dict literal read as `glmm_cases.CASES` by src/tests/test_rma_glmm.py; wrap in types.MappingProxyType(...)
- src/tests/test_rma_mv.py:80 - `CASES` mutable dict literal read as `test_rma_mv.CASES` by src/tests/test_mv_opt.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_extras_cases.py:31 - `CASES` mutable dict literal read as `glmm_extras_cases.CASES` by src/tests/test_glmm_extras.py; wrap in types.MappingProxyType(...)
- src/tests/glmm_extras_cases.py:103 - `PREDICT` mutable dict literal read as `glmm_extras_cases.PREDICT` by src/tests/test_glmm_extras.py; wrap in types.MappingProxyType(...)
- src/tests/test_rma_ls.py:99 - `FITS` mutable dict literal read as `test_rma_ls.FITS` by bin/platform-probe.py; wrap in types.MappingProxyType(...)
- src/tests/test_nchg.py:18 - `ARGS` mutable dict literal read as `test_nchg.ARGS` by bin/platform-probe.py; wrap in types.MappingProxyType(...)
- src/tests/correctness/test_properties.py:203 - `MV_CONTROL` mutable dict literal read as `correctness.test_properties.MV_CONTROL` by bin/platform-probe.py; wrap in types.MappingProxyType(...)

Confirmation: 328 files inspected; 261 module-level dict/list/set literals, comprehensions or bare-builtin calls, each checked for `from <module> import <name>`, `<module>.<name>` and `sys.modules['<module>'].<name>` readers across src/metafopy, src/audit, src/tests and bin; 17 read elsewhere, of which 6 are the sweep's accepted `mutable-module-level-exported` hits (src/audit/rlib.py:46, src/audit/rlib.py:866, src/metafopy/effects/measures.py:304, src/metafopy/mfopt.py:56, src/metafopy/models/rma_mv.py:426, src/metafopy/printing.py:100) and are not repeated.

### A21 (sampled): Lazy init race

No hits

Confirmation: 328 files inspected; 1 file with a matching construct: src/metafopy/core/rng.py:266 (`get_rng`, `if GLOBAL is None: GLOBAL = RRandom(...)`), not reached on a concurrent path (the parallel and `cl` paths of vif, ls_vif and vif_mv draw per-iteration seeds in the calling thread before dispatch; the audit's threaded script runner runs translated Python in subprocesses).

### A22 (sampled): Fall-through after partial handling

- src/audit/rlib.py:73 - `if name.startswith('transf.'): try: return resolve_function(...) except MissingFunctionError: pass` falls through to the SAME_FUNCTIONS / BASE / METAFOR_FUNCTIONS lookups as if the name were not a `transf.` name

Confirmation: 328 files inspected; 34 `try` blocks directly inside an if/elif/else branch (or a with-block in one) examined, 1 with a returning body and a non-terminating handler.

### A23 (sampled): Discarded multi-return

No hits

Confirmation: 328 files inspected; 247 functions annotated to return a fixed-length tuple, plus dataclass-returning `sel_lik.evaluate`; files with matching constructs (index or discard of a project tuple return): src/metafopy/methods/hettest.py, src/metafopy/methods/vif.py, src/metafopy/methods/ls_vif.py, src/metafopy/methods/vif_mv.py, src/metafopy/methods/deltamethod.py, src/metafopy/core/glmmadaptive.py, src/metafopy/core/glmm.py, src/metafopy/core/clubsandwich.py, src/metafopy/core/clogit.py, src/metafopy/core/nchg.py, src/metafopy/models/mv_lik.py, src/metafopy/models/rma_peto.py, src/metafopy/models/mv_opt.py, src/metafopy/models/selmodel.py, src/metafopy/models/sel_lik.py, src/metafopy/plot/*.py (plain_plot ticks), src/tests/test_audit8_values_a.py, src/tests/test_harness_compare.py, src/tests/correctness/formulas.py, src/tests/correctness/glmm_formulas.py, src/tests/correctness/test_derived.py; no discarded component signals hidden state (compvif's `used` indices are fixed by the group, already recorded by the main path).

### A25 (sampled): Shallow immutable wrapper

- src/audit/steps.py:217 - `STEPS` tuple read as `audit.steps.STEPS` by src/audit/cli.py and src/audit/report.py holds instances of `Step`, a dataclass defined without `frozen=True` (src/audit/steps.py:61)

Confirmation: 328 files inspected; 303 module-level `types.MappingProxyType(...)`, `frozenset(...)`, `tuple(...)` or tuple-literal containers, 82 read elsewhere, members checked for dict/list/set, one tuple deep, and non-frozen project dataclasses; 2 with mutable members, of which src/metafopy/plot/setlab_table.py:14 (`SETLAB`) is the sweep's accepted `mutable-members-in-immutable-wrapper` hit and is not repeated.

### S1-S4 (full tree)

### S1. Type annotations on every signature

No hits (non-test source: src/metafopy, src/audit, bin).

- src/tests: 0 missing parameter annotations, 1 missing return annotation (src/tests/test_harness_scripts.py:40 `stand_in`), 1 module affected, 133 of 134 modules fully annotated.

No-hit confirmation: 194 non-test files checked with `ruff check --select ANN` (ANN401 `Any` excluded as not a missing annotation); every one passed.

### S2. PascalCase / UPPER_SNAKE / lower_snake

No hits

No-hit confirmation: every class, function, argument and module-level name in the 328 in-scope files scanned by AST; every non-snake function or argument name is a metafor name or matrix symbol (`QE`, `logLik`, `X`, `REML`, `nAGQ`, `withS`), which the project keeps by design (ruff N802/N803/N806 ignored for metafor-style names), or a protocol name (`T`, `visit_BoolOp`); every class is PascalCase; script names in bin/ are consistently hyphenated.

### S3. TYPE_CHECKING block placement

No hits

No-hit confirmation: three `if typing.TYPE_CHECKING:` blocks (src/metafopy/methods/permutest.py:41, src/metafopy/methods/emm_grid.py:45, src/metafopy/models/rma_uni.py:39), each directly after the last import and before `__all__`; no other form is used in the tree.

### S4. Project-specific rule files (AGENTS.md conventions and claude/rules/metafopy.md)

Rule: randomised procedures use `metafopy.core.rng`, never `numpy.random`, for ported procedures.

- src/metafopy/plot/selmodel.py:179 - `ci='boot'` bands of metafor's `plot.selmodel` drawn with `numpy.random.default_rng(...).multivariate_normal` instead of R's generator (`metafopy.methods.simulate.mvrnorm` with `RRandom` exists)
- src/metafopy/methods/emm_adjust.py:143 - `mvt` adjustment (emmeans/mvtnorm `pmvt`, randomised QMC) seeded with `numpy.random.default_rng(MVT_SEED)`
- src/metafopy/methods/emm_adjust.py:147 - same, multivariate t two-sided branch
- src/metafopy/methods/emm_adjust.py:157 - same, normal one-sided branch
- src/metafopy/methods/emm_adjust.py:159 - same, multivariate t one-sided branch

Rule: result classes are frozen dataclasses with a metafor-style `__str__`.

- src/metafopy/methods/anova.py:161 - `ListAnovaResult` (metafor `list.anova.rma`) is a mutable `list` subclass, not a frozen dataclass
- src/metafopy/methods/confint.py:107 - `ConfintList` (metafor `list.confint.rma`) is a mutable `list` subclass, not a frozen dataclass
- src/metafopy/methods/profile.py:40 - `ProfileList` (metafor's list of profiles) is a mutable `list` subclass, not a frozen dataclass
- src/metafopy/plot/baujat.py:35 - public `BaujatResult` has no `__str__`
- src/metafopy/plot/cumul.py:42 - public `PlotCumulResult` has no `__str__`
- src/metafopy/plot/forest.py:60 - public `ForestResult` has no `__str__`
- src/metafopy/plot/funnel.py:61 - public `FunnelResult` has no `__str__`
- src/metafopy/plot/gosh.py:36 - public `PlotGoshResult` has no `__str__`
- src/metafopy/plot/infl.py:36 - public `PlotInflResult` has no `__str__`
- src/metafopy/plot/labbe.py:43 - public `LabbeResult` has no `__str__`
- src/metafopy/plot/llplot.py:37 - public `LlplotResult` has no `__str__`
- src/metafopy/plot/permutest.py:33 - public `PlotPermutestResult` has no `__str__`
- src/metafopy/plot/qqnorm.py:37 - public `QQNormResult` has no `__str__`
- src/metafopy/plot/radial.py:36 - public `RadialResult` has no `__str__`
- src/metafopy/plot/regplot.py:41 - public `RegplotResult` has no `__str__`
- src/metafopy/plot/rma.py:38 - public `PlotRmaResult` has no `__str__`
- src/metafopy/plot/selmodel.py:36 - public `SelPlotResult` has no `__str__`
- src/metafopy/plot/vifplot.py:28 - public `PlotVifResult` has no `__str__`

Rule: no leading-underscore names (AGENTS.md).

- src/metafopy/printing.py:229 - `_repr_html_` defined (name fixed by IPython's display protocol; its marker names the sweep rule, not S4)
- src/metafopy/models/rma_uni.py:313 - `_repr_html_` defined (as above)
- src/metafopy/core/labelled.py:328 - `_constructor` property defined (name fixed by pandas' subclassing protocol; no marker)
- src/metafopy/core/labelled.py:388 - `_constructor` property defined (as above)

Rule: link the R source and papers in docstrings (AGENTS.md). Modules that port R or metafor code and contain no link anywhere:

- src/metafopy/core/bfgs.py:1 - port of R's `vmmin` (`src/appl/optim.c`) names the file, no link
- src/metafopy/core/bvn.py:1 - port of mvtnorm `src/mvt.f` names the file, no link
- src/metafopy/core/farebrother.py:1 - port of CompQuadForm `ruben.cpp` names the file, no link
- src/metafopy/core/optim.py:1 - port of R's `optimize`/Nelder-Mead/`optimhess` names the sources, no link
- src/metafopy/core/rng.py:1 - port of R's Mersenne-Twister and `sample()`, no link
- src/metafopy/core/zeroin.py:1 - port of R's `R_zeroin2`, no link
- src/metafopy/effects/rtet.py:1 - port of metafor's `.rtet`, no link to the R source
- src/metafopy/metadata.py:1 - mirrors argument handling of metafor's `rma.uni`/`rma.mv`, no link
- src/metafopy/models/rma_ls.py:1 - metafor's `rma.ls` scale model, no link to the R source
- src/metafopy/models/rma_ls_opt.py:1 - port of R's `constrOptim`, no link
- src/metafopy/models/rma_ls_result.py:1 - metafor's `rma.ls` result and printout, no link
- src/metafopy/models/sel_result.py:1 - metafor's selection-model printout, no link
- src/metafopy/models/tau2.py:1 - follows metafor's `rma.uni()` tau^2 code path, no link
- src/metafopy/models/robust_print.py:1 - ports `print.rma.uni`/`print.rma.mv` robust lines, no link
- src/metafopy/plot/histplot.py:1 - R's `hist()` and metafor's `.coltail()`, no link
- src/metafopy/plot/setlab_table.py:1 - metafor's `.setlab` table, no link to the R source

No-hit confirmation: imports use full module paths (no `import x as y` or `from` import outside package `__init__.py` re-exports), no `from __future__`, domain errors in `metafopy.errors` (the one other exception class in the package, src/metafopy/transf.py:313 `NonFiniteError`, is an internal signal caught in the same module), every other result class is a frozen dataclass with a `__str__`, R metafor-as-second-oracle rule not checkable per file; single quotes are enforced by ruff Q.

### P1 (sampled, full): O(n^2) on real-size inputs

- src/metafopy/methods/robust.py:400 - loop over every level of each random-effects factor scans the whole cluster-code array per level (`codes[f == level]`): O(levels x k), quadratic when levels approach k; one groupby pass would do
- src/metafopy/methods/rcalc.py:272 - loop over every study rescans the full `ids` array (`ids == level`): O(studies x rows)
- src/metafopy/effects/reshape.py:752 - loop over every study does a full-frame boolean `.loc[study_rank == s]`: O(studies x rows)
- src/metafopy/effects/reshape.py:772 - second per-study loop rescans the full `study_rank` array (`flatnonzero(study_rank == s)`): O(studies x rows)

Confirmed run: every nested Python loop in src/metafopy, src/audit and bin listed by AST and each per-iteration full-array `==` scan checked; the rest are bounded by parameter count, grid size or colours, are linear in cells, or index k x k matrices whose O(k^2) is inherent.

### P2 (sampled, full): Blocking I/O in hot paths

No hits

Confirmed run: the only library file I/O is the cached dataset read in src/metafopy/datasets/base.py; no training loop, request handler or callback in scope does I/O.

### Sec1 (sampled, full): Secrets in source

No hits

Confirmed run: key, token, password, secret, private-key and provider-prefix patterns grepped across every scoped .py file and bin/*.sh.

### Sec2 (sampled, full): Input validation at system boundaries

- src/audit/cli.py:103 - `--recall` value parsed with bare `int(n)` and looked up in `corpus.audits[number]` (line 108) without validation: a non-number or unknown audit number ends in a ValueError or KeyError traceback, not a usage error

Confirmed run: every argparse parser and `sys.argv` read in src/audit and bin checked (gallery.py, logo.py, platform-probe.py, metafor-audit.py, audit/cli.py, audit/pyrun.py, audit/scripts.py); `--jobs` is clamped downstream, `--steps` is checked, `--group` uses choices.

### Sec3 (sampled, full): Command injection

No hits

Confirmed run: every subprocess call in scope passes an argument list without shell=True; bin/*.sh quote every expansion and pass arrays to docker and bash -c as positional arguments.

### Sec4 (sampled, full): Path traversal

- src/metafopy/datasets/base.py:53 - public `load(name)` / `load_tree(name)` join the caller's dataset name into the package data path (`joinpath(f'{name}.csv')`) with no containment check, so `'dat./../../x'` reads any .csv reachable relative to the package

Confirmed run: every joinpath and path built from an argument in src/metafopy, src/audit and bin checked; audit job paths come from the harness's own job files.

### Sec5 (sampled, full): PII in logs

No hits

Confirmed run: no `logging` or logger use anywhere in scope.

### T3 (sampled, full tree): Pipeline orchestrators have an end-to-end test

- src/audit/cli.py:118 - `main()`, the audit pipeline orchestrator run by `bin/metafor-audit.sh` (steps in sequence, then `audit.report.write`), is called by no test; tests cover individual steps, the engine and the report writer only
- bin/logo.py:265 - `main()` orchestrates the logo build (SVGs, PNG renders, favicons, ICO, animated SVG/GIF) and has no test

Confirmed: `bin/platform-probe.py` `main()` is run end-to-end by `src/tests/test_platform_probe.py`; no other orchestrator found.

### D1 (sampled, full tree): Every imported third-party module is declared in pyproject.toml

- src/tests/test_showcase_notebook.py:6 - imports `nbformat`, not declared in pyproject.toml (present only transitively through `nbclient`)
- bin/logo.py:28 - imports `fontTools`, not declared in pyproject.toml (declared only in the script's inline PEP 723 block)
- bin/logo.py:31 - imports `PIL`, not declared in pyproject.toml (declared only in the script's inline PEP 723 block)
- bin/logo.py:32 - imports `uharfbuzz`, not declared in pyproject.toml (declared only in the script's inline PEP 723 block; absent from uv.lock)

Confirmed: every other third-party import (numpy, pandas, scipy, matplotlib, formulaic, altair, vl_convert, hypothesis, pytest, nbclient) is declared.

### D2 (sampled, full tree): Unused declarations

- pyproject.toml:45 - dev dependency `ipykernel` is not imported, not run as a command, and registers no pytest11 entry point (used only implicitly as the `python3` kernel nbclient starts)
- pyproject.toml:48 - dev dependency `pandas-stubs` is not imported, run or loaded through an entry point (type stubs read by mypy)
- pyproject.toml:54 - dev dependency `scipy-stubs` is not imported, run or loaded through an entry point (type stubs read by mypy)

Confirmed: formulaic, matplotlib, numpy, pandas, scipy, altair, vl-convert-python, hypothesis and nbclient are imported; mypy, ruff, pytest and pre-commit are run in `.github/workflows/ci.yml` or `.pre-commit-config.yaml`; pytest-cov and pytest-mpl register `pytest11` entry points and no `-p no:` disables them. No mkdocs host is declared in pyproject.toml.

### D3: Lockfile coherence

No hits

Confirmed: /tmp/audit-metafopy.jsonl has no lockfile-drift hit.

### Doc3 (sampled, full tree): README.md and AGENTS.md consistent

- README.md:52 - calls R metafor "the numerical reference", while AGENTS.md:14 makes it the second oracle, after the publication defining each method
- README.md:57 - fixture regeneration runs plain `Rscript`, while AGENTS.md:14 requires `LC_ALL=C.UTF-8 Rscript`
- README.md:58 - says `generate_fixtures.R` refreshes `src/tests/reference/*.json`, while AGENTS.md:14 says the fixtures are generated by `src/tests/r/*.R`

Confirmed: Python version, metafor reference version 5.2-1 and the `source bin/activate.sh` then pytest/ruff/mypy workflow agree between the two files.

### Spec review (step 5)

Each spec in `agents/specs/` was checked against the specdev validation criteria. Of the 23 checked, 19 have no gaps.

- `agents/specs/auditme-clean/spec.md:65` - API: the only function signature, `def fsn(x, vi=None, *, ...) -> FsnResult`, has unannotated parameters.
- `agents/specs/correctness-ls-extensions/spec.md:27` - names the ls-profile-starts and ls-profile-feasible-start specs in prose with no relative link.
- `agents/specs/correctness-ls-extensions/spec.md:54` - cites "correctness-mv-robust Decisions" with no link.
- `agents/specs/correctness-publication-bias/spec.md:14` - names three sibling specs as code spans, not links.
- `agents/specs/correctness-uni-diagnostics/spec.md:65` - cites "correctness-mv-robust, Decisions" with no link.

## Phase B findings (consolidated)

There are 750 individual findings, so they are grouped below by question and cause. Each group's `path:line` list is the per-question block above.

### F1. User-guide output and claims the code contradicts

- **Category:** Doc
- **Question:** Doc5
- **Severity:** medium
- **Location:** 29 candidates; affected pages:
  - `docs/getting-started.md`
  - `docs/guide/advanced.md`
  - `docs/guide/datasets.md`
  - `docs/guide/diagnostics.md`
  - `docs/guide/effect-sizes.md`
  - `docs/guide/emmeans.md`
  - `docs/guide/models.md`
  - `docs/differences.md`
  - `docs/differences-notes.md`
- **Description:**
  - Printed output in examples has moved since it was captured: R-style `predict` and `emmeans` tables, `list_datasets()[:5]` and the second `hettest` block. Some header lines are missing as well.
  - Some statements contradict the code:
    - unused `control=` entries are said to raise in `confint`, `hettest` and `permutest`, but they are ignored;
    - `differences-notes.md` describes `ni`, `ddf`, `progbar`, `res.control` and `p` wrongly for `rma_mv`;
    - "not ported" options such as `abbrev` and `retdat` have since been ported;
    - the `method` and `type` tables are incomplete.
  - `src/tests/test_docs_examples.py` runs the examples but never compares their output, which is how the drift got through.
- **Recommendation:** re-run every example on the affected pages, update the shown output and the contradicted sentences, and leave the dated records in `docs/differences.md` alone.

### F2. Domain errors missing from `Raises:` sections

- **Category:** A
- **Question:** A4
- **Severity:** medium
- **Location:** 26 candidates, among them:
  - `conv_2x2`, `conv_delta`, `conv_wald`, `conv_fivenum`, `escalc`, `to_long`, `to_table`, `collect`, `prepare_tables`, `rma_mh`, `rma_peto`: `NotFoundError` and `ArgumentError` from `Inputs.get` and `metadata.column`;
  - `forest`, `funnel`, `labbe`, `fitted_values`, `rcalc_list`, `wald_between`, `confint_rma_ls`, `profile_rma_ls`, `part_vif`;
  - extras on `hettest`, `build_design`, `minimise_standin`, `records_call`, `profile`, `robust`.
- **Description:** the sweep cannot follow these routes, which go through instance methods and dispatch, so callers are not told about errors that do reach them.
- **Recommendation:** add the missing names and remove the extras. For `profile` and `robust`, wait for decision D1.

### F3. `assert` used as runtime validation

- **Category:** A
- **Question:** A12
- **Severity:** medium
- **Location:** `src/metafopy/plot/qqnorm.py:242`, `:243`, `src/metafopy/methods/profile.py:309`
- **Description:**
  - In `qqnorm.py` the asserts are the only check that rejects an unrecognised `label` string, so under `python -O` the check disappears.
  - In `profile.py` an `except AssertionError` handler relies on the assert to fall back to another range, so under `-O` an `AttributeError` escapes instead.
- **Recommendation:** replace the asserts with `if ...: raise metafopy.errors.InputError(...)`, and replace the assert-based fallback with an explicit `None` check.

### F4. README contradicts AGENTS.md

- **Category:** Doc
- **Question:** Doc3
- **Severity:** medium
- **Location:** `README.md:52`, `:57`, `:58`
- **Description:** the README calls R the "numerical reference", where AGENTS.md makes the publication first and R second. It also regenerates fixtures with plain `Rscript` instead of `LC_ALL=C.UTF-8 Rscript`. Finally, it says that `generate_fixtures.R` alone refreshes `src/tests/reference/*.json`, when most of the 58 entries in `src/tests/r/` (as of 2026-11-21) write fixtures there.
- **Recommendation:** make the README match `AGENTS.md`.

### F5. Imported but undeclared third-party modules

- **Category:** D
- **Question:** D1
- **Severity:** medium (`nbformat`), low (`bin/logo.py`)
- **Location:** `src/tests/test_showcase_notebook.py:6` (`nbformat`); `bin/logo.py:28`, `:31`, `:32` (`fontTools`, `PIL`, `uharfbuzz`)
- **Description:** `nbformat` arrives only as a dependency of `nbclient`. `bin/logo.py` declares its three packages in its own inline PEP 723 block, which is how `uv run` installs them.
- **Recommendation:** declare `nbformat` in the dev group (running `uv lock`). Dismiss the `bin/logo.py` hits, because its PEP 723 block is its declaration.

### F6. Specs that no longer match the code

- **Category:** Doc
- **Question:** Doc4
- **Severity:** low
- **Location:** 30 candidates, in these specs:
  - `audit-shard-survival`
  - `auditme-clean` (the old `bnomis/agents` hook source and `AGENTS_DEPLOY_KEY`, changed in `31d809f`; `check_class`)
  - `correctness-glmm`
  - `correctness-mv-robust`
  - `correctness-publication-bias`
  - `correctness-tests`
  - `correctness-uni-diagnostics`
  - `cross-platform-numerics`
  - `r-oracle-pin`
  - `rma-mv-reml-stop`
  - `scripts-shard-survival`
- **Description:** each statement names a function, constant, test file or return type that the code now contradicts.
- **Recommendation:** amend each spec, bump its version, and leave the dated sections alone.

### F7. Standard-library errors raised where a domain error exists

- **Category:** A
- **Question:** A3
- **Severity:** low
- **Location:** `src/metafopy/methods/predict.py:245`; 11 sites in `src/audit` (`pycall.py:99`, `rlib.py:801`, `:807`, `recall.py:59`, `steps.py:84`, `:87`, `:105`, `:110`, `rdriver.py:48`, `scripts.py:67`, `:209`)
- **Recommendation:** raise `metafopy.errors.InputError` in `predict.py`. Raise the harness's own error classes in `src/audit`, adding a step-failure class if none fits; decision D2 covers the same question for A1.

### F8. Literal chains without `assert_never`

- **Category:** A
- **Question:** A11
- **Severity:** low
- **Location:** 36 chains, among them:
  - `regtest`, `tes`, `trimfill`, `glm_fit`, `glmm`;
  - `influence`, `ls_confint`, `ls_predict`, `matreg`, `rma_ls_fit`;
  - `forest.addpoly`, `forest.arrow`, `plot_rma`, `sel_lik.evaluate`;
  - `records_call`, and the test `check_fit`.
- **Recommendation:** give each `Literal` value its own arm and end the chain with `typing.assert_never`.

### F9. File and subprocess calls with no translating envelope

- **Category:** A
- **Question:** A1
- **Severity:** low
- **Location:** 61 candidates: 59 in `src/audit` and `bin`, and `src/metafopy/datasets/base.py::read` (`pandas.read_csv`, twice)
- **Description:** an `OSError` or pandas parser error escapes untranslated.
- **Recommendation:** see decision D2.

### F10. Package-surface drift

- **Category:** A
- **Question:** A17
- **Severity:** low
- **Location:** `src/metafopy/__init__.py`, `datasets/__init__.py`, `effects/__init__.py`, `interactive/__init__.py`, `plot/__init__.py`
- **Description:**
  - These read names from submodules when the package exports them: `src/audit/valuetree.py` (`metafopy.metadata.MetaData`, `metafopy.effects.reshape.TableArray`), `rma_uni.py` (`metafopy.effects.measures.MEASURES`), `binary.py`, `rma_uni.py` and `sel_result.py` (`metafopy.plot.rma`, `metafopy.plot.selmodel`), and `profile.py:179`.
  - Many `__all__` names have no consumer inside the repository.
- **Recommendation:** read the exported names through the package; for the unconsumed `__all__` names, see decision D3.

### F11. Functions with branches and no direct test

- **Category:** T
- **Question:** T1
- **Severity:** medium
- **Location:** 279 candidates in `bin/`, `src/audit/` and `src/metafopy/` (all listed above)
- **Recommendation:** see decision D4.

### F12. Pure functions with no property test

- **Category:** T
- **Question:** T2
- **Severity:** low
- **Location:** 144 candidates (all listed above), among them numerical cores that have simple oracles: `bvn`, `dbinom`, `farebrother`, `tukey`, `nchg`, `emm_adjust` distributions, the `sel_spec` weight functions, `map_delta`/`map_inverse` and the formatting helpers.
- **Recommendation:** see decision D5.

### F13. Small design defects

- **Category:** A, P
- **Questions:** A5, A7, A8, A9, A10, A14, A22, A25, P3
- **Severity:** low
- **Location:**
  - A5: `forest.py:2020`
  - A7: `mv_lik.py:128`
  - A8: `generate.py:351`, `report.py:212`, `rma_mv.py:826`, `rma_mv.py:894`
  - A9: `steps.py:74`, `rng.py:263`, `pycall.py:65`, `correctness/cases.py:509`
  - A10: `selmodel.py:57`, `rlib.py:1238`
  - A14: `generate.py:197`
  - A22: `rlib.py:73`
  - A25: `steps.py:217`
  - P3: `rma_ls_fit.py:289`, `glmm.py:622`
- **Recommendation:**
  - Fix the A5, A8, A10, A14, A22, A25 and P3 sites in one task list.
  - Rename the A9 functions, or move their I/O out.
  - Accept A7 with a scoped marker. `Factorization` exists to factorise once, so doing it in its constructor is the point of the class.

### F14. Duplicated construction of result objects

- **Category:** A
- **Question:** A15
- **Severity:** low
- **Location:** 17 sites (listed in the A15 block above)
- **Recommendation:** factor each duplicated assembly into one helper per result class, so that adding a field cannot leave one copy behind.

### F15. Same value passed to two slots

- **Category:** A
- **Question:** A13
- **Severity:** low
- **Location:** 12 sites (listed above), such as `p=m, parms=m` and `p=p_eff, p_eff=p_eff`
- **Description:** each pair mirrors two fields that metafor's object carries, and they are equal by definition in that case.
- **Recommendation:** add a one-line docstring note on each result field saying when it equals the other. No change to the code.

### F16. Rescanning loops

- **Category:** P
- **Question:** P1
- **Severity:** low
- **Location:** `robust.py:400`, `rcalc.py:272`, `reshape.py:752`, `reshape.py:772`
- **Recommendation:** group the rows once (`numpy.unique(..., return_inverse=True)` or `groupby`) instead of masking the whole array per level.

### F17. Unchecked inputs

- **Category:** Sec
- **Questions:** Sec2, Sec4
- **Severity:** low
- **Location:** `src/audit/cli.py:103` (`--recall` value), `src/metafopy/datasets/base.py:53` (dataset name joined into the package path)
- **Recommendation:**
  - Make `--recall` a usage error for a non-number or an unknown audit.
  - Reject a dataset name that is not in `list_datasets()` before joining it into the path.

### F18. Project rules the sweep does not check

- **Category:** S
- **Question:** S4
- **Severity:** medium (RNG), low (the rest)
- **Location:**
  - `numpy.random` in a ported procedure: `plot/selmodel.py:179` (the `ci='boot'` bands) and `methods/emm_adjust.py:143`, `:147`, `:157`, `:159` (seeding for the `mvt` adjustment).
  - `ListAnovaResult`, `ConfintList` and `ProfileList` are `list` subclasses, not frozen dataclasses.
  - Leading underscores required by an outside protocol: `_repr_html_` at `printing.py:229` and `rma_uni.py:313`; `_constructor` at `labelled.py:328` and `:388`.
  - 16 ported modules with no link to their R source or paper.
- **Recommendation:**
  - Use `metafopy.core.rng` for the bootstrap bands. For `mvt`, document that SciPy's integrator is seeded on its own and cannot match R draw for draw.
  - Accept the `list` subclasses with a scoped marker: R returns lists here and the result must index like one.
  - Add scoped markers for the protocol names.
  - Add the missing links.

### F19. Other low-cost fixes

- **Category:** S, A, Doc
- **Severity:** low
- **Location:**
  - S1: one missing return annotation at `src/tests/test_harness_scripts.py:40`.
  - A20: 11 test-module dicts read from other modules: `glmm_cases.py:44`, `:46`, `:48`, `:50`, `:58`, `test_rma_mv.py:80`, `glmm_extras_cases.py:31`, `:103`, `test_rma_ls.py:99`, `test_nchg.py:18`, `correctness/test_properties.py:203`.
  - Spec gaps: the 5 rows under "Spec review" above.
  - Doc1: `AGENTS.md` "Audit" says `bin/metafor-audit.sh` writes `wrong-or-missing.md` without saying where. It goes under the ignored `.audit/runs/<timestamp>/`.
- **Recommendation:**
  - Annotate the test helper.
  - Wrap the dicts in `types.MappingProxyType`.
  - Add the spec links and annotations.
  - Name the output path in `AGENTS.md`.

### F20. Findings that need no change

- **Category:** T, D
- **Questions:** T3, D2
- **Severity:** low
- **Location:**
  - T3: `src/audit/cli.py:118`, `bin/logo.py:265`
  - D2: `pyproject.toml:45` (`ipykernel`), `:48` (`pandas-stubs`), `:54` (`scipy-stubs`)
- **Description:**
  - `bin/metafor-audit.sh --recall` is the audit's end-to-end check, and the logo script is a one-off build tool.
  - `ipykernel` is the kernel `nbclient` starts, and the two stub packages are read by mypy. None of the three falls under D2's carve-outs, but each is used.
- **Recommendation:** dismiss these with their reasons, and file a skill request for D2 carve-outs covering stub packages and notebook kernels.

## Decisions

### D1. Whether a `singledispatch` generic's `Raises:` owes its implementations' errors

- **What is decided:** whether A4 judges a `functools.singledispatch` entry point (`anova`, `profile`, `robust`, `vif`, `predict`, `permutest`, `influence`, `rstandard` and the others) on its own body, or on the errors its registered implementations raise.
- **Background (2026-11-21):** the A4 brief does not settle this, and batches split.
  - Most judged the body only. That flags `InputError` as extra on `profile` and `robust`, which document their implementations' errors.
  - One batch judged by dispatch, which clears `vif`, whose docstring says "raised by the implementation".
  - A caller of `metafopy.vif()` does receive the implementations' errors.
- **What it affects:** A4 verdicts on about 15 generics.
- **Options:**
  - *Judge by dispatch (Recommended):* the generic owes what its implementations raise. `profile`, `robust` and `vif` stay clean, and the generics that declare only `ArgumentError` are judged again and gain their implementations' errors. Roughly 12 more A4 findings, all docstring edits.
  - *Body only:* keep the current verdicts. `profile` and `robust` lose their documented `InputError`, and `vif` is flagged the same way.
- **Recommendation:** judge by dispatch, because a `Raises:` section is for the caller, and the caller sees the dispatched errors. The brief should then say so too (skill request).

### D2. File and subprocess errors in the audit harness

- **What is decided:** what to do with the 59 A1 findings in `src/audit` and `bin`.
- **Background (2026-11-21):** the brief counts an envelope only if it raises the package's domain error. The harness's only error classes are case-level ones (`MissingFunctionError` and the others), and none of them fits a file that cannot be read. Today the harness lets `OSError` end the run with a traceback, and where it does catch, it raises `RuntimeError` (`scripts.py`).
- **What it affects:** `src/audit/*.py` and `bin/*.py`: about 60 call sites, or 60 ledger dismissals.
- **Options:**
  - *Dismiss with a reason (Recommended):* these are developer tools whose failures are meant to stop the run with a traceback. Each finding is recorded with `phaseb.py dismiss` and stays out of later audits. The two `datasets/base.py::read` findings in the library stay open and go into the task list.
  - *Add a harness error class:* add an `audit.errors.HarnessError` and wrap every file and subprocess call at step boundaries, which is a task list of its own. The A3 `RuntimeError` sites (F7) would use the same class.
- **Recommendation:** dismiss. Translating I/O errors matters for a library's callers, not for a command-line harness that only its developer runs.

### D3. Names in `__all__` with no consumer in the repository

- **What is decided:** whether A17's "name in `__all__` with no consumer" findings on `src/metafopy/__init__.py`, `datasets/__init__.py` and `interactive/__init__.py` are defects.
- **Background:** these names are metafopy's public API, and they are used by users, not by the repository.
- **What it affects:** the A17 ledger only; no code changes.
- **Options:**
  - *Dismiss as public API (Recommended):* record the reason in the ledger, and keep the submodule-bypass findings (F10) as fixes.
  - *Drop the names:* remove from `__all__` every name nothing in the repository reads. That would break users' code.
- **Recommendation:** dismiss.

### D4. The untested branchy functions (T1)

- **What is decided:** how the 279 T1 findings are taken on.
- **Background:** this is the first time T1 has been run, so these findings are catch-up, not regressions. Most are public helpers that the tests reach only through a higher-level function, which the brief does not count for a public function.
- **What it affects:** a large body of new tests.
- **Options:**
  - *File `TODO.md` items by package (Recommended):* one item each for `src/metafopy`, `src/audit` and `bin`. Each goes through `/todo-do`, which writes a spec choosing which helpers deserve a direct test. The findings stay open in the ledger meanwhile, as carried findings rather than regressions.
  - *Task list now:* direct tests for all 279 in this review's task list.
  - *Dismiss helpers reached through tested callers:* record them as covered by caller tests, which needs a reason per id.
- **Recommendation:** file the items. 279 tests is too much for one task list, and many helpers are better left tested through their callers.

### D5. The pure functions with no property test (T2)

- **What is decided:** how the 144 T2 findings are taken on.
- **Background:** the repository's only hypothesis tests are fit-level ones in `src/tests/correctness/test_properties.py`.
- **What it affects:** new hypothesis tests.
- **Options:**
  - *File one `TODO.md` item, starting with the numerical cores (Recommended):* that is, `bvn`, `dbinom`, `farebrother`, `tukey`, `nchg`, the `emm_adjust` distributions, the selection weight functions and `map_delta`/`map_inverse`. The formatting helpers come later. The findings stay open in the ledger.
  - *Task list now* for all 144.
- **Recommendation:** file the item, since the numerical cores are where a property catches a real bug.

## Files reviewed (Phase B)

Every listed question ran on its lister's candidates. Every sampled question ran on each of the 328 `.py` files under `src/metafopy`, `src/audit`, `src/tests` and `bin`, with questions A1-A25, S1-S4, P1-P3, Sec1-Sec5, T1-T3 and D1-D3. Doc1-Doc5 also ran on `AGENTS.md`, `README.md`, `docs/` and `agents/specs/`. The file list is in `agents/audit-reviews/ledger/` (listed questions) and in the "Confirmation" line under each sampled question above.

## Prior findings re-verified

None: this is the first review, and there was no `summary.md` or `agents/tasks/audit-reviews/`.

## Recommendations

1. F1, F2, F3, F4 and F5 (`nbformat`): fix in the review's task list first.
2. F18 (the bootstrap bands' RNG), F6, F7, F8, F10, F13, F16, F17 and F19: the rest of the task list.
3. F14 (duplicated construction): its own task, last, as it touches many result classes.
4. F11 and F12: `TODO.md` items, following decisions D4 and D5.
5. F9, F15, F20 and D3: dismissals or markers, with reasons.

## Answers

Answered 2026-11-21 on the review page (https://claude.ai/artifact/XWmcd8XMA6kdWdv8F4Wtd8), every answer with no comment:

- **F1-F20:** each approved, with the recommended action.
- **D1. Whether a `singledispatch` generic owes its implementations' errors:** judge by dispatch. The 17 generics were re-judged on 2026-11-21 under that ruling, counting implementations registered by call (`generic.register(Type, fn)`) as well as by decorator. 16 now miss names their implementations raise; `vif` matches. `profile` and `robust` lose their "extra `InputError`" findings. A4 open findings went from 26 to 40, measured 2026-11-21, and the A4 ledger holds the new verdicts.
- **D2. File and subprocess errors in the audit harness:** dismiss with a reason. The 59 A1 findings in `src/audit` and `bin` were dismissed in the ledger on 2026-11-21; the two `datasets/base.py::read` findings stay open.
- **D3. Names in `__all__` with no consumer:** dismiss as public API. The `datasets` and `interactive` A17 candidates are dismissed. The `metafopy`, `effects` and `plot` candidates stay open for their submodule bypasses (F10).
- **D4. Untested branchy functions (T1):** file `TODO.md` items by package (`src/metafopy`, `src/audit`, `bin`).
- **D5. Pure functions without a property test (T2):** file one `TODO.md` item, starting with the numerical cores.
