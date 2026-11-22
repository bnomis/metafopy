# Measurements: direct tests for the audit harness

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->
Measured on 2026-11-22 at `73ffe43` with `python3 ~/.claude/skills/auditme/checks/phaseb.py report . --question T1` and `--question T2`,
which read `agents/audit-reviews/ledger/T1.jsonl` and `T2.jsonl`. Every open candidate under `src/audit` and `bin` is listed below: 117 for
T1 and 12 for T2, 124 functions in all, since five are open on both questions. No open candidate is under `src/tests`; test files appear in
the report only as the evidence a verdict names. No candidate under `src/audit` or `bin` is dismissed.

The verdicts are stale. The T1 brief is at version 6 and its ledger rows were judged under versions 1 (2026-11-21) and 3 (2026-11-22); the T2
brief is at version 3 and its rows under version 1. `phaseb.py plan` at `73ffe43` therefore re-plans every candidate of both questions (1230
for T1 and 193 for T2, none reused). One T1 candidate under `src/audit` is new since the ledger, `src/audit/cli.py::parse_args`, and a direct
test already calls it (`src/tests/test_harness_compare.py::test_recall_value_that_names_no_audit_is_a_usage_error`). Brief version 6 counts
a test that loads a `bin` script by path and runs an entry point reaching the candidate, so some `bin/platform-probe.py` verdicts may turn
covered on re-judging (`environment` is reached by `main()` in `src/tests/test_platform_probe.py`).

Privacy as the checker decides it at `73ffe43`: the `src/audit/rlib.py` candidates are private (`rlib.py` has an `__all__`), the `bin`
candidates are private as script helpers, and every other `src/audit` candidate is public, because no other `src/audit` module has an
`__all__`. A public candidate needs a test that calls it; a private one is also covered by a test of a caller that reaches it.

## Groups

The groups are this spec's, not the checker's. A task assigns a candidate by the first rule that matches:

| Group | Rule | Oracle |
|---|---|---|
| A. R-library shims | the candidate is in `src/audit/rlib.py` | R itself (spec decision D2) |
| B. Transpiler | the candidate is in `src/audit/transpile.py` | a hand-written table of parse trees, with R's values and R's deparsed text |
| D. Drivers | the candidate, or a function it calls on every path, starts a process (`subprocess`, `Rscript`, `git`), opens a socket or a URL, installs a signal handler or timer, or writes outside a directory it is given | none in-process; covered end to end by `bin/metafor-audit.sh --recall` |
| E. Scripts | the candidate is under `bin/` | per script (spec decision D4) |
| C. Pure helpers | every other candidate under `src/audit`: its result depends only on its arguments and committed data files | hand-built expected values from the docstring and `src/audit/README.md`; for T2, the property its ledger `reason` states |

| Group | T1 | T2 |
|---|---|---|
| A | 37 | 0 |
| B | 11 | 0 |
| C | 47 | 10 |
| D | 13 | 0 |
| E | 9 | 2 |

## Every open candidate (measured 2026-11-22 at `73ffe43`)

| Group | Candidate | Question | What it does |
|---|---|---|---|
| A | `src/audit/rlib.py::Named.__array_wrap__` | T1 | keeps names on 2-D ufunc results of a Named matrix |
| A | `src/audit/rlib.py::accepts` | T1 | whether a callable (or its singledispatch implementation) takes an argument |
| A | `src/audit/rlib.py::as_array` | T1 | Series or frame to numpy, other values through `numpy.asarray` |
| A | `src/audit/rlib.py::format_cell` | T1 | a paste cell: whole floats without decimals |
| A | `src/audit/rlib.py::r_as_dataframe` | T1 | R `as.data.frame` of frames, results, emmeans summaries and grids |
| A | `src/audit/rlib.py::r_as_matrix` | T1 | R `as.matrix` of `list.rma` results, frames and vectors |
| A | `src/audit/rlib.py::r_as_numeric` | T1 | R `as.numeric`: factor codes from 1, strings, numbers |
| A | `src/audit/rlib.py::r_assign_names` | T1 | R `names<-` on a frame, Series, dict or other value |
| A | `src/audit/rlib.py::r_attr` | T1 | R `attr` |
| A | `src/audit/rlib.py::r_capture_output` | T1 | R `capture.output` |
| A | `src/audit/rlib.py::r_cbind` | T1 | R `cbind` |
| A | `src/audit/rlib.py::r_class` | T1 | R `class` |
| A | `src/audit/rlib.py::r_cor` | T1 | R `cor` |
| A | `src/audit/rlib.py::r_dataframe` | T1 | R `data.frame` from vectors, frames and named columns, recycling |
| A | `src/audit/rlib.py::r_diag` | T1 | R `diag` |
| A | `src/audit/rlib.py::r_equal` | T1 | R `==`; a number against text compares as text |
| A | `src/audit/rlib.py::r_head` | T1 | R `head` |
| A | `src/audit/rlib.py::r_index` | T1 | R `[` on frames (one and two subscripts, names, logical, drop), Series, dicts, `list.rma`, matrices |
| A | `src/audit/rlib.py::r_index2` | T1 | R `[[` by name and position; a missing name raises |
| A | `src/audit/rlib.py::r_inherits` | T1 | R `inherits` |
| A | `src/audit/rlib.py::r_isna` | T1 | R `is.na` of a frame, list or array |
| A | `src/audit/rlib.py::r_lapply` | T1 | R `lapply` |
| A | `src/audit/rlib.py::r_length` | T1 | R `length` |
| A | `src/audit/rlib.py::r_list` | T1 | R `list` |
| A | `src/audit/rlib.py::r_matrix` | T1 | R `matrix` |
| A | `src/audit/rlib.py::r_names` | T1 | R `names` |
| A | `src/audit/rlib.py::r_par` | T1 | R `par` |
| A | `src/audit/rlib.py::r_paste` | T1 | R `paste` |
| A | `src/audit/rlib.py::r_rbind` | T1 | R `rbind` |
| A | `src/audit/rlib.py::r_rep` | T1 | R `rep` |
| A | `src/audit/rlib.py::r_seq` | T1 | R `seq` |
| A | `src/audit/rlib.py::r_split` | T1 | R `split` |
| A | `src/audit/rlib.py::r_stopifnot` | T1 | R `stopifnot` |
| A | `src/audit/rlib.py::r_tail` | T1 | R `tail` |
| A | `src/audit/rlib.py::r_var` | T1 | R `var` |
| A | `src/audit/rlib.py::recycled` | T1 | R recycling of a Series operand of another length |
| A | `src/audit/rlib.py::truth` | T1 | an `if` condition; length zero raises UnsupportedError |
| B | `src/audit/transpile.py::args_code` | T1 | Python code of a call's named and unnamed arguments |
| B | `src/audit/transpile.py::callee_name` | T1 | the name of a symbol callee, None otherwise |
| B | `src/audit/transpile.py::replacement` | T1 | code of the new value after `head(x, ...) <- v` (`$<-`, `[<-`, `f<-`) |
| B | `src/audit/transpile.py::tr` | T1 | translates one parse-tree node by its type |
| B | `src/audit/transpile.py::tr_assign` | T1 | translates `x <- v` and nested replacement assignments |
| B | `src/audit/transpile.py::tr_call` | T1 | translates a call: operators, `if`, `for`, `function`, lazy and NSE calls |
| B | `src/audit/transpile.py::tr_function` | T1 | translates `function(...)` with and without defaults |
| B | `src/audit/transpile.py::tr_sym` | T1 | translates a symbol: reserved names, constants, variables |
| B | `src/audit/transpile.py::unparse` | T1 | the R text of a parse tree (formulas) |
| B | `src/audit/transpile.py::walk` | T1 | every node of a tree except plain assignment targets |
| B | `src/audit/transpile.py::with_model` | T1 | passes the model to `pairmat()` inside a model call |
| C | `src/audit/apidiff.py::callables_for` | T1 | every metafopy callable that handles an R generic for an object, with singledispatch |
| C | `src/audit/apidiff.py::comparable_default` | T1 | an R default (TRUE, NULL, "text", number, other) as a comparable Python value |
| C | `src/audit/apidiff.py::compare_function` | T1 | R formals against the Python parameters and defaults: missing-function and missing-arg findings |
| C | `src/audit/apidiff.py::parameters` | T1 | parameter names of a callable and whether it takes **kwargs |
| C | `src/audit/apidiff.py::python_defaults` | T1 | the defaults of a callable or a partial |
| C | `src/audit/apidiff.py::run` | T1 | the api step over every fixture and generic of a universe |
| C | `src/audit/apidiff.py::source_mentions` | T1 | whether any callable's source names an argument |
| C | `src/audit/apidiff.py::spellings_for` | T1 | the Python names an R generic or S3 method may have (`summary_escalc`, plot functions) |
| C | `src/audit/domains.py::refs_in` | T1 | the fixtures a JSON argument value refers to (`$ref`, `$col`, nested) |
| C | `src/audit/engine.py::py_stop` | T1 | finding kind and detail for a Python shard that stalled, passed the memory limit or died |
| C | `src/audit/engine.py::r_stop_reason` | T1 | the reason text for an R shard that stalled, passed the memory limit or died |
| C | `src/audit/generate.py::args_of` | T1 | formals plus the arguments read from `...`, plus graphics arguments for drawing functions |
| C | `src/audit/generate.py::default_string` | T1, T2 | the text of a quoted R default, else None |
| C | `src/audit/generate.py::option_cases` | T1 | one case per value of each `getOption` setting a function reads |
| C | `src/audit/generate.py::option_values` | T1 | the values to try for one argument, or why there are none (skip, uncovered) |
| C | `src/audit/generate.py::sweep` | T1 | one case per value of each argument of a function, with companions |
| C | `src/audit/generate.py::unaccounted_functions` | T1 | exported functions no case reaches and no recipe waives |
| C | `src/audit/ledger.py::Ledger.classify` | T1 | the known.toml entry that matches a finding, counting its use |
| C | `src/audit/ledger.py::bracket_sets` | T2 | the `[...]` sets of an fnmatch glob, as fnmatch.translate finds them |
| C | `src/audit/printdiff.py::close_numbers` | T1, T2 | two printed numbers agree to the last printed digit or within drift |
| C | `src/audit/printdiff.py::decimals` | T2 | the number of decimals of a printed number, exponent ignored |
| C | `src/audit/printdiff.py::first_difference` | T2 | the first line where two printouts differ, else None |
| C | `src/audit/printdiff.py::same_line` | T2 | two printed lines agree token by token |
| C | `src/audit/printdiff.py::same_token` | T1, T2 | two printed tokens agree, numbers inside tokens compared as numbers |
| C | `src/audit/pycall.py::check_arguments` | T1 | raises MissingArgumentError for names a callable does not take |
| C | `src/audit/pycall.py::coef_for` | T1 | R `coef` of an object: `coef_summary` for a summary, the `coef` method otherwise |
| C | `src/audit/pycall.py::getitem` | T1 | R `x[i]` and `x[i, j]` on frames, Series, matrices and result dataclasses |
| C | `src/audit/pycall.py::head` | T1 | R `head` of a result that selects rows, or of its frame |
| C | `src/audit/pycall.py::integral_arguments` | T1 | whole floats passed to `int` parameters become ints |
| C | `src/audit/pycall.py::match_positional` | T1 | R argument matching: positional arguments fill the formals left after the named ones |
| C | `src/audit/pycall.py::option_value` | T1 | `na.action` setting `na.omit` becomes `omit` |
| C | `src/audit/pycall.py::print_call` | T1 | R `print(x, ...)` as text: arguments go to `summary()`, digits branches |
| C | `src/audit/pycall.py::shift_index` | T1 | 1-based R indices to 0-based, through lists, bools unchanged |
| C | `src/audit/pycall.py::summary_for` | T1 | R `summary` of an object: `summary_escalc` for a frame, the method otherwise, wrapped as text |
| C | `src/audit/pycall.py::tail` | T1 | R `tail` of a result that selects rows, or of its frame |
| C | `src/audit/pyrun.py::drawn_text` | T1 | the strings the open matplotlib figures hold that contain a letter |
| C | `src/audit/pyrun.py::plain_text` | T2 | the drawn strings that are plain text, plotmath and mathtext left out |
| C | `src/audit/pyrun.py::readable` | T1 | a plotmath `plain(...)` or p-value range as the text it draws |
| C | `src/audit/report.py::cause_key` | T1 | what makes two findings one cause: kind and path, or the masked message |
| C | `src/audit/report.py::function_of` | T1 | the R function a finding is about, through the fixture's function for construct cases |
| C | `src/audit/report.py::line_of` | T1 | the report line of one cause, with at most four functions named |
| C | `src/audit/report.py::normal_path` | T2 | a difference path without positions and data-derived names |
| C | `src/audit/report.py::result_class` | T1 | the result class a missing field is reported under, per step |
| C | `src/audit/steps.py::generate_cases` | T1 | the generated cases of a run, built once and cached on the context |
| C | `src/audit/steps.py::selected` | T2 | whether a case id matches any `--only` pattern (glob or substring) |
| C | `src/audit/universe.py::Universe.draws` | T1 | whether a generic or any of its methods draws |
| C | `src/audit/universe.py::Universe.method_for` | T1 | the S3 method R dispatches to for a class chain |
| C | `src/audit/universe.py::Universe.refuses` | T1 | whether R's `.chkclass` checks reject these classes for a generic |
| C | `src/audit/valuetree.py::compare_vector` | T1 | differences of two vector nodes: shape, labels, values, index shift |
| C | `src/audit/valuetree.py::index_shifted` | T1 | integer vectors that differ by exactly one throughout |
| C | `src/audit/valuetree.py::label_fields` | T1 | names, row and column names of a vector or matrix node from Python labels |
| C | `src/audit/valuetree.py::named_vector` | T1 | a mapping, `digits` or metafopy table as R's named vector |
| C | `src/audit/valuetree.py::num_cell` | T1 | a number as a node cell: NaN to None, infinities to `Inf` text |
| C | `src/audit/valuetree.py::vector_node` | T1 | a Python vector as a node in the shape of R's guide node |
| D | `src/audit/cli.py::main` | T1 | the audit command line: recall, rerender, list, then the steps into a run directory |
| D | `src/audit/pyrun.py::case_result` | T1 | one case run in Python under the seed and classified against R's outcome |
| D | `src/audit/pyrun.py::execute` | T1 | a case's call under `run_guarded`, or a FixtureError record when a needed fixture failed |
| D | `src/audit/pyrun.py::run_guarded` | T1 | a call under SIGALRM, capturing warnings, output and drawn text |
| D | `src/audit/rdriver.py::cluster_port` | T1 | a free TCP port not yet given to another R process of the run |
| D | `src/audit/rdriver.py::run` | T1 | runs an R script with Rscript; HarnessError with stderr on failure |
| D | `src/audit/recall.py::checkout` | T1 | a detached git worktree of metafopy at a commit, reused between runs |
| D | `src/audit/recall.py::run_audit` | T1 | runs the harness as a subprocess against an earlier audit's commit |
| D | `src/audit/report.py::git_head` | T1 | short git HEAD of the metafopy source, `unknown` when git fails |
| D | `src/audit/scripts.py::fetch_source` | T1 | downloads the pinned metafor source and checks its sha256 |
| D | `src/audit/scripts.py::run_py_scripts` | T1 | the Python side of a scripts shard in its own process, surviving a dead one |
| D | `src/audit/steps.py::build_universe` | T1 | the universe step: runs `universe.R`, compares with the committed snapshot |
| D | `src/audit/steps.py::scripts_step` | T1 | the scripts step: `--only` patterns, then `scripts.run`, findings and tally |
| E | `bin/logo.py::animated_gif` | T1 | renders the logo animation frames with headless Chromium and saves the GIF |
| E | `bin/logo.py::simulate` | T1 | deterministic bounce of the logo dots (mulberry32), one position list per frame |
| E | `bin/logo.py::smoothstep` | T1, T2 | quintic smoothstep: 0 below a, 1 above b, smooth between |
| E | `bin/metafor-audit.py::metafopy_source` | T1, T2 | reads `--metafopy-src` (two spellings) from argv before metafopy is imported |
| E | `bin/platform-probe.py::case_by_id` | T1 | the correctness case with an id, or KeyError |
| E | `bin/platform-probe.py::environment` | T1 | platform, Python, numpy, scipy, BLAS and LAPACK identity, libc on Linux |
| E | `bin/platform-probe.py::ls_runs` | T1 | runs every start of the location-scale reference fit and records each result |
| E | `bin/platform-probe.py::measure_selmodel_decreasing` | T1 | probe group: selmodel decreasing-weights fit against the reference maximum |
| E | `bin/platform-probe.py::score_norm` | T1 | largest absolute location-scale score at alpha, NaN where undefined |
<!-- doc-counts: accept-end -->
