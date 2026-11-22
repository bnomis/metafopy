# Open candidates after the re-judge

Measured on 2026-11-22 at `54df067` with `phaseb.py plan`, a fresh verdict for every candidate under brief version 6 (T1) and 3 (T2), `phaseb.py check`, `phaseb.py merge`, then `phaseb.py report . --question T1` and `--question T2`. It is a dated record: never edit it.

Groups are those of `measurements.md` (first rule that matches). Group A is split in the order listed here: the first half (rounded up) is Task 4, the rest Task 5. A group C row open on T1 goes to Task 7 for `printdiff`, `valuetree`, `report`, `steps` and `apidiff`, otherwise to Task 8; a group C row open on T2 only goes to Task 9. A row open on both goes to its T1 task, which also writes the T2 property test.

Reviewers judged the cases the brief leaves unsettled as "not flagged" (for example candidates reached only through a dispatch table).

## Open `src/audit` and `bin` candidates

| Group | Candidate | Question | Task |
|---|---|---|---|
| A | `src/audit/rlib.py::Named.__array_wrap__` | T1 | 4 |
| A | `src/audit/rlib.py::format_cell` | T1 | 4 |
| A | `src/audit/rlib.py::r_as_dataframe` | T1 | 4 |
| A | `src/audit/rlib.py::r_as_matrix` | T1 | 4 |
| A | `src/audit/rlib.py::r_as_numeric` | T1 | 4 |
| A | `src/audit/rlib.py::r_assign_names` | T1 | 4 |
| A | `src/audit/rlib.py::r_attr` | T1 | 4 |
| A | `src/audit/rlib.py::r_capture_output` | T1 | 4 |
| A | `src/audit/rlib.py::r_cbind` | T1 | 4 |
| A | `src/audit/rlib.py::r_class` | T1 | 4 |
| A | `src/audit/rlib.py::r_cor` | T1 | 4 |
| A | `src/audit/rlib.py::r_dataframe` | T1 | 4 |
| A | `src/audit/rlib.py::r_diag` | T1 | 4 |
| A | `src/audit/rlib.py::r_equal` | T1 | 4 |
| A | `src/audit/rlib.py::r_head` | T1 | 4 |
| A | `src/audit/rlib.py::r_index` | T1 | 4 |
| A | `src/audit/rlib.py::r_index2` | T1 | 4 |
| A | `src/audit/rlib.py::r_inherits` | T1 | 4 |
| A | `src/audit/rlib.py::r_isna` | T1 | 5 |
| A | `src/audit/rlib.py::r_lapply` | T1 | 5 |
| A | `src/audit/rlib.py::r_length` | T1 | 5 |
| A | `src/audit/rlib.py::r_list` | T1 | 5 |
| A | `src/audit/rlib.py::r_matrix` | T1 | 5 |
| A | `src/audit/rlib.py::r_names` | T1 | 5 |
| A | `src/audit/rlib.py::r_par` | T1 | 5 |
| A | `src/audit/rlib.py::r_paste` | T1 | 5 |
| A | `src/audit/rlib.py::r_rbind` | T1 | 5 |
| A | `src/audit/rlib.py::r_rep` | T1 | 5 |
| A | `src/audit/rlib.py::r_seq` | T1 | 5 |
| A | `src/audit/rlib.py::r_split` | T1 | 5 |
| A | `src/audit/rlib.py::r_stopifnot` | T1 | 5 |
| A | `src/audit/rlib.py::r_tail` | T1 | 5 |
| A | `src/audit/rlib.py::r_var` | T1 | 5 |
| A | `src/audit/rlib.py::recycled` | T1 | 5 |
| A | `src/audit/rlib.py::truth` | T1 | 5 |
| E | `bin/logo.py::animated_gif` | T1 | 9 |
| E | `bin/logo.py::simulate` | T1 | 9 |
| E | `bin/logo.py::smoothstep` | T1, T2 | 9 |
| E | `bin/metafor-audit.py::metafopy_source` | T1, T2 | 9 |
| E | `bin/platform-probe.py::ls_runs` | T1 | 9 |
| E | `bin/platform-probe.py::measure_selmodel_decreasing` | T1 | 9 |
| E | `bin/platform-probe.py::score_norm` | T1 | 9 |
| C | `src/audit/apidiff.py::callables_for` | T1 | 7 |
| C | `src/audit/apidiff.py::comparable_default` | T1 | 7 |
| C | `src/audit/apidiff.py::compare_function` | T1 | 7 |
| C | `src/audit/apidiff.py::parameters` | T1 | 7 |
| C | `src/audit/apidiff.py::python_defaults` | T1 | 7 |
| C | `src/audit/apidiff.py::run` | T1 | 7 |
| C | `src/audit/apidiff.py::source_mentions` | T1 | 7 |
| C | `src/audit/apidiff.py::spellings_for` | T1 | 7 |
| D | `src/audit/cli.py::main` | T1 | 10 |
| C | `src/audit/domains.py::refs_in` | T1 | 8 |
| C | `src/audit/engine.py::py_stop` | T1 | 8 |
| C | `src/audit/engine.py::r_stop_reason` | T1 | 8 |
| C | `src/audit/generate.py::default_string` | T2 | 9 |
| C | `src/audit/ledger.py::Ledger.classify` | T1 | 8 |
| C | `src/audit/ledger.py::bracket_sets` | T2 | 9 |
| C | `src/audit/printdiff.py::close_numbers` | T2 | 9 |
| C | `src/audit/printdiff.py::decimals` | T2 | 9 |
| C | `src/audit/printdiff.py::first_difference` | T2 | 9 |
| C | `src/audit/printdiff.py::same_line` | T2 | 9 |
| C | `src/audit/printdiff.py::same_token` | T2 | 9 |
| C | `src/audit/pycall.py::check_arguments` | T1 | 8 |
| C | `src/audit/pycall.py::coef_for` | T1 | 8 |
| C | `src/audit/pycall.py::getitem` | T1 | 8 |
| C | `src/audit/pycall.py::head` | T1 | 8 |
| C | `src/audit/pycall.py::integral_arguments` | T1 | 8 |
| C | `src/audit/pycall.py::match_positional` | T1 | 8 |
| C | `src/audit/pycall.py::option_value` | T1 | 8 |
| C | `src/audit/pycall.py::print_call` | T1 | 8 |
| C | `src/audit/pycall.py::run_call` | T1 | 8 |
| C | `src/audit/pycall.py::summary_for` | T1 | 8 |
| C | `src/audit/pycall.py::tail` | T1 | 8 |
| C | `src/audit/pyrun.py::plain_text` | T2 | 9 |
| C | `src/audit/pyrun.py::readable` | T1 | 8 |
| D | `src/audit/pyrun.py::run_guarded` | T1 | 10 |
| D | `src/audit/rdriver.py::run` | T1 | 10 |
| D | `src/audit/recall.py::checkout` | T1 | 10 |
| D | `src/audit/recall.py::run_audit` | T1 | 10 |
| C | `src/audit/report.py::normal_path` | T2 | 9 |
| C | `src/audit/report.py::result_class` | T1 | 7 |
| D | `src/audit/scripts.py::fetch_source` | T1 | 10 |
| D | `src/audit/steps.py::build_universe` | T1 | 10 |
| C | `src/audit/steps.py::generate_cases` | T1 | 7 |
| D | `src/audit/steps.py::scripts_step` | T1 | 10 |
| C | `src/audit/steps.py::selected` | T2 | 9 |
| C | `src/audit/universe.py::Universe.draws` | T1 | 8 |
| C | `src/audit/universe.py::Universe.method_for` | T1 | 8 |
| C | `src/audit/universe.py::Universe.refuses` | T1 | 8 |

## Outside this list (under `src/metafopy`; not tested here)

| Candidate | Question |
|---|---|
| `src/metafopy/bias/fsn.py::FsnResult.summary` | T1 |
| `src/metafopy/bias/hc.py::HcResult.summary` | T1 |
| `src/metafopy/bias/regtest.py::RegtestResult.summary` | T1 |
| `src/metafopy/bias/tes.py::TesResult.summary` | T1 |
| `src/metafopy/bias/trimfill.py::TrimfillResult.summary` | T1 |
| `src/metafopy/core/labelled.py::RFrame.text` | T1 |
| `src/metafopy/core/labelled.py::SparseVCovMat.text` | T1 |
| `src/metafopy/core/numderiv.py::hessian_calculus` | T1 |
| `src/metafopy/models/mv_result.py::RMAMVResult.R` | T1 |
| `src/metafopy/models/mv_result.py::formula_text` | T1 |
