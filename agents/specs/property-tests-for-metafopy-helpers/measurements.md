# Measurements: property tests for metafopy helpers

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->
Measured on 2026-11-22 at `7c8bfeb` from `agents/audit-reviews/ledger/T2.jsonl` (verdicts judged 2026-11-21 at `31d809f`): the 132 open,
undismissed T2 findings under `src/metafopy`. Each ledger row's `reason` states the property to test; the spec's tasks read it from there.
`phaseb.py plan . --question T2` at `7c8bfeb` re-plans 121 of the 193 T2 candidates (72 reused), so the open set is re-judged before any test
is written (spec, Implementation step 1).

The groups are this spec's, not the checker's: A holds the functions that compute a probability, a density, a weight or a statistic; B the
functions that turn numbers into text or lay text out, and small numeric helpers of printing and plotting; C the functions that match names,
parse formulas or rewrite strings.

## A. Numerical kernels (43, measured 2026-11-22)

| Module | Functions |
|---|---|
| `src/metafopy/bias/fsn.py` | `rnd_fsn` |
| `src/metafopy/bias/ranktest.py` | `pkendall` |
| `src/metafopy/core/bvn.py` | `bvn_rect`, `bvnu` |
| `src/metafopy/core/dbinom.py` | `bd0`, `dbinom_raw`, `pow1p`, `stirlerr` |
| `src/metafopy/core/farebrother.py` | `farebrother`, `ruben` |
| `src/metafopy/core/nchg.py` | `fisher_log_probability`, `ln_fac`, `tabulated_log_probability` |
| `src/metafopy/core/tukey.py` | `ptukey`, `qtukey`, `wprob` |
| `src/metafopy/effects/fivenum_estimators.py` | `weibull_shape_start` |
| `src/metafopy/effects/rtet.py` | `rtet` |
| `src/metafopy/methods/emm_adjust.py` | `pf_cdf`, `pf_sf`, `pt_sf`, `qdunnx`, `qf`, `qt` |
| `src/metafopy/methods/ls_anova.py` | `omnibus` |
| `src/metafopy/models/sel_lik.py` | `apply_named`, `exp_safe`, `map_delta`, `map_inverse` |
| `src/metafopy/models/sel_spec.py` | `s_halfnorm`, `s_halfnorm1`, `s_halfnorm2`, `s_logistic`, `s_logistic1`, `s_logistic2`, `s_negexp`, `s_negexp1`, `s_negexp2`, `s_negexppow`, `s_power`, `s_power1`, `s_power2` |
| `src/metafopy/models/tau2.py` | `i2_from_q` |

## B. Number formatting, layout and small numeric helpers (40, measured 2026-11-22)

| Module | Functions |
|---|---|
| `src/metafopy/bias/inputs.py` | `print_vector` |
| `src/metafopy/core/labelled.py` | `decimals`, `format_real`, `scientific_places`, `trim_zeros` |
| `src/metafopy/core/timing.py` | `format_time`, `number` |
| `src/metafopy/effects/summary.py` | `level_to_alpha` |
| `src/metafopy/formatters.py` | `r_as_character` |
| `src/metafopy/formula.py` | `r_number` |
| `src/metafopy/methods/blup.py` | `format_r_frame` |
| `src/metafopy/methods/deltamethod.py` | `get_digits` |
| `src/metafopy/methods/ls_common.py` | `center_stars` |
| `src/metafopy/methods/vif.py` | `format_btt`, `format_vector` |
| `src/metafopy/models/mv_factor.py` | `format_number` |
| `src/metafopy/models/rma_uni.py` | `coef_span`, `level_to_alpha` |
| `src/metafopy/models/selmodel.py` | `r_as_character` |
| `src/metafopy/plot/forest.py` | `anno_symbol`, `finish_annotext` |
| `src/metafopy/plot/gosh.py` | `hex_color` |
| `src/metafopy/plot/infl.py` | `n2mfrow` |
| `src/metafopy/plot/profile.py` | `per_panel` |
| `src/metafopy/plot/rutils.py` | `color_ramp`, `fmtx1`, `level_to_alpha`, `r_pretty`, `rround`, `usr_range` |
| `src/metafopy/printing.py` | `fmt_df_scalar`, `fmt_named_vector`, `fmt_num`, `fmt_pval`, `fmt_table`, `fmt_wrapped_table`, `number_rows`, `signif_footer`, `signif_stars`, `zero_below` |

## C. Name matching, formula parsing and string rewriting (49, measured 2026-11-22)

| Module | Functions |
|---|---|
| `src/metafopy/core/padjust.py` | `match_adjust` |
| `src/metafopy/effects/datatools.py` | `charmatch`, `make_names`, `shorten_names` |
| `src/metafopy/effects/fivenum_estimators.py` | `sum_stat_index` |
| `src/metafopy/effects/measures.py` | `make_unique_dot1` |
| `src/metafopy/formula.py` | `call_spans`, `parse_random`, `r_deparse`, `r_logical_calls`, `r_logical_formula`, `r_power`, `r_tokens` |
| `src/metafopy/interactive/forest.py` | `unique_labels` |
| `src/metafopy/metadata.py` | `make_unique` |
| `src/metafopy/methods/common.py` | `match_arg` |
| `src/metafopy/methods/emm_adjust.py` | `match_method`, `parse_side` |
| `src/metafopy/methods/emm_grid.py` | `appearance` |
| `src/metafopy/methods/emm_nesting.py` | `factor_id`, `strip_supersets`, `term_info` |
| `src/metafopy/methods/hettest.py` | `expand_boot` |
| `src/metafopy/methods/matreg.py` | `match_columns`, `parse_formula`, `pmatch`, `resolve_x`, `resolve_y`, `subsequence_distance` |
| `src/metafopy/methods/predict.py` | `is_subsequence`, `level_indices`, `match_names` |
| `src/metafopy/methods/rcalc.py` | `formula_variables`, `match_nfun` |
| `src/metafopy/models/call.py` | `split_terms`, `update_formula`, `update_side` |
| `src/metafopy/models/mv_factor.py` | `split_top_level` |
| `src/metafopy/models/mv_hessian.py` | `inner_names`, `lower_triangle_order` |
| `src/metafopy/models/mv_print.py` | `abbreviate`, `abbreviate_once` |
| `src/metafopy/models/rma_mv.py` | `replace_call_notation` |
| `src/metafopy/models/rma_uni.py` | `abbreviate_names` |
| `src/metafopy/models/sel_opt.py` | `choose_optimizer`, `partial_match` |
| `src/metafopy/models/sel_spec.py` | `match_type` |
| `src/metafopy/plot/forest.py` | `parse_predtail` |
| `src/metafopy/plot/infl.py` | `abbreviate` |

<!-- doc-counts: accept-end -->
