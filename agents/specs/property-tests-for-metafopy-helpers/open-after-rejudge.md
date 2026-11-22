# Open T2 candidates after the re-judge

Measured on 2026-11-22 at `f1bad68` from `phaseb.py report . --question T2` after `phaseb.py merge`: the open `src/metafopy` candidates, each with its group (A, B or C, as in `measurements.md`) and the task that tests it. A dated record, not a claim about the present. `no (new)` marks a candidate that `measurements.md` does not list; it takes the group the spec's definitions give and the last task of that group (4, 6 or 8), except where a task list names its module.

| Candidate | Group | Task | In measurements.md |
|---|---|---|---|
| `src/metafopy/bias/fsn.py::rnd_fsn` | A | 3 | yes |
| `src/metafopy/bias/inputs.py::fp` | B | 6 | no (new) |
| `src/metafopy/bias/inputs.py::print_vector` | B | 6 | yes |
| `src/metafopy/bias/ranktest.py::pkendall` | A | 3 | yes |
| `src/metafopy/core/bvn.py::bvn_rect` | A | 3 | yes |
| `src/metafopy/core/bvn.py::bvnu` | A | 3 | yes |
| `src/metafopy/core/clubsandwich.py::f_sf` | A | 4 | no (new) |
| `src/metafopy/core/clubsandwich.py::hotelling` | A | 4 | no (new) |
| `src/metafopy/core/dbinom.py::bd0` | A | 3 | yes |
| `src/metafopy/core/dbinom.py::dbinom_raw` | A | 3 | yes |
| `src/metafopy/core/dbinom.py::pow1p` | A | 3 | yes |
| `src/metafopy/core/dbinom.py::stirlerr` | A | 3 | yes |
| `src/metafopy/core/farebrother.py::farebrother` | A | 3 | yes |
| `src/metafopy/core/farebrother.py::ruben` | A | 3 | yes |
| `src/metafopy/core/labelled.py::decimals` | B | 5 | yes |
| `src/metafopy/core/labelled.py::format_real` | B | 5 | yes |
| `src/metafopy/core/labelled.py::scientific_places` | B | 5 | yes |
| `src/metafopy/core/labelled.py::trim_zeros` | B | 5 | yes |
| `src/metafopy/core/nchg.py::fisher_log_probability` | A | 3 | yes |
| `src/metafopy/core/nchg.py::ln_fac` | A | 3 | yes |
| `src/metafopy/core/nchg.py::tabulated_log_probability` | A | 3 | yes |
| `src/metafopy/core/padjust.py::match_adjust` | C | 8 | yes |
| `src/metafopy/core/timing.py::format_time` | B | 5 | yes |
| `src/metafopy/core/timing.py::number` | B | 5 | yes |
| `src/metafopy/core/tukey.py::ptukey` | A | 3 | yes |
| `src/metafopy/core/tukey.py::qtukey` | A | 3 | yes |
| `src/metafopy/core/tukey.py::wprob` | A | 3 | yes |
| `src/metafopy/effects/datatools.py::charmatch` | C | 8 | yes |
| `src/metafopy/effects/datatools.py::make_names` | C | 8 | yes |
| `src/metafopy/effects/datatools.py::r_round` | C | 8 | no (new) |
| `src/metafopy/effects/datatools.py::shorten_names` | C | 8 | yes |
| `src/metafopy/effects/fivenum_estimators.py::sum_stat_index` | C | 8 | yes |
| `src/metafopy/effects/fivenum_estimators.py::weibull_shape_start` | A | 3 | yes |
| `src/metafopy/effects/measures.py::make_unique_dot1` | C | 8 | yes |
| `src/metafopy/effects/rtet.py::rtet` | A | 3 | yes |
| `src/metafopy/effects/summary.py::level_to_alpha` | B | 6 | yes |
| `src/metafopy/formatters.py::fmtt` | B | 5 | no (new) |
| `src/metafopy/formatters.py::r_as_character` | B | 5 | yes |
| `src/metafopy/formula.py::call_spans` | C | 7 | yes |
| `src/metafopy/formula.py::parse_random` | C | 7 | yes |
| `src/metafopy/formula.py::r_deparse` | C | 7 | yes |
| `src/metafopy/formula.py::r_logical_calls` | C | 7 | yes |
| `src/metafopy/formula.py::r_logical_formula` | C | 7 | yes |
| `src/metafopy/formula.py::r_number` | B | 5 | yes |
| `src/metafopy/formula.py::r_power` | C | 7 | yes |
| `src/metafopy/formula.py::r_tokens` | C | 7 | yes |
| `src/metafopy/interactive/forest.py::unique_labels` | C | 8 | yes |
| `src/metafopy/metadata.py::make_unique` | C | 8 | yes |
| `src/metafopy/methods/anova.py::check_test` | C | 8 | no (new) |
| `src/metafopy/methods/blup.py::format_r_frame` | B | 6 | yes |
| `src/metafopy/methods/common.py::match_arg` | C | 8 | yes |
| `src/metafopy/methods/deltamethod.py::get_digits` | B | 6 | yes |
| `src/metafopy/methods/emm_adjust.py::match_method` | C | 8 | yes |
| `src/metafopy/methods/emm_adjust.py::parse_side` | C | 8 | yes |
| `src/metafopy/methods/emm_adjust.py::pf_cdf` | A | 4 | yes |
| `src/metafopy/methods/emm_adjust.py::pf_sf` | A | 4 | yes |
| `src/metafopy/methods/emm_adjust.py::pt_sf` | A | 4 | yes |
| `src/metafopy/methods/emm_adjust.py::qdunnx` | A | 4 | yes |
| `src/metafopy/methods/emm_adjust.py::qf` | A | 4 | yes |
| `src/metafopy/methods/emm_adjust.py::qt` | A | 4 | yes |
| `src/metafopy/methods/emm_grid.py::appearance` | C | 8 | yes |
| `src/metafopy/methods/emm_nesting.py::factor_id` | C | 8 | yes |
| `src/metafopy/methods/emm_nesting.py::strip_supersets` | C | 8 | yes |
| `src/metafopy/methods/emm_nesting.py::term_info` | C | 8 | yes |
| `src/metafopy/methods/hettest.py::expand_boot` | C | 8 | yes |
| `src/metafopy/methods/ls_anova.py::omnibus` | A | 3 | yes |
| `src/metafopy/methods/ls_common.py::center_stars` | B | 6 | yes |
| `src/metafopy/methods/matreg.py::match_columns` | C | 8 | yes |
| `src/metafopy/methods/matreg.py::parse_formula` | C | 7 | yes |
| `src/metafopy/methods/matreg.py::pmatch` | C | 8 | yes |
| `src/metafopy/methods/matreg.py::resolve_x` | C | 8 | yes |
| `src/metafopy/methods/matreg.py::resolve_y` | C | 8 | yes |
| `src/metafopy/methods/matreg.py::subsequence_distance` | C | 8 | yes |
| `src/metafopy/methods/predict.py::is_subsequence` | C | 8 | yes |
| `src/metafopy/methods/predict.py::level_indices` | C | 8 | yes |
| `src/metafopy/methods/predict.py::match_names` | C | 8 | yes |
| `src/metafopy/methods/predict.py::resolve_predtype` | C | 8 | no (new) |
| `src/metafopy/methods/rcalc.py::formula_variables` | C | 7 | yes |
| `src/metafopy/methods/rcalc.py::match_nfun` | C | 7 | yes |
| `src/metafopy/methods/robust.py::check_options` | C | 8 | no (new) |
| `src/metafopy/methods/vif.py::format_btt` | B | 6 | yes |
| `src/metafopy/methods/vif.py::format_vector` | B | 6 | yes |
| `src/metafopy/mfopt.py::option_space` | B | 6 | no (new) |
| `src/metafopy/models/call.py::split_terms` | C | 7 | yes |
| `src/metafopy/models/call.py::update_formula` | C | 7 | yes |
| `src/metafopy/models/call.py::update_side` | C | 7 | yes |
| `src/metafopy/models/mv_factor.py::format_number` | B | 5 | yes |
| `src/metafopy/models/mv_factor.py::split_top_level` | C | 7 | yes |
| `src/metafopy/models/mv_hessian.py::inner_names` | C | 7 | yes |
| `src/metafopy/models/mv_hessian.py::lower_triangle_order` | C | 7 | yes |
| `src/metafopy/models/mv_hessian.py::numbered` | C | 7 | no (new) |
| `src/metafopy/models/mv_inputs.py::formula_variables` | C | 7 | no (new) |
| `src/metafopy/models/mv_print.py::abbreviate` | C | 7 | yes |
| `src/metafopy/models/mv_print.py::abbreviate_once` | C | 7 | yes |
| `src/metafopy/models/rma_glmm.py::check_arguments` | C | 8 | no (new) |
| `src/metafopy/models/rma_mv.py::replace_call_notation` | C | 7 | yes |
| `src/metafopy/models/rma_mv.py::resolve_rscale` | C | 8 | no (new) |
| `src/metafopy/models/rma_uni.py::abbreviate_names` | C | 7 | yes |
| `src/metafopy/models/rma_uni.py::coef_span` | B | 6 | yes |
| `src/metafopy/models/rma_uni.py::first_df` | B | 6 | no (new) |
| `src/metafopy/models/rma_uni.py::level_to_alpha` | B | 6 | yes |
| `src/metafopy/models/rma_uni.py::pfmt` | B | 6 | no (new) |
| `src/metafopy/models/robust_print.py::decorate` | B | 6 | no (new) |
| `src/metafopy/models/sel_lik.py::apply_named` | A | 4 | yes |
| `src/metafopy/models/sel_lik.py::exp_safe` | A | 4 | yes |
| `src/metafopy/models/sel_lik.py::map_delta` | A | 4 | yes |
| `src/metafopy/models/sel_lik.py::map_inverse` | A | 4 | yes |
| `src/metafopy/models/sel_opt.py::choose_optimizer` | C | 8 | yes |
| `src/metafopy/models/sel_opt.py::partial_match` | C | 8 | yes |
| `src/metafopy/models/sel_spec.py::match_type` | C | 8 | yes |
| `src/metafopy/models/sel_spec.py::s_halfnorm` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_halfnorm1` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_halfnorm2` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_logistic` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_logistic1` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_logistic2` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_negexp` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_negexp1` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_negexp2` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_negexppow` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_power` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_power1` | A | 4 | yes |
| `src/metafopy/models/sel_spec.py::s_power2` | A | 4 | yes |
| `src/metafopy/models/selmodel.py::r_as_character` | B | 5 | yes |
| `src/metafopy/models/tau2.py::i2_from_q` | A | 3 | yes |
| `src/metafopy/plot/forest.py::anno_symbol` | B | 6 | yes |
| `src/metafopy/plot/forest.py::finish_annotext` | B | 6 | yes |
| `src/metafopy/plot/forest.py::parse_predtail` | B | 6 | yes |
| `src/metafopy/plot/gosh.py::hex_color` | B | 6 | yes |
| `src/metafopy/plot/infl.py::abbreviate` | C | 7 | yes |
| `src/metafopy/plot/infl.py::n2mfrow` | B | 6 | yes |
| `src/metafopy/plot/profile.py::per_panel` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::color_ramp` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::fmtx1` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::level_to_alpha` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::r_pretty` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::rround` | B | 6 | yes |
| `src/metafopy/plot/rutils.py::usr_range` | B | 6 | yes |
| `src/metafopy/printing.py::fmt_df_scalar` | B | 5 | yes |
| `src/metafopy/printing.py::fmt_named_vector` | B | 5 | yes |
| `src/metafopy/printing.py::fmt_num` | B | 5 | yes |
| `src/metafopy/printing.py::fmt_pval` | B | 5 | yes |
| `src/metafopy/printing.py::fmt_table` | B | 5 | yes |
| `src/metafopy/printing.py::fmt_wrapped_table` | B | 5 | yes |
| `src/metafopy/printing.py::number_rows` | B | 5 | yes |
| `src/metafopy/printing.py::signif_stars` | B | 5 | yes |
| `src/metafopy/printing.py::zero_below` | B | 5 | yes |
