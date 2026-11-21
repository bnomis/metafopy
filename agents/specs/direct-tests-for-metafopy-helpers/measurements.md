# Measurements: direct tests for metafopy helpers

<!-- doc-counts: accept-start measured on 2026-11-21 - a dated record, not a claim about the present -->
Measured on 2026-11-21 from `agents/audit-reviews/ledger/T1.jsonl` (verdicts judged at `31d809f`) and from `metafopy.__all__` and each module's `__all__` at `a6251d1`: the 162 open T1 findings under `src/metafopy`, by how public each function is. Groups follow `auditme.core.is_private` in the auditor checker: a function is private when no `__all__` above it (its module's, or any enclosing package `__init__.py`'s) names its first qualname segment.

## Exported from `metafopy` (functions) (2)

- `effects/measures.py::replmiss`
- `plot/rutils.py::fmtx`

## Methods of classes exported from `metafopy` (28)

- `metadata.py::MetaData.design`
- `metadata.py::MetaData.names_all`
- `metadata.py::MetaData.ni_f`
- `metadata.py::MetaData.weights_f`
- `methods/confint.py::ConfintResult.ci_null`
- `methods/leave1out.py::Leave1outResult.to_frame`
- `methods/matreg.py::MatregResult.stat`
- `methods/matreg.py::MatregResult.table_lines`
- `methods/permutest.py::PermutestResult.to_frame`
- `methods/predict.py::PredictResult.__post_init__`
- `methods/rcalc.py::RCalcResult.__post_init__`
- `methods/selmodel.py::SelProfileResult.delta`
- `models/glmm_result.py::GLMMResult.heterogeneity_lines`
- `models/glmm_result.py::GLMMResult.render`
- `models/glmm_result.py::GLMMResult.title_lines`
- `models/mv_result.py::RMAMVResult.R`
- `models/mv_result.py::RMAMVResult.__post_init__`
- `models/mv_result.py::RMAMVResult.dimnames_of`
- `models/mv_result.py::RMAMVResult.named_inner`
- `models/mv_result.py::RMAMVResult.random`
- `models/mv_result.py::RMAMVResult.weight_matrix`
- `models/rma_ls_result.py::RMALSResult.__post_init__`
- `models/rma_ls_result.py::RMALSResult.hat_matrix`
- `models/rma_mh.py::MHResult.render`
- `models/rma_mh.py::MHResult.weight_values`
- `models/rma_uni.py::RMAResult.hat_matrix`
- `models/sel_result.py::SelmodelResult.__post_init__`
- `models/tau2.py::Control.__getattr__`

## Named only in their own module's `__all__` (functions) (108)

- `bias/ranktest.py::kendall_test`
- `core/bvn.py::bvnu`
- `core/calculus.py::derivative_mixed`
- `core/calculus.py::hessian`
- `core/clogit.py::clogistic`
- `core/clogit.py::clogit`
- `core/clogit.py::clogit_settings`
- `core/clogit.py::coxph_ties`
- `core/clubsandwich.py::contrast_df`
- `core/clubsandwich.py::vcov_cr`
- `core/clubsandwich.py::wald_test`
- `core/dbinom.py::dbinom_raw`
- `core/farebrother.py::farebrother`
- `core/glm.py::aliased_columns`
- `core/glm.py::check_link`
- `core/glm.py::deviance_residuals`
- `core/glm.py::family_constant`
- `core/glm.py::glm_fit`
- `core/glm.py::saturated_loglik`
- `core/glm.py::valid_mu`
- `core/glmm.py::fit_glmm`
- `core/glmmadaptive.py::fit_adaptive`
- `core/labelled.py::design_matrix`
- `core/labelled.py::matrix_text`
- `core/linalg.py::drop_redundant`
- `core/linalg.py::is_symmetric`
- `core/nchg.py::study_loglik`
- `core/numderiv.py::grad_scalar`
- `core/parallel.py::map_sims`
- `core/parallel.py::tick`
- `core/rng.py::get_rng`
- `core/stationarity.py::newton_decrement`
- `core/timing.py::timed`
- `core/zeroin.py::zeroin2`
- `effects/fivenum_estimators.py::estimate`
- `effects/rtet.py::rtet`
- `formatters.py::format_c`
- `formula.py::formula_context`
- `formula.py::r_factor`
- `methods/binary.py::refit_tables`
- `methods/emm_adjust.py::critical_values`
- `methods/emm_adjust.py::p_values`
- `methods/emm_contrasts.py::build_contrasts`
- `methods/emm_links.py::measure_link`
- `methods/emm_nesting.py::nonempty_display`
- `methods/emm_summary.py::summary`
- `methods/fitted.py::hat_values`
- `methods/fitted.py::rows_of`
- `methods/fitview.py::check_method`
- `methods/listrma.py::rows`
- `methods/mv_diagnostics.py::refit_mv`
- `methods/pairmat.py::resolve`
- `methods/vcov.py::vcov_frame`
- `methods/weights.py::weights_mv`
- `methods/weights.py::weights_uni`
- `mfopt.py::option_digits`
- `mfopt.py::option_space`
- `mfopt.py::reduce_digits`
- `models/binary.py::fit_statistics`
- `models/binary.py::format_header`
- `models/glmm_fit.py::fit_model`
- `models/mv_factor.py::evaluate_variable`
- `models/mv_factor.py::indicator`
- `models/mv_factor.py::interaction`
- `models/mv_factor.py::split_top_level`
- `models/mv_factor.py::to_factor`
- `models/mv_hessian.py::variance_hessian`
- `models/mv_inputs.py::build_x`
- `models/mv_inputs.py::extract_random`
- `models/mv_inputs.py::prepare_v`
- `models/mv_inputs.py::prepare_w`
- `models/mv_inputs.py::prepare_yi`
- `models/mv_inputs.py::take_square`
- `models/mv_lik.py::evaluate`
- `models/mv_lik.py::final_fit`
- `models/mv_lik.py::quad_form`
- `models/mv_print.py::abbreviate`
- `models/mv_print.py::format_result`
- `models/mv_structs.py::con_e`
- `models/mv_structs.py::con_vcov_un`
- `models/mv_structs.py::process_after_rmna`
- `models/mv_structs.py::process_after_sub`
- `models/rma_ls.py::fit_scale`
- `models/rma_ls_fit.py::fit_rma_ls`
- `models/rma_ls_fit.py::make_ls_control`
- `models/rma_ls_opt.py::constr_optim`
- `models/rma_ls_opt.py::minimise`
- `models/rma_ls_opt.py::polish_identity`
- `models/rma_ls_print.py::coef_table`
- `models/rma_ls_print.py::format_result`
- `models/robust_print.py::counts_lines`
- `models/robust_print.py::legend_lines`
- `models/sel_lik.py::evaluate`
- `models/sel_lik.py::map_delta`
- `models/sel_lik.py::map_inverse`
- `models/sel_opt.py::minimize`
- `models/sel_result.py::format_selmodel`
- `models/sel_spec.py::build_spec`
- `models/tables.py::pair`
- `models/tables.py::prepare_tables`
- `plot/histplot.py::draw_histogram`
- `plot/histplot.py::draw_tail`
- `plot/histplot.py::panels`
- `printing.py::drop_stars`
- `printing.py::fmt_df`
- `printing.py::fmt_df_scalar`
- `printing.py::fmt_wrapped_table`
- `printing.py::signif_footer`

## Methods of classes named only in their own module's `__all__` (19)

- `core/labelled.py::EscalcFrame.digit_for`
- `core/labelled.py::EscalcFrame.text`
- `core/labelled.py::LabelledMatrix.__str__`
- `core/labelled.py::LabelledMatrix.transpose`
- `core/labelled.py::VCovMat.text`
- `core/rng.py::RRandom.prob_sample_no_replace`
- `core/rng.py::RRandom.prob_sample_replace`
- `core/rng.py::RRandom.rbits`
- `core/rng.py::RRandom.sample_int`
- `core/rng.py::RRandom.unif_index`
- `core/rng.py::RRandom.unif_rand`
- `core/rng.py::RRandom.walker`
- `methods/anova.py::ListAnovaResult.__init__`
- `methods/ls_anova.py::LSWaldResult.stat_line`
- `methods/ls_predict.py::LSPredictResult.__post_init__`
- `methods/mv_profile.py::MVProfileResult.component_grid`
- `models/mv_lik.py::Problem.penalty`
- `models/rma_ls_fit.py::LSProblem.score_parts`
- `plot/funnel.py::FunnelResult.to_frame`

## Already private (named in no `__all__`), and no test reaches them (5)

- `methods/permutest.py::fixed_btt`
- `methods/predict.py::fitted_levels`
- `models/mv_inputs.py::block_diagonal`
- `models/mv_result.py::formula_text`
- `models/mv_structs.py::distance_from_matrix`

<!-- doc-counts: accept-end -->
