# Open after privatising: direct tests for metafopy helpers

<!-- doc-counts: accept-start measured on 2026-11-21 - a dated record, not a claim about the present -->
Measured on 2026-11-21 at `2b52cd2` (the commit of Task 2, which made the internal helpers private), after Task 3 re-judged the 162 candidates of `measurements.md` by the T1 brief and the rulings in `agents/audit-reviews/review-1.md`, and merged the verdicts into `agents/audit-reviews/ledger/T1.jsonl`. These are the candidates that `phaseb.py report . --question T1` still shows open under `src/metafopy`: 71 of the 162, 30 public and 41 private. The other candidates have a test that counts. A private helper is covered by a test of a caller one call away, so Tasks 6 and 7 add caller tests for the private list. Tasks 4 and 5 add direct tests for the public list.

A batch line is public or private by its `private` flag.

## Public (30)

- `effects/measures.py::replmiss`
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
- `plot/funnel.py::FunnelResult.to_frame`
- `plot/rutils.py::fmtx`

## Private (41)

- `core/calculus.py::derivative_mixed`
- `core/calculus.py::hessian`
- `core/clogit.py::clogistic`
- `core/clogit.py::clogit`
- `core/clogit.py::clogit_settings`
- `core/clogit.py::coxph_ties`
- `core/clubsandwich.py::contrast_df`
- `core/clubsandwich.py::vcov_cr`
- `core/clubsandwich.py::wald_test`
- `core/farebrother.py::farebrother`
- `core/glm.py::aliased_columns`
- `core/glm.py::deviance_residuals`
- `core/glm.py::family_constant`
- `core/glm.py::glm_fit`
- `core/glm.py::saturated_loglik`
- `core/glm.py::valid_mu`
- `core/glmm.py::fit_glmm`
- `core/glmmadaptive.py::fit_adaptive`
- `core/labelled.py::EscalcFrame.digit_for`
- `core/labelled.py::EscalcFrame.text`
- `core/labelled.py::LabelledMatrix.__str__`
- `core/labelled.py::LabelledMatrix.transpose`
- `core/labelled.py::VCovMat.text`
- `core/labelled.py::design_matrix`
- `core/labelled.py::matrix_text`
- `core/numderiv.py::grad_scalar`
- `core/parallel.py::map_sims`
- `core/rng.py::RRandom.prob_sample_no_replace`
- `core/rng.py::RRandom.prob_sample_replace`
- `core/rng.py::RRandom.rbits`
- `core/rng.py::RRandom.sample_int`
- `core/rng.py::RRandom.unif_index`
- `core/rng.py::RRandom.unif_rand`
- `core/rng.py::RRandom.walker`
- `core/stationarity.py::newton_decrement`
- `core/timing.py::timed`
- `effects/fivenum_estimators.py::estimate`
- `methods/permutest.py::fixed_btt`
- `models/mv_inputs.py::block_diagonal`
- `models/mv_result.py::formula_text`
- `models/mv_structs.py::distance_from_matrix`

<!-- doc-counts: accept-end -->
