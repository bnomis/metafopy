# Changelog

## Unreleased (1.0 candidate)

- Add `selmodel` (selection models for publication bias) with `predict`, `confint`, `profile`, `anova` and `plot_selmodel`.
- Add location-scale models via `rma_uni(..., scale=...)`.
- `rma_mv`: new arguments `optbeta` (with `beta`, `lambda1`, `lambda2`), `vccon`, `cvvc` and `restart`; `RMAMVResult` fills `hessian`, `vvc` and `vc_names`; `MVControl` gains `hessian_d`, `hessian_eps`, `hessian_r`.
- `robust`: `coef_test='saddlepoint'` and the `HTA`, `HTB`, `EDF`, `EDT` Wald tests (clubSandwich), for `rma_uni` and `rma_mv`.
- `rma_glmm`: experimental measures `RR`, `RD`, `PR`, `PLN`; `predict()` for `rma_glmm` fits.
- `forest()`/`addpoly()`: `predstyle` `'shade'` and `'dist'`, `predlim`, `preddist`, `predtail`, `fonts`; `predict()` results carry `pi_se`, `pi_dist`, `pi_ddf`.
- `vif(sim=, seed=, parallel=, ncpus=, cl=, progbar=)` for location-scale models; `profile(..., parallel=, ncpus=, cl=, progbar=)` for selection models; `metafopy.core.parallel`; R fixtures and tests for `selmodel(type='stepcon')`.
- Add a documentation site (MkDocs Material).
