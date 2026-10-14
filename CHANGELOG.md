# Changelog

## Unreleased (1.0 candidate)

- Add the metafor plots that were missing: `regplot` (meta-analytic scatterplot), `forest()` of a `cumul` result (`forest_cumul`), `plot_cumul` (plot of cumulative results), `llplot`, `plot_permutest` and `plot_vif`. Each is checked against the geometry R records. `cumul` results carry `measure`, `level` and `slab_null`. The gallery shows the six next to metafor's figures.
- Add the BMJ, RevMan and superscript forest plot styles to the gallery, drawn from `forest()` with extra annotations as on the metafor pages. `forest()` of a fitted model takes `na_action='pass'` (metafor's `options(na.action='na.pass')`) to keep the studies with missing values as rows of the plot, which the RevMan style needs; `fail` raises an error if there are any.
- Add a plot gallery to the user manual (`docs/guide/gallery.md`) with the plots of the metafor gallery drawn by metafopy next to the R figures (`bin/gallery.py`, `src/tests/r/gallery.R`), and the datasets `dat.damico2009` and `dat.pignon2000`.
- `plot_infl(mfrow=)` sets the layout of the panels; `addpoly()` accepts a fitted `rma_mv` model; the zero tick of a plot is exactly zero; overlapping tick labels of the categorical x axes are left out as R does.
- Add `selmodel` (selection models for publication bias) with `predict`, `confint`, `profile`, `anova` and `plot_selmodel`.
- Add location-scale models via `rma_uni(..., scale=...)`.
- `rma_mv`: new arguments `optbeta` (with `beta`, `lambda1`, `lambda2`), `vccon`, `cvvc` and `restart`; `RMAMVResult` fills `hessian`, `vvc` and `vc_names`; `MVControl` gains `hessian_d`, `hessian_eps`, `hessian_r`.
- `robust`: `coef_test='saddlepoint'` and the `HTA`, `HTB`, `EDF`, `EDT` Wald tests (clubSandwich), for `rma_uni` and `rma_mv`.
- `rma_glmm`: experimental measures `RR`, `RD`, `PR`, `PLN`; `predict()` for `rma_glmm` fits.
- `forest()`/`addpoly()`: `predstyle` `'shade'` and `'dist'`, `predlim`, `preddist`, `predtail`, `fonts`; `predict()` results carry `pi_se`, `pi_dist`, `pi_ddf`.
- `vif(sim=, seed=, parallel=, ncpus=, cl=, progbar=)` for location-scale models; `profile(..., parallel=, ncpus=, cl=, progbar=)` for selection models; `metafopy.core.parallel`; R fixtures and tests for `selmodel(type='stepcon')`.
- Add a documentation site (MkDocs Material): installation, a tutorial, a user guide, worked analyses, a guide for users coming from R and a function reference generated from the docstrings. `src/tests/test_docs_examples.py` runs every Python example in `docs/`.
- Make `metafopy.models` a regular package (it had no `__init__.py`), so that the documentation tools can resolve the public names.
