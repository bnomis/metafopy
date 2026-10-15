# Changelog

## Unreleased (1.0 candidate)

- `hettest` (metafor 5.2-1): score, likelihood ratio and Wald-type tests of heteroscedastic heterogeneity, and Kolmogorov-Smirnov and Anderson-Darling tests of the Pearson residuals, with parametric bootstrap p-values from R's generator (`src/tests/r/generate_hettest.R`).
- `residuals(type=, na_action=)` with `'response'`, `'rstandard'`, `'rstudent'`, `'pearson'` and `'cholesky'` residuals (as `res.residuals(...)` and `metafopy.residuals(res, ...)`), following metafor's rules for which fit classes allow which types; `weights_(type='matrix', na_action=)` for `rma_uni` and location-scale fits, and `'matrix'` for `rma_mv`; `rma_glmm` and selection-model fits refuse `weights_` as in metafor. The `exclude` and `pass` actions pad to the rows of `subset`, like R.
- `plot_profile` draws the list that `profile()` returns for `rma_mv`, location-scale and selection models (one panel per component) and single location-scale profiles; `profile()` of location-scale fits takes `plot=` and `ax=`.
- `rma_glmm`: the `clogit` and `clogistic` optimisers of `CM.EL` (`src/metafopy/core/clogit.py`) and the `link` and `family` overrides.
- `anova(object2=)` and `robust(clubSandwich=)` accept R's argument names.
- `emmeans`: nested factors (`emmprep(nesting=)`, detected from the data and the model terms like emmeans' `.find_nests`, with `contrast`/`pairs` within groups of different size), the `side`, `delta` and `null` arguments (one-sided, equivalence and non-inferiority tests and intervals), and matrix `weights`. Checked against emmeans 2.0.4 on R fixtures (`src/tests/r/generate_emmeans.R`).
- `conv_fivenum`: the `qe`, `bc`, `mln` (estmeansd) and `blue` (metaBLUE) methods, with `seed` and `verbose` arguments. `bc`, `mln` and `blue` reproduce R to 1e-9 relative, `qe` to about 1e-6 (`src/tests/r/generate_fivenum_methods.R`).
- Add a native port of emmeans for `rma` fits: `emmprep`, `emmeans`, `contrast`, `pairs`, `regrid`, with the contrast families and multiplicity adjustments of emmeans (`tukey`, `dunnettx`, `sidak`, `scheffe`, `mvt`, `p_adjust` methods), back-transformation with optional bias adjustment, and an exact port of R's `ptukey`/`qtukey` (`metafopy.core.tukey`). Checked against emmeans and metafor on R fixtures (`src/tests/r/generate_emmeans.R`).
- `rma_glmm(control={'package': 'GLMMadaptive'})` and `{'package': 'glmmTMB'}`: the two other back-ends of metafor 5.2-1, checked against R fixtures (`src/tests/r/generate_glmm_packages.R`); GLMMadaptive reproduces `mixed_fit`'s EM and BFGS iterations. `optim_bfgs` takes `gr` and `parscale`.
- Fill the remaining gaps to metafor 5.2-1 found by comparing every method on every fit class in R and Python:
  - Data helpers: `aggregate_escalc`, `conv_2x2`, `conv_delta`, `conv_fivenum`, `conv_wald`, `to_long`, `to_table`, `to_wide`, `contrmat`, `vec2mat`, `dfround`, a public `replmiss`, and `deltamethod`; datasets `dat.hasselblad1998`, `dat.ishak2007` and `dat.senn2013`.
  - `matreg` (regression on correlation and covariance matrices) with `confint` and `predict`.
  - `rma_mv`: `profile`, `ranef`, `cooks_distance`, `dfbetas`, `rstandard`, `rstudent`; `hatvalues(type=, na_action=)`.
  - `forest`, `funnel`, `ranktest`, `radial`, `baujat`, `qqnorm`, `labbe`, `gosh`, `tes` and `predict` accept every fit class metafor accepts (`rma_mh`, `rma_peto`, `rma_mv`, `rma_glmm`, location-scale and selection models), and raise metafor's message where metafor refuses a class. `addpoly()` takes `predict` results. New `plot_rma` (the four-panel plot of `plot.rma`) and `.plot()` methods.
  - Arguments: `predict(hetvar=, prob=, addx=, vcov=, adjust=, tau2_levels=)`, `anova(adjust=, refit=)` with an exact port of `p.adjust`, `parallel`/`ncpus`/`cl`/`progbar` for `profile` and `gosh`, `plot_permutest(alpha=, QS=)`, and `verbose` on the model functions, `confint`, `trimfill` and `fsn`.
  - `simulate()` with R-exact draws, `update()`, `se()`, `nobs()`, `df_residual()`, `model_matrix()` and `.formula()`; `fitted()`/`residuals()` for `rma_glmm`; the formatters `fmtx`, `fmtp`, `fmtp2`, `fmtt` and `setmfopt`/`getmfopt`.
- `rma_mv` accepts a `V` with missing covariances in mirrored positions and drops rows with missing values as metafor's `.anyNAv()` does (it rejected the `rcalc` output of `dat.craft2003`).
- `escalc(measure='HR'/'HD')` passes `yi` and `vi` through as metafor does.
- `inspect.signature(metafopy.rma_uni)` (and `help()`) no longer raises.
- Register the `slow` pytest marker.

- `vif(sim=)` of `rma_uni` models shuffles like metafor with R's generator (`seed + l` in iteration `l`, or the stream of `set_seed` without `seed`), shuffling the variables of the data for a model given as a formula (`sim_type='data'`), so `set_seed(s)` gives the simulated VIFs of R's `set.seed(s)`. The gallery figure uses `sim=1000`, the 1000 iterations R's `sim=TRUE` runs.
- `plot_permutest(legend=...)` draws the legend in the first panel only, as metafor does. The gallery figure of the permutation distributions is seeded with `set_seed(1234)` like the R script, so its axes and histograms now match R's.
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
