# Methods for fitted models

Functions that take a fitted model. Each is a single-dispatch function: the generic function chooses the implementation from the class of the fitted model, and the implementations are documented below the generic. A call with a class that has no implementation raises `metafopy.errors.ArgumentError`.

## Predictions and intervals

::: metafopy.predict
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.predict.predict_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.predict.predict_rma_glmm
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.predict.predict_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_predict.predict_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.PredictResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.confint
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.confint.confint_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.confint.confint_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_confint.confint_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.binary.confint_fixed_binary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.ConfintResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Tests and profiles

::: metafopy.anova
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.anova.anova_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.anova.anova_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_anova.anova_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.WaldResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.LRTResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.hettest
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.HettestResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.permutest
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.permutest.permutest_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_permutest.permutest_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.PermutestResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.profile
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.profile.profile_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_profile.profile_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_profile.profile_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.ProfileResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.SelProfileResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_profile.MVProfileResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.vif
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.vif.vif_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_vif.vif_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.VIFResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.VIFEntry
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Cluster-robust inference

::: metafopy.robust
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.robust.robust_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.robust.robust_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Residuals and influence

::: metafopy.rstandard
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.influence.rstandard_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.binary.rstandard_fixed_binary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_resid.rstandard_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.rstandard_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.rstudent
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.influence.rstudent_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.binary.rstudent_fixed_binary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.rstudent_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.influence
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.influence.influence_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.influence_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.cooks_distance
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.cooks_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.dfbetas
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.dfbetas_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.InfluenceResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.ResidualResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_diagnostics.ClusterResidualResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Leave-one-out and cumulative analyses

::: metafopy.leave1out
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.leave1out.leave1out_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.binary.leave1out_fixed_binary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.Leave1outResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.cumul
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.leave1out.cumul_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.binary.cumul_fixed_binary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.CumulResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Random effects and GOSH

::: metafopy.blup
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.blup.blup_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.blup.blup_rma_glmm
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_resid.blup_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.BlupResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.ranef
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ranef.ranef_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ranef.ranef_rma_glmm
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.ls_resid.ranef_rma_ls
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.mv_ranef.ranef_rma_mv
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.RanefResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.gosh
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.methods.gosh.gosh_rma_uni
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.GoshResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Sampling variance-covariance matrices

::: metafopy.vcalc
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.rcalc
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.RCalcResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.bldiag
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.blsplit
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.pairmat
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Regression on correlation and covariance matrices

::: metafopy.matreg
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.MatregResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.MatregConfintResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.MatregPredictResult
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Estimated marginal means

::: metafopy.emmprep
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.emmeans
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.contrast
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.pairs
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.regrid
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.EmmGrid
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: metafopy.EmmSummary
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Random number generator

::: metafopy.set_seed
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false
