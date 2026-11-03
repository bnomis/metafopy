# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


## gosh gives different subsets and results from R on every fit

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). Every `gosh` fit differs from R, in which subsets are included and in every result of them. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. Every fit is affected, so the likely cause is the choice or order of subsets in `src/metafopy/methods/gosh.py` (random subsets when there are more than `subsets` of them), not one statistic.

Fix direction: compare the subset matrix `incl` with R's for one small fit first; the printed summaries follow from it.

Spec: agents/specs/r-fused-arithmetic/spec.md (implemented); tasks: agents/tasks/r-fused-arithmetic/tasks-1.md.

Causes, from the run's `wrong-or-missing-full.md`:

- `$incl` element 259: R False, Python True (86 cases; `gosh`, `print`)
- `$res$H2` element 9: R 2.01139580879891, Python 22.871620858684505 (relative 0.91) (86 cases; `gosh`, `print`)
- `$res$I2` element 9: R 50.2832811112825, Python 95.62776942579347 (relative 0.47) (86 cases; `gosh`, `print`)
- `$res$QE` element 9: R 10.0569790439946, Python 114.35810429342253 (relative 0.91) (86 cases; `gosh`, `print`)
- `$res$k` element 3: R 4, Python 5.0 (relative 0.2) (86 cases; `gosh`, `print`)
- `$res$estimate` element 6: R -0.0270396616679308, Python -0.5357504809424486 (relative 0.95) (50 cases; `gosh`, `print`)
- `$res$tau` element 0: R 0.384719728838867, Python 0.0 (relative 1) (60 cases; `gosh`)
- `$res$tau2` element 0: R 0.148009269757851, Python 0.0 (relative 1) (60 cases; `gosh`)
- `$res$intrcpt` element 0: R -0.408765362241168, Python 0.3991816712497156 (relative 2) (36 cases; `gosh`)
- `[]$ci.lb$intrcpt` element 3: R None, Python -1.3381058095114846 (33 cases; `profile`)
- `[]$ci.ub$intrcpt` element 3: R None, Python 0.07707642670447312 (32 cases; `profile`)
- `$fit` element 0: R False, Python True (24 cases; `gosh`)
- `$res$ablat` element 2: R 0.00233066315964768, Python -0.012250480672567066 (relative 1.2) (24 cases; `gosh`)
- `$res$factor.alloc.random` element 1: R -0.651139340705579, Python None (12 cases; `gosh`)
- `$res$factor.alloc.systematic` element 1: R -0.58522921765392, Python None (12 cases; `gosh`)
- `$res$year` element 0: R None, Python 0.014467269976428554 (12 cases; `gosh`)
- `$beta$sim` element 1152: R None, Python 1.0193929566782072 (2 cases; `vif`)
- `$log.year...1900...1000.` element 53: R None, Python 0.0633444763804744 (1 cases; `dfbetas`)
- `[]$beta$<name>` element 2: R None, Python 5.766776911948009 (1 cases; `profile`)
- `[]$beta$intrcpt` element 2: R None, Python -40.15864014777602 (1 cases; `profile`)
- `[]$ci.lb$<name>` element 2: R None, Python -13.618798384495301 (1 cases; `profile`)
- `[]$ci.ub$<name>` element 2: R None, Python 25.15235220839132 (1 cases; `profile`)
- `[]$ll` element 2: R None, Python -9.81822951023776 (1 cases; `profile`)
- line 4: R 'k         6.0000  2.0000  5.0000  6.0000  7.0000  10.0000', Python 'k         6.2500  2.0000  5.0000  6.0000   7.2500  10.0000' (36 cases; `gosh`, `print`)
- line 4: R 'k         6.1000  3.0000  4.7500  6.0000   8.0000  11.0000', Python 'k         7.5000  4.0000  5.7500  8.0000   9.2500  11.0000' (14 cases; `gosh`)
- line 2: R 'Model fits succeeded: 17', Python 'Model fits succeeded: 19' (12 cases; `gosh`)
- line 4: R 'k        6.7500  4.0000  5.0000  6.5000  8.0000 11.0000', Python 'k        6.7000  3.0000  5.7500  7.0000  8.0000 11.0000' (11 cases; `gosh`)
- line 4: R 'k         6.4000    3.0000   5.0000   6.0000  7.2500 11.0000', Python 'k         6.7500    4.0000   5.0000   6.5000  8.0000 11.00 (10 cases; `gosh`)
- line 3: R '1.0000', Python '1.00' (4 cases; `vif`)
- line 4: R 'k                        5.5000  3.0000  4.2500  5.5000  6.7500  8.0000', Python 'k                        6.2000  4.0000  5.0000 (1 cases; `gosh`)
- line 4: R 'k         5.5000    3.0000    4.2500   5.5000  6.7500   8.0000', Python 'k          6.2000    4.0000   5.0000   6.0000  7.0000 10 (1 cases; `gosh`)
- line 4: R 'k        7.2000  5.0000  6.0000  7.0000  8.7500 10.0000', Python 'k        6.8000  3.0000  6.0000  7.0000  8.0000  9.0000' (1 cases; `gosh`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. The `dfbetas` cause (`$log.year...1900...1000.`) is now matched by the ledger entry of `matrix/dfbetas/mv_wide`, which names element 53 where it named element 17, so a rerender no longer lists it.

Note 2026-11-03: `bin/metafor-audit.sh --only gosh` on the pinned R 4.6.2 with the fused `dbinom` (metafopy `5b09d65`, run `.audit/runs/20261103-204725`, local) lists no `$incl`, `$res` or summary-line (`k`, `Model fits succeeded`) difference for a `gosh` case: the causes that came from different random subsets have gone. The only `gosh` findings left are `matrix/gosh/uni_digits` (`$digits` R 2, Python 4.0, and its `print` line) and the `scripts/gosh/*` missing fields (`chksum*`, `verbose`), all matched by the ledger. Two ledger entries (`matrix/gosh/*` and `options/gosh/trimfill_bcg/*` in `src/audit/data/known.toml`) still give the reason "R 4.3.3"; they hide any `matrix/gosh` difference, so their reason is stale and the match should be narrowed to `uni_digits`. The slice did not run the `vif`, `dfbetas` or `profile` causes listed above; they stay open here until a whole audit lists them or not.

## Location-scale fits differ from R in estimates, permutation tests, profile bounds and print

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). Location-scale fits (`rma` with `scale`; the `ls_*` fits) differ in the last digits of their standard errors and tests, in `permutest` (permuted coefficients of the opposite sign), in `confint` scale bounds where R gives `NA`, in `vif`, in four optimizers metafopy refuses, and in the headings of their print. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced; several may share one. The findings point at `src/metafopy/models/rma_ls_fit.py` (convergence), `src/metafopy/methods/ls_permutest.py`, `src/metafopy/methods/ls_confint.py`, `src/metafopy/methods/ls_vif.py` and `src/metafopy/models/rma_ls_result.py` (print). The R process killed for memory is R's own limit on `permutest(exact=TRUE)`, not a metafopy fault.

Fix direction: start with the fit itself (the 211-case `QM` and 127-case `se` differences on `ls_year` and `ls_na`), since the tests, `confint` and `permutest` are built on it; then `permutest`'s signs, the `confint` bounds where R gives up, the optimizers, and the print headings. Split this item if the parts turn out to have separate causes.

Causes, from the run's `wrong-or-missing-full.md`:

- R: R passed the memory limit of 4 GiB (killed) (2 cases; `permutest`)
- EstimationError: Optimizer (Rcgmin) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (Rvmmin) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (lbfgsb3c) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (optimParallel) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- `$QM.perm` element 14: R 0.0200910239625718, Python 11.119548182959777 (relative 1) (50 cases; `permutest`)
- `$beta.perm$intrcpt` element 19: R -47.5589351508886, Python 45.83744578027836 (relative 2) (50 cases; `permutest`)
- `$beta.perm$year` element 19: R 0.0238767715964596, Python -0.023592544278131733 (relative 2) (50 cases; `permutest`)
- `$zval.perm$intrcpt` element 19: R -1.32883778217893, Python 1.9383745471523741 (relative 1.7) (50 cases; `permutest`)
- `$zval.perm$year` element 19: R 1.3115090918244, Python -1.9512703736744204 (relative 1.7) (50 cases; `permutest`)
- `$beta$prop` element 1: R 0.998998998998999, Python 0.998 (relative 0.001) (2 cases; `vif`)
- `$vcovmat` element 10: R 0.600577420109585, Python 0.6005034810249658 (relative 0.00012) (2 cases; `rma`)
- `$alpha$prop` element 1: R 0.698396793587174, Python 0.6717325227963525 (relative 0.038) (1 cases; `vif`)
- `$alpha$sim` element 1990: R 1.00000000064568, Python None (1 cases; `vif`)
- `$ci.lb.alpha` element 1: R -0.137767763314913, Python -0.13771275253750662 (relative 0.0004) (1 cases; `rma`)
- `$ci.ub.alpha` element 0: R 3.49331755351219, Python 3.4915535605161026 (relative 0.0005) (1 cases; `rma`)
- `$pval.alpha` element 0: R 0.626571724710284, Python 0.6264407445288733 (relative 0.00021) (1 cases; `rma`)
- `$se.alpha` element 1: R 0.0701795309043885, Python 0.07015146427298927 (relative 0.0004) (1 cases; `rma`)
- `$va` element 2: R -0.162084992095922, Python -0.1619553584200976 (relative 0.0008) (1 cases; `rma`)
- `$zval.alpha` element 1: R -0.00311216515101929, Python -0.0031133933120681464 (relative 0.00039) (1 cases; `rma`)
- line 12: R 'QM(df = 2) = 1.2830, p-val = 0.5265', Python 'QM(df = 2) = 1.2828, p-val = 0.5266' (211 cases; `permutest`, `print`, `rma`, `rma.glmm` and 2 more)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0622  -21.4105  0.5339  .', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0622  -21.4102   (127 cases; `print`, `rma`, `summary`, `update`)
- line 2: R 'alpha.1 -10.4383    NA -2.9930', Python 'alpha.1 -10.4383 -27.0874 -2.9930' (44 cases; `confint`)
- line 2: R 'alpha.1  -3.0288 -18.7856 0.4381', Python 'alpha.1  -3.0288 -25.1411 0.4381' (43 cases; `confint`)
- line 14: R 'intrcpt           -3.0288  2.1031  -1.4401  0.1498   -7.1508   1.0933', Python 'intrcpt           -3.0288  2.1032  -1.4401  0.14 (6 cases; `rma`)
- line 6: R '0.3083  0.8897', Python '0.3077  0.8898' (5 cases; `vif`)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646   8  0.0992  -23.3477  2.4711  .', Python 'intrcpt  -10.4383  5.5980  -1.8646   8  0.0992  -2 (5 cases; `rma`)
- line 12: R 'Test of Scale Coefficients (coefficients 2:3):', Python 'Model Results (Scale):' (3 cases; `rma`)
- line 6: R 'ablat   -0.0189  0.0017  -11.0003  <.0001  -0.0223  -0.0155  ***', Python '-0.0189  0.0017  -11.0003  <.0001  -0.0223  -0.0155  * (3 cases; `rma`)
- line 14: R 'intrcpt           -5.4749  27.8296  -0.1967  0.8440  -60.0200  49.0701', Python 'intrcpt           -5.4749  27.8241  -0.1968  0. (3 cases; `rma`)
- line 13: R 'I(ni/100)     -1.00  0.61  -1.64  0.05   -2.19   0.19  *', Python 'I(ni/100)     -1.00  0.61  -1.64  0.05   -2.19   0.19  .' (2 cases; `permutest`)
- line 2: R 'alpha.1    -3.03 -18.79  0.44', Python 'alpha.1    -3.03 -25.14  0.44' (2 cases; `confint`)
- line 2: R 'alpha.1   -10.44    NA -2.99', Python 'alpha.1   -10.44 -27.09 -2.99' (2 cases; `confint`)
- line 1: R "'log Lik.' -3.214979 (df=4)", Python "'log Lik.' -3.21498 (df=4)" (1 cases; `logLik`)
- line 2: R '-10.3388 5.5402 -21.1974 0.5199', Python '-10.3388 5.5401 -21.1971 0.5196' (1 cases; `predict`)
- line 2: R '-3.5286 1.7967 -7.0501 -0.0071', Python '-3.5286 1.7970 -7.0506 -0.0066' (1 cases; `predict`)
- line 3: R 'I(ni/100)  1.0094  0.28', Python 'I(ni/100)  1.0094  0.30' (1 cases; `vif`)
- line 2: R 'QS(df = 1) = 0.5476, p-val = 0.4593', Python 'QS(df = 1) = 0.5474, p-val = 0.4594' (1 cases; `anova`)
- line 4: R 'Test of Location Coefficients (coefficients 2:3):', Python 'Model Results (Location):' (1 cases; `rma`)
- line 15: R 'Test of Scale Coefficients (coefficients 2:3):', Python 'intrcpt' (1 cases; `rma`)
- line 6: R 'allocrandom      2.2723  0.50', Python 'allocrandom      2.2723  0.97' (1 cases; `vif`)
- line 6: R 'alpha.3   1.1903 <-10.7124    NA', Python 'alpha.3   1.1903 <-10.7124 >13.0929' (1 cases; `confint`)
- line 2: R 'alpha.1  -3.0288 -17.9949 -0.1956', Python 'alpha.1  -3.0288 -25.2283 -0.1956' (1 cases; `confint`)
- line 2: R 'alpha.1 -10.4383    NA    NA', Python 'alpha.1 -10.4383    NA -2.9930' (1 cases; `confint`)
- line 17: R 'imag         2.1262  1.1857   1.7932  0.0729  -0.1978   4.4502   .', Python 'imag         2.1258  1.1857   1.7929  0.0730  -0.19 (1 cases; `rma`)
- line 12: R 'intrcpt                                              0.0876  -1.2308  0.0846  .', Python 'intrcpt                                (1 cases; `rma`)
- line 14: R 'intrcpt           -3.4482  6.0407  -0.5708  0.5681  -15.2878   8.3913', Python 'intrcpt           -3.4482  6.0391  -0.5710  0.56 (1 cases; `rma`)
- line 8: R 'intrcpt       6.9574  27.4113   0.2538  0.7996  -46.7677  60.6825', Python 'intrcpt       6.9587  27.4109   0.2539  0.7996  -46.7 (1 cases; `rma`)
- line 3: R 'intrcpt     0.2303 0.0655  3.5166  <.1  0.1019  0.3586     NA     NA', Python 'intrcpt     0.2303 0.0655  3.5166 0.0004  0.1019   (1 cases; `vif`)
- line 15: R 'intrcpt     -2.3458  0.8353  -2.8082  0.0050  -3.9830  -0.7085  **', Python 'intrcpt     -2.3456  0.8354  -2.8079  0.0050  -3.98 (1 cases; `rma`)
- line 3: R 'intrcpt   0.2677 0.2247  1.1914  0.2 -0.1727  0.7080     NA     NA', Python 'intrcpt   0.2677 0.2247  1.1914 0.2335 -0.1727  0.70 (1 cases; `vif`)
- line 11: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0500   -21.4105  0.5339  *', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0500   -21.4102 (1 cases; `permutest`)
- line 11: R 'intrcpt  -10.4383  5.5982  -1.8646  0.1500   -21.4105  0.5339', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.1500   -21.4102  0 (1 cases; `permutest`)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0622  -19.6465  -1.2301  .', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0622  -19.6462  (1 cases; `rma`)
- line 3: R 'intrcpt -35.8515 40.6744 -0.8814  0.4 -115.5718 43.8688     NA     NA', Python 'intrcpt -35.8509 40.6745 -0.8814 0.3781 -115.5715 (1 cases; `vif`)
- line 3: R 'intrcpt -81.2229 19.3071 -4.2069  <.1 -119.0641 -43.3817     NA     NA', Python 'intrcpt -81.2229 19.3071 -4.2069 <.0001 -119.064 (1 cases; `vif`)
- line 1: R 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub X.intrcpt X.I.ni.100. X.meta', Python 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub X.intr (1 cases; `predict`)
- line 1: R 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub Z.intrcpt Z.I.ni.100. Z.imag', Python 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub Z.intr (1 cases; `predict`)
- line 2: R 'year  379.7309 223.8329 -58.9736 818.4354', Python '1 379.7302 223.8271 -58.9629 818.4233' (1 cases; `predict`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. One cause appeared that belongs to no item: `R: creation of server socket failed: port 11347 cannot be opened` (`profile`, R's own failure to open a port while shards ran; not a difference of metafopy).

## Selection models, trimfill and profile differ from R in small values and print

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). `selmodel` fits differ in standard errors and tests in the fourth digit, `profile` on them gives values where R gives none, and `trimfill` gives a different `se.k0`. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The findings point at `src/metafopy/models/selmodel.py` and `src/metafopy/models/sel_opt.py` (convergence), `src/metafopy/methods/profile.py` and `src/metafopy/bias/trimfill.py`.

Fix direction: compare `selmodel` on bcg with R at a tighter tolerance first; then `trimfill`'s `se.k0`.

Causes, from the run's `wrong-or-missing-full.md`:

- `element #: R #   < p <= #     #   #   NA    NA    NA     NA     NA   , Python #   < p <= #` element 25: R 0.1   < p <= 0.5     2   98.1051   NA    NA    NA     NA     NA   , Python 0.1   < p <= 0.5     2   98.1050   NA    NA    NA   (5 cases; `print`)
- `$se.k0` element 0: R 2.42672899037311, Python 2.5278357875775743 (relative 0.04) (3 cases; `trimfill`)
- `$ys` element 941: R 5.43592478667369e-05, Python 5.434893707014758e-05 (relative 0.00019) [plot(sav, ylim = c(0, 50), ci = TRUE, bty = "l", seed  (2 cases; `the examples and tests of metafor`)
- `$pval.delta` element 1: R 9.65330124664416e-05, Python 9.648235818601725e-05 (relative 0.00052) (1 cases; `selmodel`)
- `$se` element 0: R 0.0695218232173794, Python 0.06949089045726904 (relative 0.00044) (1 cases; `selmodel`)
- `$zval` element 0: R -2.3598482640218, Python -2.360898712835864 (relative 0.00044) (1 cases; `selmodel`)
- `[]$ci.ub$ablat` element 2: R -0.0128066816910312, Python -0.012807994996842143 (relative 0.0001) (1 cases; `profile`)
- `element #: R #   < p <= #     #   #   NA    NA    NA     NA     NA , Python #   < p <= #  ` element 22: R 0.1   < p <= 0.5     2   98.1051   NA    NA    NA     NA     NA , Python 0.1   < p <= 0.5     2   98.1050   NA    NA    NA     (1 cases; `print`)
- line 5: R '-0.1641  0.0695  -2.3598  0.0183  -0.3003  -0.0278  *', Python '-0.1641  0.0695  -2.3609  0.0182  -0.3003  -0.0279  *' (3 cases; `rma`, `selmodel`)
- line 1: R 'Estimated number of missing studies on the left side: 0 (SE = 2.4267)', Python 'Estimated number of missing studies on the left s (3 cases; `trimfill`)
- line 17: R '0.0037  2.1304  0.0017  0.9986  0.0000  4.1793', Python '0.0037  2.1307  0.0017  0.9986  0.0000  4.1798' (2 cases; `selmodel`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished; the stale `delta.1` print-diff ledger entry of `options/selmodel/*` was removed, because the pinned R no longer shows that print difference.

## Multivariate fits differ from R in the Hessian, ranef, rstudent and confint

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). `rma.mv` fits return a Hessian of a different size, `ranef` accepts what R refuses, and `rstudent` and `confint` give values where R gives `NA`. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The findings point at `src/metafopy/models/rma_mv.py`, `src/metafopy/methods/mv_ranef.py` and `src/metafopy/methods/mv_diagnostics.py`.

Fix direction: start with the Hessian's size (R 9, Python 4 on `mv_sp`), which decides what the other results are built on.

Causes, from the run's `wrong-or-missing-full.md`:

- R: Non-numeric argument to mathematical function (2 cases; `ranef`)
- `$hessian` length R 9, Python 4 (3 cases; `rma.mv`)
- `$vvc` length R 9, Python 4 (3 cases; `rma.mv`)
- `$sim_1` element 28: R 0.00452385962595871, Python 0.004523223001831139 (relative 0.00014) (1 cases; `simulate`)
- `[]$random` element 4: R 0.0626269108525104, Python 0.06260801509732174 (relative 0.0003) (1 cases; `confint`)
- line 3: R '2       NA     NA      NA', Python '2  -0.3409 0.4999 -0.6819' (1 cases; `rstudent`)
- line 8: R 'tau^2.3   0.0007    NA    NA', Python 'tau^2.3   0.0007 0.0002 0.0062' (1 cases; `confint`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. The `rstudent` cause (line 3) is now matched by the ledger entry of `matrix/rstudent/mv_wide`, which names line 3 where it named line 19, so a rerender no longer lists it.

## Generalised linear mixed models differ from R in tests and a warning

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). `rma.glmm` fits differ in their tests (`Wld` 268.5990 against 191.7124 for `model="CM.EL"`), in small values, and do not give R's Hessian warning for the saturated model. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The findings point at `src/metafopy/models/rma_glmm.py` and `src/metafopy/models/glmm_fit.py`.

Fix direction: start with the `Wld` difference, which is far beyond rounding.

Causes, from the run's `wrong-or-missing-full.md`:

- `$prob` element 0: R 0.00197382147421394, Python 0.0019733636664781558 (relative 0.00023) (1 cases; `predict`)
- line 11: R 'QM(df = 1) = 76.2183, p-val < .0001', Python 'QM(df = 1) = 76.2266, p-val < .0001' (5 cases; `rma`, `rma.glmm`)
- line 11: R '-0.7138  0.1691  -4.2219  <.0001  -1.0451  -0.3824  ***', Python '-0.7138  0.1690  -4.2225  <.0001  -1.0451  -0.3825  ***' (2 cases; `rma.glmm`)
- line 3: R 'tau^2 (estimated amount of residual heterogeneity):     0.2733', Python 'tau^2 (estimated amount of residual heterogeneity):      (2 cases; `rma.glmm`)
- line 5: R 'I^2 (residual heterogeneity / unaccounted variability): 61.79%', Python 'I^2 (residual heterogeneity / unaccounted variability):  (1 cases; `rma.glmm`)
- line 8: R 'Wld(df = 12) = 268.5990, p-val < .0001', Python 'Wld(df = 12) = 191.7124, p-val < .0001' (1 cases; `rma.glmm`)
- line 14: R 'intrcpt     -10.2984  18.9350  -0.5439  0.5865  -47.4102  26.8135', Python 'intrcpt     -10.3059  18.9351  -0.5443  0.5863  -47. (1 cases; `rma.glmm`)
- line 14: R 'intrcpt  -61.2823  29.0390  -2.1103  0.0348  -118.1976  -4.3670  *', Python 'intrcpt  -61.2807  29.0369  -2.1104  0.0348  -118.1 (1 cases; `rma.glmm`)
- Cannot invert the Hessian for the saturated model. (4 cases; `rma.glmm`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. `rma_glmm(model='UM.RS')` no longer differs from R in se, so its value-diff ledger entry was removed. The warning `Cannot invert the Hessian for the saturated model.` is now a `warning-extra` ledger entry for the `control` optimizer cases (R inverts, metafopy warns); the `warning-missing` entry for the other cases stays, and the refused-where-R-works entry for `glmm_cm` was removed.

## Small numeric differences from R in rma, hettest and emmeans

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). A few single cases differ from R in the fourth digit or in one coefficient, outside the families above. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced; each may be separate. The cases name `rma`, `hettest` and `contrast_adjust_mvt`.

Fix direction: check each case against R; ledger the ones that are rounding below the threshold with a reason, and fix the rest.

Causes, from the run's `wrong-or-missing-full.md`:

- `$ci.ub` element 1: R -0.00033139675441687, Python -0.00033145363275964915 (relative 0.00017) (2 cases; `rma`)
- `$H` element 3: R 0.978416389473902, Python 0.9791954341103023 (relative 0.0008) (1 cases; `rma`)
- `$pval` element 0: R 0.928, Python 0.927 (relative 0.0011) (1 cases; `hettest`)
- `element #: R #, Python # (relative #) [expect_equivalent(c(residuals(res)), c(dat$yi - coe` element 5: R 0.0904795669122727, Python 0.09049455961258412 (relative 0.00017) [expect_equivalent(c(residuals(res)), c(dat$yi - coef(res)))] (1 cases; `the examples and tests of metafor`)
- line 20: R '---', Python 'intrcpt' (7 cases; `rma`)
- line 5: R 'X^2(df = 12) = 5.6201, p = 0.9280', Python 'X^2(df = 12) = 5.6201, p = 0.9270' (1 cases; `hettest`)
- line 15: R 'allocrandom      -22.3482  55186.2609  -0.0004  0.9997  -108185.4320', Python 'allocrandom      -27.1962  626049.2249   0.0000   (1 cases; `rma`)
- line 2: R 'alternate - random        0.448 0.516 Inf   0.868  0.6580', Python 'alternate - random        0.448 0.516 Inf   0.868  0.6578' (1 cases; `contrast_adjust_mvt`)
- line 13: R 'intrcpt                   -0.5506  0.2503  -2.2003  0.0278  -1.0411  -0.0601  *', Python 'intrcpt                   -0.5506  0.2 (1 cases; `rma`)
- line 11: R 'intrcpt   -1.1536  2.3709  -0.4866  0.6266  -5.8005  3.4933', Python 'intrcpt   -1.1536  2.3700  -0.4867  0.6264  -5.7987  3.491 (1 cases; `rma`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished and none appeared.

## Printed output differs from R in layout, row names and the printed call

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). Some objects print differently from R, with no difference in their values: the call printed by `update`, row names from 0 where R starts at 1, the repr of a result where R prints its elements, and the column names of `predict`. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The findings point at the print methods of each result class (`src/metafopy/models/call.py` for the printed call).

Fix direction: fix them one class at a time, starting with the printed call of `update`, which affects every fitting function.

Causes, from the run's `wrong-or-missing-full.md`:

- line 2: R '1        11      1     1 1976 -0.180 0.118', Python '0         11       1      1  1976 -0.18  0.118' (2 cases; `identity`, `print`)
- line 1: R '$slab', Python "RegplotResult(ax=<Axes: xlabel='ablat', ylabel='Log Risk Ratio'>, xi=array([44.," (2 cases; `print`, `regplot`)
- line 1: R 'trial               author year tpos  tneg cpos  cneg ablat      alloc', Python 'trial                author  year  tpos  ...     (4 cases; `$<-`)
- R prints the elements under $names, Python prints the repr of a dict (1 cases; `rcalc`)
- line 2: R '1        11      1     1 1976 -0.180 0.118  1', Python '0         11       1      1  1976 -0.18  0.118   1' (1 cases; `$<-`)
- line 15: R 'X.factor.alloc.random X.factor.alloc.systematic', Python 'X.factor(alloc)random X.factor(alloc)systematic' (1 cases; `predict`)
- line 315: R '[34,] .           .           .            1e-03  6e-04  6e-04  6e-04  6e-04', Python '[34,] .           .           .          (1 cases; `vcalc`)
- line 1: R '(function (yi, vi, sei, weights, ai, bi, ci, di, n1i, n2i, x1i,', Python "rma_uni(yi='yi', vi='vi', data=<EscalcFrame>)" (12 cases; `update`)
- line 1: R '(function (yi, V, W, mods, data, slab, subset, random, struct = "CS",', Python "rma_mv(yi='yi', V='vi', data=<DataFrame>, random= (2 cases; `update`)
- line 1: R 'print failed: argument must be coercible to non-negative integer', Python 'estimate     se    zval   pval   ci.lb   ci.ub         (6 cases; `leave1out`)
- line 1: R 'selmodel.rma.uni(x = list(b = -0.714532348365101, beta = -0.714532348365101,', Python "selmodel(x=<RMAResult REML k=13 p=1>, type (1 cases; `update`)
- line 1: R '(function (ai, bi, ci, di, n1i, n2i, data, slab, subset, add = 1/2,', Python "rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg' (1 cases; `update`)
- line 1: R '(function (ai, bi, ci, di, n1i, n2i, x1i, x2i, t1i, t2i, measure = "OR",', Python "rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cn (1 cases; `update`)
- line 1: R '(function (ai, bi, ci, di, n1i, n2i, x1i, x2i, t1i, t2i, xi,', Python "rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', measu (1 cases; `update`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished and none appeared.
