# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


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

Note 2026-11-04: re-checked on the slice `.audit/runs/20261104-105354` (local, metafopy `a7b8700`); every cause above is listed again except the Hessian warning, which the ledger matches. On `construct/glmm_bcg/mods=~ alloc` (UM.FS) R's log-likelihood is -70.4602835 and metafopy's -70.4652470: metafopy's fit stops short of R's maximum (tau2 0.27290 against 0.27334, QM 1.8844 against 1.8981), so these differences are a defect of metafopy's fit, not a stop point of R's optimiser. Start there.

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

Note 2026-11-04: the `profile` causes the gosh item carried belong here. The slice `--only vif --only dfbetas --only profile` (`.audit/runs/20261104-071144`, local) lists `[]$ci.lb$intrcpt` and `[]$ci.ub$intrcpt` element 3 with R None, Python a value, on `matrix/profile/sel_bcg`, `matrix/profile/sel_digits` and the `options/profile/sel_bcg/*` cases, and `[]$ci.lb$intrcpt` element 2 (R -0.402157171730945, Python -0.40209684929512357) and `[]$ci.ub$ablat` on `matrix/profile/sel_mods_bcg`.

Note 2026-11-04: re-checked on the slice `.audit/runs/20261104-105354` (local, metafopy `a7b8700`); every cause above is listed again. `selmodel(type='beta')` of `trimfill_bcg` has R's log-likelihood (-99.049066667) and `delta`, and R's `optimizer='BFGS'` gives R's se too; only the se (0.06952 against 0.06949) differs, so it is the numerical Hessian. In `profile()` of `sel_bcg` R's refits converge but R cannot invert their Hessian ('Error when trying to invert the Hessian'), so its `ci.lb`/`ci.ub` are NA where metafopy's Hessian inverts. `trimfill`'s `se.k0` is not yet traced.

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

Note 2026-11-04: the `profile` causes on `matrix/profile/mv_wide` that the gosh item carried belong here: the slice `.audit/runs/20261104-071144` (local) lists `[]$ll`, `[]$beta$intrcpt`, `[]$beta$log(year - 1900 + 1000)` and `[]$ci.lb`/`[]$ci.ub` of both coefficients, element 2, R None, Python a value - the same pattern as `rstudent` and `confint` here.

Note 2026-11-04: re-checked on the slice `.audit/runs/20261104-105354` (local, metafopy `a7b8700`); every cause above is listed again. In `profile()` of `mv_wide` R's refit fails at one grid point (`ll` NA) and metafopy's converges: the pattern accepted for location-scale fits on 2026-11-04, a decision that named those fits only. The Hessian of `mv_sp` (R 9 entries, metafopy 4) is a different set of parameters, not an optimiser stop point.

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

Note 2026-11-04: `line 20: R '---', Python 'intrcpt'` (7 cases) and the `intrcpt -0.5506` row (1 case) were the wrapping of a wide coefficient table: `metafopy.printing.fmt_table` counted a blank column of significance stars two characters wide where R counts one. Fixed with the location-scale print (`test_print_wraps_a_wide_table_as_r`); the `construct` step lists neither since (`.audit/runs/20261104-100918`, local).

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

Note 2026-11-04: from the location-scale item (now in `DONE.md`): `predict()` of `ls_na` with a named `newscale` prints the row name `1` where R prints `year` (`options/predict/ls_na/newscale={"$named":{"ablat":30,"year":1960}}`, `.audit/runs/20261104-092750`, local).
