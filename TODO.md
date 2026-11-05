# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


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

Note 2026-11-04: the `rma` causes still listed by the whole `construct` run `.audit/runs/20261104-100918` (re-rendered with today's ledger) are location-scale fits: `uni_reml` with `scale = ~ ablat` (`$H`, the scale row `-1.1536 2.3709`; log-likelihood -12.2023666 in both), `uni_mods_na` with `scale = ~ alloc` (the degenerate `alloc` scale of `ls_bcg`) and `uni_mods_year` with `scale = ~ 0 + I(1|ablat)` (`$ci.ub`; see the item on `I()` reading `|` as a bitwise OR). The first two fall under the operator's decision of 2026-11-04 on location-scale fits. `hettest`, the emmeans contrast and the script residual were not re-run.

Note 2026-11-04 (correcting the note above): the case `uni_mods_year` has `scale = ~ 0 + I(1/ablat)`; case ids show `/` as `|` (`label()`, `src/audit/domains.py:116`), so it has nothing to do with the item on `I()` reading `|` as a bitwise OR (now in `DONE.md`). The `rma` causes of location-scale fits are worked in the item "Accepted location-scale differences that no ledger entry matches"; `hettest`, the emmeans contrast and the script residual stay here.

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

Note 2026-11-04: a print defect found by the construct run `.audit/runs/20261104-193457` (local, metafopy `548759d`): a scale model with one coefficient other than the intercept (`scale = ~ 0 + ablat`, `~ 0 + I(1/ablat)`, on `uni_reml`, `uni_mods`, `uni_na`, `uni_mods_na`, `uni_mods_year`, `uni_factor`, `uni_counts`, `ls_bcg`, `ls_year` and `ls_na`) prints the row without a name in R (R '-0.0338  0.0167  -2.0262  0.0427  -0.0664  -0.0011  *', Python 'ablat   -0.0338  0.0167  ...'). R's `print.rma.uni` prints any one-row scale table as a vector (`if (nrow(res.table) == 1L)`); `src/metafopy/models/rma_ls_print.py` drops the name only for an intercept-only scale model (`res.Z_int_only`). Start here: one condition, then the printed call of `update`.
