# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


## vif() of location-scale fits differs from R with sim, reestimate and table

<!-- doc-counts: accept-start found on 2026-11-05 - a dated record, not a claim about the present -->
Found on 2026-11-04 by the slice `.audit/runs/20261104-092750` and again on 2026-11-05 by `.audit/runs/20261105-162814` (local, `--only scale= --only ls_`, options step, metafopy `e23afc5`'s parent), where every case below is listed as new. The location-scale item in `DONE.md` (2026-11-04) carried some of them in its body, but its Done line records only the fix of the `vif` print digits, and no ledger entry matches them.

- `options/vif/ls_bangert/sim=true`: `$beta$prop` element 1 R 0.282, Python 0.296; `$alpha$prop` element 1 R 0.698396793587174, Python 0.6717325227963525; `$beta$sim` element 1285 R 1.83095037872554, Python 1.0234467077414686; `$alpha$sim` element 1990 R 1.00000000064568, Python None; print line 3 R 'I(ni/100)  1.0094  0.28', Python 'I(ni/100)  1.0094  0.30'
- `options/vif/ls_year/sim=true`: `$beta$prop` element 1 R 0.998998998998999, Python 0.998; `$beta$sim` element 1152 R None, Python 1.0193929566782072
- `options/vif/ls_bcg/sim=true`: print line 6 R 'allocrandom      2.2723  0.50', Python 'allocrandom      2.2723  0.97'
- `options/vif/ls_year/reestimate=true`: print line 6 R '0.3083  0.8897', Python '0.3077  0.8898'
- `options/vif/ls_na/table=true`: print line 7 R 'intrcpt -10.4383 5.5982 -1.8646  <.1 -21.4105 0.5339     NA     NA', Python 'intrcpt -10.4383 5.5980 ...'
<!-- doc-counts: accept-end -->

Cause: not traced. Suspected (unconfirmed): the `sim` proportions and the `None` entries (a refit R kept and metafopy dropped, and the reverse) suggest the simulated refits fail or are dropped differently, as R's `nlminb` fails on the flat scale likelihoods accepted for location-scale fits on 2026-11-04; the `reestimate` and `table` rows differ in the fourth digit, as the accepted location-scale estimates do. Code: `src/metafopy/methods/ls_vif.py`. A documented difference is a lead for the `sim` cases: the docstring of `vif_rma_ls` (`src/metafopy/methods/ls_vif.py:257`) says that, unlike metafor, a model given by formulas has the columns of its model matrix permuted, not the underlying data variables, so the simulated distributions (and `prop`) of `ls_bcg`'s `alloc` dummies cannot match R's.

Fix direction: first decide whether `sim` should permute the data variables as metafor does (a change to a documented behaviour); then trace `options/vif/ls_bangert/sim=true` refit by refit against R with a fixed seed: count the refits each side drops and compare the draws they start from. Ledger what is a failure or stop point of R's optimiser under the operator's decision of 2026-11-04 on location-scale fits; fix what is a difference of metafopy's own (a draw, a dropped refit, a proportion's denominator: R's `$beta$prop` 0.998998998998999 is 998/999, metafopy's 0.998 is 998/1000).

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

Note 2026-11-05: the one-row scale table now prints without a row name for any one-row scale model (`src/metafopy/models/rma_ls_print.py`, `test_rma_ls.py::test_print_one_row_scale_table_without_a_name`), as R's `print.rma.uni` does; the slice `.audit/runs/20261105-162814` (local, `--only scale= --only ls_`, steps construct, matrix and options, re-rendered) lists none of those row names. Its fourth-digit print of `uni_mods_year` with `scale = ~ 0 + I(1/ablat)`, which the row name hid, is the accepted value-diff of the `construct/uni_mods*/scale=*` entry and has a print entry beside it. The printed call of `update` and the `predict` row name of `ls_na` remain.
