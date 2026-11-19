# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Find the Hooper data behind Table 5 of Bagos and Nikolopoulos (2009)

<!-- doc-counts: accept-start probed on 2026-11-18 - a dated record, not a claim about the present -->
Narrowed on 2026-11-18 from "Find the data behind Table 5 of Bagos and Nikolopoulos (2009)" (`agents/tasks/bagos2009-table5/tasks-1.md`; decision
D2 (a) of `agents/reviews/order-todo-items-1-5-specs.md`, answered 2026-11-18): the ER-visit and school-absence columns now have cases; the
Hooper column (dietary fat and combined cardiovascular events) does not. Probed on 2026-11-18: the ten trials of Table 1 of Hooper et al. (2001,
BMJ 322:757) match Bagos and Nikolopoulos's person-years but carry 567 intervention and 638 control events against their 553 and 653, and a
fixed-effects Poisson fit gives an IRR of 0.858 against the printed 0.819; no swap of arms or rounding of the London corn/olive control events
reproduces it.
<!-- doc-counts: accept-end -->

Fix direction: find the full Cochrane review, Hooper L et al. (2001), Dietary fat intake and prevention of cardiovascular disease, Cochrane
Database Syst Rev CD002137, DOI 10.1002/14651858.CD002137 (the 2001 issue), and its per-trial events and person-years; repository copies at UEA
ePrints and LSHTM Research Online were not checked. If they reproduce Table 5's Hooper column, write its cases in
`src/tests/correctness/cases/glmm_published.toml` beside the `glmm.published.bagos2009.table5.*` cases.

<!-- doc-counts: accept-start checked on 2026-11-19 - a dated record, not a claim about the present -->
Note 2026-11-19: the 2012 update, "Reduced or modified dietary fat for preventing cardiovascular disease" (Cochrane Database Syst Rev 2012,
Issue 5, DOI 10.1002/14651858.CD002137.pub3, in the papers folder), cannot reproduce the column: its history says the 2011 update re-ran every
analysis "as relative risks (rather than as rate ratios)", Analysis 1.3 (combined cardiovascular events) gives events and participants with no
person-years, and it pools 23 comparisons (2020 and 2867 events, WHI 2006 included) where the first version had 27 studies and 30901 person-years.
Only the 2001 issue (DOI 10.1002/14651858.CD002137, no `.pub` suffix; the Cochrane Library's "Other versions" of the review) will do.
<!-- doc-counts: accept-end -->

<!-- doc-counts: accept-start checked on 2026-11-19 - a dated record, not a claim about the present -->
Note 2026-11-19 (later): the premise above is corrected, not rewritten. Bagos and Nikolopoulos's reference "Hooper et al. (2001)" is the BMJ
article (BMJ 322(7289):757-763), not the Cochrane review, and the first Cochrane version (2000, Issue 2, DOI 10.1002/14651858.CD002137; reprints of
2009 and 2010 in the papers folder) carries the same per-trial data with person-years to one more decimal (all-trials totals 643 control and 573
intervention combined events). Their printed totals are intervention 6636 people, 10649 person-years, 553 events, control 6631, 10426, 653. A search
over every trial of that Table 1, each left out, as printed or with its arms swapped, matches the participants but misses the events by 10 or more
and the person-years by about 9; 653 control events exceed every total the data allow. The 2011 (Issue 7) and 2012 updates have relative risks
only. The authors' `metapoisson.do` (Google Drive folder linked from https://sites.google.com/compgen.org/lab/tools/poisson-meta-analysis) holds
only the Wolf et al. ER-visit data, and the article's Stata appendix the Collins et al. ibuprofen data. Leads left: the BMJ article's web-extra
trial table (doi:10.1136/bmj.322.7289.757), and asking the authors (Pantelis Bagos, pbagos@compgen.org) for the data behind the Hooper column.
<!-- doc-counts: accept-end -->

## Correctness coverage the extension specs left for later

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. `agents/specs/correctness-location-scale/spec.md` (out of scope) leaves for a later spec `profile` beyond the
property, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha` and `beta`, and `alpha_min` and `alpha_max`;
`agents/specs/correctness-location-scale/mismatches.md` adds REML with `optbeta=True`, the deviance and `vb` of `optbeta` fits.
`agents/specs/correctness-mv-robust/spec.md` says a later spec can add the `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef`.
No spec, task list or item covers them; they are checked against R metafor only.
<!-- doc-counts: accept-end -->

Fix direction: deferred. When taken up, one spec per area with `/specdev`, listing the defining references and the coverage items in
`src/tests/correctness/test_coverage.py`.
