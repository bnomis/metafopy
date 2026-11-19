# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Judge the published-value mismatches recorded "not yet reviewed"

<!-- doc-counts: accept-start found on 2026-11-19 - a dated record, not a claim about the present -->
Found on 2026-11-19 by a TODO review: strict expected-failure cases where metafopy and R metafor 5.2-1 agree with each other but not with the printed
value are recorded "not yet reviewed", and no review or item covers them. In `agents/specs/correctness-publication-bias/mismatches.md` (written
2026-11-19 by `agents/tasks/citkowicz2017-cases/tasks-1.md`): `published.citkowicz2017.baskerville2012.article.unadjusted.se`, `.article.beta.b`,
`.article.blind.unadjusted` and `.article.blind.beta.b_lrt`. In `agents/specs/correctness-glmm/mismatches.md` (recorded 2026-11-18):
`glmm.published.bagos2009.table5.school.rs.mismatch`. The last review of such cases was `agents/reviews/correctness-harness-checks-mismatches.md`.
<!-- doc-counts: accept-end -->

Cause: not a known defect; both implementations agree, so each difference is suspected to lie in the source's own fit or rounding.

Fix direction: a review page with one item per case (accept as a source difference, or investigate), its answers copied into the review document and
into each `mismatches.md` entry with the date; an "investigate" answer becomes its own item.

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
