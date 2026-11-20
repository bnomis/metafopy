# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## No correctness case covers the rma_uni diagnostics

<!-- doc-counts: accept-start found on 2026-11-19 - a dated record, not a claim about the present -->
Found on 2026-11-19 while writing `agents/specs/correctness-mv-diagnostics/spec.md`: no case in `src/tests/correctness` names the `rma_uni`
diagnostics `cooks_distance`, `dfbetas`, `rstudent` or `ranef`, so only the R parity fixtures check them. Viechtbauer and Cheung (2010, Res Synth
Methods 1:112-125, doi:10.1002/jrsm.11) define them directly. Decision D1 of `agents/reviews/correctness-mv-diagnostics-spec.md` (answered
2026-11-19) keeps them out of the rma_mv spec and files them here.
<!-- doc-counts: accept-end -->

Fix direction: a spec with `/specdev`, modelled on `agents/specs/correctness-mv-diagnostics/spec.md` (brute-force leave-one-out references in
`formulas.py`, coverage items in `src/tests/correctness/test_coverage.py`), its definitions checked against Viechtbauer and Cheung (2010).

Spec: `agents/specs/correctness-uni-diagnostics/spec.md`; tasks: `agents/tasks/correctness-uni-diagnostics/tasks-1.md`.

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

Spec: `agents/specs/correctness-mv-diagnostics/spec.md` (the rma_mv part); tasks: `agents/tasks/correctness-mv-diagnostics/tasks-1.md`.

Note 2026-11-20: the rma_mv diagnostics part is done (`agents/specs/correctness-mv-diagnostics/spec.md`,
`agents/tasks/correctness-mv-diagnostics/tasks-1.md`); the location-scale part remains.
