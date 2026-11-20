# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## A property test still excludes identical effects for a fixed bug

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 by a TODO review: `meta_data` in `src/tests/correctness/test_properties.py:79-80` skips every draw whose effects are identical
(`hypothesis.assume(numpy.ptp(effects) > 0.01)`), with the comment that identical effects make Q zero, "which the library does not handle for an
equal-effects fit (see TODO.md)". No open item covers that: "rma_uni raises ZeroDivisionError for an equal-effects fit when every effect is
identical" has been in `DONE.md` since 2026-11-14 (fixed in `src/metafopy/models/rma_uni.py`, case `property.i2-zero-q.identical`). Not yet
checked whether every property that uses `meta_data` holds for identical effects.
<!-- doc-counts: accept-end -->

Fix direction: remove the `assume` and its comment, run `pytest src/tests/correctness/test_properties.py`; if a property fails for identical
effects, keep a narrower exclusion with a comment naming why, or file the failure as a defect.

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
