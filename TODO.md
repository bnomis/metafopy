# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

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

## No correctness case covers user weights with a scale model

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 while writing `agents/specs/correctness-ls-extensions/spec.md`: `rma_uni(..., weights=..., scale=...)` runs in metafopy, and no
case in `src/tests/correctness` passes user `weights` with `scale`, so only R parity checks it. Decision D2 of
`agents/reviews/correctness-ls-extensions-spec.md` (answered 2026-11-20) files it here rather than in that spec. Whether a publication defines
user weights for location-scale models is not yet checked.
<!-- doc-counts: accept-end -->

Fix direction: a spec with `/specdev`, after the location-scale extensions list: find the defining source (or, under the rule of 2026-11-13, metafor's
documentation, or an exemption if only its code defines it) and add the coverage items in `src/tests/correctness/test_coverage.py`.
