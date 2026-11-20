# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## A location-scale fit with a start on the alpha_max bound returns a non-maximum

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 by the case `ls.alpha.bounds.ML.bcg.wide` in `src/tests/correctness/cases/location_scale.toml` (review answered 2026-11-20, D1 of
`agents/specs/correctness-ls-extensions/mismatches.md`, *follow the reference*). `rma_uni` on `dat.bcg` (RR, `mods = ~ ablat`, `scale = ~ ablat`, ML,
log link) with `alpha_min = [-10, -1.2]` and `alpha_max = [0, 1.2]` returns alpha [0, -0.0795806] (log-likelihood -8.6762) with only the "(almost)
equal to their bound" warning; the maximum inside the box is alpha [-4.95749, 0.03859] (log-likelihood -7.5393). The start is clipped onto the upper
bound, where the logistic map used for the box sends it to infinity. R metafor 5.2-1 stops with "Optimizer (nlminb) did not achieve convergence".
<!-- doc-counts: accept-end -->

Fix direction: move the start strictly inside the box, flip the case to a passing one (remove `status = "mismatch"` and its `reason`), mark the R
parity fixtures if any exist for it, add a `src/audit/data/known.toml` entry for the case metafor cannot fit, and add a section to
`docs/differences.md` (metafor errors here, metafopy returns the maximum). Run `pytest src/tests/correctness -m 'not slow'` and the whole test suite.

## No correctness case covers user weights with a scale model

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 while writing `agents/specs/correctness-ls-extensions/spec.md`: `rma_uni(..., weights=..., scale=...)` runs in metafopy, and no
case in `src/tests/correctness` passes user `weights` with `scale`, so only R parity checks it. Decision D2 of
`agents/reviews/correctness-ls-extensions-spec.md` (answered 2026-11-20) files it here rather than in that spec. Whether a publication defines
user weights for location-scale models is not yet checked.
<!-- doc-counts: accept-end -->

Fix direction: a spec with `/specdev`, after the location-scale extensions list: find the defining source (or, under the rule of 2026-11-13, metafor's
documentation, or an exemption if only its code defines it) and add the coverage items in `src/tests/correctness/test_coverage.py`.
