# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## A location-scale fit accepts optbeta=True with user weights

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 at `0cc8edc` while writing `agents/specs/correctness-ls-weights/spec.md` (decision D2 of
`agents/reviews/correctness-ls-weights-spec.md`, answered 2026-11-20: file it here). `rma_uni(yi, vi, weights="tpos", mods="~ ablat",
scale="~ ablat", optbeta=True)` on `dat.bcg` (RR) returns a fit (`beta` (0.33244, -0.02878)), where R metafor 5.2-1 stops with "Cannot use custom
weights when 'optbeta=TRUE'." (`rma.uni`, the check after "Extracting custom weights"). Cause: `fit_rma_ls` in
`src/metafopy/models/rma_ls_fit.py:405` checks `optbeta` against `weighted` only; the custom-weights check exists in
`src/metafopy/models/rma_mv.py:698`, not in the location-scale fit.
<!-- doc-counts: accept-end -->

Fix direction: small, no spec. Raise `metafopy.errors.InputError("Cannot use custom weights when 'optbeta=True'.")` in `fit_rma_ls` when the user
gives `weights` with `optbeta=True`, with a regression test in `src/tests/test_rma_ls.py` (failing first); check whether an R parity fixture or a
`src/audit/data/known.toml` entry records the old behaviour.

## No correctness case covers user weights with a scale model

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 while writing `agents/specs/correctness-ls-extensions/spec.md`: `rma_uni(..., weights=..., scale=...)` runs in metafopy, and no
case in `src/tests/correctness` passes user `weights` with `scale`, so only R parity checks it. Decision D2 of
`agents/reviews/correctness-ls-extensions-spec.md` (answered 2026-11-20) files it here rather than in that spec. Whether a publication defines
user weights for location-scale models is not yet checked.
<!-- doc-counts: accept-end -->

Fix direction: a spec with `/specdev`, after the location-scale extensions list: find the defining source (or, under the rule of 2026-11-13, metafor's
documentation, or an exemption if only its code defines it) and add the coverage items in `src/tests/correctness/test_coverage.py`.
