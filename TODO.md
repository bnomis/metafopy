# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## No correctness case covers user weights with a scale model

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
Found on 2026-11-20 while writing `agents/specs/correctness-ls-extensions/spec.md`: `rma_uni(..., weights=..., scale=...)` runs in metafopy, and no
case in `src/tests/correctness` passes user `weights` with `scale`, so only R parity checks it. Decision D2 of
`agents/reviews/correctness-ls-extensions-spec.md` (answered 2026-11-20) files it here rather than in that spec. Whether a publication defines
user weights for location-scale models is not yet checked.
<!-- doc-counts: accept-end -->

Fix direction: a spec with `/specdev`, after the location-scale extensions list: find the defining source (or, under the rule of 2026-11-13, metafor's
documentation, or an exemption if only its code defines it) and add the coverage items in `src/tests/correctness/test_coverage.py`.

Spec: `agents/specs/correctness-ls-weights/spec.md`; tasks: `agents/tasks/correctness-ls-weights/tasks-1.md`.
