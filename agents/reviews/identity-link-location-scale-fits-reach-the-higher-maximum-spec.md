# Review: the identity-link-location-scale-fits-reach-the-higher-maximum spec

**Review page:** https://claude.ai/artifact/BKmKMVHKKBrZjvFvSJMWhS
**Date:** 2026-11-26

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/identity-link-location-scale-fits-reach-the-higher-maximum/spec.md`,
for the `TODO.md` item "Identity-link location-scale fits can stop below a higher maximum on the face `tau2 = 0`". It extends the operator's choice of
2026-11-25 (metafopy reports the higher maximum, departing from R metafor 5.2-1) to the identity link: after the identity polish, one start on each
face `z_i' alpha = 0` and one reflection per scale slope, each polished by `polish_identity` alone, kept only at a higher regular maximum on its face.
Approve or flag the spec and each of the choices it made; there are no open decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
- Pass: the item is still open at `f0f6d44`: `fit_rma_ls` runs the restarts on the log link only.
- Pass: grid truth on all 1287 bcglat permutations: 85 identity fits are below the true maximum, 81 of them with it on a face `tau2_g = 0`.
- Pass: the prototype brings all 85 to the true maximum and changes no other permutation; none ends above the grid maximum.
- Pass: the prototype changes no fit in the 701 location-scale tests and correctness cases, and every one passes.
- Pass: R metafor 5.2-1 gives metafopy's default values on permutations 18, 68, 679 and 914, and reaches the higher maximum from `alpha.init` on
  18, 68 and 914.
- Pass: the spec has 174 lines, no non-ASCII punctuation, and passes the pre-commit hooks.
<!-- doc-counts: accept-end -->

## The items

### The spec: identity-link location-scale fits reach the higher maximum

- **File:** agents/specs/identity-link-location-scale-fits-reach-the-higher-maximum/spec.md, version 1.0, proposed 2026-11-26.
- **What it does:** face starts (the nearest `alpha` in the metric `Z'Z` with `r' alpha = 0`, per distinct row `r`, near-zero entries exactly 0,
  kept when feasible) and a midrange reflection per non-constant column; each polished by `polish_identity`; the best replaces the fit when its
  log-likelihood is higher by more than 1e-6 and the information on its face has eigenvalue ratio above 1e-5. Same conditions as the log-link
  restart. Tests in `test_rma_ls_restarts.py`, a correctness case on permutation 18, `docs/differences.md`, a scoped audit before and after.
- **Output changes (prototype, 2026-11-26):** no fit in the test suite; the 85 bcglat identity permutations move up (permutation 18: R's z
  `0.2490` to `-3.5966`). Cost: one fit 39.4 to 97.2 ms; the exact permutest about 51 to about 125 s.
- **Recommendation: approve.**

### Choice: the identity link follows the choice of 2026-11-25 and shares `control['restarts']`

One setting returns both links to R's single start. **Recommendation: approve.**

### Choice: polish only, no optimiser refit

The barrier optimiser kept a face maximum from a start on the face on 4 of 85, and drifts off it on permutation 679 from 1e-3 inside (as R does).
**Recommendation: approve.**

### Choice: the reflection start is kept

It recovered 1 of the 85 (permutation 914, an interior maximum no face start reaches). **Recommendation: approve.**

### Choice: the test reference is left as it is

It reaches the higher maximum on permutation 18 (73 of the 85), and no case runs the identity permutest. **Recommendation: approve.**

### Choice: the same conditions as the log-link restart

Fixed-coefficient refits stay out: a face of dimension 1 leaves the guard nothing to check. **Recommendation: approve.**

## Decisions

None open.

## Answers

