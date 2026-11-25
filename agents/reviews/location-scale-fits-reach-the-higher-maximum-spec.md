# Review: the location-scale-fits-reach-the-higher-maximum spec

**Review page:** https://claude.ai/artifact/MsBFkxwxF87RoVgnkLtSKv
**Date:** 2026-11-25

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md`, for the
`TODO.md` item "Location-scale fits can stop at a lower local maximum than the test reference". On 2026-11-25 the operator chose that metafopy
reports the higher of two interior maxima, departing from R metafor 5.2-1, which stops at whichever maximum its single start leads to. The spec
adds one derived restart per scale slope, kept only when it reaches a higher regular maximum. Approve or flag the spec and each of the choices it
made; there are no open decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
- Pass: the item is still open at `9bb0947`: `ls.permutest.location.exact.ML.z.bcglat` is a strict-xfail mismatch naming it
  (`src/tests/correctness/cases/location_scale.toml:895`).
- Pass: R metafor 5.2-1 gives metafopy's values on permutations 433 and 1278 from its default start, and the higher maximum from `alpha.init`.
- Pass: the prototype changes exactly the 8 permutations of the item, puts both p-values of the case inside the reference bounds, and changes no
  main fit among the 10697 location-scale fits the test suite makes.
- Pass: the spec has 177 lines and no non-ASCII punctuation.
<!-- doc-counts: accept-end -->

## The items

### The spec: location-scale fits reach the higher maximum

- **File:** agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md, version 1.0, proposed 2026-11-25.
- **What it does:** after the usual log-link fit, one restart per non-constant column of Z (the slope moved towards and past zero by 4 / sd of its
  column); the restart's end point replaces the fit when its log-likelihood is higher by more than 1e-6 and its information on the standardised Z
  has eigenvalue ratio above 1e-5. Runs only with every scale coefficient free, no alpha bounds, no optbeta and no user weights. New
  `control={'restarts': False}` gives R's single start and is carried into permutest and other refits. The test reference gets the same restarts;
  the bcglat case loses its mismatch status; a new correctness case on permutation 433; R-parity tests pass `restarts=False`;
  `docs/differences.md`; a scoped audit before and after.
- **Output changes (prototype, 2026-11-25):** no main fit; 43 permuted refits with all coefficients free; p-values: bcglat lat 0.23932 to 0.23776,
  `test_permutest[less]` 0.98 to 0.96, `[px2]` 0.12 to 0.16, `[btt]` pval_alpha 0.65 to 0.675, `[t]` pval_alpha 0.52 to 0.54. Cost: one fit 4.7 to
  13.0 ms, the exact permutest 7.5 to 18.6 s (prototype; the real information is cheaper).
- **Recommendation: approve.**

### Choice: default on, `restarts=False` as the way back to R

Follows from the operator's choice that metafopy reports the higher maximum. Reversible by changing the default. **Recommendation: approve.**

### Choice: R-parity tests pass `restarts=False`

The parity tests check the port of metafor's procedure, so they keep its single start; the new correctness case on permutation 433 checks the
departure. The other way would record each changed permutation of each parity test as a departure. **Recommendation: approve.**

### Choice: the identity link is left out and filed as its own TODO item

On the bcglat data 83 of 1287 identity-link permutations have a higher maximum, most on the face `tau2 = 0` of one group (a valid boundary
estimate); no start kind tried recovered more than 12 of them (2026-11-25). It needs its own start design and a guard on the active face.
**Recommendation: approve.**

### Choice: fits with a fixed coefficient (profile, confint) are left out

With one free coefficient the information has a single eigenvalue, so the guard checks nothing; the prototype changed 100 such refits that it could
not tell from ridge points. **Recommendation: approve.**

## Decisions

None open.

## Answers

Answered on 2026-11-25 on the review page (every answer the recommended option, no comments):

- The spec: approved.
- Default on, `restarts=False` as the way back to R: approved.
- R-parity tests pass `restarts=False`: approved.
- The identity link left out and filed as its own TODO item: approved.
- Fits with a fixed coefficient left out: approved.
