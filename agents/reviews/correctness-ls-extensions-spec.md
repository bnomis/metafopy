# Review: the correctness-ls-extensions spec

**Review page:** https://claude.ai/artifact/ABF4DGdhLvJFY1Y86xAf9R
**Date:** 2026-11-20

**What is being reviewed and how.** One new spec, `agents/specs/correctness-ls-extensions/spec.md` (with `measurements.md`), written by `/todo-do`
for the location-scale part of the `TODO.md` item "Correctness coverage the extension specs left for later", its last open part. It adds
correctness cases for location-scale `profile`, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha` and `beta`, the
`alpha_min`/`alpha_max` box, REML `optbeta` and the `vb` of `optbeta` fits, which only R parity fixtures check today. No metafopy code changes.
Approve or flag the spec, answer the decisions, and supply the one missing paper before the run.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
- Pass: every gap is still open at `936021a` (no case calls these functions or options on a location-scale fit).
- Pass: measured against independent references, metafopy agrees to 4.6e-6 or better (most to 1e-8 or better), except the two expected mismatches
  below (measurements.md, R 4.6.2 with metafor 5.2-1).
- Missing: Fox and Monette (1992), JASA 87:178-183, doi:10.1080/01621459.1992.10475190, the GVIF definition the `vif` cases cite, is not in the
  papers folder `/Users/simonb/src/metafopy papers`.
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **References:** derived cases from Viechtbauer and Lopez-Lopez (2022) (joint likelihood, fixed coefficients, profiles, marginal residuals),
  Henderson's mixed model equations for `blup` and `ranef`, Fox and Monette for `vif`; the documented options (`weighted=False`, the box,
  conditional `rstandard`, `vif(reestimate=True)`) coded from metafor's help pages under the rule of 2026-11-13. Exempt as code-only: the deviance,
  the fit statistics of `weighted=False` fits, fixed `beta` under REML, the `vb` of REML `optbeta` fits. The paper prints no new values, so no
  `published` case.
- **A premise corrected:** the location-scale mismatch record of 2026-11-15 deferred REML `optbeta` as "a different objective"; measured on
  2026-11-20 its maximiser is the ordinary REML fit's (3.6e-8), so it becomes a derived case.
- **Expected mismatches** (strict expected failures for the review step):
  - `ls.alpha.bounds.ML.bcg.wide`: with a box that does not bind, metafopy returns alpha (0, -0.0796), log-likelihood -8.676, with only a "near the
    bound" warning; the maximum inside the box is (-4.957, 0.0386), -7.539. The start clipped onto the bound is the cause; R stops with a
    convergence error instead.
  - `ls.profile.ML.identity.bcg`: identity-link profile values up to 1.06e-8 below the reference (8.3e-9 allowed); R is further off. The answer of
    2026-11-15 (follow the reference, implemented as the identity-link polish) stands; the review step asks only about the remainder.
- **Stops in the task list:** the sources step, only if Fox and Monette is still missing; the mismatch review.
- **Recommendation:** approve.

## Decisions

### D1. Which covariance is the reference for `vb` of an `optbeta` fit

- **What is decided:** the reference the `ls.optbeta.vb.ML` cases compare metafopy's `vb` against.
- **Background:** metafopy, like metafor, inverts the numerical Hessian of the joint likelihood (observed information). The paper's eq 10 gives
  `(X'WX)^-1`, the beta block of the expected information (Harvey 1976). On 2026-11-20 the two differed by 2.0e-3 to 4.4e-3 relative; metafopy
  matched the analytic observed information to 7.7e-8.
- **Affects:** two cases; under B, metafopy's `optbeta` standard errors, tests and intervals, and `docs/differences.md`.
- **Options:** *A, observed information (Recommended)*: the cases pass, no output change; *B, equation 10*: the cases are mismatches and, if
  followed, `optbeta` fits report the ordinary fit's `vb`; *C, exempt*: R parity alone.
- **Recommendation:** A: both estimate the same asymptotic covariance, the paper uses the observed-information rule for the scale part, and B departs
  from metafor.

### D2. User weights with a scale model

- **What is decided:** whether to file the gap the draft found outside this item: `rma_uni(..., weights=..., scale=...)` runs in metafopy and no
  correctness case covers it.
- **Background:** found 2026-11-20 while writing this spec; it is not in the `TODO.md` item, so the spec leaves it out.
- **Options:** *File a TODO.md item (Recommended)*: a new item after this one, for a later spec; *Leave it*: it stays checked by R parity only and
  unrecorded.
- **Recommendation:** file it, so the gap is recorded rather than lost.

## Prerequisite

Put Fox J, Monette G (1992), Generalized collinearity diagnostics, JASA 87:178-183, doi:10.1080/01621459.1992.10475190, in
`/Users/simonb/src/metafopy papers/`. The `vif` cases cite its definition of the generalised variance inflation factor; no paper already in the
folder defines it (Viechtbauer and Lopez-Lopez do not), so the sources step stops without it.

## Answers

Answered 2026-11-20 on the review page (https://claude.ai/artifact/ABF4DGdhLvJFY1Y86xAf9R), each with no comment:

- **1. The spec:** approved.
- **D1. Which covariance is the reference for `vb` of an `optbeta` fit:** A, observed information.
- **D2. User weights with a scale model:** file a `TODO.md` item.

Applied in version 1.1 of the spec (2026-11-20); the `TODO.md` item is filed. Fox and Monette (1992) was not yet in the papers folder on 2026-11-20.
