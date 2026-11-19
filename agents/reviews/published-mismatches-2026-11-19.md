# Review: published-value mismatches recorded "not yet reviewed"

**Review page:** https://claude.ai/artifact/WuoGvtFtLsHAL8HqHDxXAR
**Date:** 2026-11-19
**Source:** `TODO.md`, "Judge the published-value mismatches recorded "not yet reviewed""

## What is reviewed and how

Five strict expected-failure cases compare metafopy with a value a publication prints, and miss it by more than half a unit of the last printed
digit. In each, metafopy and R metafor 5.2-1 agree with each other, so the difference lies between both implementations and the source. For each case,
choose **accept as a source difference** (the case stays a strict expected failure, its `reason` gains "accepted as a source difference on <date>", as
the Jackson et al. (2018) and Stijnen et al. (2010) cases were on 2026-11-14) or **investigate** (a new `TODO.md` item to find what the source did;
the case stays as it is until then).

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
- Pass: the cases whose `status = "mismatch"` reason lacks "accepted" are exactly these five (`src/tests/correctness/cases/*.toml`, 2026-11-19).
- Pass: all five are strict expected failures in the full run of 2026-11-19 at `2084573` (8618 passed, 37 skipped, 18 xfailed).
- Probe, 2026-11-19: no other estimator reproduces the Citkowicz and Vevea unadjusted values. On `dat.baskerville2012` (23 effects) the mean-only
  `se` is 0.0633 (ML), 0.0652 (REML), 0.0640 (DL), 0.0548 (equal effects) against the printed 0.069; for the blinding model REML gives an intercept
  `se` of 0.0991 (printed 0.098, ML 0.0925) but not the printed means.
<!-- doc-counts: accept-end -->

## The items

Recommendation rule: accept where metafopy and R metafor agree and no setting found reproduces the source, since the difference is then in the
source's own computation; investigate only where a lead exists.

1. **Citkowicz and Vevea (2017), mean-only unadjusted standard error** (`published.citkowicz2017.baskerville2012.article.unadjusted.se`, in
   `selmodel_published.toml`): printed 0.069; metafopy and R 0.0633. The article fitted with its own `nlminb` code; its estimates of the same model
   reproduce. Recommendation: accept.
2. **Citkowicz and Vevea (2017), mean-only beta weights b** (`...article.beta.b`): printed 4.459; metafopy and R 4.4613. a, the mean, its se and the
   likelihood ratio reproduce. Recommendation: accept (an optimiser stopping point on a flat likelihood in b).
3. **Citkowicz and Vevea (2017), blinding model without selection** (`...article.blind.unadjusted`): printed 0.574, 0.098, 0.538, 0.090, 0.017;
   metafopy and R 0.5731, 0.0925, 0.5397, 0.0868, 0.0163. REML matches one standard error but not the rest. Recommendation: accept.
4. **Citkowicz and Vevea (2017), blinding model, beta weights b and LRT** (`...article.blind.beta.b_lrt`): printed 5.097 and 9.045; metafopy and R
   5.0960 and 9.0432. Recommendation: accept (two units of the last digit on a flat likelihood).
5. **Bagos and Nikolopoulos (2009) Table 5, school absences, random study effects** (`glmm.published.bagos2009.table5.school.rs.mismatch`, in
   `glmm_published.toml`): printed CI (0.579, 0.897); metafopy (0.579941, 0.896414), R (0.579940, 0.896409). The source used Stata `gllamm`; the upper
   limit is 0.0006 from the printed value against a tolerance of 0.0005. Recommendation: accept.

## What an answer does

- Accept: the case's `reason` gains "; accepted as a source difference on <date> (metafopy and R metafor agree)" in place of "not yet reviewed", and
  its entry in `agents/specs/correctness-publication-bias/mismatches.md` or `agents/specs/correctness-glmm/mismatches.md` gains a dated note.
- Investigate: a `TODO.md` item for the case, with these figures; the case and its record stay as they are.

## Answers

Answered on the review page on 2026-11-19 (18:24 UTC, the accept-all button), read back from its `answers` store; no comments:

| Case | Answer |
|---|---|
| 1. Citkowicz and Vevea (2017), mean-only unadjusted se | accept |
| 2. Citkowicz and Vevea (2017), mean-only beta weights b | accept |
| 3. Citkowicz and Vevea (2017), blinding model without selection | accept |
| 4. Citkowicz and Vevea (2017), blinding model b and LRT | accept |
| 5. Bagos and Nikolopoulos (2009) Table 5, school absences | accept |

Applied on 2026-11-19: each case's `reason` says "accepted as a source difference on 2026-11-19", and both `mismatches.md` files record the verdict.
