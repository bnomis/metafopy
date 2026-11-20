# Review: the correctness-ls-weights spec

**Review page:** https://claude.ai/artifact/Se6qrZ8g859sv2L2MRy5PH
**Date:** 2026-11-20

**What is being reviewed and how.** One new spec, `agents/specs/correctness-ls-weights/spec.md` (with `measurements.md`), written by `/todo-do`
for the `TODO.md` item "No correctness case covers user weights with a scale model". It adds correctness cases for user `weights` in `rma_uni`,
which only R parity fixtures check today. No metafopy code changes. Approve or flag the spec, and answer the two decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
- Pass: the gap is still open at `0cc8edc` (no case passes user `weights` to `rma_uni` except the `GENQ` and `GENQM` estimators).
- Pass: measured against independent references, metafopy agrees to 4.4e-8 or better for `alpha` and 1e-15 for `beta` and `vb`; plain REML fits
  agree to 6.7e-6 at the default control settings, which the tight control of the existing REML cases removes (measurements.md, R 4.6.2 with
  metafor 5.2-1).
- Pass: every source the cases cite is in the papers folder `/Users/simonb/src/metafopy papers`.
- Found: `optbeta=True` with `weights` fits without an error in metafopy, where R stops with "Cannot use custom weights when 'optbeta=TRUE'."
  (decision D2).
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **References:** Hedges, Tipton and Johnson (2010), equations 3 and 4: the weighted least squares estimate and its exact variance for "an
  arbitrary weight matrix W" (already in the papers folder). The location-scale likelihoods (Viechtbauer and Lopez-Lopez 2022, equations 5 and 9)
  carry no user weights, so `alpha` is checked against the reference fit without them; the `rma.uni` help page's "`weighted=FALSE` is the same as
  constant weights" becomes a property. Exempt as code-only: the fit statistics with user weights (evaluated at the weighted `beta`, not at the
  maximum) and `test="knha"` with user weights.
- **Cases:** six location-scale cases (`dat.bcg` and `dat.bangertdrowns2004`, ML and REML, log and identity links) with a new check
  `ls_weights`; under D1, four plain `rma_uni` cases through the existing `expect` formulas, which need their user-weights branch widened from
  `GENQ` to any method; two properties (constant weights equal `weighted=False`; multiplying the weights by a constant changes nothing).
- **Expected mismatches:** none.
- **Stops in the task list:** the mismatch review, which needs no answer if no mismatch appears.
- **Recommendation:** approve.

## Decisions

### D1. Plain `rma_uni` fits with user weights

- **What is decided:** whether this spec also covers user `weights` in a plain `rma_uni` fit (no scale model).
- **Background:** checked 2026-11-20: no case passes user `weights` to a plain fit except `GENQ` and `GENQM`, `coverage_items()` has no
  `rma_uni.weights` item, and no `TODO.md` item records the gap. The construction is the same as for the location-scale fits (the estimator's
  `tau2`, then equations 3 and 4); metafopy agrees with it to 1e-15 for `beta` and `vb`.
- **What it affects:** `formulas.fixed_effects_fit` (its user-weights branch taken for any method, not only `GENQ`), four cases in
  `cases/inference.toml`, the item `rma_uni.weights` and two exemptions.
- **Options:**
  - *Include them (Recommended):* this spec covers both; the task list adds one formulas change and four cases.
  - *File a separate item:* a new `TODO.md` item for plain fits; this spec covers location-scale fits only.
- **Recommendation:** include them: the same reference and the same review, and splitting would leave a second spec for four cases.

### D2. `optbeta=True` with `weights` fits without an error

- **What is decided:** where the fix for this metafopy defect goes.
- **Background:** R stops with "Cannot use custom weights when 'optbeta=TRUE'."; metafopy (2026-11-20, `0cc8edc`) returns a fit
  (`dat.bcg`, `mods = scale = ~ ablat`, weights `tpos`). The check exists in `src/metafopy/models/rma_mv.py` but not in
  `src/metafopy/models/rma_ls_fit.py`.
- **What it affects:** `rma_ls_fit.py` (one `InputError`) and a test in `src/tests/test_rma_ls.py`.
- **Options:**
  - *File its own TODO.md item (Recommended):* a small product fix, done with `/todo-do` before or after this spec's list.
  - *Add it to this spec's task list:* one more task, which changes product code in a list otherwise made of tests.
- **Recommendation:** file it: the spec is about correctness tests, and a product fix with its own test is a separate commit either way.

## Answers

Answered 2026-11-20 on the review page (https://claude.ai/artifact/Se6qrZ8g859sv2L2MRy5PH), each with no comment:

- **1. The spec:** approved.
- **D1. Plain `rma_uni` fits with user weights:** include them.
- **D2. `optbeta=True` with `weights` fits without an error:** file its own `TODO.md` item.

Applied in version 1.1 of the spec (2026-11-20); the `TODO.md` item is filed.
