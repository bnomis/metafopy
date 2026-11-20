# Correctness tests: user weights

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-20

This spec adds correctness cases for user-supplied `weights` in `rma_uni`: with a scale model (`rma_uni(..., weights=..., scale=...)`, a
location-scale fit) and (decision D1) without one (a plain `rma_uni` fit). It reuses the harness, reference fit and tolerances of the
[location-scale spec](../correctness-location-scale/spec.md) and the [location-scale extensions spec](../correctness-ls-extensions/spec.md)
unchanged, and the `rma_uni` reference functions already in `formulas.py`. The dated figures behind it are in
[measurements.md](measurements.md).

**Approved** by the operator on 2026-11-20 on the review page of `agents/reviews/correctness-ls-weights-spec.md` (no comment), with the
recommended option of each decision. Version 1.1 records the approval and the decisions (see Decisions).

## Motivation

<!-- doc-counts: accept-start found on 2026-11-20 - a dated record, not a claim about the present -->
The `TODO.md` item "No correctness case covers user weights with a scale model" (found 2026-11-20 while writing
`agents/specs/correctness-ls-extensions/spec.md`, filed by decision D2 of `agents/reviews/correctness-ls-extensions-spec.md`, answered
2026-11-20) records that `rma_uni(..., weights=..., scale=...)` runs in metafopy and no case in `src/tests/correctness` passes user `weights` with
`scale`, so only R parity checks it, and that whether a publication defines user weights for location-scale models was not yet checked.
Checked 2026-11-20 while writing this spec: no case passes user `weights` to a plain `rma_uni` fit either, except the `GENQ` and `GENQM`
estimators, whose weights are the estimator's own; `coverage_items()` has no `rma_uni.weights` item, and no `TODO.md` item records that gap.
<!-- doc-counts: accept-end -->

**The defining publication outranks metafor** (`claude/rules/metafopy.md`): a disagreement becomes a `mismatch` case and metafopy's output changes
only after the user confirms it in the review step.

## Module location

- `src/tests/correctness/formulas.py` - `fixed_effects_fit` gains user weights for every method (no new function).
- `src/tests/correctness/test_derived.py` - one new check, `ls_weights`.
- `src/tests/correctness/test_properties.py` - two new properties.
- `src/tests/correctness/test_coverage.py` - the items and exemptions.
- `src/tests/correctness/cases/location_scale.toml` (the location-scale cases and exemptions) and `src/tests/correctness/cases/inference.toml`
  (the plain `rma_uni` cases and exemptions).
- `agents/specs/correctness-ls-weights/mismatches.md` - the review document of the review step.

## Definitions and references

Hedges LV, Tipton E, Johnson MC (2010), Robust variance estimation in meta-regression with dependent effect size estimates, Res Synth Methods
1:39-65, doi:10.1002/jrsm.5 (in the papers folder as `Robust variance estimation in meta-regression with dependent effect size estimates.pdf`),
define, for "an arbitrary weight matrix W", the weighted least squares estimate `b = (X' W X)^-1 X' W T` (equation 3) and its exact variance
`(X' W X)^-1 X' W Sigma W X (X' W X)^-1` (equation 4). With one estimate per study, `Sigma = diag(v_i + tau2_i)`; the reference evaluates it at
the fitted `tau2_i`, as the existing `ls_unweighted` check does. Jackson, Turner, Rhodes and Viechtbauer (2014), BMC Med Res Methodol 14:103,
give the same `b` with arbitrary positive weights, without its variance (read 2026-11-20).

The likelihood and restricted likelihood of a location-scale model (Viechtbauer and Lopez-Lopez 2022, equations 5 and 9) contain no user weights,
so `alpha` is their maximiser whatever the weights; likewise `tau2` of a plain fit is its estimator's, which takes no user weights except `GENQ`
and `GENQM`. The `rma.uni` help page (metafor 5.2-1, Details) says `weighted=FALSE` "is the same as setting the weights equal to a constant".

Defined only in metafor's code, so exempt with reason "defined only in metafor's code; R parity covers it":

- the fit statistics with user weights: the log-likelihood is equation 5 (ML) or 9 with `log|X'X|` (REML) evaluated at the user-weighted `beta`
  rather than at the maximum (measured: metafopy reproduces R to 1e-8), and the deviance, AIC, BIC and AICc built from it;
- `test="knha"` with user weights (Knapp and Hartung define the adjustment for inverse-variance weights only).

## API

```python
# formulas.py: the branch for user weights is taken whenever ``weights`` is given, not only for GENQ and GENQM
def fixed_effects_fit(
    yi: FloatArray, vi: FloatArray, method: str, weights: FloatArray | None = None, mods: str | None = None, data: object | None = None
) -> tuple[float, FloatArray, FloatArray]:
    """``(tau2, beta, se)``: ``tau2`` of ``method`` (``rma_tau2``, which uses ``weights`` only for GENQ and GENQM), then the weighted least squares
    coefficients and their standard errors: with ``weights``, Hedges, Tipton and Johnson (2010) equations 3 and 4 with ``W = diag(weights)`` and
    ``Sigma = diag(vi + tau2)``; without, the inverse-variance weights ``1 / (vi + tau2)`` and ``(X' W X)^-1``."""


# test_derived.py
def ls_weights(case: correctness.cases.Case, result: typing.Any) -> None:
    """User ``weights`` with a scale model: ``alpha`` against ``ls_reference`` of the case (equation 7 or 9, which carry no weights) at
    ``LS_RELATIVE``; ``beta`` and ``vb`` against ``formulas.gls(yi, diag(vi + tau2_i), X, diag(weights))`` at metafopy's ``tau2_i``
    (``numbers_agree``)."""


# test_properties.py
def test_weights_constant_is_unweighted(...) -> None:
    """For a plain and a location-scale fit, ``weights`` all equal to a constant ``c`` give the ``beta`` and ``vb`` of ``weighted=False``
    (the rma.uni help page); hypothesis draws ``c``."""


def test_weights_rescale_invariant(...) -> None:
    """For a plain and a location-scale fit, ``weights`` and ``c * weights`` give the same ``beta`` and ``vb`` (equations 3 and 4 are unchanged when
    ``W`` is multiplied by ``c``); hypothesis draws ``c``."""
```

Both properties use `@hypothesis.settings(max_examples=20, deadline=10000)` and draw `c` from `floats(1e-3, 1e3)`; they compare at
`DERIVED_TOLERANCE`.

## Coverage list

- `ls_items()` gains `ls.weights`, and the exemptions `ls.weights.fitstats` and `ls.weights.knha`.
- By decision D1, a new `uni_weights_items()`, called from `coverage_items()`, returns `rma_uni.weights` and the exemptions
  `rma_uni.weights.fitstats` and `rma_uni.weights.knha`. The existing `rma_uni.<f>.weights` exemptions of the diagnostics are unchanged.

## Cases

Every case is `kind = "derived"`, `function = "rma_uni"`, with a `[case.source]` citing Hedges, Tipton and Johnson (2010), equations 3 and 4 (and
Viechtbauer and Lopez-Lopez 2022, equation 5 or 9, for the location-scale cases). The `dat.bcg` cases use `escalc` RR as the existing cases do;
`dat.bangertdrowns2004` uses its own `yi` and `vi`, its sample size `ni` as the weights.

| Case | Data, weights | Model | Check |
|---|---|---|---|
| `ls.weights.ML.log.bcg` | `dat.bcg`, `tpos` | `mods = scale = ~ ablat`, ML, log | `test_derived.ls_weights` |
| `ls.weights.REML.log.bcg` | `dat.bcg`, `tpos` | REML, log | `test_derived.ls_weights` |
| `ls.weights.ML.identity.bcg` | `dat.bcg`, `tpos` | ML, identity | `test_derived.ls_weights` |
| `ls.weights.REML.identity.bcg` | `dat.bcg`, `tpos` | REML, identity | `test_derived.ls_weights` |
| `ls.weights.ML.log.bd` | `dat.bangertdrowns2004`, `ni` | `mods = scale = ~ imag`, ML, log | `test_derived.ls_weights` |
| `ls.weights.REML.identity.bd` | `dat.bangertdrowns2004`, `ni` | REML, identity | `test_derived.ls_weights` |
| `uni.weights.dl.bcg` (D1) | `dat.bcg`, `tpos` | DL | `expect` tau2, beta, se |
| `uni.weights.dl.bcg.ablat` (D1) | `dat.bcg`, `tpos` | DL, `~ ablat` | `expect` tau2, beta, se |
| `uni.weights.reml.bcg.ablat` (D1) | `dat.bcg`, `tpos` | REML, `~ ablat`, tight control | `expect` tau2, beta, se |
| `uni.weights.reml.bd` (D1) | `dat.bangertdrowns2004`, `ni` | REML, tight control | `expect` tau2, beta, se |

The `uni.weights.*` cases use `expect = { tau2 = "formulas.rma_tau2", beta = "formulas.rma_beta", se = "formulas.rma_se" }`, which pass the
case's `weights` column through `formula_value`; the REML ones pass `control = { threshold = 1e-13, tol = 1e-13, maxiter = 1000, tau2_max = 1e6 }`
as the existing REML `tau2` cases do (at the default `threshold = 1e-5`, `tau2` differs by up to 6.7e-06, measurements.md).

**Expected mismatches:** none. On 2026-11-20 every case above agreed with its reference (largest relative difference 4.4e-08 for `alpha`,
8.6e-06 for a plain REML fit at the default control settings, which the tight control removes).

## Error handling

- `optbeta=True` with `weights`: R stops with "Cannot use custom weights when 'optbeta=TRUE'."; metafopy (measured 2026-11-20 at `0cc8edc`) fits
  without an error (`dat.bcg`, `mods = scale = ~ ablat`, weights `tpos`: `beta` (0.33244, -0.02878)), since only `src/metafopy/models/rma_mv.py`
  raises that `InputError`, not `src/metafopy/models/rma_ls_fit.py`. That is a defect in metafopy, not a correctness case; decision D2 says where
  it goes.
- The identity-link `dat.bcg` fits warn "Error when trying to invert the Hessian." and have `va` NaN; `ls_weights` does not compare `va`.
- `ls_weights` fails naming the case when the reference fit raises `RuntimeError`, as the other location-scale checks do.

## Implementation

1. `fixed_effects_fit`: take the user-weights branch whenever `weights` is given; keep the `GENQ`/`GENQM` requirement that `weights` is given; cite
   Hedges, Tipton and Johnson (2010) beside Jackson et al. (2014). Run the existing `tau2.genq.*` cases, which must still pass.
2. The check `ls_weights`, the two properties, the items and exemptions in `test_coverage.py`, failing until step 3.
3. The cases above with their sources (locator: "equations 3 and 4, an arbitrary weight matrix W"; for the location-scale cases also "equation 5
   (ML) or 9 (REML), which carry no user weights").
4. Review: `mismatches.md` with one item per mismatch and its review page, per the shared `reviews.md` rule. **Work stops** until each has an
   answer; if no mismatch remains when the step runs, `mismatches.md` says so and the step needs no answer.
5. Close: each *follow the reference* answer becomes a `TODO.md` item; the TODO item "No correctness case covers user weights with a scale model"
   moves to `DONE.md`; this spec's status becomes implemented in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pandas`, `pytest`, `hypothesis` (existing); no new dependency, no R at test time.
- Every source is in the papers folder `/Users/simonb/src/metafopy papers/` except Jackson et al. (2014), which is cited only beside the
  primary source and needs no case.

## Decisions

Answered 2026-11-20 (`agents/reviews/correctness-ls-weights-spec.md`):

- **D1, plain `rma_uni` fits with user weights: included.** This spec covers them: `rma_uni.weights`, the four `uni.weights.*` cases, two
  exemptions, and the change to `fixed_effects_fit` (step 1 of Implementation).
- **D2, `optbeta=True` with `weights` fits without an error: filed** as its own `TODO.md` item, "A location-scale fit accepts optbeta=True with
  user weights"; this spec's task list does not fix it.
