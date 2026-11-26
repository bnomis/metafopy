# Correctness tests: location-scale extensions

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-20

This spec adds correctness cases for the parts of location-scale models (`rma_uni(..., scale=...)`) that the
[location-scale spec](../correctness-location-scale/spec.md) left for later: `profile`, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`,
fixed `alpha` and `beta`, the `alpha_min` and `alpha_max` constraints, `optbeta=True` under REML, and the `vb` and deviance of `optbeta` fits. It
reuses that spec's harness, reference fit and tolerances unchanged. One spec covers them all: every gap is a thin check around the existing
`formulas.ls_reference_fit` and `formulas.ls_profile_ll`, so no split is needed.

**Approved** by the operator on 2026-11-20 on the review page of `agents/reviews/correctness-ls-extensions-spec.md` (no comment), with the
recommended option of each decision. Version 1.1 records the approval and the decisions (see Decisions).

## Motivation

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
The `TODO.md` item "Correctness coverage the extension specs left for later" (found 2026-11-15 by a TODO review) records that the location-scale
spec's "Out of scope, for a later spec" list (`profile` beyond the property, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed
`alpha`/`beta`, `alpha_min`/`alpha_max`) and the "Left for a later spec" list of its `mismatches.md` (REML with `optbeta=True`, the deviance and `vb`
of `optbeta` fits) are checked against R metafor only. Its note of 2026-11-20 says the `rma_mv` part is done and the location-scale part remains.
<!-- doc-counts: accept-end -->

Re-checked on 2026-11-20 at `936021a`: no case in `src/tests/correctness` calls `profile`, `vif`, `blup`, `ranef` or `rstandard` on a location-scale
fit, or passes `weighted`, `alpha`, `beta`, `alpha_min` or `alpha_max` with `scale`; the `ls.optbeta.*` cases are ML only and skip `vb`;
`ls_fitstats` skips the deviance. The [ls-profile-starts](../ls-profile-starts/spec.md) and [ls-profile-feasible-start](../ls-profile-feasible-start/spec.md) specs (2026-11-18) changed `profile_fit`, which `confint` and
`profile` share, and are checked only through `ls.confint.scale.*` (the bounds) and `property.ls.profile.*` (the reference's own profile peaks at
metafopy's estimate); neither compares `profile()`'s output.

**The defining publication outranks metafor** (`claude/rules/metafopy.md`): a disagreement becomes a `mismatch` case and metafopy's output changes
only after the user confirms it. Figures quoted below are in [measurements.md](measurements.md) (measured 2026-11-20).

## Module location

- `src/tests/correctness/formulas.py` - the functions under API; `numpy` and `scipy` only, no metafopy import.
- `src/tests/correctness/test_derived.py`, `test_properties.py` - the new checks; `ls_optbeta` takes the method from the case (it was ML only when this spec began; the `ls.optbeta.REML.*` cases now run it under REML).
- `src/tests/correctness/test_formulas_ls.py` - harness tests of the new formulas.
- `src/tests/correctness/cases/location_scale.toml` - a new section with the cases and `[[exempt]]` entries.
- `src/tests/correctness/test_coverage.py` - the new items in `ls_items()`.
- `agents/specs/correctness-ls-extensions/mismatches.md` - the review document.

No change to `src/metafopy`.

## Definitions and references

Viechtbauer and Lopez-Lopez (2022) define: the joint likelihood of `beta` and `alpha` (eq 5), maximised "simultaneously over the p + q + 2
location and scale coefficients" (`optbeta`); the restricted likelihood (eq 9) and `beta` given `alpha` (eq 6), whose maximiser the `optbeta` REML
objective shares (measurements.md); the likelihood with some scale coefficients "constrained ... to arbitrary values" (eq 17, fixed `alpha`) and the
reduced model with location coefficients constrained (eq 15, fixed `beta` under ML); the profile of each `alpha_j` (section 2.6); the hat values and
`Var[e_i] = (1 - h_i)(v_i + tau2_i)` (section 2.8, marginal `rstandard`). Henderson (1975) defines BLUPs for any `G`, here `diag(tau2_i)`.
Fox and Monette (1992) define the GVIF. metafor's help pages define the rest: `rstandard(type="conditional")` as `y_i` minus the BLUP over its
standard error (`residuals.rma`), `vif(reestimate=TRUE)` as `Var[b_j] / Var[b'_j]` from a refit (`vif`), `weighted=FALSE` as unit weights and the
box `alpha.min`/`alpha.max` (`rma.uni`); those are coded from the documentation (rule decided 2026-11-13, [correctness-mv-robust](../correctness-mv-robust/spec.md) Decisions).

Defined only in metafor's code, so exempt with reason "defined only in metafor's code; R parity covers it": the deviance of a location-scale fit
(`-2 (ll - ll_sat)` under ML, `-2 ll` under REML; no help page defines it), the fit statistics of a `weighted=False` fit (equation 5 at the OLS
`beta`, not the maximised likelihood), and the `optbeta` REML objective's own Hessian (`vb` of an REML `optbeta` fit, fixed `beta` under REML).

| Source | Used for | Folder | Locator |
|---|---|---|---|
| Viechtbauer W, Lopez-Lopez JA (2022). Location-scale models for meta-analysis. Res Synth Methods 13:697-715. doi:10.1002/jrsm.1562 | all items | yes, read 2026-11-20 | eqs 5-10, 15, 17, 18; sections 2.5.2, 2.6, 2.8 |
| Harvey AC (1976). Estimating regression models with multiplicative heteroscedasticity. Econometrica 44:461-465. doi:10.2307/1913974 | `optbeta` `vb` (decision D1) | yes, read 2026-11-20 (page images) | eqs 11, 12 |
| Verbyla AP (1993). Modelling variance heterogeneity: REML and diagnostics. JRSS B 55:493-508. doi:10.1111/j.2517-6161.1993.tb01918.x | `ls.optbeta.REML` | yes, text extracted 2026-11-20 | the restricted likelihood (section 2) |
| Henderson CR (1975). Best linear unbiased estimation and prediction under a selection model. Biometrics 31:423-447. doi:10.2307/2529430 | `blup`, `ranef` | yes, read 2026-11-19 (rma_mv spec) | eq 5, the text after eq 8 |
| Searle SR, Casella G, McCulloch CE (1992). Variance Components. Wiley. doi:10.1002/9780470316856 | `blup`, `ranef` | yes | section 7.6 |
| Fox J, Monette G (1992). Generalized collinearity diagnostics. JASA 87:178-183. doi:10.1080/01621459.1992.10475190 | `ls.vif.location`, `ls.vif.scale` | **supply** | the GVIF definition |
| metafor 5.2-1 help pages `rma.uni`, `vif`, `residuals.rma`, `ranef`, `fitstats` (no DOI) | documented options, the exemptions | read 2026-11-20 | Details |

DOIs checked against Crossref on 2026-11-20. **No published values**: the paper's Tables 1 to 3 and its text print only quantities the existing
`published.vl2022.*` cases cover, and its Figure 3 profile plots print no numbers.

## API

```python
# formulas.py
def ls_constrained_fit(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    Z: FloatArray,
    link: str,
    method: str,
    fixed: FloatArray,
    lower: FloatArray | None = None,
    upper: FloatArray | None = None,
) -> dict[str, typing.Any]:
    """Equation 7 or 9 maximised over the alpha that ``fixed`` leaves NaN, the others held (eq 17), within ``[lower, upper]`` when given
    (SLSQP with Z alpha >= 0 for the identity link, L-BFGS-B for the log link). Starts include the fitted-line start of ``ls_profile_ll``. Returns alpha, beta (eq 6), ll and the inverse information of the free
    alpha. Raises ``RuntimeError`` with the optimiser's message when no start ends at a finite value."""


def ls_joint_information(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, beta: FloatArray, alpha: FloatArray, link: str) -> FloatArray:
    """Analytic observed information of (beta, alpha) from eq 5 (minus its Hessian); checked against finite differences in test_formulas_ls.py."""


def ls_joint_fit(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, beta_fixed: FloatArray) -> dict[str, typing.Any]:
    """Equation 5 (ML) maximised over the free beta (NaN in ``beta_fixed``) and every alpha (eq 15 with the constraint at a given value), from the
    profile fit; returns beta, alpha, ll, free_beta (and the vb and va blocks) and the inverse of ``ls_joint_information`` over the free coefficients."""


def ls_blups(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray) -> dict[str, FloatArray]:
    """Henderson's mixed model equations with R = diag(vi), Z = I, G = diag(tau2): ``u`` and its prediction error se, ``X b + u`` and its se."""


def ls_rstandard(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray, conditional: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Residual, se and z: ``e_i / sqrt((1 - h_i)(v_i + tau2_i))`` (section 2.8), or ``(1 - lambda_i) e_i`` over ``(1 - lambda_i)`` times that se."""


def gvif(vcov: FloatArray, group: list[int]) -> float:
    """Fox and Monette (1992): det(R_gg) det(R_rr) / det(R) on the correlation matrix of ``vcov`` without its first (intercept) row and column."""
```

Checks, one per item group, each comparing at the tolerances of the location-scale spec (`LS_RELATIVE` 1e-6 for estimates, `LS_LOGLIK` 1e-9
for log-likelihoods, `LS_NUMERIC_HESSIAN` 1e-4 where metafopy's `va` enters, `DERIVED_TOLERANCE` for formulas evaluated at metafopy's own `tau2_i`):
`test_derived.ls_profile` (`ll` at each grid value of `profile(result, alpha=j, steps=7)` against `ls_profile_ll`, NaN where the reference gives
-1e10, and `maxll`), `ls_vif`, `ls_vif_reestimate`, `ls_blup`, `ls_ranef`, `ls_rstandard`, `ls_unweighted` (alpha against the reference fit; OLS
`beta` and `(X'X)^-1 X' M X (X'X)^-1` at metafopy's `tau2_i`), `ls_alpha_fixed`, `ls_alpha_bounds`, `ls_beta_fixed`, `ls_optbeta_vb`, and
`ls_optbeta` generalised to REML; `test_properties.ls_optbeta_va` (an `optbeta` fit's `va` equals the ordinary fit's, the alpha block of the inverse
joint information being the inverse profile information) and `ls_rstandard_types_agree` (marginal and conditional `z` are equal).

## Coverage list

`ls_items()` gains, the links read from the code as for `ls.fit`, the `rstandard` types from
`typing.get_args(inspect.signature(metafopy.methods.ls_resid.rstandard_rma_ls).parameters['type'].annotation)`, and one constant
`LS_VIF = ('location', 'reestimate', 'scale')`:

- `ls.profile.<link>`; `ls.vif.<kind>`; `ls.blup`, `ls.ranef`; `ls.rstandard.<type>`;
- `ls.unweighted`, `ls.alpha.fixed`, `ls.alpha.bounds`, `ls.beta.fixed.ML`, `ls.optbeta.REML`, `ls.optbeta.vb.ML` (decision D1);
- `[[exempt]]`, reason "defined only in metafor's code; R parity covers it": `ls.fitstats.deviance`, `ls.unweighted.fitstats`, `ls.beta.fixed.REML`,
  `ls.optbeta.vb.REML`.

Out of scope: `vif(sim=...)` (each draw is the GVIF the `vif` items check, drawn with the R-exact generator), `profile` of `optbeta`
fits (metafopy raises `InputError`, as metafor does), presentation arguments (`exp`, `xlim`, `digits`, `transf`). User `weights` with `scale` is
not in the TODO item and has no case either; decision D2 filed it as its own `TODO.md` item.

## Cases

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
Datasets as in measurements.md (`bd`, `bcg`, `bd2`, `bcg2`); log link unless named. Measured agreement is the largest difference found.

| Case | Fit | Covers | Measured |
|---|---|---|---|
| `ls.profile.ML.log.bd`, `ls.profile.REML.log.bcg` | `profile`, both alphas | `profile.log` | 1.4e-14, 1.2e-12 |
| `ls.profile.REML.identity.bd` | identity link | `profile.identity` | 7.1e-15 |
| `ls.profile.ML.identity.bcg` | identity link | `profile.identity` | **mismatch**: 1.06e-08 below the reference (allowed 8.3e-09) |
| `ls.vif.ML.bd2`, `ls.vif.REML.bcg2` | `vif()` | `vif.location`, `vif.scale` | 9.1e-07 |
| `ls.vif.reestimate.ML.bd2`, `ls.vif.reestimate.ML.bcg2` | `vif(reestimate=True)` | `vif.reestimate` | 4.6e-06 |
| `ls.blup.ML.z.bd`, `ls.blup.REML.t.bcg` | `blup()` | `blup` | 4.6e-15 |
| `ls.ranef.REML.z.bd`, `ls.ranef.ML.t.bcg` | `ranef()` | `ranef` | 3.9e-15 |
| `ls.rstandard.marginal.REML.bd`, `ls.rstandard.conditional.ML.bcg` | `rstandard()` | `rstandard.*` | 2.1e-15 |
| `ls.unweighted.ML.bd`, `ls.unweighted.REML.bcg` | `weighted=False` | `unweighted` | 4.5e-08 (alpha) |
| `ls.alpha.fixed.ML.bd`, `ls.alpha.fixed.REML.bcg` | `alpha = [nan, 2.7]`; `[-3.5, nan]` | `alpha.fixed` | 7.2e-09 |
| `ls.beta.fixed.ML.bd`, `ls.beta.fixed.ML.bcg` | `optbeta`, `beta = [nan, 0.05]`; `[nan, 0.02]` | `beta.fixed.ML` | 1.7e-07 |
| `ls.alpha.bounds.ML.bd`, `ls.alpha.bounds.REML.bd`, `ls.alpha.bounds.ML.bcg` | boxes (-10, -10; 10, 10), (-20, -10; 20, 1.9), (-20, -1; 20, 0) | `alpha.bounds` | 6.8e-09 |
| `ls.alpha.bounds.ML.bcg.wide` | box (-10, -1.2; 0, 1.2), not binding | `alpha.bounds` | **mismatch**: ll 1.14 below the maximum |
| `ls.optbeta.REML.bd`, `ls.optbeta.REML.bcg` | `optbeta`, REML | `optbeta.REML` | 3.6e-08 |
| `ls.optbeta.vb.ML.bd`, `ls.optbeta.vb.ML.bcg` | `optbeta`, ML | `optbeta.vb.ML` | 7.7e-08 (D1: observed information) |
| `property.ls.optbeta.va.ML.bd`, `property.ls.rstandard.types.REML.bcg` | properties | - | 4.2e-07, 2.0e-15 |

Every case cites its source, DOI and locator. None is `slow`: the profile cases take 2 to 7 s with their reference, every other case under 0.1 s.
<!-- doc-counts: accept-end -->

**Expected mismatches** (strict xfail until answered):

- `ls.alpha.bounds.ML.bcg.wide`: metafopy returns alpha 0, -0.0795806 (ll -8.6762) with only the "(almost) equal to their ... bound" warning; the
  bounded maximum is -4.95749, 0.03859 (ll -7.5393, the unconstrained optimum, inside the box). A start inside the box reaches it, so the start (the
  HE-based start clipped onto the upper bound, where the logistic map sends it to infinity) is the cause. R metafor 5.2-1 does not agree with
  metafopy: it stops with "Optimizer (nlminb) did not achieve convergence". Wider boxes take both programs to the `tau2 = 0` plateau (ll -9.4736).
- `ls.profile.ML.identity.bcg`: the identity-link profile refits end up to 1.06e-08 below the reference where the optimum lies on `tau2_i = 0`.
  R metafor is further off (1.3e-08 to 1.2e-07 below, and every alpha[0] refit fails). The review restates the answer of 2026-11-15
  (`../correctness-location-scale/mismatches.md`, item `ls_identity_opt`: *follow the reference*, since implemented as the identity-link polish) and
  asks only whether the remaining shortfall needs a further fix or a tolerance.

## Error handling

- The reference functions raise `RuntimeError` naming the optimiser's message rather than return a non-optimum; a check that cannot fit fails naming
  case, dataset and step, as the location-scale checks do.
- `ls_profile` treats a reference value of -1e10 (no feasible `tau2_i`) and metafopy's NaN as agreeing; NaN on one side only fails naming the grid value.
- `ls_vif_reestimate` skips a coefficient whose reference refit raises and says so; no case is chosen where that happens (`bcg2` REML scale).
- The existing loader errors apply unchanged.

## Implementation

1. The formulas, with harness tests: `ls_joint_information` against central finite differences of eq 5; `gvif` of a two-moderator OLS against
   `Var[b_j] / Var[b'_j]`; `ls_blups` with constant `tau2` against `uni_blups`.
2. The checks, and the items and exemptions in `test_coverage.py`, failing until step 3.
3. The cases above with their sources.
4. Sources: every source is in the folder except Fox and Monette (1992). **Work stops here** only if the `vif` cases cannot cite it by then;
   it is listed in `mismatches.md` under "Sources needed" with what is wanted (the GVIF definition and its section).
5. Review: `mismatches.md` with one item per mismatch (source, reference, metafopy and metafor values, options *follow the reference*,
   *keep metafor*, *investigate*) and its review page, per the shared `reviews.md` rule. **Work stops** until each has an answer; if no mismatch
   remains when the step runs, `mismatches.md` says so and the step needs no answer.
6. Close: each *follow the reference* answer becomes a `TODO.md` item (fix, flip the case, parity fixtures, `known.toml`, `docs/differences.md`);
   a dated note on the TODO item "Correctness coverage the extension specs left for later" says the location-scale part is done; since that was its
   last part, the item moves to `DONE.md`; this spec's status becomes implemented in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pandas`, `pytest` (existing); no new dependency, no R at test time.
- The PDF of Fox and Monette (1992), from the operator.

## Decisions

Answered by the operator on 2026-11-20 on the review page of `agents/reviews/correctness-ls-extensions-spec.md`, each with no comment:

- **D1, the reference for `vb` of an `optbeta` fit: A, the observed information** (`ls_joint_information`). The `ls.optbeta.vb.ML` cases compare
  against it; no output change. Equation 10 of the paper (the expected information) is not used.
- **D2, user `weights` with `scale`: filed** as the `TODO.md` item "No correctness case covers user weights with a scale model", after the item
  this spec closes; it stays out of this spec.
