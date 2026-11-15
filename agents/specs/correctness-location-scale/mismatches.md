# Correctness tests, location-scale models: mismatches and sources needed

**Spec:** `agents/specs/correctness-location-scale/spec.md`
**Task list:** `agents/tasks/correctness-location-scale/tasks-1.md` (Task 11 completes this document and publishes its page)
**Review page:** https://claude.ai/artifact/B6BxeoKp7BYY9en6w6QuFf (published 2026-11-15 by the operator session)

## What is reviewed and how

The case-writing tasks compared metafopy with the defining publication first and R metafor second. This document holds every case where the
reference and metafopy disagree (one mismatch item may cover several cases with one cause), and every source that could not be read. For each mismatch
you choose *follow the reference*, *keep metafor* or *investigate*. The list stops until every item has an answer.

## Automatic checks

Run 2026-11-15 from the worktree:

| Check | Result |
|---|---|
| Every `status = "mismatch"` case in `cases/location_scale*.toml` has an item above | pass: the 14 cases (`ls.fit.*.identity.*`, `property.ls.fit.*.identity.*`, `property.ls.links.*`, `ls.confint.scale.*.bcg`, `published.vl2022.table1.plci.size`, `published.vl2022.table2.plci.soc`) are all named; `cases/location_scale_other.toml` has none |
| `git diff main -- src/audit/data/r-versions.json` is empty | pass |
| `mypy` | pass |
| `pytest src/tests/correctness -m 'not slow'` | **fail**: `test_properties.py::test_scale_invariance[SJIT]` raises `EstimationError: Iterative SJ estimator did not converge` on a Hypothesis example (`yi = [0, -0.5625, 0, 0.875, -0.7734375, 0]`, `vi = [0.5, 0.625, 0.5, 0.5, 0.125, 1]`, scale 2.0). It is not a location-scale case and no location-scale module is involved; it is filed as Task 13 of the task list |
| Both gate tests (derived-case gate, coverage) | pass, inside the run above |

## Mismatches

### The identity-link fit stops about 1e-5 short of the maximum

- **Cases** (all in `src/tests/correctness/cases/location_scale.toml`, strict expected failures): `ls.fit.ML.identity.bd`, `ls.fit.REML.identity.bd`,
  `ls.fit.ML.identity.bcg`, `ls.fit.REML.identity.bcg` (the fit against the reference) and `property.ls.fit.ML.identity.bd`,
  `property.ls.fit.REML.identity.bd`, `property.ls.fit.ML.identity.bcg`, `property.ls.fit.REML.identity.bcg` (metafopy's log-likelihood is not below
  the reference's by more than 1e-9), and `property.ls.links.ML.bd`, `property.ls.links.REML.bd` (the identity-link and log-link fits of a
  two-group scale model, which span the same variances, differ in `beta` by about 1e-5: identity ML 0.20520075, 0.00347308 against log ML 0.20520475,
  0.00348536; identity REML 0.20742984, -0.0239243 against log REML 0.20743445, -0.02393794; their log-likelihoods agree to 1e-6).
- **Source:** Viechtbauer W, Lopez-Lopez JA (2022). Location-scale models for meta-analysis. Res Synth Methods 13:697-715, doi:10.1002/jrsm.1562,
  equations 3 (identity link), 7 (ML) and 9 (REML); the reference is `formulas.ls_reference_fit` (BFGS, SLSQP with `Z alpha >= 0` for the identity
  link, then Nelder-Mead), which checks its own result against the analytic score.
- **Reference values** (`dat.bangertdrowns2004` with `imag` in both parts, and `dat.bcg` log risk ratio with `ablat` in both parts):

  | Case | alpha, reference | alpha, metafopy | log-likelihood, reference | log-likelihood, metafopy |
  |---|---|---|---|---|
  | ML, `bd` | 0.03402053, 0.23031694 | 0.03401541, 0.23040472 | -17.058054672888 | -17.058054743801 |
  | REML, `bd` | 0.03699443, 0.42990534 | 0.03698802, 0.42979705 | -16.893376921444 | -16.893376988271 |
  | ML, `bcg` | -0.02665867, 0.00205067 | -0.02667603, 0.002052 | -7.139288817730 | -7.139288984088 |
  | REML, `bcg` | 0.33913546, -0.0061661 | 0.33905883, -0.00616471 | -7.988765268276 | -7.988765308005 |

  The coefficients differ by about 1e-5 relative to the largest (the case tolerance is 1e-6) and the log-likelihood by 1e-8 to 2e-7 (the tolerance is
  1e-9). Under the log link every comparison agrees to better than 1e-9. The reference ends at a higher log-likelihood than metafopy in all four fits,
  and it satisfies `Z alpha >= 0` (the `bcg` ML optimum is on the boundary, `tau2_i = 0` for the study with the smallest `ablat`, in both fits).
- **metafor:** no parity fixture in `src/tests/reference/phase5/location_scale.json` fits these models with the identity link, so metafor's own
  values are not on record here; `src/tests/test_rma_ls.py` accepts the identity-link fits at `OPT = 1e-4` against metafor. Task 10 runs R only for
  glmmTMB, which has no identity link. The convergence setting of metafopy's identity-link optimiser follows metafor's `constrOptim`.
- **Options:** follow the reference (a `TODO.md` item to polish the identity-link optimum, for example by a final Newton step on the score, after
  which the cases pass); keep metafor (the cases are loosened to 1e-4, the tolerance the metafor parity tests use, so the optimiser stays as
  metafor's); investigate (run metafor's identity-link fit on these four models, compare its log-likelihood with the reference, then decide).

### The profile-likelihood interval of the `dat.bcg` scale intercept differs from the reference

- **Cases** (strict expected failures in `src/tests/correctness/cases/location_scale.toml`): `ls.confint.scale.ML.bcg`, `ls.confint.scale.REML.bcg`.
  The same check passes on `dat.bangertdrowns2004` (both methods).
- **Source:** Viechtbauer and Lopez-Lopez (2022), doi:10.1002/jrsm.1562, section 2.5.3; Hardy and Thompson (1996), Stat Med 15:619-629. The reference is
  `formulas.ls_profile_bounds` (`brentq` on `2 (ll_max - ll_j(a)) = chi2_{1, 0.95}`, the other alpha re-maximised at each `a` from two starts, one
  keeping the fitted line of `Z alpha`).
- **Reference values** (log link, `ablat` in both parts, alpha.1 is the intercept, estimate -4.957 under ML):

  | Case | Bound | Reference | metafopy |
  |---|---|---|---|
  | ML | lower | `nan` (not crossed within 20 standard errors) | -23.309450 |
  | ML | upper | 1.803009 | 1.022354 |
  | REML | lower | -16.010862 | -16.010862 |
  | REML | upper | 3.402673 | 3.286219 |

  At a = 1.0 the reference profile gives `2 (ll_max - ll_j(a))` = 3.04 (below the 3.84 cut-off), so metafopy's ML upper bound of 1.02 lies where the
  profile has not yet crossed; at a = -23.3 the reference gives 1.99, so metafopy's lower bound is not a crossing either. The refit at a fixed intercept
  appears to end at a lower log-likelihood than the reference's.
- **metafor:** the parity fixture `src/tests/reference/phase5/location_scale.json` was not checked for a profile interval of this model in this task.
- **Options:** follow the reference (a `TODO.md` item: refit the profile from the fitted line, as the reference does); keep metafor (the bounds come
  from metafor's refit, whose starting values metafopy follows); investigate (run metafor's `confint` on this model and compare the two profiles).

### Decision: how close the identity-link fit must be to the maximum

- **What is decided:** whether metafopy's identity-link fit must reach the maximum to 1e-9 in the log-likelihood, as the log-link fit does, or only to
  the tolerance of metafor's optimiser.
- **Background:** the cases are strict expected failures. The spec sets 1e-6 relative for coefficients and 1e-9 for the log-likelihood, a tolerance it
  chose for a maximum the optimiser reaches; the log-link fit meets it. The identity-link fit is constrained, and metafopy's constrained optimiser stops
  earlier (dated 2026-11-15, on the four fits above).
- **What it affects:** `src/metafopy/models/rma_ls_opt.py` and `rma_ls_fit.py` (if followed), the identity-link tolerance of the parity tests, and
  `docs/differences.md`.
- **Options:**
  - *Follow the reference*: a `TODO.md` item to tighten the identity-link optimum; the eight cases pass after the fix.
  - *Keep metafor*: the cases are rewritten at 1e-4; the optimiser stays as metafor's, and `docs/differences.md` records the tolerance.
  - *Investigate*: the cases stay strict expected failures and nothing is filed; metafor's value is read first.
- **Recommendation:** *investigate*, because metafor's own identity-link log-likelihood on these four models is not on record, and the answer decides
  whether this is a metafopy defect or the shared optimiser's tolerance.

### The profile-likelihood intervals of the scale coefficients differ from the paper's where the profile is flat

- **Cases** (strict expected failures in `src/tests/correctness/cases/location_scale.toml`): `published.vl2022.table1.plci.size` and
  `published.vl2022.table2.plci.soc`. The printed bounds that metafopy does reproduce are in `published.vl2022.table1.plci` and
  `published.vl2022.table2.plci`, which pass.
- **Source:** Viechtbauer and Lopez-Lopez (2022), doi:10.1002/jrsm.1562, Tables 1 and 2 (95% PLCI columns), read 2026-11-15 from the PDF page images
  (the text layer loses the minus signs); fitted with the paper's models on `dat.bangertdrowns2004` (REML, log link, `knha`).
- **Values:**

  | Table | Coefficient | Bound | Paper | metafopy | Independent reference (`formulas.ls_profile_bounds`) |
  |---|---|---|---|---|---|
  | 1 | sample size | lower | -7.159 | -5.93003 | -7.159061 |
  | 1 | social sciences vs. math | lower, upper | `< -10`, `> 10` | -3.61002, 4.41224 | no crossing (`nan`, `nan`) |
  | 2 | social sciences vs. math | upper | 7.700 | 5.72067 | not computed (the reference fit of this model does not converge) |
  | 2 | social sciences vs. math | lower | `< -10` | -4.68066 | not computed |

  Bounds the paper prints as `< -8`, `> 10` and `< -10` lie beyond the range of its profile plots and are not numbers, so they have no case. metafopy
  agrees with them where it was checked: the Table 1 intercept lower bound is -22.09 (paper `< -8`), the science vs. math upper bound is 22.76 (paper
  `> 10`). Every Wald interval, coefficient, p-value, omnibus test, fit statistic, prediction and the likelihood ratio test of both tables agree with the
  paper's printed digits.
- **Cause:** the same as the `dat.bcg` item above: where the profile is flat, metafopy's bound is not a crossing of the profile. For the Table 1
  sample-size bound the independent reference reproduces the paper (-7.159061), so this is a metafopy defect, not a misprint.
- **metafor:** the paper's numbers were computed with metafor (`confint(res5, alpha=2, xlim=c(-8, 1))`), so metafor gives the printed values; the
  parity fixture `src/tests/reference/phase5/location_scale.json` was not checked for this model in this task.
- **Options:** follow the reference (a `TODO.md` item: refit the profile as metafor and the reference do, after which both cases pass); keep metafor
  (not applicable: metafor agrees with the paper); investigate (run metafor's `confint` on the Table 2 model to read the bound the paper prints).

## Left for a later spec

- `optbeta=True` under REML: `ls_reference_fit` has no joint objective, and a joint maximisation over beta and alpha under REML is a different
  objective from equation 9, so `ls.optbeta` is derived under ML only.
- The deviance of `ls.fitstats.*`: the paper defines none, so the derived cases compare the log-likelihood, AIC, BIC and AICc only.
- `vb` of an `optbeta=True` fit: metafopy takes it from the joint information matrix; no reference covers that matrix, so the `ls.optbeta` cases compare
  `alpha`, `beta`, `tau2` and the log-likelihood.

## Sources needed

None so far (Task 2 found every source present).

### Recommendations on the page (operator session, 2026-11-15)

The page has one item per mismatch above. It recommends *follow the reference* for all three, changing the identity-link recommendation from
*investigate*: the reference reaches a higher feasible log-likelihood than metafopy in all four fits, so metafopy's identity-link estimate is not the
maximum whatever metafor gives. For the two profile-interval items, the paper (computed with metafor) and the independent reference agree against
metafopy. The `SJIT` failure is Task 13, not a page item.

## Answers

Answered by the user on the review page on 2026-11-15 (saved at 05:11 UTC); copied here by the operator session the same day.

| Item | Answer | Comment |
|---|---|---|
| `ls_identity_opt`: the identity-link fits stop short of the maximum (10 cases) | follow the reference | - |
| `ls_plci_bcg`: the profile-likelihood interval of the `dat.bcg` scale intercept, ML and REML (2 cases) | follow the reference | - |
| `ls_plci_vl2022`: the profile-likelihood intervals in Viechtbauer and Lopez-Lopez (2022) Tables 1 and 2 (2 cases) | follow the reference | - |
