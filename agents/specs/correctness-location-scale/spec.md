# Correctness tests: location-scale models

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-12

Location-scale models (`rma_uni(..., scale=...)`, metafor's `rma.ls`) are today checked only against R metafor: the parity fixtures and
`bin/metafor-audit.sh` prove that metafopy reproduces metafor, not that either is right. A `TODO.md` item found on 2026-11-12 ("Extend the correctness
tests beyond the core models") records that the [correctness-tests spec](../correctness-tests/spec.md) covers only `escalc`, `rma_uni`, `rma_mh` and
`rma_peto`, and says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and `robust`", with no such spec
written. This spec is the `rma_ls` one. Correctness means agreement with the publications that define each method, ranked above agreement with metafor
(`NOTES.md`: "if metafor has bugs we don't want to reproduce them"; `claude/rules/metafopy.md`).

It reuses the harness of the correctness-tests spec unchanged: the case file format, the kinds of evidence (`published`, `derived`, `property`,
`simulation`, `other`) and their tolerances, the scope gate (each coverage item needs a `published` or `derived` case), `formulas.py` importing nothing
from metafopy, and the mismatch flow (strict xfail, then a review document and the user's answer before metafopy's output changes). It adds coverage
items, case files, formulas and `check` functions.

**The defining paper is by metafor's author.** Viechtbauer and Lopez-Lopez (2022) define the model and its inference, and every number it prints was
computed with metafor. Its printed values show that metafopy agrees with metafor as the paper ran it, not that the method is coded right. So in this
spec `derived` cases (formulas coded from the paper's equations, maximised independently) and `property` cases carry the weight, and the published values
are a cross-check (see Decisions).

## Module location

- `src/tests/correctness/cases/location_scale.toml` - every case of this spec, with its `[[exempt]]` entries.
- `src/tests/correctness/formulas.py` - new location-scale functions (API below), still `numpy` and `scipy` only.
- `src/tests/correctness/test_derived.py`, `test_properties.py`, `test_simulation.py`, `test_other_software.py` - new `check` functions, since most
  cases fit a model and then call a method on it (`confint`, `predict`, `anova`, `permutest`), which a single `function` call cannot express.
- `src/tests/correctness/test_coverage.py` - the new items in `coverage_items()`.
- `src/tests/correctness/r/other_software_ls.R` - generator of the `other` cases from R `glmmTMB` (see Evidence).
- `src/metafopy/models/rma_ls_fit.py` - one refactor: the inline `('ML', 'REML')` check becomes a module constant `LS_METHODS`, so the gate reads it.
- `agents/specs/correctness-location-scale/mismatches.md` - the review document of disagreements and sources needed.

**Prerequisite:** the `TODO.md` item "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals" (its own
task list) adds `check` functions for back-transformed values and interval bounds. The published predicted `tau2` values and their intervals on the
`tau2` scale need it. Check: that item has moved from `TODO.md` to `DONE.md`. The `derived` and `property` cases do not depend on it.

## API

```python
# metafopy/models/rma_ls_fit.py
LS_METHODS: tuple[str, ...] = ('ML', 'REML')  # the methods fit_rma_ls accepts; it raises InputError for any other


# formulas.py - coded from Viechtbauer and Lopez-Lopez (2022), equations 5 to 15; no metafopy import
def ls_tau2(Z: FloatArray, alpha: FloatArray, link: str) -> FloatArray:
    """tau2_i = exp(Z alpha) (log link) or Z alpha (identity link)."""


def ls_loglik(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> float:
    """The profile log-likelihood (eq. 7, beta at its WLS value, eq. 6) or the restricted log-likelihood (eq. 9) at alpha."""


def ls_reference_fit(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str) -> dict[str, typing.Any]:
    """Maximise ls_loglik with scipy.optimize (BFGS from several starts, then Nelder-Mead; SLSQP with Z alpha >= 0 for the identity link).

    Returns alpha, beta, vb (eq. 10), tau2, ll and the analytic observed information of alpha (va), independent of metafopy's numerical Hessian.
    """


def ls_wald(estimate: FloatArray, vcov: FloatArray, idx: list[int], ddf: float | None) -> tuple[float, float]:
    """The Wald statistic of eq. 12/14 and its p-value (chi-square, or F with (m, ddf) after division by m for the t-type tests)."""


def ls_knha_s2w(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray, adhoc: bool) -> float:
    """The Knapp-Hartung factor of section 2.5.4: RSS / (k - p), at least 1 with adhoc (Knapp and Hartung, 2003)."""


def ls_profile_bounds(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, j: int, level: float) -> tuple[float, float]:
    """Profile-likelihood interval of alpha_j (section 2.5.3): brentq on 2 (ll_max - ll_j(a)) = chi2_{1, level}, refitting the other alphas each time."""


def ls_lrt(ll_full: float, ll_reduced: float, df: int) -> tuple[float, float]:
    """The likelihood-ratio statistic of eq. 15 and its chi-square p-value."""


def ls_predict(x0: FloatArray, z0: FloatArray, fit: dict[str, typing.Any], link: str, crit: float) -> dict[str, float]:
    """pred, se, CI and PI (section 2.7) at x0 and z0, and the predicted tau2 with its CI from z0' va z0 (section 2.5.2)."""


def ls_exact_permutation_p(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str, part: str) -> FloatArray:
    """Exact permutation p-values by enumerating every distinct row permutation of X ('location') or Z ('scale'), refitting with ls_reference_fit."""


# test_derived.py (pattern of the existing confint_* checks)
def ls_confint_scale(case: correctness.cases.Case, result: typing.Any) -> None:
    """confint(result, tol=1e-12) for each alpha against formulas.ls_profile_bounds."""
```

The other `check` functions follow the same one-line pattern, one per coverage item that is not a single attribute: `ls_vcov_alpha`, `ls_qs`,
`ls_predict_*`, `ls_anova_*`, `ls_permutest_exact` (derived); `ls_reduces_to_rma_uni`, `ls_subgroups_are_separate_fits`, `ls_score_is_zero`,
`ls_invariance`, `ls_profile_peaks_at_estimate`, `ls_optimisers_agree` (property).

## Coverage list

`coverage_items()` gains, read from the code where the code lists the options:

- `ls.fit.<method>.<link>` for each `method` in `metafopy.models.rma_ls_fit.LS_METHODS` and each `link` in
  `typing.get_args(inspect.signature(metafopy.rma_uni).parameters['link'].annotation)` (alpha, beta, tau2 and the log-likelihood);
- `ls.test.<test>` for each value of `metafopy.models.rma_uni.Test`, read at test time (the location Knapp-Hartung factor and the t/F reference for both
  parts, section 2.5.4);
- `ls.optbeta` (location and scale coefficients optimised jointly);
- `ls.scale.vcov` (inverse negative Hessian of alpha) and `ls.scale.QS` (omnibus Wald test of the scale part, eq. 14);
- `ls.fitstats.ML`, `ls.fitstats.REML` (log-likelihood, deviance, AIC, BIC, AICc);
- `ls.heterogeneity.<statistic>` for `QE`, `I2`, `H2`;
- `ls.confint.scale` (profile likelihood) and `ls.confint.location` (Wald, `fixed=True`);
- `ls.predict.<situation>` for the situations `metafopy.methods.ls_predict` documents: `fitted`, `newmods`, `newscale`, `both`;
- `ls.anova.<form>` for `btt`, `att`, `X`, `Z`, `lrt.ML`, `lrt.REML`;
- `ls.permutest.location.exact`, `ls.permutest.scale.exact`.

The situations of `predict` and the forms of `anova` have no list in the code, so they are constants in `test_coverage.py`, as `HETEROGENEITY` is.
Random permutation tests are not gate items: they draw with the R-exact generator (tested by the RNG parity tests) and compute the same statistics as the
exact test, so a `simulation` case checks that the random p-value lies within 4 binomial standard errors of the exact one. Out of scope, for a later
spec: `profile` (beyond the property below), `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha`/`beta` values and the `alpha_min`
and `alpha_max` constraints.

## Evidence

Every DOI below was checked against Crossref on 2026-11-12. "In folder" means the PDF is in `/Users/simonb/src/metafopy papers` (outside the
repository, never copied in).

| Items | Source | Access |
|---|---|---|
| All: model, eqs 5-15, KH generalisation, PI, worked example | Viechtbauer W, Lopez-Lopez JA (2022). Location-scale models for meta-analysis. Res Synth Methods 13:697-715. doi:10.1002/jrsm.1562 | Open (CC BY-NC 4.0, PMC9790740); not in folder |
| `ls.fit.REML.*` | Verbyla AP (1993). Modelling variance heterogeneity: residual maximum likelihood and diagnostics. JRSS B 55:493-508. doi:10.1111/j.2517-6161.1993.tb01918.x | Paywalled; not in folder |
| `ls.fit.*.log` (log-linear dispersion) | Smyth GK (1989). Generalized linear models with varying dispersion. JRSS B 51:47-60. doi:10.1111/j.2517-6161.1989.tb01747.x; Harvey AC (1976). Estimating regression models with multiplicative heteroscedasticity. Econometrica 44:461-465. doi:10.2307/1913974 | Paywalled; not in folder |
| Background: heterogeneity as a function of covariates | Raudenbush SW, Bryk AS (1987). Examining correlates of diversity. J Educ Stat 12:241-269. doi:10.3102/10769986012003241 | Paywalled; not in folder |
| `ls.test.knha`, `ls.test.adhoc` | Knapp G, Hartung J (2003). Improved tests for a random effects meta-regression with a single covariate. Stat Med 22:2693-2710. doi:10.1002/sim.1482 | In folder |
| `ls.confint.scale` | Hardy RJ, Thompson SG (1996). A likelihood approach to meta-analysis with random effects. Stat Med 15:619-629. doi:10.1002/(SICI)1097-0258(19960330)15:6<619::AID-SIM188>3.0.CO;2-A | Paywalled; not in folder |
| `ls.heterogeneity.*` | Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Stat Med 21:1539-1558. doi:10.1002/sim.1186 | In folder |
| `ls.permutest.*` | Higgins JPT, Thompson SG (2004). Controlling the risk of spurious findings from meta-regression. Stat Med 23:1663-1682. doi:10.1002/sim.1752; Follmann DA, Proschan MA (1999). Valid inference in random effects meta-analysis. Biometrics 55:732-737. doi:10.1111/j.0006-341X.1999.00732.x | Paywalled; not in folder |
| `other` cases | Brooks ME et al. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated generalized linear mixed modeling. R Journal 9:378-400. doi:10.32614/RJ-2017-066 | Open |

**Published values** (read from the PMC text on 2026-11-12; the inline numbers of the text and of Tables 1 and 2 are missing there, so the PDF is
needed): the worked example on `dat.bangertdrowns2004` (48 studies, REML, log link, KH for location and t/F for scale), with total sample size
rescaled and subject recoded into math, science and social science (28, 9 and 11 studies). Table 3 prints the log-likelihood, AIC, BIC and AICc of five
models under ML and REML, two decimals (`ls.fitstats.*`); the text prints the predicted `tau2` of math and social science studies at 100 participants with
CIs 0.006 to 0.121 and 0.004 to 0.437 (`ls.predict.newscale`); the PDF adds the coefficients, CIs and profile CIs of Tables 1 and 2 and the ML LRT
on 3 df (`ls.anova.lrt.ML`). The recoding of `subject` and the rescaling of `ni` are read from the authors' analysis code, which the paper links, and
written inline in the case file. The paper's DOI page hosts the PDF; the user supplies it to the folder (it is not reachable through the PubMed tool).

**Properties the paper states** (strong, independent of any printed number): an intercept-only scale model gives the `rma_uni` fit with the same
method (`beta`, `se`, `tau2 = exp(alpha_0)`, log-likelihood; not `se` of `tau2`, which `rma_uni` takes from the expected and the LS model from the
observed information); a single categorical moderator in both parts gives the separate `rma_uni` fits of each category (`beta` and `tau2` per category,
ML and REML); the LRT under REML is meaningful only for models with the same location part. Further properties: the score of `ls_loglik` is zero at
metafopy's estimates; centring or rescaling a scale moderator changes alpha linearly and leaves `tau2_i` and the log-likelihood unchanged; the profile
log-likelihood of each alpha peaks at the estimate; every optimiser in `metafopy.models.rma_ls_fit.OPTIMIZERS` that metafopy implements reaches the
same maximum; the log and identity links agree when the scale model is a set of group indicators.

**Other software.** `glmmTMB` (not by metafor's author) fits the log-link model: the known sampling variances as an `equalto` random-effect covariance
(new in glmmTMB 1.1.14), the heterogeneity as the residual dispersion with `dispformula = ~ z`, ML and `REML = TRUE`. It has no identity link. glmmTMB
1.1.15.2 is already in the audit's pinned library (`src/audit/data/r-versions.json`), so the generator runs against `METAFOPY_AUDIT_RLIB`, refuses any
other glmmTMB version, and compares `tau2_i`, `beta` and the log-likelihood (whether its dispersion is `sigma` or `sigma^2` is read from its
documentation when the generator is written; `alpha` is compared after that factor). `nlme::gls` cannot express a known variance plus an estimated
one, and `brms` and `blsmeta` are Bayesian, so neither is used.

## Error handling

- The existing loader errors apply to `location_scale.toml` unchanged (malformed case, unknown `kind`, duplicate id, `mismatch` without `reason`).
- A `check` function that cannot fit (metafopy raises `EstimationError`, or `formulas.ls_reference_fit` does not converge from any start) fails its case naming
  the case, the dataset and the step; `ls_reference_fit` raises `RuntimeError` with the optimiser's message rather than returning a non-optimum.
- `ls_profile_bounds` returns `nan` for a bound that the profile does not cross within `+-` 20 standard errors, as the paper reports for flat profiles;
  a case expects `nan` there only when the source says so.
- `other_software_ls.R` stops unless `packageVersion("glmmTMB")` equals the version pinned at its top, and writes that version and R's into each case's
  `source`.

## Implementation

1. `LS_METHODS` in `rma_ls_fit.py`, used by `fit_rma_ls`'s method check (no behaviour change); the new items in `coverage_items()`, all failing.
2. `formulas.py` location-scale functions from the paper's equations. The analytic gradient and observed information of `ls_loglik` are coded by hand
   and checked against `scipy.optimize.approx_fprime` in a harness test, so the derived `ls.scale.vcov` does not share metafopy's numerical Hessian.
3. Derived cases on `dat.bangertdrowns2004` (the paper's models) and `dat.bcg` (a second, smaller dataset), each `ls.fit` item with one moderator in
   both parts. Tolerance per case: `1e-6` relative for alpha, beta and their variances, `1e-9` for the log-likelihood (the optimum of a multi-dimensional
   surface agrees to the optimiser's tolerance, not to `1e-10`); a property case asserts metafopy's log-likelihood is not below `ls_reference_fit`'s by more than
   `1e-9`. `ls.heterogeneity.QE` includes a location model without an intercept: metafopy, like metafor 5.2-1 (`rma.uni.r`, `QE <- crossprod(Ymc, P) %*%
   Ymc`), centres `yi` before the quadratic form, which equals Cochran's `y' P y` only when the model has an intercept (on 2026-11-12, `dat.bangertdrowns2004`
   with `mods=ni/100`, `intercept=False`: 95.742 against 135.998). The fix runs earlier in this order (see Decisions), so the case expects
   Cochran's value.
4. Property cases listed under Evidence; the intercept-only and the subgroup reductions use `rma_uni` from metafopy, so they are `property`, not
   `derived`.
5. Exact permutation cases on a subset of `dat.bcg` small enough to enumerate (`slow`), and the `simulation` case for random permutations.
6. Published cases, once the operator has supplied the paper's PDF; **work stops here until the source is supplied** (it is listed in
   `mismatches.md` under "Sources needed" with what is wanted from it).
7. `other_software_ls.R` and its generated cases, committed; tests never run R.
8. `mismatches.md` with its review page, as the correctness-tests spec lays out; **work stops until each mismatch has an answer**, then each *follow the
   reference* answer becomes a `TODO.md` item (fix, flip the case, mark the parity fixtures, `known.toml`, `docs/differences.md`), and the spec's status
   becomes implemented.

## External dependencies

- `numpy`, `scipy`, `pytest` (existing); no new Python dependency.
- R `glmmTMB` 1.1.15.2 from the audit's pinned library, for step 7 only.
- The PDF of Viechtbauer and Lopez-Lopez (2022), from the operator.

## Decisions

Answered on 2026-11-13 in the [review](../../reviews/order-correctness-follow-ups-specs.md) (its D1 and D3).

- **The paper's printed values count as `published`, and every location-scale item also needs a `derived` case.** `location_scale.toml` sets the
  per-file gate setting that requires a `derived` case; the published values add to it.
- **The QE defect in models without an intercept is fixed before this spec runs.** It is filed in `TODO.md` ("QE and the moment tau2 estimators
  centre yi in models without an intercept") and its task list runs earlier in the same order, so `ls.heterogeneity.QE` without an intercept is
  expected to equal Cochran's `y' P y`; if it does not, the case is a mismatch for the review in step 8.
