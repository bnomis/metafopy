# Correctness tests for rma_glmm

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-12

`rma_glmm` fits generalised linear mixed models to binomial and Poisson counts: the unconditional models with fixed or random study effects (`UM.FS`,
`UM.RS`), the conditional models with the approximate and exact likelihood (`CM.AL`, `CM.EL`), and the one-group logistic and Poisson models. Today its
numbers are checked only against R metafor (`src/tests/reference/` fixtures and `bin/metafor-audit.sh`), which proves that metafopy reproduces metafor,
not that either maximises the likelihood the defining papers write down. A `TODO.md` item found on 2026-11-12 ("Extend the correctness tests beyond the
core models") records that the [correctness-tests spec](../correctness-tests/spec.md) promises "Later specs extend the same harness to `rma_mv`,
`rma_glmm`, `rma_ls`, the bias methods, `selmodel` and `robust`", and that no such spec existed. This spec is the `rma_glmm` one; sibling specs cover
the other areas.

It reuses that harness unchanged in kind: the same case format, the same five kinds of evidence (`published`, `derived`, `property`, `simulation`,
`other`) and their ranking, the same gate (each coverage item needs a `published` or `derived` case, or an `[[exempt]]` entry with a reason), and the
same mismatch flow. **Correctness comes first, metafor compatibility second** (`NOTES.md`, decided 2026-11-11; `claude/rules/metafopy.md`): where
metafopy agrees with metafor but not with the defining reference, the reference wins once the user has confirmed the case.

## Prerequisites

- The `TODO.md` item "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals" is done (moved to
  `DONE.md`), so a case can check a back-transformed estimate (the odds ratios and rates Stijnen et al. 2010 print) and `predict` intervals. Check:
  `grep -n '^def ' src/tests/correctness/test_derived.py` lists a back-transform check beside the existing `predict_pi`. Until then, cases that need
  it are not written and their items stay on the gate's missing list; no other step waits for it.

## Module location

- `src/tests/correctness/cases/glmm.toml` - `derived` and `property` cases; `glmm_published.toml` - `published` cases; `glmm_other.toml` - generated
  `other` cases.
- `src/tests/correctness/glmm_formulas.py` - the independent likelihoods and fits below; imports `numpy` and `scipy` only, never `metafopy`.
- `src/tests/correctness/cases.py` - formula names resolve in every module of `FORMULA_MODULES` (`'formulas'`, `'glmm_formulas'`), written
  `"<module>.<name>"`; `test_coverage.py`'s import check covers each module of the tuple. A sibling spec may make the same change first; then only the
  module is added to the tuple.
- `src/tests/correctness/test_coverage.py` - the `rma_glmm` items added to `coverage_items()`.
- `src/tests/correctness/test_derived.py`, `test_properties.py` - the new `check` functions.
- `src/tests/correctness/r/other_software_glmm.R` - generator of `glmm_other.toml`.
- `agents/specs/correctness-glmm/mismatches.md` - the review document of disagreements and sources needed.

## Coverage list

Read from the code at test time wherever the code names the options, so a newly ported option fails the gate until it has a case:

- `rma_glmm.fit.<measure>.<model>.<FE|ML>`: every measure in `metafopy.models.rma_glmm.SUPPORTED` and `EXPERIMENTAL`; for the two-group measures
  every model in `MODELS` (only `UM.FS`, `UM.RS` for those in `UNCONDITIONAL_ONLY`), for those in `metafopy.models.glmm_fit.ONE_GROUP` the model
  `none`; `FE` for the fixed-effects fit and `ML` for the mixed-effects fit. Each case checks `beta`, `se` and, for `ML`, `tau2` (and `sigma2`
  for `UM.RS`). `IRR` with `CM.AL` is fitted as `CM.EL`; its case shows that.
- `rma_glmm.method.<m>` for every value of `METHODS`: `EE` and `CE` give the `FE` fit.
- `rma_glmm.link.<link>` for every value of `metafopy.core.glm.Link` (the `link` argument of the unconditional and one-group models).
- `rma_glmm.package.<p>` for every value of `metafopy.models.glmm_fit.PACKAGES`: the log-likelihood at the estimates is the package's approximation of
  the marginal likelihood (adaptive Gauss-Hermite quadrature with `nAGQ` points for `lme4` and `GLMMadaptive`, the Laplace approximation for
  `glmmTMB` and for `lme4` with `UM.RS`).
- `rma_glmm.optimizer.<o>` for every value of `metafopy.models.glmm_fit.OPTIMIZERS` (they apply to `CM.EL` for `OR` only). An optimiser that raises
  `NotSupportedError` gets an `[[exempt]]` entry giving that reason, and a harness test checks that each such exemption still raises, so it goes
  stale when support is added.
- Inference: `rma_glmm.test.z`, `rma_glmm.test.t`; `rma_glmm.coding.<c>` for the codings `0.5`, `1`, `0`; `rma_glmm.cor` (`UM.RS` with `cor=True`,
  `rho`); `rma_glmm.heterogeneity.<s>` for `QE_Wld`, `QE_LRT`, `QM`, `I2`, `H2`; `rma_glmm.i2def.<d>` for `'1'`, `'2'`; `rma_glmm.tau2_ci` (the Wald
  interval on `log tau2` of `CM.EL` with `ML`: `se_tau2`, `ci_lb_tau2`, `ci_ub_tau2`); `rma_glmm.predict.pi`. The test, coding and `i2def` values are
  literals inside `check_arguments` and `i2_h2`, not named constants, so `test_coverage.py` lists them as it lists `HETEROGENEITY`.

Not covered: `confint`, `blup` and `ranef` refuse `rma_glmm` fits (as in metafor), so they have no items.

## Kinds of evidence for these models

- **derived** is the core. `glmm_formulas.py` codes each likelihood from its paper: binomial and Poisson log-likelihoods with each link, including the
  family constants that `logLik()` reports; Fisher's non-central hypergeometric likelihood of `CM.EL` by direct summation over the support (van
  Houwelingen et al. 1993; Stijnen et al. 2010); and the normal mixture of each, integrated three ways: exactly (`scipy.integrate.quad`,
  `dblquad` for `UM.RS`), by adaptive Gauss-Hermite quadrature with `n` points centred at the conditional mode, and by the Laplace approximation
  (Pinheiro and Bates 1995). The formulas fit each model with their own `scipy.optimize`
  search from their own start, never metafopy's.
- **Optimiser dependence.** Every `ML` fit, every `CM.EL` fit and every `UM.RS` fit is the result of a numerical search, and metafopy's search stops at
  a gradient tolerance (`GLMMControl.gtol`), the ported `optim` routines at R's `reltol`. Two optima that agree to `1e-5` in the estimates differ by
  about `1e-10` in the log-likelihood, because the likelihood is flat at its maximum. So a derived `ML` case makes two checks: the formulas' log-likelihood
  **at metafopy's reported estimates** equals metafopy's `ll` and the formulas' maximum (`rtol=1e-8`), and its gradient there is below `1e-4`; and the
  estimates agree with the formulas' optimum at a looser, stated tolerance (`1e-5` for `nlminb` and the mixed models, `1e-4` for the optimiser items).
  The first is the stronger check: it fails for a wrong likelihood even when both searches stop in the same wrong place.
- **Fixed-effects fits** have closed forms where the paper gives one (`PLO` and `IRLN` without moderators: `logit(sum xi / sum ni)`,
  `log(sum xi / sum ti)`); otherwise the formulas solve the score equations with `scipy.optimize.root`. Tolerance `1e-8` relative on `beta`, because
  metafopy's `glm` stops at R's deviance criterion (`epsilon = 1e-8`); the `CM.EL` `FE` standard error comes from the analytic conditional information
  (the sum of the conditional variances) and is compared at `1e-6`, the accuracy of the `numDeriv` Hessian metafopy uses.
- **Integration as a convention.** Adaptive quadrature with `nAGQ = 7`, the Laplace approximation that `lme4` forces for `UM.RS`, and `CM.EL`'s default
  integration tolerance (`int_rel_tol = eps^0.25`, about `1.2e-4`) are documented approximations of the exact marginal likelihood the papers define.
  They are conventions, not mismatches (as the earlier spec treats a published variance approximation): the `package` cases pin each approximation
  through the formula of the same approximation, and separate cases run with an accurate integral (`GLMMadaptive` with a large `nAGQ`, `control`
  `int_rel_tol = 1e-10`) and compare with the exact likelihood. A difference between the approximation and the exact fit that changes a printed digit of
  a published value is reported in `mismatches.md` as a convention, for the user to see.
- **property**: swapping the groups negates `beta` and keeps `tau2` (`UM.FS` with coding `0.5`, `CM.AL`, `CM.EL`); swapping events and non-events
  negates the log odds ratio; multiplying both person-times by one constant leaves `IRR` unchanged; `IRR` `CM.AL` equals `CM.EL`; the codings give one
  `FE` fit; `ML` with `tau2` below `tau2tol` gives the `FE` estimates; `I2 = 100 (H2 - 1) / H2`. Properties add to the derived cases, never replace them.
- **simulation**: none. No `rma_glmm` estimator has an exact finite-sample property (unbiasedness, exact coverage) that a seeded check could assert.
- **other**: see Other software.

## Seed sources

Every DOI was checked against Crossref on 2026-11-12. Open access means a full text reachable through the connected tools; the papers the user supplies
go in `/Users/simonb/src/metafopy papers` (outside the repository, never copied in). Of the files already there, only Higgins and Thompson (2002) applies
(the typical within-study variance behind `I2` and `H2`, `i2def='1'`).

| Area | Source | Worked example | Access |
|---|---|---|---|
| All models, `CM.EL`, IRR, one-group | Stijnen T, Hamza TH, Ozdemir P (2010). Random effects meta-analysis of event outcome in the framework of the generalized linear mixed model with applications in sparse data. Stat Med 29(29):3046-3067. doi:10.1002/sim.4040 | `dat.nielweise2007` (OR, proportions), `dat.nielweise2008` (IRR) | user supplies |
| Seven models compared, `UM.FS`, `UM.RS`, `CM.AL`, `CM.EL` | Jackson D, Law M, Stijnen T, Viechtbauer W, White IR (2018). A comparison of seven random-effects models for meta-analyses that estimate the summary odds ratio. Stat Med 37(7):1059-1085. doi:10.1002/sim.7588 | Table 3 prints the measles data (inline `[case.data]`); Table 4 prints the log odds ratio, its SE and tau2 per model, outcomes 1, 2, 4 and 4b (4b drops the double-zero study, a `drop00` check) | open (CC BY, PMC5841569) |
| `UM.FS` (fixed study effects) | Simmonds MC, Higgins JPT (2016). A general framework for the use of logistic regression models in meta-analysis. Stat Methods Med Res 25(6):2858-2877. doi:10.1177/0962280214534409 | to be read | user supplies |
| `UM.RS`, GLMM framework | Turner RM, Omar RZ, Yang M, Goldstein H, Thompson SG (2000). A multilevel model framework for meta-analysis of clinical trials with binary outcomes. Stat Med 19(24):3417-3432. doi:10.1002/1097-0258(20001230)19:24<3417::AID-SIM614>3.0.CO;2-L | to be read | user supplies |
| `CM.EL`, `CM.AL` | van Houwelingen HC, Zwinderman KH, Stijnen T (1993). A bivariate approach to meta-analysis. Stat Med 12(24):2273-2284. doi:10.1002/sim.4780122405 | `dat.collins1985a` (fixed and random conditional logistic models) | user supplies |
| Poisson (`IRR`, `IRLN`) | Bagos PG, Nikolopoulos GK (2009). Mixed-effects Poisson regression models for meta-analysis of follow-up studies with constant or varying durations. Int J Biostat 5(1), article 21. doi:10.2202/1557-4679.1168 | to be read; no metafopy dataset known | user supplies |
| One-group `PLO` | Hamza TH, van Houwelingen HC, Stijnen T (2008). The binomial distribution of meta-analysis was preferred to model within-study variability. J Clin Epidemiol 61(1):41-51. doi:10.1016/j.jclinepi.2007.03.016; Zhou XH, Brizendine EJ, Pritz MB (1999). Methods for combining rates from several studies. Stat Med 18(5):557-566. doi:10.1002/(SICI)1097-0258(19990315)18:5<557::AID-SIM53>3.0.CO;2-F | `dat.pritz1997` (Zhou et al.) | user supplies |
| Quadrature, Laplace | Pinheiro JC, Bates DM (1995). Approximations to the log-likelihood function in the nonlinear mixed-effects model. J Comput Graph Stat 4(1):12-35. doi:10.1080/10618600.1995.10474663 | formulas only | user supplies |
| Conditional MLE of a common OR | Breslow N (1981). Odds ratio estimators when the data are sparse. Biometrika 68(1):73-84. doi:10.1093/biomet/68.1.73 | formulas only | user supplies |
| `I2`, `H2` | Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Stat Med 21(11):1539-1558. doi:10.1002/sim.1186 | formula for the typical variance | in the papers folder |

Package papers, for the `package` items' conventions (open): Bates D, Maechler M, Bolker B, Walker S (2015). Fitting linear mixed-effects models using
lme4. J Stat Softw 67(1). doi:10.18637/jss.v067.i01; Brooks ME et al. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated
generalized linear mixed modeling. R J 9(2):378-400. doi:10.32614/RJ-2017-066. `i2def='2'` (`1 / mean(1 / vi)`) is metafor's definition; the case
writer looks for a published source and exempts the item with that reason if none is found.

Jackson et al. (2018) computed their Table 4 with metafor 1.9-9 and `glmer`, so its values confirm the paper's numbers, not an independent
implementation; the derived cases carry the independent check (see Decisions). Published values are facts and are quoted with their citation; no
source text or tables are copied beyond the numbers and the inline data used.

### Other software

- R `stats::mantelhaen.test(exact = TRUE)` (base R) gives the conditional maximum likelihood estimate of a common odds ratio and its exact interval,
  independent of metafor: `other` cases for `CM.EL` with `FE` on `dat.nielweise2007`, `dat.collins1985a` and the Jackson data.
- R `glmmML` (Brostrom; Gauss-Hermite quadrature for random-intercept binomial and Poisson models) fits the one-group models and `CM.AL` independently
  of `lme4`. It is not in the audit's pinned library (`src/audit/data/r-versions.json`), so it goes in the library named by
  `METAFOPY_CORRECTNESS_RLIB`, as R `meta` did; nothing is installed by this spec's tasks.
- Not independent, so not used: `lme4::glmer`, `GLMMadaptive::mixed_model` and `glmmTMB::glmmTMB` (metafor calls them for the very fits in question,
  so they are what the R parity tests already compare with); `survival::clogit` and `Epi::clogistic` (metafor's `optimizer` choices); R `meta`'s
  `metabin(method = "GLMM")` and `metainc(method = "GLMM")` (they call `rma.glmm`).

`r/other_software_glmm.R` stops unless `packageVersion("glmmML")` equals the version pinned at its top, writes that version and R's into each case's
`source`, and is run as `METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_glmm.R`.
The generated values are committed; tests never run R.

## API

```python
# src/tests/correctness/glmm_formulas.py
def binomial_loglik(y: FloatArray, size: FloatArray, eta: FloatArray, link: str) -> float:
    """Binomial log-likelihood with its binomial coefficients, for the linear predictor ``eta`` under ``link``."""


def nchg_loglik(ai: FloatArray, n1i: FloatArray, ci: FloatArray, n2i: FloatArray, log_or: FloatArray) -> float:
    """Sum of Fisher's non-central hypergeometric log-probabilities of ``ai`` given the margins (van Houwelingen et al. 1993)."""


def marginal_loglik(conditional: collections.abc.Callable[[FloatArray], FloatArray], tau2: float, rule: str, points: int = 7) -> float:
    """Normal-mixture log-likelihood of a random-intercept model; ``rule`` is ``'exact'``, ``'agq'`` (``points``) or ``'laplace'``."""


def glmm_fit(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray, model: str, method: str, rule: str = 'agq', points: int = 7) -> GLMMReference:
    """The formulas' own maximum likelihood fit: estimates, standard errors from the observed information, ``tau2``, log-likelihood."""


# src/tests/correctness/test_derived.py
def glmm_likelihood_at_estimate(case: correctness.cases.Case, result: GLMMResult) -> None:
    """The formulas' log-likelihood at the reported estimates equals ``result.fit_stats_ml.ll`` and the formulas' maximum; the gradient is small."""
```

Each model gets a `glmm_fit`-style entry point (`poisson_`, `one_group_`, `conditional_` variants with the arguments of its measure); `GLMMReference` is
a frozen dataclass in `glmm_formulas.py`. Case `args` are `rma_glmm`'s own arguments, so `formula_value` passes them by name as it does for `escalc`.

## Error handling

- The harness's errors are unchanged (malformed case files raise `ValueError` naming file and case; an unknown function, dataset or attribute fails the
  case). A formula name whose module is not in `FORMULA_MODULES` is refused by the loader.
- `glmm_formulas.py` turns `scipy.integrate.IntegrationWarning` and a non-converged `scipy.optimize` result into `ArithmeticError` naming the model, so a
  reference value is never computed from a failed search.
- An exempted optimiser that no longer raises `NotSupportedError` fails `test_coverage.py`, naming the exemption to replace with a case.
- `r/other_software_glmm.R` stops when `METAFOPY_CORRECTNESS_RLIB` is unset or holds another `glmmML` version.

## Implementation

1. Harness: `FORMULA_MODULES` in `cases.py` and the import check over each module; the `rma_glmm` items in `coverage_items()`; the exemption test for
   unsupported optimisers. The gate then lists every new item as missing.
2. `glmm_formulas.py`: the likelihoods, the three integration rules, the fits, the saturated-model log-likelihoods for `QE_LRT` and the Wald statistic
   for `QE_Wld`, the typical variance for `I2`/`H2`. Unit-check the rules against each other on a dataset where all three agree to `1e-6`.
3. Derived cases in `glmm.toml`, one per coverage item, on `dat.nielweise2007`, `dat.nielweise2008`, `dat.collins1985a`, `dat.pritz1997` and the
   Jackson et al. data inline; each states its tolerance.
4. Property cases in `glmm.toml`.
5. Published cases in `glmm_published.toml`: Jackson et al. (2018) Table 4 first (open access); then each supplied paper. **Stop for the operator**
   when a source must be supplied: list it in `mismatches.md` under "Sources needed" with the table or example wanted, and go on with the rest.
6. Other-software cases from `r/other_software_glmm.R`, once the operator has made `glmmML` available outside the pinned library.
7. `mismatches.md` and its review page (shared `reviews.md` rule): every case committed as `status = "mismatch"` (strict xfail, `reason` naming both
   values, metafor's value checked with the audit's R library), every source needed and every convention difference. **Stop for the mismatch review**:
   metafopy's output changes only after the user's answer; each *follow the reference* answer becomes a `TODO.md` item.
8. Close: file the answers, set this spec's status to implemented in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pytest` (existing); no new Python dependency.
- R base `stats` and `glmmML` (pinned in the generator, outside the audit's library) for step 6 only.

## Decisions

Answered on 2026-11-13 in the [review](../../reviews/order-correctness-follow-ups-specs.md) (its D3 and D4).

1. **Kind of the Jackson et al. (2018) values: `published`, with a derived case for every item as well.** The printed values count for the gate,
   and each coverage item also has a `derived` case, so an error metafor shares is still caught by the derived likelihood check.
2. **`glmmML` as other software: installed.** `glmmML` 1.1.7 (CRAN) was installed on 2026-11-13 in the library `METAFOPY_CORRECTNESS_RLIB` names, beside R `meta`, outside the audit's pinned
   library; `r/other_software_glmm.R` pins 1.1.7 at its top. The one-group models and `CM.AL` get `other` cases from it.
