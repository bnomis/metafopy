# Correctness tests: rma_mv and robust

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-12

Found on 2026-11-12 while reviewing `TODO.md` (item "Extend the correctness tests beyond the core models"): the introduction of the
[correctness-tests spec](../correctness-tests/spec.md) says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods,
`selmodel` and `robust`", and no such spec existed, so multivariate and multilevel models and cluster-robust inference are checked only against R
metafor. A metafor bug in them passes every test. This spec extends the harness in `src/tests/correctness/` to `rma_mv` (estimation, variance structures,
inference, profile intervals) and `robust` (the plain sandwich and the `club_sandwich` small-sample corrections). **The defining publication outranks
metafor** (`NOTES.md` and `claude/rules/metafopy.md`, decided 2026-11-11): where metafopy matches metafor but not the reference, the reference wins once
the user confirms the case. Sibling specs cover `rma_glmm`, `rma_ls` and the publication-bias methods.

Everything the correctness-tests spec settles is reused unchanged: the case format, the five kinds of evidence (`published`, `derived`, `property`,
`simulation`, `other`) and their tolerances, the rule that only `published` and `derived` cases satisfy the gate, the `formulas.py` import ban, strict
xfail for `mismatch` cases, and the review of every mismatch before metafopy's output changes. This spec adds coverage items, case files, formulas, check
functions and two small harness features (a follow-on call and two ways to build `V`).

**Prerequisite:** the `TODO.md` item "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals" adds
`check` functions for back-transformed estimates, `predict` intervals and `confint` I2/H2 bounds. Only the `published` cases of `rma_mv.predict.pi`
depend on it; check that the item has moved to `DONE.md` and its `predict` check function exists before writing them, and leave them to a later task if
not (the `derived` case uses the existing `test_derived.predict_pi` pattern).

## Module location

- `src/tests/correctness/cases.py` - the `then` key and the `vblocks` and `vconst` setups (API below).
- `src/tests/correctness/formulas.py` - multivariate likelihood, variance structures and sandwich estimators (still `numpy` and `scipy` only).
- `src/tests/correctness/cases/rma_mv.toml`, `cases/robust.toml` - the new case catalogue; `cases/other_mv.toml` - generated other-software cases.
- `src/tests/correctness/test_derived.py`, `test_properties.py`, `test_simulation.py` - new check functions; `test_coverage.py` - new items.
- `src/tests/correctness/r/other_software_mv.R` - generator of `other_mv.toml`.
- `src/metafopy/models/rma_mv.py`, `src/metafopy/methods/robust.py` - option constants for the gate (no change of behaviour).
- `agents/specs/correctness-mv-robust/mismatches.md` - the review document of mismatches and sources needed.

## API

The gate reads its lists from the code. Three lists exist only as literals inside the validation today, so they become module constants that the
validation uses, and the gate reads them:

```python
# src/metafopy/models/rma_mv.py
METHODS = (*FIXED_METHODS, 'ML', 'REML')  # replaces the literal in `if method not in (...)`
TESTS = ('z', 't', 'knha', 'adhoc')  # after the 'hksj' alias is mapped to 'knha'
DFS = ('residual', 'contain')  # the string values of `dfs`
# src/metafopy/methods/robust.py
ADJUST = (False, True, 'Stata1', 'Stata2')  # 'Stata' stays an accepted alias of 'Stata1', not an item
```

Case-file additions (`cases.py`):

```python
@dataclasses.dataclass(frozen=True)
class Step:
    """A follow-on call: ``metafopy.<function>(result, **args)`` on the result of the case's call."""

    function: str
    args: dict[str, typing.Any]


# Case gains: then: Step | None = None. run_case applies it and returns the follow-on result, which `expect` and `check` then see.
# setup gains two keys besides `escalc`, each passed to the call as `V` (computed with numpy and pandas, never metafopy):
#   vblocks = { cluster = "trial", columns = ["v1i", "v2i"] }  block-diagonal V, block j = the rows of cluster j, the named columns
#   vconst = { cluster = "studyid", vi = "vi", rho = 0.8 }      V[i, l] = rho * sqrt(vi[i] * vi[l]) within a cluster, vi on the diagonal


def build_v(frame: pandas.DataFrame, setup: dict[str, typing.Any], where: str) -> FloatArray:
    """The ``V`` matrix a ``vblocks`` or ``vconst`` setup describes, rows in data order."""
```

A `robust` case is then `function = "rma_mv"` (or `"rma_uni"`) with `then = { function = "robust", args = { cluster = "district", club_sandwich = true,
vcov = "CR2" } }`. `expect` keys are the existing attribute accessors (`beta[0]`, `se[1]`, `sigma2[0]`, `rho[0]`, `QM`, `QE`, `ddf[0]`, `pval[0]`).

New `formulas.py` functions, each coded from the equations of the source named in its docstring:

```python
def struct_matrix(struct: str, inner: FloatArray, tau2: FloatArray, rho: FloatArray, dist: FloatArray | None = None) -> FloatArray:
    """G for one outer level: CS, HCS, UN, ID, DIAG, AR, HAR (Wolfinger 1993), CAR (rho ** |t_i - t_j|), the spatial structures from
    distances (Cressie 1993, section 2.3.1), GEN and GDIAG (random coefficients, Laird and Ware 1982)."""


def mv_loglik(yi: FloatArray, V: FloatArray, X: FloatArray, M_extra: FloatArray, method: str) -> float:
    """The ML or REML log-likelihood of y ~ N(X b, V + M_extra) at the GLS b (Harville 1977, equations for the restricted likelihood)."""


def mv_fit(yi: FloatArray, V: FloatArray, X: FloatArray, terms: list[RandomTerm], method: str) -> MVFit:
    """Maximise mv_loglik over the components of ``terms`` with scipy.optimize.minimize (BFGS on log variances and atanh correlations);
    returns the components, b, vb = (X' M^-1 X)^-1 and the log-likelihood."""


def cluster_robust_vcov(X: FloatArray, W: FloatArray, resid: FloatArray, cluster: IntArray, vcov: str, working: FloatArray | None = None) -> FloatArray:
    """CR0 and CR1 (Hedges, Tipton and Johnson 2010, equations 6 and 8), the Stata factors (Cameron and Miller 2015, section VI.B),
    CR2 (Bell and McCaffrey 2002; Pustejovsky and Tipton 2018, equation 6), CR3 (the jackknife form, MacKinnon and White 1985)."""


def satterthwaite_dfs(...) -> FloatArray: ...        # Tipton (2015), equation 11; Pustejovsky and Tipton (2018), section 3
def hotelling_wald(...) -> tuple[float, float, float]: ...  # HTA, HTB, HTZ, EDF, EDT of Tipton and Pustejovsky (2015), section 3
def saddlepoint_pvalue(...) -> float: ...            # Kuonen (1999), equations 2 to 4
```

`RandomTerm` and `MVFit` are small frozen dataclasses in `formulas.py`. The check functions in `test_derived.py` read the case's `random`, `struct`, `mods`
and data, build `X`, the grouping codes and the inner levels with `pandas.factorize` and a regular expression for the supported forms (`~ 1 | a`,
`~ 1 | a/b`, `~ inner | outer`, `~ inner || outer`), and call these formulas. Each `rma_mv` derived case makes three comparisons, its tolerances stated in
the case: the log-likelihood of `formulas.mv_loglik` at metafopy's components equals metafopy's (`rtol=1e-10`); `beta` and `se` of the GLS formula at
metafopy's components equal metafopy's (`1e-10`); `mv_fit`'s optimum gives components within `rtol=1e-6` and a log-likelihood no more than `1e-8`
above metafopy's (a multi-parameter optimiser, not a root of one equation).

## Coverage list

`test_coverage.coverage_items()` gains, each list read at test time:

- `rma_mv.method.<m>` for `rma_mv.METHODS`; `rma_mv.struct.<s>` for `metafopy.models.mv_structs.STRUCTS`; `rma_mv.Rscale.<r>` for `rma_mv.RSCALES`;
- `rma_mv.test.<t>` for `rma_mv.TESTS`; `rma_mv.dfs.<d>` for `rma_mv.DFS`;
- named items: `rma_mv.random.sigma2` (`~ 1 | f`), `rma_mv.random.nested` (`~ 1 | a/b`), `rma_mv.random.two_terms` (a G and an H term),
  `rma_mv.random.R` (a known correlation matrix), `rma_mv.weights` (`W`), `rma_mv.QE`, `rma_mv.QM`, `rma_mv.confint.pl` (profile intervals of each kind of
  component), `rma_mv.confint.fixed`, `rma_mv.predict.pi`;
- `robust.rma_uni` and `robust.rma_mv` (the plain sandwich on each class); `robust.adjust.<a>` for `robust.ADJUST`;
- `robust.vcov.<v>` for `metafopy.core.clubsandwich.CR_TYPES`, `robust.coef_test.<t>` for `robust.COEF_TESTS`, `robust.conf_test.<t>` for
  `robust.CONF_TESTS`, `robust.wald_test.<w>` for `metafopy.core.clubsandwich.WALD_TESTS`.

Out of scope, each a metafor extension with no defining publication or no change of the model: `lambda1`/`lambda2` penalties, `vccon`, `optbeta`
(a property case checks it reaches the same optimum, but it is not an item), `sparse`, `cvvc`, the optimiser choices, and the `rma_mv` diagnostics
(`cooks_distance`, `dfbetas`, `rstudent`, `ranef`), which a later spec can add.

## Seed sources

Every DOI was checked against Crossref on 2026-11-12 (Pustejovsky and Tipton is dated 2017 there, its online date; the issue is 2018). "Folder" means the
user's papers folder `/Users/simonb/src/metafopy papers` (outside the repository; never copied in). Values are read and recorded as the earlier spec says.

| Items | Source | Data | Access |
|---|---|---|---|
| UN, REML, ML, `QE` | Berkey CS, Hoaglin DC, Antczak-Bouckoms A, Mosteller F, Colditz GA (1998). Meta-analysis of multiple outcomes by regression with random effects. Stat Med 17:2537-2550. doi:10.1002/(SICI)1097-0258(19981130)17:22<2537::AID-SIM953>3.0.CO;2-C | `dat.berkey1998` (`vblocks`) | supply |
| sigma2, nested, CS | Konstantopoulos S (2011). Fixed effects and variance components estimation in three-level meta-analysis. Res Synth Methods 2:61-76. doi:10.1002/jrsm.35 | `dat.konstantopoulos2011` | supply |
| sigma2, nested | Assink M, Wibbelink CJM (2016). Fitting three-level meta-analytic models in R: a step-by-step tutorial. Quant Methods Psychol 12:154-174. doi:10.20982/tqmp.12.3.p154 | `dat.assink2016` | open access |
| UN, `QM`, `confint.fixed` | Kalaian HA, Raudenbush SW (1996). A multivariate mixed linear model for meta-analysis. Psychol Methods 1:227-235. doi:10.1037/1082-989X.1.3.227 | `dat.kalaian1996` | supply |
| AR, HAR, CAR | Ishak KJ, Platt RW, Joseph L, Hanley JA, Caro JJ (2007). Meta-analysis of longitudinal studies. Clin Trials 4:525-539. doi:10.1177/1740774507083567 | `dat.ishak2007` (wide; inline long data) | supply |
| UN (bivariate) | van Houwelingen HC, Arends LR, Stijnen T (2002). Advanced methods in meta-analysis: multivariate approach and meta-regression. Stat Med 21:589-624. doi:10.1002/sim.1040 | `dat.colditz1994` arm-level log odds | supply |
| UN, review | Jackson D, Riley RD, White IR (2011). Multivariate meta-analysis: potential and promise. Stat Med 30:2481-2498. doi:10.1002/sim.4172 | formulas only | open access (PMC3470931) |
| FE with known V | Gleser LJ, Olkin I (2009). Stochastically dependent effect sizes. Chapter 19 of Cooper H, Hedges LV, Valentine JC (eds), The Handbook of Research Synthesis and Meta-Analysis, 2nd ed. Russell Sage Foundation (no DOI; not checked against a registry) | tables of the chapter (inline data) | folder |
| CS, HCS, UN, ID, DIAG, AR, HAR | Wolfinger R (1993). Covariance structure selection in general mixed models. Commun Stat Simul Comput 22:1079-1106. doi:10.1080/03610919308813143 | formulas only | supply |
| spatial structures | Cressie N (1993). Statistics for Spatial Data, revised ed. Wiley. doi:10.1002/9781119115151 | formulas only | supply |
| GEN, GDIAG | Laird NM, Ware JH (1982). Random-effects models for longitudinal data. Biometrics 38:963-974. doi:10.2307/2529876 | formulas only | supply |
| `R`, `Rscale` | Hadfield JD, Nakagawa S (2010). General quantitative genetic methods for comparative biology. J Evol Biol 23:494-508. doi:10.1111/j.1420-9101.2009.01915.x; Nakagawa S, Santos ESA (2012). Methodological issues and advances in biological meta-analysis. Evol Ecol 26:1253-1274. doi:10.1007/s10682-012-9555-5 | `dat.moura2021` with its tree | supply |
| REML likelihood | Harville DA (1977). Maximum likelihood approaches to variance component estimation and to related problems. JASA 72:320-338. doi:10.1080/01621459.1977.10480998 | formulas only | supply |
| `confint.pl` | Hardy RJ, Thompson SG (1996). A likelihood approach to meta-analysis with random effects. Stat Med 15:619-629. doi:10.1002/(SICI)1097-0258(19960330)15:6<619::AID-SIM188>3.0.CO;2-A | formulas only | supply |
| `test.knha`, `test.adhoc` | Knapp G, Hartung J (2003), Stat Med 22:2693-2710. doi:10.1002/sim.1482 (univariate; the `rma_mv` form is metafor's generalisation) | formulas only | folder |
| `dfs.contain` | SAS/STAT User's Guide, PROC MIXED, `DDFM=CONTAIN` (documentation, no DOI) | formulas only | open (SAS website) |
| `robust.rma_uni`, CR1 | Sidik K, Jonkman JN (2005). A note on variance estimation in random effects meta-regression. J Biopharm Stat 15:823-838. doi:10.1081/BIP-200067915 | to check | supply (not the folder's other Sidik and Jonkman 2005 paper) |
| CR0, CR1, `robust.rma_mv` | Hedges LV, Tipton E, Johnson MC (2010). Robust variance estimation in meta-regression with dependent effect size estimates. Res Synth Methods 1:39-65. doi:10.1002/jrsm.5 | worked example, data to check | supply |
| CR2, Satterthwaite | Tipton E (2015). Small sample adjustments for robust variance estimation with meta-regression. Psychol Methods 20:375-393. doi:10.1037/met0000011 | worked example, data to check | supply |
| HTA, HTB, HTZ, EDF, EDT | Tipton E, Pustejovsky JE (2015). Small-sample adjustments for tests of moderators and model fit using robust variance estimation in meta-regression. JEBS 40:604-634. doi:10.3102/1076998615606099 | worked example, data to check | supply |
| CR2, naive-tp, Wald tests | Pustejovsky JE, Tipton E (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. J Bus Econ Stat 36:672-683. doi:10.1080/07350015.2016.1247004 | formulas only | supply (check for a preprint) |
| CR2 origin | Bell RM, McCaffrey DF (2002). Bias reduction in standard errors for linear regression with multi-stage samples. Survey Methodology 28:169-181 (no DOI) | formulas only | open (Statistics Canada) |
| CR3 | MacKinnon JG, White H (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. J Econometrics 29:305-325. doi:10.1016/0304-4076(85)90158-7 | formulas only | supply |
| Stata1, Stata2, CR1S | Cameron AC, Miller DL (2015). A practitioner's guide to cluster-robust inference. J Hum Resour 50:317-372. doi:10.3368/jhr.50.2.317 | formulas only | supply |
| saddlepoint | Kuonen D (1999). Saddlepoint approximations for distributions of quadratic forms in normal variables. Biometrika 86:929-935. doi:10.1093/biomet/86.4.929 | formulas only | supply |
| RVE worked example | Tanner-Smith EE, Tipton E, Polanin JR (2016). Handling complex meta-analytic data structures using robust variance estimates: a tutorial in R. J Dev Life Course Criminol 2:85-112. doi:10.1007/s40865-016-0026-5 | `dat.tannersmith2016` (`vconst`) | supply |

Already in the folder and applicable: Knapp and Hartung (2003), and the Handbook of Research Synthesis and Meta-Analysis (2nd ed., 2009) for Gleser and
Olkin's chapter. Every other row is supplied by the user or read open access with the connected PubMed, Paperclip and alphaXiv tools.

## Error handling

- `then` with keys other than `function` and `args`, or a `setup` with both `vblocks` and `vconst`, or either without `cluster`, raises `ValueError`
  naming the file and case id, as every malformed case does; an unknown follow-on function raises `LookupError` naming it.
- `build_v` raises `ValueError` naming the case when a `vblocks` cluster's row count differs from the number of `columns`, or a block is not symmetric.
- `mv_fit` that does not converge (scipy's `success` false, or a gradient norm above `1e-6`) fails the case naming the structure and the optimiser
  message; it is never compared.
- `r/other_software_mv.R` stops unless each package version equals the one pinned at its top, and unless `tools::package_dependencies` shows that
  none of `mvmeta`, `nlme`, `robumeta` and `sandwich` depends on metafor.

## Implementation

1. Constants: add `METHODS`, `TESTS`, `DFS` to `rma_mv.py` and `ADJUST` to `robust.py`, used by the existing validation; the existing `rma_mv` and
   `robust` tests pass unchanged.
2. Harness: `then`, `vblocks`, `vconst` in `cases.py`, with inline-file tests in `test_coverage.py` (a valid case of each, each malformed form above).
3. `formulas.py`: the structures, `mv_loglik`, `mv_fit`, GLS, the KH factor `r' M^-1 r / (k - p)` and its ad hoc floor at 1, containment degrees of
   freedom, profile bounds (Hardy and Thompson 1996: the component where the likelihood ratio statistic equals the chi-square(1) quantile, solved with
   `brentq`), and the sandwich family. Spatial structures use metafor's documented parametrisation by range where it differs from Cressie's; the case's
   source says which, as a convention, not a mismatch.
4. Derived cases in `rma_mv.toml` and `robust.toml` for every item, on the seed datasets.
5. Property cases: `rma_mv` with `~ 1 | id` (one row per id) equals `rma_uni` (beta, se, tau2, log-likelihood); a three-level fit with one level's
   sigma2 fixed at 0 equals the two-level fit; with two inner levels UN and HCS give the same fit; AR equals CAR at equally spaced times; location and
   scale invariance; REML and ML score zero at metafopy's optimum; `optbeta=True` reaches the same optimum; CR0 with one row per cluster equals the HC0
   sandwich; CR2 is exactly unbiased when the working model is the true one (Pustejovsky and Tipton 2018, theorem 1), checked in closed form from
   `(I - H) M (I - H)'`, not by simulation.
6. Simulation cases (`slow`, `numpy.random.default_rng(seed)`, 4-standard-error bounds): exact coverage of the `method="FE"` Wald interval with known
   `V`; unbiasedness of the GLS estimate.
7. Sources: fetch the open-access rows; list every other row in `mismatches.md` under "Sources needed" with the table or example wanted. **Work stops**
   until the operator supplies or drops each one.
8. Published cases from the supplied sources, one source at a time, each with its locator and date read.
9. Other software: `r/other_software_mv.R` writes `other_mv.toml` from `mvmeta` (UN on `dat.berkey1998`), `nlme` (`lme` with `varFixed(~ vi)` and
   `sigma` fixed at 1 for the three-level fit), `robumeta` (correlated-effects RVE with Tipton's correction on `dat.tannersmith2016`, compared with
   `robust(club_sandwich=True)` given robumeta's weights as `W`) and `sandwich::vcovCL` (CR0, CR1 and the Stata factors on a weighted `lm`). The values
   are committed; tests never run R.
10. Gate: add the coverage items to `test_coverage.py`; every item has a `published` or `derived` case or an `[[exempt]]` entry with a reason.
11. Review: every `mismatch` case gets an item in `mismatches.md` (source and locator, reference, metafopy and metafor values, options *follow the
    reference*, *keep metafor*, *investigate*), with its review page, per the shared `reviews.md` rule. **Work stops** until each item has an answer.
12. Close: file each *follow the reference* answer as a `TODO.md` item (fix, flip the case, mark parity fixtures, `known.toml` entry, `docs/differences.md`
    under "Where metafor departs from its reference"); set this spec's status to implemented in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pandas`, `pytest` (existing).
- R packages for step 9 only, all already in the audit's pinned library (`src/audit/data/r-versions.json`: `mvmeta` 1.0.3, `nlme` 3.1-171, `robumeta`
  2.1, `sandwich` 3.1-3); the generator reads that library through `METAFOPY_AUDIT_RLIB`, read-only, and changes no pin. Unlike R `meta` in the earlier
  spec, nothing is installed outside it. `clubSandwich` 0.7.0 is also pinned but is not used: metafor calls it and `metafopy.core.clubsandwich` ports
  it, so agreement with it is parity, not independent evidence.
- Sources through the connected literature tools; paywalled papers from the user.

## Decisions

Answered on 2026-11-13 in the [review](../../reviews/order-correctness-follow-ups-specs.md) (its D2).

**Options that only software documentation defines: derive them from the documentation.** `rma_mv` `test="knha"` and `"adhoc"`, `dfs="contain"`,
the `Rscale` values and the Stata adjustments of `robust` each get a `derived` case coded from the documented formula, its `source` naming the
documentation (metafor's help page, the SAS documentation, the Stata manual) and saying the convention is that software's. An option with no written
formula anywhere, only metafor's code, gets an `[[exempt]]` entry reading "defined only in metafor's code; R parity covers it".
