# Correctness tests: publication bias

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-12

The [correctness tests](../correctness-tests/spec.md) (implemented) check `escalc`, `rma_uni`, `rma_mh` and `rma_peto` against the publications that
define them. Their introduction says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and `robust`";
on 2026-11-12 a review of `TODO.md` found that no such spec existed (item "Extend the correctness tests beyond the core models"), so the
publication-bias methods are checked only against R metafor. This spec covers them: `regtest` (Egger's regression test and its predictors),
`ranktest` (Begg and Mazumdar), `trimfill` (Duval and Tweedie), `fsn` (fail-safe N), `tes` (test of excess significance), `hc` (Henmi and Copas) and
`selmodel` (step-function, parametric and truncation selection models). The rule is unchanged (`NOTES.md`, decided 2026-11-11; `claude/rules/metafopy.md`):
the defining publication is the first oracle and metafor the second, and metafopy's output changes only after the user confirms each disagreement.
The sibling specs `correctness-mv-robust`, `correctness-glmm` and `correctness-location-scale` cover the other models.

The spec reuses the harness as it stands: the case format, the five kinds of evidence and their tolerances, the rule that only `published` and
`derived` cases satisfy the gate, `formulas.py` importing nothing from metafopy, and the mismatch flow (strict xfail, a review document, the user's
answer before any fix). It adds case files, formulas, `check` functions, a fitted-model `setup` and coverage items.

**Prerequisite:** the `TODO.md` item "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals" adds
`check` functions for a back-transformed estimate, `predict` intervals and `confint` I2 and H2 bounds. Borenstein et al. and Preston et al. print
pooled ratios, so the published cases that use them need it. Check it is in place: `grep -n 'def ' src/tests/correctness/test_derived.py` lists a
back-transform check. Until then those cases wait; the others do not.

## Module location

- `src/tests/correctness/cases/bias.toml` - `regtest`, `ranktest`, `trimfill`, `fsn`, `tes` and `hc` cases; `cases/selmodel.toml` - `selmodel` cases.
- `src/tests/correctness/formulas.py` - new re-derivations (Implementation, step 2).
- `src/tests/correctness/cases.py` - the `setup` extension (API).
- `src/tests/correctness/test_derived.py`, `test_properties.py`, `test_simulation.py` - new `check` functions and simulations.
- `src/tests/correctness/test_coverage.py` - the new coverage items.
- `src/tests/correctness/r/other_software_bias.R` - generator of the other-software values (separate from `other_software.R`, which is pinned to
  `meta` alone).
- `agents/specs/correctness-publication-bias/mismatches.md` - the review document of disagreements and sources needed.

## API

`run_case` calls `metafopy.<function>(**args, data=frame)`. That serves `regtest`, `ranktest`, `fsn` and `tes`, which take `x = "yi"`,
`vi = "vi"` and `data`; `trimfill`, `hc` and `selmodel` need a fitted model as `x`. `setup` gains two keys beside `escalc`:

```toml
setup = { where = { Gender = "Males" }, escalc = { measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i" }, rma_uni = { method = "ML" } }
```

- `where`: a table of column = value, applied to the dataset before `escalc` (the rows kept are those equal in every column); `drop`: a list of
  0-based row indices removed before `escalc` (Egger's example leaves out ISIS-4).
- `rma_uni`: arguments of `metafopy.rma_uni(yi='yi', vi='vi', data=frame, **rma_uni)`; the fit becomes the first positional argument of
  `<function>` and `data` is not passed.

```python
def case_frame(case: Case) -> pandas.DataFrame | None:
    """The case's data: the dataset or inline columns, filtered by ``setup.where`` and ``setup.drop``, with the ``escalc`` of ``setup`` applied."""


def run_case(case: Case) -> typing.Any:
    """Build the frame; with ``setup.rma_uni`` fit the model and call ``metafopy.<function>(fit, **args)``, else ``metafopy.<function>(**args, data=frame)``."""


def literal_options(func: collections.abc.Callable[..., typing.Any], parameter: str) -> tuple[str, ...]:
    """The string values of the ``Literal`` in the annotation of ``parameter``, through type aliases and ``| None``; in ``test_coverage.py``."""
```

New `check` functions (module `test_derived`), each comparing with a formula at the case's tolerance and naming case, quantity and both values:
`kendall_z` (Begg and Mazumdar's normal deviate from `tau` and k, for sources that print Z), `trimfill_filled` (the filled effects and variances,
order-free), `selmodel_fit` (maximises `formulas.selection_loglik` with `scipy.optimize` from metafopy's start values and compares `beta`, `tau2`,
`delta` and `ll`) and `hc_interval` (Henmi and Copas's interval by their integral). `fetch` is unchanged: every value checked is a result attribute
(`zval`, `pval`, `est`, `tau`, `k0`, `se_k0`, `p_k0`, `fsnum`, `O`, `E`, `X2`, `theta_lim`, `beta`, `ci_lb`, `tau2`, `delta[i]`, `LRT`).

## Coverage list

`test_coverage.py`'s `coverage_items()` adds the items below, reading each option list from the code, so a newly ported option fails the gate
until it has a case. Module constants are read with `importlib.import_module` (`metafopy.bias.regtest` as an attribute is the
function `regtest`, which the package re-exports over its submodule; checked 2026-11-12). Each needs a `published` or `derived` case (`pass` or `mismatch`) or an `[[exempt]]` entry with a reason.

- `regtest.<model>.<predictor>`: every `model` in `literal_options(metafopy.regtest, 'model')` with every predictor in
  `literal_options(metafopy.regtest, 'predictor')`; and `regtest.limit` (the limit estimate `est` and its interval).
- `ranktest.tau`, `ranktest.pval.exact` and `ranktest.pval.normal` (the two values of `exact`).
- `trimfill.k0.<estimator>` and `trimfill.se_k0.<estimator>` for every value of `literal_options(metafopy.trimfill, 'estimator')`; `trimfill.p_k0`
  (the R0 test); `trimfill.side` (the side chosen when `side=None`); `trimfill.fill` (the filled studies and the refitted estimate).
- `fsn.<type>` for every value of `literal_options(metafopy.fsn, 'type')`, and `fsn.pool.<pool>` for every value of `literal_options(metafopy.fsn, 'pool')`.
- `tes.test.<test>` for every value of `literal_options(metafopy.tes, 'test')`; `tes.E` (the powers and their sum); `tes.tau2` (the power averaged
  over heterogeneous true effects); `tes.theta_lim`.
- `hc.estimate` and `hc.ci`.
- `selmodel.<type>` for every type in `metafopy.models.sel_spec.TYPE_OPTIONS`; `selmodel.alternative.<a>` for `metafopy.models.selmodel.ALTERNATIVES`;
  `selmodel.prec.<p>` for `metafopy.models.selmodel.PRECISIONS`; `selmodel.fixed_delta` (a priori weights); `selmodel.decreasing`; `selmodel.LRT`.

Inference choices the core already covers (`test`, `method`, `level`) are not repeated. Items whose only definition is metafor itself are settled
under Decisions.

## Seed sources

Every DOI was checked against Crossref on 2026-11-12. "Folder" means the file is already in `/Users/simonb/src/metafopy papers` (outside the
repository, never copied in); "PMC" means open in PubMed Central; everything else is to be supplied by the user.

| Item | Source | Access | Worked example on a metafopy dataset |
|---|---|---|---|
| regtest lm, sei | Egger M, Davey Smith G, Schneider M, Minder C (1997). Bias in meta-analysis detected by a simple, graphical test. BMJ 315:629-634. doi:10.1136/bmj.315.7109.629 | PMC2127453 | Check the table for the magnesium review (`dat.egger2001`) |
| regtest rma | Sterne JAC, Egger M (2005). Regression methods to detect publication and other bias in meta-analysis. In Rothstein, Sutton, Borenstein (eds), Publication Bias in Meta-Analysis, Wiley, 99-110. doi:10.1002/0470870168.ch6 | supply | to check |
| regtest ni, ninv | Macaskill P, Walter SD, Irwig L (2001). Stat Med 20:641-654. doi:10.1002/sim.698; Peters JL et al. (2006). JAMA 295:676-680. doi:10.1001/jama.295.6.676 | supply | none known |
| regtest vi, limit | Moreno SG et al. (2009). BMC Med Res Methodol 9:2. doi:10.1186/1471-2288-9-2; Stanley TD, Doucouliagos H (2014). Res Synth Methods 5:60-78. doi:10.1002/jrsm.1095 | PMC2649158; supply | to check |
| regtest sqrtninv | Pustejovsky JE, Rodgers MA (2019). Res Synth Methods 10:57-71. doi:10.1002/jrsm.1332 | supply | to check |
| ranktest | Begg CB, Mazumdar M (1994). Operating characteristics of a rank correlation test for publication bias. Biometrics 50:1088-1101. doi:10.2307/2533446 | supply | to check |
| trimfill L0, R0 | Duval S, Tweedie R (2000). Trim and fill. Biometrics 56:455-463. doi:10.1111/j.0006-341X.2000.00455.x; Duval S, Tweedie R (2000). A nonparametric "trim and fill" method. JASA 95:89-98. doi:10.1080/01621459.2000.10473905 | supply | to check |
| trimfill Q0 | Duval S (2005). The trim and fill method. In Rothstein et al. (eds), 127-144. doi:10.1002/0470870168.ch8 | supply | to check |
| fsn Rosenthal | Rosenthal R (1979). The file drawer problem and tolerance for null results. Psychol Bull 86:638-641. doi:10.1037/0033-2909.86.3.638 | supply | none |
| fsn Orwin | Orwin RG (1983). A fail-safe N for effect size in meta-analysis. J Educ Stat 8:157-159. doi:10.3102/10769986008002157 | supply | none |
| fsn Rosenberg | Rosenberg MS (2005). The file-drawer problem revisited. Evolution 59:464-468. doi:10.1111/j.0014-3820.2005.tb01004.x | supply | to check |
| fsn Fisher pool | Becker BJ (2005). Failsafe N or file-drawer number. In Rothstein et al. (eds), 111-125. doi:10.1002/0470870168.ch7 | supply | to check |
| fsn General | Viechtbauer W (2024). A fail-safe N computation based on the random-effects model. SRSM talk, https://www.wvbauer.com/lib/exe/fetch.php/talks:2024_viechtbauer_srsm_fail_safe_n.pdf (no DOI) | open (URL resolved 2026-11-12) | to check |
| tes | Ioannidis JPA, Trikalinos TA (2007). An exploratory test for an excess of significant findings. Clin Trials 4:245-253. doi:10.1177/1740774507079441 | supply | none known |
| hc | Henmi M, Copas JB (2010). Confidence intervals for random effects meta-analysis and robustness to publication bias. Stat Med 29:2969-2983. doi:10.1002/sim.4029 | supply | `dat.lee2004` (metafor's help says the funnel plot is "as in Henmi and Copas") |
| stepfun | Iyengar S, Greenhouse JB (1988). Stat Sci 3:109-117. doi:10.1214/ss/1177013012; Hedges LV (1992). Stat Sci 7:246-255. doi:10.1214/ss/1177011364; Vevea JL, Hedges LV (1995). Psychometrika 60:419-435. doi:10.1007/BF02294384 | supply (Project Euclid access not confirmed) | to check |
| fixed delta | Vevea JL, Woods CM (2005). Sensitivity analysis using a priori weight functions. Psychol Methods 10:428-443. doi:10.1037/1082-989X.10.4.428 | supply | `dat.cohen1981` (metafor's help reproduces its table) |
| stepfun, selection | Hedges LV, Vevea J (2005). Selection method approaches. In Rothstein et al. (eds), 145-174. doi:10.1002/0470870168.ch9 | supply | to check |
| beta | Citkowicz M, Vevea JL (2017). A parsimonious weight function for modeling publication bias. Psychol Methods 22:28-41. doi:10.1037/met0000119; truncated beta: Pustejovsky (2024), https://jepusto.com/posts/beta-density-selection-models/ | supply; open | `dat.baskerville2012` |
| halfnorm, negexp, logistic, power | Preston C, Ashby D, Smyth R (2004). Adjusting for publication bias: modelling the selection process. J Eval Clin Pract 10:313-322. doi:10.1111/j.1365-2753.2003.00457.x | supply | `dat.hahn2001` |
| trunc, truncest | metafor's `?selmodel` (5.2-1); similar models: Rust RT et al. (1990). J Mark Res 27:220-226. doi:10.1177/002224379002700209; Formann AK (2008). Contemp Clin Trials 29:732-739. doi:10.1016/j.cct.2008.05.004 | supply | none |
| several | Borenstein M et al. (2009). Introduction to Meta-Analysis, ch. 30. doi:10.1002/9780470743386 | folder | `dat.hackshaw1998` (37 studies; prints Rosenthal 398, Orwin 103, Begg, Egger, trim and fill) |
| several | Sutton AJ (2009). Publication bias. Ch. 23 of Cooper, Hedges, Valentine (eds), The Handbook of Research Synthesis and Meta-Analysis, 2nd ed. (no DOI) | folder | `dat.raudenbush1985` (rank test, Egger, Rosenthal, trim and fill L0) |
| data only | Lehmann GK et al. (2018). Evol Psychol 16(4). doi:10.1177/1474704918802412 (3PSM, `dat.lehmann2018`) | PMC10480976 | to check |

Before writing each case, read the value and locator from the source and confirm the data match the dataset. The earlier spec found that
`dat.curtis1998` did not reproduce its paper; such a source gives no case. `docs/differences.md` records that `selmodel(type='beta')` on
`dat.baskerville2012` has a flat likelihood, so Citkowicz and Vevea's values may not be reproducible.

## Error handling

- `setup` with a key other than `where`, `drop`, `escalc`, `rma_uni`, a `where` column the dataset lacks, or a `drop` index out of range raises
  `ValueError` naming file and case.
- `literal_options` on a parameter whose annotation is not a `Literal` (or `Literal | None`) raises `TypeError` naming the function and parameter,
  so a refactor that removes the `Literal` fails the gate instead of shrinking it.
- `selmodel_fit` fails the case if `scipy.optimize` does not converge, naming the case and the optimiser's message; it never passes silently.
- A metafopy call that raises (`metafopy.errors.InputError`, `EstimationError`) fails that case with the exception, as now.
- `r/other_software_bias.R` stops unless each package version equals the one pinned at its top, and writes the package and R versions into each
  generated case's `source`.

## Implementation

1. Harness: the `setup` keys `where`, `drop` and `rma_uni` in `cases.py`; `literal_options` and the new items in `test_coverage.py` (the gate fails
   at once with every new item listed). Tests of the harness for each new error. Commit.
2. `formulas.py`, from the publications only: Egger's regression in its original form (standard normal deviate on precision, ordinary least squares),
   so it checks metafopy's weighted form independently; the predictors as published; Begg and Mazumdar's standardised deviates and Kendall statistic;
   the L0, R0 and Q0 estimators and their standard errors, and the R0 null distribution; Rosenthal (Stouffer), Fisher pooling, Orwin, Rosenberg;
   TES powers, the chi-square, binomial and exact (Poisson-binomial) tests; Henmi and Copas's interval; and one general selection-model
   log-likelihood (Iyengar and Greenhouse; Vevea and Hedges 1995) taking a weight function, with the weight functions of each published type. Then
   the `derived` cases, one or more per coverage item. Tolerances: closed forms `1e-10`; iterative results `1e-8`; `selmodel_fit` states its own per
   case (the optimum of a selection likelihood is not reached to `1e-8` in the parameters; `ll` is compared at `1e-8`).
3. Published cases from the folder (Borenstein ch. 30, Sutton ch. 23) and the open sources first, then the supplied ones, one source at a time with
   locator and date. A source that uses another convention (a two-tailed alpha for Rosenthal; Peters' weights rather than `1/vi`) sets the case's
   `args` to that convention where metafopy offers it, and the case's `source` says so; where metafopy cannot express it, it is a mismatch item.
   Points to read with care, from the code as of 2026-11-12: `trimfill` computes `p_k0` for R0 as `0.5 ** (k0 + 1)`; check it against the R0 null
   distribution of Duval and Tweedie (2000, JASA).
4. Properties: `selmodel` with every `delta` fixed at no selection equals the ML `rma_uni` fit; reflecting `yi` to `-yi` negates `regtest`'s `zval`
   and the limit estimate, keeps `ranktest`'s p-value, and moves `trimfill`'s missing studies to the other side with the same `k0`; location shifts
   leave `ranktest` unchanged; `trimfill` with `k0 = 0` returns the input model; `hc` with a DL `tau2` of 0 gives the equal-effects interval.
5. Simulation (`slow`, `numpy.random.default_rng(seed)`), exact results only: the size of `regtest(model='lm')` under the weighted normal linear model
   it assumes; the size of the exact TES test, at most `alpha`, with `theta` known.
6. Other software, generated by `r/other_software_bias.R` with `LC_ALL=C.UTF-8 Rscript` against the library named by `METAFOPY_CORRECTNESS_RLIB`
   (outside the audit's pinned library in `src/audit/data/r-versions.json`, which this spec does not change); the values are committed and tests never
   run R. **The work stops for the operator** to install the packages there (External dependencies).
7. `mismatches.md` with its review page, as the earlier spec's: each mismatch with source and locator, the reference, metafopy's and metafor's values,
   and the options *follow the reference*, *keep metafor* or *investigate*; and the table of sources needed. **The work stops for the operator twice**:
   after step 2, when the paywalled sources must be supplied before step 3 can finish, and here, until every mismatch has an answer.
8. Close: copy the answers into `mismatches.md`, file each *follow the reference* answer as a `TODO.md` item (fix, flip the case, mark the parity
   fixture, `known.toml`, `docs/differences.md`, as the earlier spec lays out), set this spec's status to implemented and update `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pytest`, `tomllib` (existing).
- For step 6, in `METAFOPY_CORRECTNESS_RLIB`, each pinned in the generator:
  - R `weightr` 2.0.2 (CRAN, published 2019-07-06; depends on `ggplot2` and `scales`, not metafor): Vevea-Hedges step-function models, estimated and
    a priori weights. Independent of metafor, so its cases check `selmodel` stepfun.
  - R `meta` 8.5-0 (the version `other_software.R` already pins): `metabias` (Egger, Begg, Peters, Harbord) and `trimfill` (L0, R0) are meta's own
    code, but meta imports metafor and takes tau2 from `rma.uni`, so a random-effects refit checks only the pipeline around tau2.
  - Not used: R `puniform` 0.2.8 imports metafor and implements p-uniform methods metafopy does not port; `metasens` (Copas, limit meta-analysis)
    covers no function here.
- Source access through the connected PubMed tools and the user's folder.

## Decisions

Answered on 2026-11-13 in the [review](../../reviews/order-correctness-follow-ups-specs.md) (its D2 and D4).

**Items defined only by metafor: documented ones derived, code-only ones exempt.** `trunc`, `truncest` and `tes.theta_lim`, which metafor's help page
defines in formulas, get `derived` cases whose `source` names that help page. `fsn` type `Binomial` and the `selmodel` types `halfnorm1`, `negexp1`,
`logistic1`, `power1`, `halfnorm2`, `negexp2`, `logistic2` and `power2` get `[[exempt]]` entries reading "defined only in metafor's code; R parity
covers it".

**R `weightr` 2.0.2: installed** on 2026-11-13, with its dependencies (`ggplot2`, `scales` and theirs), in the library `METAFOPY_CORRECTNESS_RLIB` names, beside R `meta`, outside the audit's pinned library, so the
step-function `selmodel` cases have an independent check.
