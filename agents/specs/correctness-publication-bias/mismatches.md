# Correctness tests, publication bias: mismatches and sources needed

**Spec:** `agents/specs/correctness-publication-bias/spec.md`
**Task list:** `agents/tasks/correctness-publication-bias/tasks-1.md` (Task 13 completes this document and publishes its page)
**Review page:** none yet (Task 13 publishes it)

## What is reviewed and how

The case-writing tasks compared metafopy with the defining publication first and R metafor second. This document holds every case where the reference
and metafopy disagree (one item may cover several cases with one cause), and every source that could not be read. For each mismatch you choose
*follow the reference*, *keep metafor* or *investigate*. The list stops until every item has an answer. Task 13 adds the automatic checks, the decisions
and the page; the items below are appended by the case-writing tasks as they find them.

## Mismatches

### Fail-safe N: the root search on a step function rounds up one too far

- **Cases** (strict expected failures in `src/tests/correctness/cases/bias.toml`): `bias.fsn.egger2001.rosenthal.fisher`,
  `bias.fsn.egger2001.general.target`. Companion pass case: `bias.fsn.egger2001.general.target.tight`.
- **Source:** Becker BJ (2005), Failsafe N or file-drawer number, in Rothstein et al. (eds), Publication bias in meta-analysis, doi:10.1002/0470870168.ch7
  (Fisher pooling with N studies of p = 0.5); Viechtbauer W (2024), SRSM talk "A fail-safe N computation based on the random-effects model", slides
  17 to 20 (General approach). The reference is `formulas.fsn_of`: the least integer N whose criterion holds.
- **Reference values** (`dat.egger2001` without rows 8 and 16, log risk ratios with `add = 1/2, to = "all"`, as in the talk):

  | Case | Reference | metafopy | Evidence |
  |---|---|---|---|
  | Fisher pooling, alpha 0.05 | 48 | 47 | Fisher p-value 0.0493 at N = 47, 0.0541 at N = 48 |
  | General, target log(0.95), `exact = true` | 116 | 117 | average effect -0.05112 at N = 116 (target -0.05129), -0.05154 at N = 115; metafopy gives 116 with `tol = 1e-10` |

  The talk prints 117 for the General target case, so metafopy agrees with the talk; the reference differs from it by one.
- **Cause:** metafopy mirrors metafor's root search. For Fisher pooling the number of null studies in the statistic is `int(N)` but the degrees of
  freedom use the real `N`, so the root lies below 47. For the General approach the refitted estimate is a step function of `floor(N)`; the search
  stops within its default tolerance just above the step at 116 and the result is rounded up (rounding only keeps an integer within about 1.5e-8).
- **metafor:** R metafor is not installed in the session's R (`library(metafor)` fails), so metafor's values for these two cases were not run. The
  R parity fixture `src/tests/reference/phase2/bias.json` (`hackshaw1998`, Fisher pooling 100, Rosenthal 393) agrees with the reference; the cases on
  that dataset pass.
- **Options:** follow the reference (a `TODO.md` item: make the Fisher criterion integer in N and resolve the step before rounding); keep metafor
  (the cases are corrected to metafor's convention, 47 and 117, citing the talk's printed 117); investigate (install R metafor and run both cases first).

### Selection models: a precision measure on a step function makes the likelihood not a density

- **Cases** (strict expected failures in `src/tests/correctness/cases/selmodel.toml`): `selmodel.prec.sei.bangertdrowns2004`,
  `selmodel.prec.vi.bangertdrowns2004`, `selmodel.prec.ninv.bangertdrowns2004`, `selmodel.prec.sqrtninv.bangertdrowns2004`.
- **Source:** none defines a precision measure for step functions: Hedges (1992) and Vevea and Hedges (1995) give the step function without one, and
  metafor's `?selmodel` (5.2-1) offers `prec` for the continuous types; metafor warns that adding a precision measure to a step-function model is
  "undocumented and experimental". The reference is `formulas.selection_loglik` with the weight `delta[j] / preci`, maximised by `scipy.optimize`.
- **Reference values** (`dat.bangertdrowns2004`, ML, steps 0.025, 0.5 and 1):

  | `prec` | metafopy beta | metafopy ll | reference beta | reference ll |
  |---|---|---|---|---|
  | `sei` | 0.13780 | -4.1029 | 0.10569 | -17.4217 |
  | `vi` | 0.16908 | 7.5714 | 0.10569 | -17.4217 |
  | `ninv` | 0.16769 | 7.1307 | 0.10569 | -17.4217 |
  | `sqrtninv` | 0.13711 | -4.3548 | 0.10569 | -17.4217 |

  With a consistent density the per-study precision cancels between the numerator and the normalising constant, so the reference does not depend on
  `prec`; it equals the fit of the same steps without a precision measure (passing case `selmodel.alternative.greater.bangertdrowns2004`).
- **Cause:** in `src/metafopy/models/sel_lik.py` (`ll_stepfun`) the numerator weights each p-value by `delta[j] / preci`, but the first term of the
  normalising constant has weight 1, the others `delta[j] / preci`. At metafopy's own estimate the reference log-likelihood is -21.3758 (`sei`) against
  metafopy's -4.1029; metafopy's value is reproduced exactly by a numerator `delta[j] / preci` over a constant whose first interval has weight 1.
- **metafor:** the R parity fixture `src/tests/reference/phase5/selmodel.json`, fit `hack_stepfun_sei` (`dat.hackshaw1998`, `steps = 0.025`,
  `prec = "sei"`), holds metafor's beta 0.22334, tau2 0.019944, delta 0.32096 and ll -2.0493; metafopy gives the same values, so metafor has the
  same behaviour. I did not run R metafor on the `bangertdrowns2004` cases.
- **Options:** follow the reference (a `TODO.md` item: normalise the first interval with the same weight, which changes metafopy from metafor);
  keep metafor (the four cases become exemptions "defined only in metafor's code; R parity covers it", and the continuous types carry the `prec` coverage in
  Task 6); investigate (run R metafor on these cases first).

### Selection models: `truncest` stops at a local maximum of a likelihood that is a step function of the truncation point

- **Cases** (strict expected failure in `src/tests/correctness/cases/selmodel.toml`): `selmodel.truncest.bangertdrowns2004`. The sibling case
  `selmodel.trunc.bangertdrowns2004` (truncation point fixed at 0) passes.
- **Source:** none defines an estimated truncation point. The reference is `formulas.truncation_loglik` (the truncation weight function of metafor's
  `?selmodel`, 5.2-1, after Hedges 1984), maximised by `scipy.optimize` over the coefficients, `tau2`, `delta` and the truncation point.
- **Reference values** (`dat.bangertdrowns2004`, ML):

  | Quantity | metafopy | reference |
  |---|---|---|
  | beta | 0.26491 | 0.27598 |
  | tau2 | 0.054216 | 0.054737 |
  | delta | 1.42227 | 1.48969 |
  | truncation point | 0.27073 | 0.20001 |
  | ll | -17.7909 | -17.7406 |

  The reference function reproduces metafopy's own log-likelihood at metafopy's estimate to 1e-12, so the two compute the same likelihood; the
  reference estimate has the higher value, so metafopy has not reached the maximum.
- **Cause:** the likelihood jumps at every observed effect (an effect on one side of the truncation point has weight 1, on the other weight `delta`),
  so it has a local maximum inside each stretch between two observed effects. metafopy's gradient-based search stops at one in the stretch holding its
  start value, the reference reaches a higher one at the left end of the stretch that starts at an observed effect of 0.2.
- **metafor:** the R parity fixture `src/tests/reference/phase5/selmodel.json`, fit `hack_truncest`, fixes the truncation point (`delta = [NA, 0.1]`), so
  metafor's behaviour with an estimated truncation point is not pinned by any fixture, and R metafor is not installed in the session's R.
- **Options:** follow the reference (a `TODO.md` item: search each stretch between observed effects, which changes metafopy from metafor); keep metafor
  (the case becomes an exemption "defined only in metafor's code; R parity covers it", with a note that the optimum is local); investigate (install R
  metafor and run this case first).

### Published values that differ from metafopy in the last printed digit or two (Task 7)

- **Cases** (strict expected failures in `src/tests/correctness/cases/bias.toml`): `published.borenstein2009.hackshaw1998.rosenthal`,
  `published.borenstein2009.hackshaw1998.orwin`, `published.borenstein2009.hackshaw1998.trimfill.limits`,
  `published.sutton2009.raudenbush1985.ranktest`, `published.sutton2009.raudenbush1985.trimfill.limit`, `published.sterne2005.hackshaw1998.regtest`,
  `published.sterne2005.raudenbush1985.ranktest`, `published.sterne2005.raudenbush1985.regtest`, `published.henmi2010.lee2004.dl.limit`.
- **Sources:** Borenstein et al. (2009) chapter 30 (doi:10.1002/9780470743386); Sutton (2009) chapter 23 of the Handbook of Research Synthesis and
  Meta-Analysis; Sterne and Egger (2005) Table 6.2 (doi:10.1002/0470870168.ch6); Henmi and Copas (2010) section 5 (doi:10.1002/sim.4029).
- **Reference and metafopy values:**

  | Case | Printed | metafopy |
  |---|---|---|
  | Rosenthal fail-safe N, passive smoking | 398 | 393 (metafor's parity fixture `src/tests/reference/phase2/bias.json` also 393) |
  | Orwin fail-safe N, target risk ratio 1.05 | 103 | 104 (103.9 rounded up) |
  | trim and fill limits, passive smoking | 0.085 to 0.227 (1.089 to 1.254) | 0.0842 to 0.2270 (1.0878 to 1.2548) |
  | rank test p, teacher expectancy (Sutton) | 0.08 (Z = 1.75) | 0.0740 (Z = 1.79) |
  | trim and fill lower limit, teacher expectancy | -0.045 | -0.0444 |
  | regression test p, passive smoking | 0.024 | 0.0228 |
  | rank correlation and p, teacher expectancy (Sterne) | 0.29 and 0.086 | 0.300 and 0.0740 |
  | regression test p, teacher expectancy (Sterne) | 0.055 | 0.0574 (Sutton prints 0.057) |
  | DerSimonian-Laird lower limit, acupoint review | -1.06 | -1.0664 |

- **Cause:** not investigated further. The sources were computed by other programs (a commercial package for the book; unknown for the rest), and two
  of them (Sutton and Sterne) disagree with each other on the same data. Where the same source prints other values of the same fit, those agree (the
  passing cases beside each). R metafor was not run on these cases (it is not installed in the session's R).
- **Options:** follow the reference (a `TODO.md` item per case after finding which step differs); keep metafor (the cases become exemptions or are
  loosened to the sources' digits, citing this item); investigate (install R metafor and run each first).

### Published selection-model values that differ from metafopy (Task 9)

- **Cases** (strict expected failures in `src/tests/correctness/cases/selmodel_published.toml`): the `mod1`, `mod2` and `sev2` `beta` cases and every
  `tau2` case of `published.vevea2005.cohen1981.*`, `published.vevea2005.cohen1981.unadjusted.se`, `published.vevea2005.raudenbush1985.unadjusted.se`,
  `published.hedges2005.hackshaw1998.unadjusted.spread`, `.estimated.beta`, `.estimated.weights`, `.strong1`, `.weak2`, `.strong2`,
  `published.preston2004.hahn2001.mh.ci`, `published.preston2004.hahn2001.trimfill.k0` and `published.citkowicz2017.baskerville2012.beta`.
- **Sources:** Vevea and Woods (2005), Psychological Methods 10(4):428-443, Tables 1 to 3 (doi:10.1037/1082-989X.10.4.428); Hedges and Vevea (2005),
  chapter 9 of Publication Bias in Meta-Analysis, Tables 9.2 to 9.4 (doi:10.1002/0470870168.ch9); Preston et al. (2004), J Eval Clin Pract 10(2):313-322;
  the supplementary R code of Citkowicz and Vevea (2017), run on its Table 1 data.
- **Reference and metafopy values:**

  | Case | Printed (or code output) | metafopy |
  |---|---|---|
  | Cohen, moderate one-tailed, mean and variance | 0.35 and 0.002 | 0.3577 and 0.0042 |
  | Cohen, severe one-tailed, variance (mean 0.32 agrees) | 0.005 | 0.0095 |
  | Cohen, moderate two-tailed, mean and variance | 0.35 and 0.002 | 0.3620 and 0.0028 |
  | Cohen, severe two-tailed, mean and variance | 0.32 and 0.004 | 0.3322 and 0.0057 |
  | Cohen unadjusted SE of the mean | 0.044 | 0.0447 |
  | Teacher expectancy unadjusted SE of the difference | 0.079 | 0.0731 |
  | Passive smoking unadjusted SE and variance | 0.050 and 0.019 | 0.0486 and 0.0204 |
  | Passive smoking, estimated step weights, mean (SE) | 0.13 (0.129) | 0.1217 (0.1299) |
  | Passive smoking, estimated weights and G2 (p) | 2.48, 1.01, 0.42; 7.02 (0.071) | 2.42, 0.978, 0.397; 7.066 (0.0698) |
  | Passive smoking, a priori weights, strong one-tailed | 0.12 and 0.019 | 0.1260 and 0.0175 |
  | Passive smoking, a priori weights, weak two-tailed | 0.18 and 0.018 | 0.1873 and 0.0153 |
  | Passive smoking, a priori weights, strong two-tailed | 0.14 and 0.018 | 0.1510 and 0.0092 |
  | Oral rehydration review, Mantel-Haenszel interval | 0.47 to 0.81 | 0.463 to 0.804 |
  | Oral rehydration review, trim and fill missing studies | 2 | 0 (estimator L0), 1 (R0) |
  | Citkowicz code, beta weights a and b, mean, G2 | 0.4378, 4.729, 0.1162, 6.724 | 0.4333, 4.805, 0.1128, 6.719 |

- **Cause:** not a defect found in `selmodel`. The agreeing cases beside them (all teacher expectancy selection fits, the passive smoking weak one-tailed
  fit, the Cohen severe one-tailed mean) show the model and the weights are read correctly. The disagreements are in quantities the likelihood barely
  determines: the variance component near zero (Cohen), step weights with standard errors of 1 or more (passive smoking), and the beta weights of a
  19-study fit (the supplement's own code stops at a different point from other starting values: a 1.03, b 5.50, G2 4.13 from (2, 2, 0.05, 0.3)). The
  papers' software and optimiser tolerances are not stated. No tolerance was loosened. Preston et al. print weight-function parameters (3.16, 2.66, 3.34
  for the one-tailed fits) in their own parametrisation, which does not map to metafor's `delta`, and their adjusted odds ratios appear only in a figure.
- **metafor:** R metafor is not installed in the session's R, so metafor's values for these fits were not run.
- **Options:** follow the reference (nothing to change in `selmodel` is known; a `TODO.md` item to find which optimiser setting reproduces the paper);
  keep metafor (the cases become exemptions, or are loosened to the sources' digits citing this item); investigate (install R metafor and run each first).

### Henmi-Copas interval at a DerSimonian-Laird tau2 of 0 is not the equal-effects interval (Task 10)

- **Case** (strict expected failure in `src/tests/correctness/cases/properties.toml`): `property.hc.zero-tau2.interval`. Its sibling
  `property.hc.zero-tau2.estimate` (the estimate and standard error equal the equal-effects ones) passes.
- **Source:** the spec's own property, step 4 of `agents/specs/correctness-publication-bias/spec.md`: "`hc` with a DL `tau2` of 0 gives the equal-effects
  interval". No publication states it; Henmi and Copas (2010, doi:10.1002/sim.4029, equations 14 to 21) define the interval.
- **Reference and metafopy values** (six studies with `yi` 0.30, 0.31, 0.29, 0.30, 0.32, 0.28 and `vi` 0.10, 0.08, 0.12, 0.09, 0.11, 0.10, whose
  DerSimonian-Laird `tau2` is 0):

  | Quantity | Equal-effects interval (the property) | metafopy `hc` | `formulas.hc_interval` (the paper's integral) |
  |---|---|---|---|
  | lower limit | 0.0495 | 0.0644 | 0.0644 |
  | upper limit | 0.5513 | 0.5364 | 0.5364 |

- **Cause:** not a defect in `hc`. The paper's own integral, implemented independently in `formulas.hc_interval`, gives metafopy's limits to 5 digits.
  The Henmi-Copas interval conditions on the estimated `tau2` (its quantile comes from a root search over the distribution of Q, not from the normal
  distribution), so an estimate truncated at 0 does not make it collapse to the equal-effects interval; the spec's property is a misreading of the method.
- **metafor:** R metafor is not installed in the session's R, so its interval for this case was not run; metafopy mirrors metafor's `hc` code, and the
  parity fixtures under `src/tests/reference/` do not hold a `tau2` of 0.
- **Options:** keep metafor (drop the property: the case becomes an exemption citing this item, and the estimate case stays); follow the reference
  (nothing to change in `hc`, since the reference is the spec's wrong premise: not recommended); investigate (install R metafor and run this case first).

## Decisions

Added by Task 6; Task 13 completes the section with background and the page.

### `negexppow`: how is it checked?

- **What is decided:** the spec's decisions name neither `negexppow` nor `stepcon`, so neither has a case. `negexppow` is on metafor's `?selmodel` help
  page (the weight `exp(-delta[0] * preci * p ** (1 / delta[1]))`); `src/tests/correctness/cases/selmodel.toml` holds a temporary exemption "pending the
  user's answer in mismatches.md".
- **Options:** derive from metafor's help (a `derived` case through `selmodel_fit`, as for `halfnorm`; recommended, since the help page defines it);
  exempt as code-only ("defined only in metafor's code; R parity covers it"); name a source (a publication to derive it from).

### `stepcon`: how is it checked?

- **What is decided:** `stepcon` is a step function whose weights are constrained to lie in (0, 1) (a logistic transform of each weight; metafopy's
  `src/metafopy/models/sel_spec.py`), and does not appear on metafor's help page as read for metafopy's docstring. The same temporary exemption is in
  `selmodel.toml`.
- **Options:** derive from metafor's help (only if the help defines it; check the page first); exempt as code-only (recommended: no publication or help
  page defines it, and the step-function likelihood it shares is already derived); name a source.

## Sources needed

Found by Task 7 (read 2026-11-15; PDFs in `/Users/simonb/src/metafopy papers`). No `published` case is written for these:

| Source | Why no case |
|---|---|
| Egger et al. (1997), BMJ 315:629-634, PMC2127453 | not in the folder; the PMC page fetched by the session gave only the abstract, which prints no value of `regtest` or `ranktest` |
| Sterne and Egger (2005), chapter 6, magnesium trials (text of the rank and regression tests) | the printed rank correlation, p-values and bias coefficient do not reproduce on `dat.egger2001` (for example p = 0.009 against metafopy's 0.0073 without ISIS-4), so the dataset is not the chapter's; the text extraction also lost its decimal points |
| Begg and Mazumdar (1994), examples 1 and 2 | the appendix data are printed rounded and the printed Z (1.76 and 0.19) does not reproduce (1.60 and 0.26); example 3 is a stratified test metafopy does not offer |
| Macaskill et al. (2001), Peters et al. (2006), Moreno et al. (2009), Pustejovsky and Rodgers (2019) | simulation studies with no worked example on data |
| Stanley and Doucouliagos (2014) | worked examples use data sets that are neither bundled nor printed |
| Duval and Tweedie (2000), JASA | the two examples' data are not printed and not bundled |
| Duval (2005), tables 8.3 and 8.4 | the printed overall estimates of Table 8.4 match no heterogeneity estimator, and k0 disagrees in three of the four subgroups of Table 8.3; the subgroup split of Table 8.4 also needs a range filter the harness lacks |
| Becker (2005), Fisher pooling and tables 7.2 and 7.3 | Becker's Fisher fail-safe N follows another definition (18 and 79 against metafopy's 37 and 100); the 37 rows of Table 7.2 were not typed in |
| Rosenthal (1979) large examples; Orwin (1983) | the figures depend on a mean effect over thousands of studies (not expressible as study data) and on rounding down against metafopy's rounding up (3263 against 3264; 1236 against 1237) |
| Rosenberg (2005) | no worked example reproducible by `fsn` |
| Ioannidis and Trikalinos (2007) | the study-level data of its meta-analyses are not printed |

### `selmodel` sources (checked by Task 8, 2026-11-15)

Checked against `/Users/simonb/src/metafopy papers/download_report.md` and the folder. Every other `selmodel` seed source is present, or open
(Lehmann et al. 2018 is on PMC, PMC10480976).

| Source | Citation and DOI | Table or example wanted | Status |
|---|---|---|---|
| Citkowicz and Vevea (2017) | Psychological Methods 22(1), doi:10.1037/met0000119 (needs APA access; the folder holds only the 2012 dissertation) | beta weight-function estimates on `dat.baskerville2012` | dropped 2026-11-15, supplement used (the authors' R code `MCitkowicz_suppmat.docx` and Table 1 data `MCitkowicz_suppmat_data.docx`; the case checks against that likelihood code, not printed estimates) |

Found by Task 9 (read 2026-11-15). No `published` case is written for these:

| Source | Why no case |
|---|---|
| Hedges (1992), Statistical Science 7:246-255, doi:10.1214/ss/1177011364 | the PDF in the folder is scanned images (no text layer) and no OCR tool is installed; its example data are not bundled |
| Iyengar and Greenhouse (1988), Statistical Science 3:109-117, doi:10.1214/ss/1177013012 | the same: scanned images; the data are not bundled |
| Vevea and Hedges (1995), Psychometrika 60:419-435, Tables 1 and 2 | the psychotherapy phobia subset of Smith, Glass and Miller (1980) is not bundled or printed |
| Vevea and Woods (2005), conformity example (Table 4) | the data of Cooper and Hedges (1994), Table A.1, are not bundled or printed in the paper |
| Hedges and Vevea (2005), employment interview and Tables 9.5 to 9.8 | the data of McDaniel et al. (1994) are not bundled; the Copas-Shi (a, b) models of Tables 9.5, 9.8 and 9.11 are not in metafopy |
| Hedges and Vevea (2005), teacher expectancy (Tables 9.9 to 9.11, 18 studies of Raudenbush 1984) | does not reproduce its data: the unadjusted intercept 0.349 (SE 0.079) and slope -0.371 (SE 0.089) against 0.357 and -0.378 on `dat.raudenbush1985` (19 studies, contact over one week) |
| Preston et al. (2004), Table 1 and Figure 7 | the beta estimates are in the paper's own parametrisation of the weight functions; the adjusted odds ratios are only in a figure |
| Rust et al. (1990), J Marketing Res 27:220-226; Formann (2008), Contemp Clin Trials 29:732-739 | different models (a censored normal; a left-truncated normal on normal scores); Rust's marketing data are not bundled and Formann's estimates have no counterpart in `selmodel` `trunc` |
| Lehmann et al. (2018), Evol Psychol 16(4), PMC10480976 | reports no selection-model estimate: its publication-bias analysis is funnel-plot regression (men rating women r = -.07 [-.36, .22], women rating men r = .52 [.23, .72]), and the 3PSM figures are not in the text |
