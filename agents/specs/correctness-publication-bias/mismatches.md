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
