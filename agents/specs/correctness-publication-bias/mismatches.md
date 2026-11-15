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

None found by Task 4.
