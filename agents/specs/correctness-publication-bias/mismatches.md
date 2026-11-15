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

## Sources needed

None found by Task 4.
