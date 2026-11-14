# Correctness tests for rma_mv and robust: mismatches and sources needed

**Task list:** `agents/tasks/correctness-mv-robust/tasks-1.md` (completed in Task 12)
**Review page:** https://claude.ai/artifact/MAECYpLWXXsWsT6q2grWPk (published 2026-11-14 by the operator session)

## What is reviewed and how

Each item below is a `derived` or `published` case whose reference, coded from the cited publication, disagrees with metafopy. The case is committed
with `status = "mismatch"` (a strict expected failure) and a reason naming both values. For each item you choose whether metafopy follows the reference,
keeps metafor's behaviour, or the difference is investigated further. The two items below are the only mismatches.

## Automatic checks

Checked 2026-11-14:

- **Pass:** `pytest src/tests/correctness -m 'not slow'` passes: the two cases below are strict expected failures and nothing else fails.
- **Pass:** every `status = "mismatch"` case in `src/tests/correctness/cases/*.toml` (`robust.club.vcov.CR1S`, `robust.club.coef_test.saddlepoint`) has an item below.

## Mismatches

### CR1S: the paper's factor uses N, metafopy and metafor use N - 1

- **Case:** `robust.club.vcov.CR1S` in `src/tests/correctness/cases/robust.toml`.
- **Source:** Pustejovsky and Tipton (2018), Journal of Business and Economic Statistics 36:672-683, section 2: `c_S = sqrt(m N / ((m - 1)(N - p)))`
  (the preprint in the papers folder, `Pustejovsky_2018_Smallsample_methods_for_clusterrobust_variance_preprint.pdf`).
- **Reference:** variance of the intercept 0.220873 on `dat.tannersmith2016` (`rma_mv`, `~ 1 | studyid`, `vconst` rho 0.6, `~ aget1 + aget2`, 17
  clusters, N = 113, p = 3).
- **metafopy:** 0.218918, the reference times (N - 1) / N = 112 / 113 (every entry of the variance differs by that factor).
- **metafor:** metafopy's form, `sqrt(m (N - 1) / ((m - 1)(N - p)))`, is what the R parity cases of `src/tests/test_robust.py` and
  `src/tests/test_robust_extras.py` reproduce for clubSandwich 0.7.0; the factor is Stata's (Cameron and Miller 2015).

### Saddlepoint p-values: Kuonen's equation 3 against the Lugannani-Rice form

- **Case:** `robust.club.coef_test.saddlepoint` in `src/tests/correctness/cases/robust.toml`.
- **Source:** Kuonen (1999), Biometrika 86:929-935, equations 2 to 4, applied to the ratio of quadratic forms of the t statistic (section 5).
- **Reference:** 0.201118, 0.385678 and 0.060665 for the three coefficients of the same model (equation 3, the Barndorff-Nielsen form with `q = 0`).
- **metafopy:** 0.198807, 0.385766 and 0.060105: the Lugannani-Rice form, which the paper names after equation 3 as "usually indistinguishable in
  practice". On this data the same quadratic forms give 0.198863, 0.385664 and 0.060117 with that form, so the whole difference is the choice of form
  (metafopy's root is found to the tolerance of R's `uniroot`).
- **metafor:** the same form as metafopy; `src/tests/test_robust_extras.py` reproduces clubSandwich 0.7.0's saddlepoint p-values.
- **Not a gap in the reference:** a simulation of the quadratic form (two million draws) gives 0.1962, 0.3478 and 0.0577; both forms approximate it and
  neither is exact, so the case records the difference and does not decide which form is right.

## Decisions

### D1: CR1S, which small-sample factor does `robust` use?

- **What is decided:** whether metafopy keeps `sqrt(m (N - 1) / ((m - 1)(N - p)))` or changes to the paper's `sqrt(m N / ((m - 1)(N - p)))`.
- **Background:** the paper gives the N form; metafopy matches clubSandwich 0.7.0 (the R package by the paper's own authors), which uses N - 1, and metafor's
  `robust` follows clubSandwich. The N - 1 factor is Stata's.
  The paper introduces CR1S as the factor "some functions in Stata use", and the source it cites for Stata's practice, Cameron and Miller
  (2015), Journal of Human Resources 50:317-372, equation 12, gives that factor as `G/(G - 1) (N - 1)/(N - K)` (checked 2026-11-14 in the papers
  folder, `Cameron_2015_A_practitioners_guide_to_clusterrobust.pdf`). The printed N therefore looks like a slip in describing Stata's factor, not a
  different method.
- **What it affects:** every CR1S variance differs by 112/113 on this data; `robust.py` and the parity cases would change if the reference is followed.
- **Options:**
  - *Keep metafor (Recommended):* the case is corrected to the N - 1 factor, cites where clubSandwich documents it, and passes. No code change.
  - *Follow the reference:* a `TODO.md` item changes `robust.py`, flips the case, marks the R parity fixtures and adds a `known.toml` entry and a
    `docs/differences.md` section; metafopy then departs from clubSandwich.
  - *Investigate:* a `TODO.md` item records what is in doubt, and the case stays an expected failure.
- **Recommendation:** keep metafor, since the package of the paper's authors is the practical reference and metafopy ports metafor.

### D2: saddlepoint, which approximation form is used?

- **What is decided:** whether metafopy keeps the Lugannani-Rice form or changes to Kuonen's equation 3.
- **Background:** the paper calls the forms "usually indistinguishable in practice"; neither matches a simulation exactly, and clubSandwich 0.7.0 uses
  Lugannani-Rice.
- **What it affects:** p-values differ in the third decimal at most (0.198807 against 0.201118 at the largest).
- **Options:** the same three as D1 (*keep metafor*, *follow the reference*, *investigate*), with the same consequences.
- **Recommendation:** keep metafor, for the reason in D1.

## Sources needed

None, checked 2026-11-14. Every seed source is in the user's papers folder or was read open access (nothing saved in the repository): Assink and
Wibbelink (2016) from tqmp.org, Jackson, Riley and White (2011) from PMC3470931, Bell and McCaffrey (2002) from Statistics Canada, and SAS/STAT PROC MIXED
`DDFM=CONTAIN` from the SAS documentation (smallest rank contribution among the random effects containing the fixed effect; residual df if none does).

## Answers

### Mismatches

Copied on 2026-11-14 from the review page (both answers saved 2026-11-14T17:37:45Z, by the accept-all button or by hand; no comments).

- **D1, CR1S, which small-sample factor `robust` uses** (`robust.club.vcov.CR1S`): **keep metafor**, the recommended option. The case's reference is
  recomputed with the N - 1 factor, citing Cameron and Miller (2015) equation 12 beside the paper, and it passes. No code change.
- **D2, saddlepoint, which approximation form is used** (`robust.club.coef_test.saddlepoint`): **keep metafor**, the recommended option. The case's
  reference is recomputed with the Lugannani-Rice form the paper names after equation 3, and it passes. No code change.
