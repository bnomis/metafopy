# Correctness tests for rma_mv and robust: mismatches and sources needed

**Task list:** `agents/tasks/correctness-mv-robust/tasks-1.md` (completed in Task 12)
**Review page:** none yet (published in Task 12)

## What is reviewed and how

Each item below is a `derived` or `published` case whose reference, coded from the cited publication, disagrees with metafopy. The case is committed
with `status = "mismatch"` (a strict expected failure) and a reason naming both values. For each item you choose whether metafopy follows the reference,
keeps metafor's behaviour, or the difference is investigated further. Task 12 completes this document (automatic checks, decisions, answers).

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

## Sources needed

None so far.
