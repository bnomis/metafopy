# Correctness harness checks: mismatches and sources without a usable value

**Task list:** `agents/tasks/correctness-harness-checks/tasks-1.md`
**Review page:** to be published in Task 4

Task 3 recorded the item below; Task 4 completes this document (automatic checks, decisions, answers).

## Mismatches

### Upper limit of I2 on Borenstein et al. Dataset 2: 81.09 printed, 81.0849 computed

- **Case:** `published.borenstein2009.ch18.d2.ci.i2ub` in `src/tests/correctness/cases/published.toml` (strict xfail).
- **Source:** Borenstein et al. (2009), Introduction to Meta-Analysis, chapter 18, equation 16.25, Dataset 2 (log odds ratios).
- **Printed value:** 81.09 (%).
- **metafopy:** 81.0849 (`confint(type='ht')`, DL fit). Every other printed value of the chapter's three datasets agrees: the tau2 limits, the other
  I2 limits and the prediction intervals.
- **metafor:** not run; metafopy's `ht` interval is the metafor formula, centred on H from Q since the confint-ht-centre list.
- **Cause:** the book rounds the intermediate U to four digits (2.2993) before computing 1 - 1/U^2; from the unrounded Q the value is 81.08 to two
  digits.

## Sources without a usable value

- Higgins and Thompson (2002), Table I (sclerotherapy): prints no study data, only k, Q, tau2 and I2 point values, so its H and I2 intervals cannot
  be recomputed from supplied data. `published.higgins2002.sclerotherapy` holds the point values only.
