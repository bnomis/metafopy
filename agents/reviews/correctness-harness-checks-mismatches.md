# Correctness harness checks: mismatches and sources without a usable value

**Task list:** `agents/tasks/correctness-harness-checks/tasks-1.md` (Task 4)
**Review page:** to be published by the operator session

## What is reviewed and how

Tasks 2 and 3 added published cases for back-transformed estimates, prediction intervals and the test-based intervals of tau2, I2 and H. Each case
compares metafopy with the value the source prints. One case disagrees with metafopy beyond the published tolerance and is recorded as a mismatch
below; you choose how to settle it. The list stops until the decision has an answer. The sources that print no usable value are listed for
information.

## Automatic checks

- Every `mismatch` case has an item: `grep -rn 'status = "mismatch"' src/tests/correctness/cases` finds one case,
  `published.borenstein2009.ch18.d2.ci.i2ub`, and the item below covers it. Pass.
- `pytest src/tests/correctness -m 'not slow'` on 2026-11-14: 673 passed, 4 deselected, 1 xfailed (the mismatch case, strict). Pass.

## Mismatches

### Upper limit of I2 on Borenstein et al. Dataset 2: 81.09 printed, 81.0849 computed

- **Case:** `published.borenstein2009.ch18.d2.ci.i2ub` in `src/tests/correctness/cases/published.toml` (strict xfail).
- **Source:** Borenstein et al. (2009), Introduction to Meta-Analysis, chapter 18, equation 16.25, Dataset 2 (log odds ratios).
- **Printed value:** 81.09 (%).
- **metafopy:** 81.0849 (`confint(type='ht')`, DL fit). Every other printed value of the chapter's three datasets agrees: the tau2 limits, the other
  I2 limits and the prediction intervals.
- **metafor:** not run; metafopy's `ht` interval is the metafor formula, centred on H from Q since the confint-ht-centre list.
- **Cause:** the book rounds the intermediate U to four digits (2.2993) before computing 1 - 1/U^2; from the unrounded Q the value is 81.08 to two
  digits. The difference from the printed value is 0.0051 against a tolerance of 0.005.

## Sources without a usable value

- Higgins and Thompson (2002), Table I (sclerotherapy): prints no study data, only k, Q, tau2 and I2 point values, so its H and I2 intervals cannot
  be recomputed from supplied data. `published.higgins2002.sclerotherapy` holds the point values only.

## Decisions

### Decision: what to do with the I2 upper limit that the book rounds

- **What is decided:** how the case `published.borenstein2009.ch18.d2.ci.i2ub` ends: as a recorded difference, or removed.
- **Background:** the book computes the limit from intermediate values rounded to four digits, so its last printed digit differs from the exact
  result by 0.0051. All other values of the same chapter agree with metafopy. Metafopy's value follows equation 16.25 exactly.
- **What it affects:** one case in `src/tests/correctness/cases/published.toml`; with *follow the reference* also the `ht` branch of
  `src/metafopy/methods/confint.py`, the R parity fixtures and `docs/differences.md`.
- **Options:**
  - *Follow the reference*: a `TODO.md` item to make metafopy return 81.09; the case stays a mismatch until then. This would mean reproducing the
    book's intermediate rounding, which no other source does.
  - *Keep metafor*: the case is corrected to the convention metafor documents (exact arithmetic), citing where metafor documents it, and passes.
  - *Treat it as a misprint*: the case is removed and the answer recorded here; the other printed values of the chapter stay as cases.
  - *Investigate*: the case stays a strict expected failure and a `TODO.md` item records what is in doubt.
- **Recommendation:** *treat it as a misprint*. The printed digit comes from rounding in the book, not from a different method, and metafopy and
  metafor use the exact formula; removing the one case keeps the other values of the chapter as checks.

## Answers

Left empty for the operator's verdict; each answer is copied here with its date.
