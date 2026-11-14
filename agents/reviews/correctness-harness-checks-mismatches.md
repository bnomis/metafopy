# Correctness harness checks: mismatches and sources without a usable value

**Task list:** `agents/tasks/correctness-harness-checks/tasks-1.md` (Task 4)
**Review page:** https://claude.ai/artifact/Sg3dT3n2jFbFic4kVkYcM8

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
- **Cause:** intermediate rounding in the book. With ln H and B rounded to four digits (0.3734 and 0.2343) and z = 1.96, U is 2.2994 and the
  upper limit 81.0865, printed as 81.09. Unrounded, U is 2.29930 and the limit 81.0850 (81.08 to two digits); rounding U alone to 2.2993 still
  gives 81.0849. Checked by the operator session on 2026-11-14, correcting the first explanation (U rounded to four digits), which does not
  reproduce 81.09. The difference from the printed value is 0.0051 against a tolerance of 0.005.

## Sources without a usable value

- Higgins and Thompson (2002), Table I (sclerotherapy): prints no study data, only k, Q, tau2 and I2 point values, so its H and I2 intervals cannot
  be recomputed from supplied data. `published.higgins2002.sclerotherapy` holds the point values only.

## Decisions

### Decision: what to do with the I2 upper limit that the book rounds

- **What is decided:** how the case `published.borenstein2009.ch18.d2.ci.i2ub` ends: as a recorded difference, or removed.
- **Background:** the book computes the limit from ln H and B rounded to four digits, with z = 1.96, so its last printed digit differs from the exact
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

Copied on 2026-11-14 from the review page (answer saved 2026-11-14T12:49:47Z).

- **The I2 upper limit on Borenstein et al. (2009) Dataset 2** (`published.borenstein2009.ch18.d2.ci.i2ub`, decision `d1_i2ub`): **treat it as a misprint**, the recommended option. No comment. The case is removed; the chapter's other printed values stay as cases.
