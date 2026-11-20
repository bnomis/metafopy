# Correctness tests for rma_uni diagnostics: mismatches and sources needed

**Task list:** `agents/tasks/correctness-uni-diagnostics/tasks-1.md`
**Spec:** `agents/specs/correctness-uni-diagnostics/spec.md`

## What is reviewed and how

Each item in a mismatch record would be a `derived` or `published` case whose reference, coded from the cited publication, disagrees with
metafopy; the case would be committed with `status = "mismatch"` (a strict expected failure) and a reason naming both values, and for each item
you would choose whether metafopy follows the reference, keeps metafor's behaviour, or the difference is investigated further. This work has no
such case, so there is nothing to review.

## Automatic checks

Checked 2026-11-20:

- **Pass:** `pytest src/tests/correctness -m 'not slow'` passes with no expected failure from this task list.
- **Pass:** no case in `src/tests/correctness/cases/rma_uni_diagnostics.toml` or `src/tests/correctness/cases/properties.toml` has
  `status = "mismatch"`, so no item is owed.
- **Pass:** each case's `source.locator` was checked against the extracted text of its paper in `/Users/simonb/src/metafopy papers/`:
  Viechtbauer and Cheung (2010) equations 13 to 16, 19, 20, 23 to 25 and 27, section 5.1 with Table I, section 5.2 with Table II; Henderson (1975)
  equation 5 and the text after equation 8; Searle, Casella and McCulloch (1992) section 7.6, equations 52 and 53 (Henderson's mixed model
  equations); Raudenbush (2009) Table 16.2, section 16.5.6.1 and equations 16.23 to 16.25. No locator needed correction.

## Mismatches

None.

## Decisions

None.

## Answers

None: no answer needed.
