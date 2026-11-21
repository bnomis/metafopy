# Review: the direct-tests-for-metafopy-helpers spec

**Review page:** https://claude.ai/artifact/7QZrQWpsVZkNGr5FY6P8qq
**Date:** 2026-11-21

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/direct-tests-for-metafopy-helpers/spec.md` (with
`measurements.md`), for the `TODO.md` item "Public metafopy helpers with branches have no direct test". The spec makes every branchy function under
`src/metafopy` reached by a test, in the way the auditme T1 question counts. Approve or flag the spec, and answer the two decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-21 - a dated record, not a claim about the present -->
- Pass: the item is still open at `a6251d1`. The ledger has 162 open `src/metafopy` T1 findings, and the audit-fix queue added no direct test for
  any of them.
- Pass: nothing reads a module's `__all__` except the checker's `is_private`. No module does `import *`, `docs/reference/*.md` names each
  object explicitly, and `mypy` strict's re-export rule applies only to imported names.
- Pass: in the auditor checker, `auditme.core.is_private` treats a function as private when no `__all__` above it names it. So a helper becomes
  private by leaving its module's `__all__`, with no leading underscore.
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **What it does:**
  1. Remove the 127 internal helpers (108 functions and the classes of 19 methods) from their modules' `__all__`.
  2. Re-judge T1 at HEAD with the phaseb tooling.
  3. Write direct tests for the 30 public candidates (2 functions and 28 methods of exported result classes).
  4. Write caller tests for the private helpers no test reaches, such as `block_diagonal` and `distance_from_matrix`.
  5. Close the item.
- **No behaviour change.** A test that shows a defect stops its task, and the defect goes to `TODO.md`.
- **Recommendation:** approve.

## Decisions

### D1. Internal helpers: make them private, or test each directly

- **What is decided:** what to do with the 127 helpers that are named only in their own module's `__all__`.
- **Background (2026-11-21):** T1 counts a caller's test as covering a private function, but not a public one. These helpers are in `__all__` by
  habit. They are called across modules by full path, and nothing else reads the lists (see the automatic checks).
- **What it affects:** the `__all__` lists in many modules, and the number of new tests: about 30 direct tests plus a few caller tests,
  against about 160 direct tests.
- **Options:**
  - *Make them private (Recommended):* remove them from `__all__`. A caller test then covers each one, and direct tests go to what users
    call. A helper whose removal breaks something stays public and gets a direct test.
  - *Test each directly:* keep `__all__` as it is and write about 130 more direct tests of internal helpers.
- **Recommendation:** make them private. `__all__` should say what the package offers, and a direct test of an internal helper mostly repeats
  its caller's test.

### D2. The numerical kernels: this spec or the property-test item

- **What is decided:** whether the numerical kernels that are also on the next `TODO.md` item ("Numerical cores have no property-based tests":
  `bvn`, `dbinom`, `farebrother`, `tukey`, `nchg`, `emm_adjust`, `sel_spec`, `sel_lik`) get their tests here.
- **Background:** under D1 they become private, and caller tests then cover them for T1. A property test would be a direct test too, so it
  would answer both questions.
- **What it affects:** the size of this spec, and whether the property-test item needs a spec of its own.
- **Options:**
  - *Leave them to the property-test item (Recommended):* this spec writes no hypothesis tests, and that item's spec does them later.
  - *Fold them in:* this spec also writes property tests for those kernels. It grows by about 40 tests, and the property-test item shrinks to the
    formatting helpers.
- **Recommendation:** leave them. Property tests need their own design (domains, oracles, tolerances), and this spec stays a mechanical pass.

## Answers

Answered 2026-11-21 on the review page (https://claude.ai/artifact/7QZrQWpsVZkNGr5FY6P8qq), each with no comment:

- **1. The spec:** approved.
- **D1. Internal helpers:** make them private.
- **D2. The numerical kernels:** leave them to the property-test item.

Applied in version 1.1 of the spec (2026-11-21).
