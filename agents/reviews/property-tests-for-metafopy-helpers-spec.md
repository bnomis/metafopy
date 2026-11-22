# Review: the property-tests-for-metafopy-helpers spec

**Review page:** https://claude.ai/artifact/9aMaGdbmuTYn5M6eXdHtpH
**Date:** 2026-11-22

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/property-tests-for-metafopy-helpers/spec.md` (with
`measurements.md`), for the `TODO.md` item "Numerical cores have no property-based tests". The spec gives every function under
`src/metafopy` whose domain supports a property a hypothesis test that calls it directly, as the auditme T2 question counts, numerical
kernels first. Approve or flag the spec, and answer the two decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->
- Pass: the item is still open at `7c8bfeb`. The ledger has 132 open, undismissed `src/metafopy` T2 findings, and no hypothesis test calls
  any of them.
- Pass: hypothesis (`>=6.100`) is a dev dependency and scipy a runtime one; `.hypothesis/` is in `.gitignore`.
- Pass: every function the item names exists at HEAD, including `models/sel_lik.py::map_delta` and `map_inverse`.
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **What it does:**
  1. Re-judge T2 at HEAD with the phaseb tooling (121 of the 193 candidates re-plan at `7c8bfeb`).
  2. Add shared strategies in `src/tests/rstrategies.py`.
  3. Write property tests for group A, the numerical kernels, against scipy where the finding names it.
  4. Do the same for group B, number formatting and layout.
  5. Do the same for group C, names, formulas and strings.
  6. Re-judge and close.
- **No package change.** A property that fails inside the documented domain stops its task, and the defect goes to `TODO.md`.
- **Recommendation:** approve.

## Decisions

### D1. Scope: which groups get property tests in this spec

- **What is decided:** which of the three groups in `measurements.md` this spec covers.
- **Background (2026-11-22):** the 132 split into 43 numerical kernels (A), 40 number-formatting and layout helpers (B) and 49 name, formula
  and string helpers (C). Every finding already states its property. The item's Fix direction (2026-11-21) says kernels first, then
  formatting.
- **What it affects:** the size of the task list (about 10 tasks for all three, about 5 for A alone), and whether a TODO item stays open.
- **Options:**
  - *All three groups, A then B then C (Recommended):* one list closes every `src/metafopy` T2 finding, about 130 tests.
  - *A and B; C becomes its own TODO item:* about 85 tests here, and the 49 string helpers wait for a later spec.
  - *A only; B and C become a TODO item:* about 45 tests here, and 89 helpers wait.
- **Recommendation:** all three. The properties are written already, so the work is mechanical. Group C copies R's partial-matching and
  `make.unique` rules, where a mistake is silent.

### D2. The harness T2 findings: move them to the harness item, or keep them here

- **What is decided:** where the 12 T2 findings outside the package go: 10 under `src/audit` and 2 under `bin`, counted in the ledger at
  `7c8bfeb`.
- **Background:** the next `TODO.md` item, "Audit-harness and bin helpers with branches have no direct test" (2026-11-21), covers testing
  that code for T1.
- **What it affects:** whether this item moves to `DONE.md` when the list ends, and the scope of the harness item.
- **Options:**
  - *Move them to the harness item (Recommended):* a dated note on both items records the move, and this item closes with the list.
  - *Keep them here:* this item stays open after the list, holding only the harness findings.
- **Recommendation:** move them. One harness spec then decides both its direct tests and its property tests.

## Answers

Answered 2026-11-22 on the review page (https://claude.ai/artifact/9aMaGdbmuTYn5M6eXdHtpH), each with no comment:

- **1. The spec:** approved.
- **D1. Scope:** all three groups, A then B then C.
- **D2. The harness T2 findings:** move them to the harness item.

Applied in version 1.1 of the spec (2026-11-22).
