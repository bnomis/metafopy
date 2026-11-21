# Audit Summary

**Last updated:** 2026-11-21
**Last review:** review-1
**Total reviews conducted:** 1
**Last script run hit count:** 1265 (1265 accepted), measured 2026-11-21 at `31d809f`
**Script version:** `d7c6456b5bcd470af77894a99cae6cbcd5f5aa5e4e7e86ad0db82d6c6493de5b`

## Unresolved findings

All of these were first raised in review-1 (2026-11-21). Per-candidate locations for the listed questions are in `agents/audit-reviews/ledger/<question>.jsonl`, and the full lists are in [review-1](review-1.md).

### F1. User-guide output and claims the code contradicts

- **Rule or category:** Doc5
- **Severity:** medium
- **Location:**
  - `docs/getting-started.md`
  - `docs/guide/*.md`
  - `docs/differences.md`
  - `docs/differences-notes.md`
- **First raised:** review-1 (2026-11-21)
- **Description:** example output that the code no longer prints, and statements the code contradicts.
- **Recommendation:** re-run the examples, update the pages, and keep the dated records as they are.

### F2. Domain errors missing from `Raises:` sections

- **Rule or category:** A4
- **Severity:** medium
- **Location:** see the A4 block in review-1
- **First raised:** review-1 (2026-11-21)
- **Description:** errors that reach the caller through instance methods and dispatch are not declared.
- **Recommendation:** add or remove the names; the `singledispatch` generics wait on decision D1.

### F3. `assert` used as runtime validation

- **Rule or category:** A12
- **Severity:** medium
- **Location:** `src/metafopy/plot/qqnorm.py:242`, `src/metafopy/methods/profile.py:309`
- **First raised:** review-1 (2026-11-21)
- **Description:** asserts that disappear under `python -O`.
- **Recommendation:** replace them with `InputError` checks.

### F4. README contradicts AGENTS.md

- **Rule or category:** Doc3
- **Severity:** medium
- **Location:** `README.md:52`
- **First raised:** review-1 (2026-11-21)
- **Description:** how fixtures are made and which oracle comes first.
- **Recommendation:** make the README match `AGENTS.md`.

### F5. `nbformat` imported but not declared

- **Rule or category:** D1
- **Severity:** medium
- **Location:** `src/tests/test_showcase_notebook.py:6`
- **First raised:** review-1 (2026-11-21)
- **Description:** it arrives only as a dependency of another package.
- **Recommendation:** declare it in the dev group.

### F6-F20. Lower-severity groups

- **Rule or category:**
  - Doc4, A3, A11, A1, A17, T1, T2;
  - A5-A25, P1, P3, Sec2, Sec4, S1, S4, Doc1;
  - spec review.
- **Severity:** low, except T1 (medium) and the bootstrap RNG under S4 (medium)
- **Location:** see review-1
- **First raised:** review-1 (2026-11-21)
- **Description:** see review-1.
- **Recommendation:** as review-1 recommends, once its decisions are answered.

## Recently resolved

| Finding | Severity | Raised in | Resolved in |
|---------|----------|-----------|-------------|
| (none yet) | | | |

## Recurring patterns

- (none yet: this is the first review)

## Checker changes since the last review

- (none: first review)
