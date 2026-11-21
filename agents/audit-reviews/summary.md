# Audit Summary

**Last updated:** 2026-11-21
**Last review:** review-1
**Total reviews conducted:** 1
**Last script run hit count:** every hit accepted when run on the tasks-1 branch, 2026-11-21 (run `audit.py` for the figure)
**Script version:** `d7c6456b5bcd470af77894a99cae6cbcd5f5aa5e4e7e86ad0db82d6c6493de5b`

## Unresolved findings

Both were first raised in review-1 (2026-11-21). Per-candidate lists are in `agents/audit-reviews/ledger/T1.jsonl` and `T2.jsonl`; the full lists are in [review-1](review-1.md).

### F11. Functions with branches and no direct test

- **Rule or category:** T1
- **Severity:** medium
- **Location:** see the T1 block in review-1
- **First raised:** review-1 (2026-11-21)
- **Description:** public helpers that the tests reach only through a higher-level function.
- **Recommendation:** tracked in `TODO.md` (decision D4 of review-1).

### F12. Pure functions with no property test

- **Rule or category:** T2
- **Severity:** low
- **Location:** see the T2 block in review-1
- **First raised:** review-1 (2026-11-21)
- **Description:** pure functions whose stated property no test checks.
- **Recommendation:** tracked in `TODO.md` (decision D5 of review-1).

## Recently resolved

| Finding | Severity | Raised in | Resolved in |
|---------|----------|-----------|-------------|
| F1. User-guide output and claims the code contradicts (Doc5) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F2. Domain errors missing from `Raises:` sections (A4) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F3. `assert` used as runtime validation (A12) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F4. README contradicts AGENTS.md (Doc3) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F5. `nbformat` imported but not declared (D1) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F6. Specs that no longer match the code | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F7. Standard-library errors where a domain error exists | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F8. Literal chains without `assert_never` | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F9. File and subprocess calls with no translating envelope | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F10. Package-surface drift | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F13. Small design defects | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F14. Duplicated construction of result objects | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F15. Same value passed to two slots | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F16. Rescanning loops | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F17. Unchecked inputs | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F18. Project rules the sweep does not check (incl. bootstrap RNG, S4) | medium | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F19. Other low-cost fixes | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |
| F20. Findings that need no change | low | review-1 (2026-11-21) | tasks-1 (2026-11-21) |

## Recurring patterns

- (none yet: this is the first review)

## Checker changes since the last review

- (none: first review)
