# Audit Summary

**Last updated:** 2026-11-25
**Last review:** review-1
**Total reviews conducted:** 1
**Last script run hit count:** every hit accepted when run on the tasks-1 branch, 2026-11-21 (run `audit.py` for the figure)
**Script version:** `d7c6456b5bcd470af77894a99cae6cbcd5f5aa5e4e7e86ad0db82d6c6493de5b`

## Unresolved findings

None. F11 and F12, the two left open by review-1, were closed through `TODO.md` (see the table below). The T1 ledger verdicts in `agents/audit-reviews/ledger/T1.jsonl` were not re-judged and are refreshed by the next audit (`DONE.md`, item "Ten `src/metafopy` functions newly flagged as having no direct test").

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
| F11. Functions with branches and no direct test (T1) | medium | review-1 (2026-11-21) | `direct-tests-for-metafopy-helpers` (2026-11-22), `direct-tests-for-the-audit-harness` (2026-11-22), `test/direct-tests-ten-metafopy` (2026-11-23) |
| F12. Pure functions with no property test (T2) | low | review-1 (2026-11-21) | `property-tests-for-metafopy-helpers` (2026-11-22) |
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
