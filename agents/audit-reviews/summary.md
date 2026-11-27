# Audit Summary

**Last updated:** 2026-11-26
**Last review:** review-2
**Total reviews conducted:** 2
**Last script run hit count:** 1081 (1081 accepted), measured 2026-11-26 at `34dc7fb`
**Auditor identity:** `{"runner": "0.3.0", "api": "1.0", "rules_path": "/Users/simonb/src/auditor-rules", "rules_commit": "f0ccb9adae7c25179aead8cea60cc12fd261dabd", "rules_dirty": false, "rules_digest": "3c331160f104734559cf82888c7c353cd677528ec952fc1e7eeb064079d7e75c"}`

## Unresolved findings

None.

## Recently resolved

| Finding | Severity | Raised in | Resolved in |
|---------|----------|-----------|-------------|
| F1. User docs and README that the code contradicts (Doc5) | medium | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F2. Specs that no longer match the code (Doc4) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F3. `InternalError` missing from `Raises:` sections (A4) | medium | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F4. Functions with branches and no direct test (T1) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F5. Harness file and subprocess calls with no envelope (A1) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F6. Names in `__all__` that nothing outside the module reads (A17) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F7. README example imports against `AGENTS.md` (Doc3) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F8. Lower-snake module-level hypothesis strategies (S2) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F9. A short module alias in a test (S4) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F10. A quadratic lookup in `to_wide` (P1) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F11. Repeated expensive computation in a loop (P3) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F12. Duplicated assembly of fit statistics and test problems (A15) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
| F13. Small fixes (A8, A9, Sec2) | low | review-2 (2026-11-26) | tasks-2 (2026-11-26) |
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

- Specs and user docs drifting from the code after a spec ships (Doc4, Doc5; reviews 1 and 2)
- Domain errors left out of `Raises:` sections, now `InternalError` reached through refits (A4; reviews 1 and 2)
- Untranslated file and subprocess errors in the audit harness (A1; reviews 1 and 2)
- Result objects and fit statistics assembled by hand in several places (A15; reviews 1 and 2)

## Checker changes since the last review

- From review-1 to review-2 the checker moved from the rules built into auditor to auditor 0.3.0 with the rules repository `/Users/simonb/src/auditor-rules` (`f0ccb9a`). It adds the sweep rules `discarded-tuple-return`, `assert-never-missing` and `pure-name-has-side-effect` (their hits cleared in `424d4a8` before the audit), new versions of every listed brief but T2's, and an A17 lister that reads every module's `__all__`. Review-2 is therefore not a convergence measurement; the next audit on unchanged rules is.
