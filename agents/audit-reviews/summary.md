# Audit Summary

**Last updated:** 2026-11-26
**Last review:** review-2
**Total reviews conducted:** 2
**Last script run hit count:** 1082 (1082 accepted), measured 2026-11-26 at `a48c432`
**Auditor identity:** `{"runner": "0.3.0", "api": "1.0", "rules_path": "/Users/simonb/src/auditor-rules", "rules_commit": "f0ccb9adae7c25179aead8cea60cc12fd261dabd", "rules_dirty": false, "rules_digest": "3c331160f104734559cf82888c7c353cd677528ec952fc1e7eeb064079d7e75c"}`

## Unresolved findings

From review-2 (2026-11-26), waiting on its answers (review page https://claude.ai/artifact/PtmdcNjiH7cxKjPLjoUfCp).

### F1. User docs and README that the code contradicts

- **Rule or category:** Doc5
- **Severity:** medium
- **Location:** 20 places in `README.md` and `docs/` (listed in review-2)
- **First raised:** review-2 (2026-11-26)
- **Description:** Example output the code no longer prints, claims the code contradicts, and lists that read as complete but are not.
- **Recommendation:** Regenerate the output by running it, correct each claim, mark partial lists.

### F2. Specs that no longer match the code

- **Rule or category:** Doc4
- **Severity:** low
- **Location:** 39 statements in 16 implemented specs (listed in review-2)
- **First raised:** review-2 (2026-11-26)
- **Description:** "Today" sentences from before each spec shipped, changed error classes, the auditor hook URL and deploy key, drifted API blocks.
- **Recommendation:** Correct in place, or date a "today" sentence as a record.

### F3. `InternalError` missing from `Raises:` sections

- **Rule or category:** A4
- **Severity:** medium
- **Location:** `confint`, `profile`, `profile_rma_uni`, `residuals`, `vif`, `vif_rma_uni`, `sel_lik.evaluate`, `tables.prepare_tables`, `funnel`, `labbe`
- **First raised:** review-2 (2026-11-26)
- **Description:** An `InternalError` from a refit or `escalc` reaches the caller undeclared.
- **Recommendation:** Add `InternalError` to each `Raises:` section.

### F4. Functions with branches and no direct test

- **Rule or category:** T1
- **Severity:** low
- **Location:** `audit/pycall.py::shift_index`, `audit/rlib.py::accepts` and `::shift_index`, `ConfintList.__init__`, `ConfintResult.summary`, `RMAResult.hat_matrix`
- **First raised:** review-2 (2026-11-26)
- **Description:** No test calls them directly.
- **Recommendation:** Write the six tests.

### F5. Harness file and subprocess calls with no envelope

- **Rule or category:** A1
- **Severity:** low
- **Location:** 20 sites in `src/audit` (listed in review-2)
- **First raised:** review-2 (2026-11-26)
- **Description:** Judged again because their code changed after review-1 dismissed the class.
- **Recommendation:** Decision D1 of review-2 (recommended: dismiss).

### F6. Names in `__all__` that nothing outside the module reads

- **Rule or category:** A17
- **Severity:** low
- **Location:** 61 modules (listed in review-2)
- **First raised:** review-2 (2026-11-26)
- **Description:** Catch-up after the lister widened to every module; some names are documented API.
- **Recommendation:** Decision D2 of review-2 (recommended: split).

### F7. README example imports against `AGENTS.md`

- **Rule or category:** Doc3
- **Severity:** low
- **Location:** `README.md:13`
- **First raised:** review-2 (2026-11-26)
- **Description:** `import numpy as np` and `from metafopy import ...`.
- **Recommendation:** Full module paths.

### F8. Lower-snake module-level hypothesis strategies

- **Rule or category:** S2
- **Severity:** low
- **Location:** ten test files (listed in review-2)
- **First raised:** review-2 (2026-11-26)
- **Description:** Strategy constants not in UPPER_SNAKE.
- **Recommendation:** Rename.

### F9. A short module alias in a test

- **Rule or category:** S4
- **Severity:** low
- **Location:** `src/tests/test_name_matching_properties.py:40`
- **First raised:** review-2 (2026-11-26)
- **Description:** `st = hypothesis.strategies`.
- **Recommendation:** Write the module path in full.

### F10. A quadratic lookup in `to_wide`

- **Rule or category:** P1
- **Severity:** low
- **Location:** `src/metafopy/effects/reshape.py:734`, `:738`
- **First raised:** review-2 (2026-11-26)
- **Description:** `ustudy.index(s)` once per row.
- **Recommendation:** A `{study: rank}` dict.

### F11. Repeated expensive computation in a loop

- **Rule or category:** P3
- **Severity:** low
- **Location:** `src/metafopy/methods/emm_adjust.py:239`, `src/audit/generate.py:205`
- **First raised:** review-2 (2026-11-26)
- **Description:** Per-row Monte Carlo critical values; `universe.draws` inside the subject loop.
- **Recommendation:** Compute once per distinct input.

### F12. Duplicated assembly of fit statistics and test problems

- **Rule or category:** A15
- **Severity:** low
- **Location:** `rma_mv.py:1253`, `rma_ls_fit.py:1073`, `binary.py:59`; `test_rma_ls_restarts.py:80`, `:98`
- **First raised:** review-2 (2026-11-26)
- **Description:** `FitStats` formulas and `LSProblem` built by hand in several places.
- **Recommendation:** One helper; one fixture.

### F13. Small fixes

- **Rule or category:** A8, A9, Sec2
- **Severity:** low
- **Location:** `src/tests/test_harness_compare.py:372`, `src/tests/auditsnippets.py:53`, `src/audit/cli.py:36`
- **First raised:** review-2 (2026-11-26)
- **Description:** A late dataclass mutation, a load helper that also reads the pin, an unchecked `--timeout`.
- **Recommendation:** As review-2 F13 says.

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

- Specs and user docs drifting from the code after a spec ships (Doc4, Doc5; reviews 1 and 2)
- Domain errors left out of `Raises:` sections, now `InternalError` reached through refits (A4; reviews 1 and 2)
- Untranslated file and subprocess errors in the audit harness (A1; reviews 1 and 2)
- Result objects and fit statistics assembled by hand in several places (A15; reviews 1 and 2)

## Checker changes since the last review

- From review-1 to review-2 the checker moved from the rules built into auditor to auditor 0.3.0 with the rules repository `/Users/simonb/src/auditor-rules` (`f0ccb9a`). It adds the sweep rules `discarded-tuple-return`, `assert-never-missing` and `pure-name-has-side-effect` (their hits cleared in `424d4a8` before the audit), new versions of every listed brief but T2's, and an A17 lister that reads every module's `__all__`. Review-2 is therefore not a convergence measurement; the next audit on unchanged rules is.
