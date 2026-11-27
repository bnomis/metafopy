# Review: the coefficient-names-and-print-tests-follow-r spec

**Review page:** https://claude.ai/artifact/4Vfupvo5ADv27gJf2EqpvD
**Date:** 2026-11-27

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/coefficient-names-and-print-tests-follow-r/spec.md`, for the
`TODO.md` item "`rma_mv` prints some coefficient labels unlike R, and the print tests skip them for a partly stale reason". Measuring the five
skipped cases showed one engine difference (the order of variables inside an interaction term) and four test-side causes (a test writing `C()` where
its R script writes `factor()`, a print width that differs from the width R printed at, a flat-likelihood digit, and a stale skip). The one open
question, how `C()` names its columns, was answered on 2026-11-27 (keep `allocrandom`). Approve or flag each item below; there are no open decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-27 - a dated record, not a claim about the present -->
- Pass: the item is still open at `e9a5f1d`: `NAMES_DIFFER` (`src/tests/test_rma_mv.py:186`) and the `uni_cs_btt` skip
  (`src/tests/test_robust.py:160`) are in place, and `docs/differences.md` records none of the differences.
- Pass: R 4.6.2's `model.matrix` and metafopy's `design_matrix` give the same names on sixteen formulas, except the three that write an interaction
  against first-appearance order.
- Pass: with `factor(deltype)` and the print width at 80, `ass_mods`, `ass_mods_t` and `kon_btt` print exactly as R; at width 200,
  `uni_cs_btt` prints exactly as R.
- Pass: at width 80 every other `rma_mv` print case not in `PRINT_UNSTABLE` still matches R, as it does at 200.
- Pass: `ber_un_mods`'s REML log-likelihood equals R's to 4e-12; its `QM` (76.708512 against R's 76.708353) sits either side of 76.70845.
- Pass: the spec has no non-ASCII characters and no line over 160 characters.
<!-- doc-counts: accept-end -->

## The items

### S1. Interaction terms order their variables as R does

- **What:** a new `metafopy.formula.r_term_order`, called by `design_matrix` before formulaic builds the matrix, orders each interaction term's
  variables by their first appearance in the formula, as R's `terms.formula` does; `~ 0 + outcome + I(year - 1985):outcome` then names its columns
  `outcomeAL:I(year - 1985)`.
- **Output change:** the names (and, for two factors, the column order) of an interaction written out of that order; estimates and statistics are
  unchanged. Among the repository's tests and docs only `ber_un_mods` is affected (2026-11-27). Approving this item is the confirmation
  `claude/rules/metafopy.md` asks for before output changes.
- **Checked by:** a new R-generated fixture of `model.matrix` names and matrices (`src/tests/r/generate_formula_names.R`,
  `src/tests/test_formula_names.py`), and direct tests of `r_term_order`.
- **Recommendation:** approve.

### S2. The `rma_mv` print tests compare like with like

- **What:** `ass_mods` and `ass_mods_t` write `factor(deltype)` as their R script does; `NAMES_DIFFER` goes; the module prints at width 80, the width
  `generate_rma_mv.R` printed at; the `conftest.py` docstring that says every fixture was printed at 200 is corrected.
- **Recommendation:** approve.

### S3. `ber_un_mods`'s printout joins `PRINT_UNSTABLE`

- **What:** a flat likelihood, not a defect (see the checks); its names stay checked through `coef_names` in `test_fit`. Chosen in writing the
  spec, reversible.
- **Recommendation:** approve.

### S4. The `uni_cs_btt` and `kon_btt` skips go

- **What:** both print exactly as R at the width of their fixtures.
- **Recommendation:** approve.

### S5. `docs/differences.md` records the `C()` naming

- **What:** a "By design" section: `C(alloc)` names its columns `allocrandom` where R gives `C(alloc)random` (the operator's answer of
  2026-11-27), with R and Python run side by side.
- **Recommendation:** approve.

### S6. The no-main-effect two-factor interaction is filed, not fixed

- **What:** `~ deltype:outcome` gives R seven columns and metafopy six (contrast coding, measured 2026-11-27); it becomes its own `TODO.md` item
  once the spec is approved.
- **Recommendation:** approve.

## Answers

Answered on the review page on 2026-11-27 (03:10 UTC), copied here the same day:

- S1: approved, no comment.
- S2: approved, no comment.
- S3: approved, no comment.
- S4: approved, no comment.
- S5: approved, no comment.
- S6: approved, no comment.
