# Correctness tests: mismatches and sources needed

**Spec:** `agents/specs/correctness-tests/spec.md`
**Status:** items are being collected by the case-writing tasks; Task 13 of `agents/tasks/correctness-tests/tasks-1.md` completes this document in the
review-document layout and publishes its page.

## Mismatches

### rma_uni confint type "ht" with a REML fit centres on a different H

- **Case:** `confint.ht.reml.bcg.none` in `src/tests/correctness/cases/inference.toml` (strict xfail).
- **Source:** Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine 21:1539-1558, equations 24 and 25.
- **Reference value** (derived, `formulas.higgins_thompson_bounds`, on `dat.bcg` with log risk ratios): tau2 interval 0.19972 to 0.47038. The interval is
  `ln H +- z se` with H from Q (`H2 = Q / (k - 1)` = 12.686), as the paper defines it.
- **metafopy:** 0.20274 to 0.47703. Its bounds are centred on the `H2` of the REML fit (12.856), with the standard error computed from Q. With a DL fit
  (where `H2` equals `Q / (k - 1)`) the two agree, case `confint.ht.dl.bcg.none` passes.
- **metafor:** not checked. R `metafor` is not installed in the library this session can reach, and `src/tests/reference/` holds no `ht` fixture.
- **Options:** follow the reference; keep metafor (the REML-centred form is a deliberate generalisation of the paper); investigate (run metafor, then
  decide).
