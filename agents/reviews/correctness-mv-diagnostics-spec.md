# Review: the correctness-mv-diagnostics spec

**Review page:** https://claude.ai/artifact/SzKEUeXLhMxfvYgmjuNAfu
**Date:** 2026-11-19

**What is being reviewed and how.** One new spec, `agents/specs/correctness-mv-diagnostics/spec.md` (with `measurements.md`), written by `/todo-do`
for the rma_mv part of the `TODO.md` item "Correctness coverage the extension specs left for later". It adds correctness cases for the `rma_mv`
diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef`, which only R parity fixtures check today. No metafopy code changes. Approve or flag
the spec, and answer the one decision.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
- Pass: every name the spec builds on exists (`MV_COMPONENT_TOLERANCE`, `MV_TIGHT_TOLERANCE`, `PROPERTY_TOLERANCE`, `mv_setup`, `formulas.mv_fit`,
  the cases `mv.struct.gen.slopes`, `mv.rscale.cor.moura`, `mv.two_terms.berkey`, `mv.struct.un.berkey`).
- Pass: measured at `93aace4`, metafopy matches the brute-force deletion references to 4.8e-8 (re-estimated) and 2.9e-11 (`reestimate=False`),
  and the mixed-model-equation BLUPs to 5e-15, except the one expected mismatch below.
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **References:** no publication defines the `rma_mv` versions, so every case is `derived`: explicit leave-one-row and leave-one-cluster refits
  for Cook's distance, DFBETAS and rstudent (Viechtbauer and Cheung 2010 define the rma.uni forms; Christensen, Pearson and Johnson 1992 case-set
  deletion), and Henderson's mixed model equations for `ranef` (a different algorithm from metafor's marginal formula). Property cases: one row per
  id reduces to `rma_uni`; a one-row cluster's `X2` is its `z` squared; fixed components make both `reestimate` settings agree.
- **Expected mismatch:** `ranef` on a random-coefficients (GEN) model with a second term crossing its clusters differs from the BLUP by up to 0.031
  (largest BLUP 0.453); R metafor 5.2-1 gives metafopy's values (it uses only the within-cluster block of `M^-1`). It becomes a strict expected
  failure for review, not a code change.
- **Stops in the task list:** step 4 stops until the papers marked "supply" are given or dropped: Viechtbauer and Cheung (2010,
  doi:10.1002/jrsm.11), Christensen, Pearson and Johnson (1992, doi:10.2307/1269550), Henderson (1975, doi:10.2307/2529430), Searle, Casella and
  McCulloch (1992, Variance Components, chapter 7). Robinson (1991, doi:10.1214/ss/1177011926) is open access. Putting them in the papers folder
  before the run lets that step run straight through. Step 5 stops for the mismatch review.
- **Recommendation:** approve.

## Decisions

### D1. Coverage of the rma_uni diagnostics

- **What is decided:** whether the `rma_uni` versions of the same diagnostics, which also have no correctness case (checked 2026-11-19), go in this
  spec or a separate item.
- **Background:** Viechtbauer and Cheung (2010) define the `rma_uni` diagnostics directly; the `TODO.md` item names only the `rma_mv` ones.
- **Options:** (a) a separate `TODO.md` item for an `rma_uni` diagnostics spec - this spec stays the area you chose; (b) widen this spec - its
  property case reducing `rma_mv` to `rma_uni` then rests on covered code, but the spec grows past its size limit.
- **Recommendation:** (a).

## Answers

Answered on the review page on 2026-11-19 (21:03 UTC, the accept-all button), read back from its `answers` store; no comments:

| Item | Answer |
|---|---|
| 1. The spec | approve |
| D1. Coverage of the rma_uni diagnostics | (a) a separate `TODO.md` item |

Applied on 2026-11-19: the spec is version 1.1 with the approval recorded, and `TODO.md` has the item "No correctness case covers the rma_uni
diagnostics".
