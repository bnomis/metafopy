# Review: the correctness-uni-diagnostics spec

**Review page:** https://claude.ai/artifact/QZTTqSmWomyE4FdNFWJpxe
**Date:** 2026-11-20

**What is being reviewed and how.** One new spec, `agents/specs/correctness-uni-diagnostics/spec.md` (with `measurements.md`), written by `/todo-do`
for the `TODO.md` item "No correctness case covers the rma_uni diagnostics". It adds correctness cases for the `rma_uni` diagnostics
`cooks_distance`, `dfbetas`, `rstudent` and `ranef`, which only R parity fixtures check today. No metafopy code changes. Approve or flag the spec,
and answer each decision.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
- Pass: metafopy at `1c93ae3` reproduces every value Viechtbauer and Cheung (2010) print for their BCG example (DL, latitude and year centred)
  within half the last digit; checked again by hand on 2026-11-20 for study 4: `rstudent` -1.5056, Cook's distance 9.6362, DFBETAS (-0.8677,
  -2.8575, -2.4662).
- Pass: with the tight `control` that `inference.toml` uses, every brute-force leave-one-out reference agrees with metafopy to 1.6e-11 or better;
  DL and EE to 3.7e-14 with the defaults; `ranef` against the mixed model equations to 1e-14 (measurements.md).
- Pass: every source a case cites is in the papers folder `/Users/simonb/src/metafopy papers`.
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **References:** a `published` case from the paper's BCG example (hat values, `rstudent`, Cook's distance and the change in `tau2` for studies 4, 7
  and 13, the DFBETAS of study 4); `derived` cases from the paper's equations by explicit refits for random-effects, moderator, equal-effects and
  fixed-`tau2` fits; `ranef` against Henderson's mixed model equations; a property case that a study's `rstudent` equals the z of a dummy added for
  it (DL, REML, EE; not ML, where it does not hold).
- **No mismatch is expected**, so the review step needs no answer unless the run finds one.
- **Stops in the task list:** only the mismatch review, and only if a mismatch appears. I removed the draft's stop for Raudenbush and Bryk (1985):
  no case uses it, so it is recorded as not needed.
- **Recommendation:** approve.

## Decisions

### D1. The other diagnostics the paper defines

- **What is decided:** whether `influence`'s DFFITS, covariance ratio, `tau2.del`, `QE.del` and hat values, and marginal `rstandard`, join the scope.
- **Background:** the paper defines each (eqs 13, 14, 19, 25, 27) and prints the hat values and the change in `tau2` for its example; they come from
  the same refits, and the probe agrees with metafopy to 1.6e-11 or better (2026-11-20).
- **Affects:** more coverage items and fields of the reference; no new fit.
- **Options:** *Include (Recommended)*: checked against the paper; *Leave out*: only the four functions the item names, the rest stay R-only;
  *Include and add leave1out and cumul*: also each leave-out fit against the existing fixed-effects reference.
- **Recommendation:** include: the same source defines and prints them, and the reference is already written.

### D2. blup and the Raudenbush (2009) printed BLUPs

- **What is decided:** whether `blup` joins the scope with a `published` case.
- **Background:** Raudenbush (2009, Handbook chapter, Table 16.2) prints study 4's BLUP 0.21 with range (-0.03, 0.25) for `dat.raudenbush1985`,
  REML; metafopy and R give 0.2144 and (-0.0293, 0.2485). The book's mixed-model range cannot be reproduced from its own rounded coefficients, so it
  would not be used.
- **Affects:** two coverage items, a `published` case and a small `uni_blups` reference.
- **Options:** *Include (Recommended)*; *Leave out*: `blup` stays checked against R only.
- **Recommendation:** include: it is the only `rma_uni` shrinkage estimate with printed values.

### D3. The Knapp-Hartung scaling of the diagnostics

- **What is decided:** how diagnostics of a `test="knha"` fit are treated.
- **Background:** metafopy, porting `influence.rma.uni`, scales DFBETAS and the deleted residual's variance by `s2w`; on `dat.bcg` (DL, `~ ablat`)
  this moves Cook's distance by up to 0.106 relative and `rstudent` z by 0.20. Neither the paper nor metafor's help pages define it.
- **Options:** *Exempt (Recommended)*: "defined only in metafor's code; R parity covers it", as `weighted=False` already is; *Derived against the
  unscaled equations*: a `mismatch` case for the review step.
- **Recommendation:** exempt: there is no publication to disagree with.

## Answers

Answered 2026-11-20 on the review page (https://claude.ai/artifact/QZTTqSmWomyE4FdNFWJpxe), each with no comment:

- **1. The spec:** approved.
- **D1. The other diagnostics the paper defines:** include.
- **D2. blup and the Raudenbush (2009) printed BLUPs:** include.
- **D3. The Knapp-Hartung scaling of the diagnostics:** exempt.

Applied in version 1.1 of the spec (2026-11-20).
