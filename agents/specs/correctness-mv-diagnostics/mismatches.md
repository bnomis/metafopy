# Correctness tests for rma_mv diagnostics: mismatches and sources needed

**Task list:** `agents/tasks/correctness-mv-diagnostics/tasks-1.md`
**Spec:** `agents/specs/correctness-mv-diagnostics/spec.md`

## What is reviewed and how

Each item below is a `derived` case whose reference, coded from the cited publication, disagrees with metafopy. The case is committed with
`status = "mismatch"` (a strict expected failure) and a reason naming both values. For the item you choose whether metafopy follows the reference,
keeps metafor's behaviour, or the difference is investigated further. The one item below is the only mismatch of this work.

## Automatic checks

Checked 2026-11-19:

- **Pass:** `pytest src/tests/correctness -m 'not slow'` passes: the case below is a strict expected failure and nothing else fails.
- **Pass:** every `status = "mismatch"` case added by this task list has an item below; the only one is `mv.ranef.gen.crossed`.

## Mismatches

### ranef with GEN: the BLUPs of a model whose second term crosses the clusters

- **Case:** `mv.ranef.gen.crossed` in `src/tests/correctness/cases/rma_mv.toml` (covers `rma_mv.ranef.GEN`).
- **Source:** Henderson (1975), Biometrics 31:423-447, equation 5 (the mixed model equations) and the text after equation 8 (`Var(u-hat - u) = C22
  sigma^2`); Searle, Casella and McCulloch (1992), Variance Components, section 7.6, equations 52 and 53.
- **Data and model:** `mv.struct.gen.slopes`'s data with a factor `g` (`g0`, `g1`, `g2` by row number modulo 3), `random = [~ 1 | g, ~ x | cl]`,
  `struct = ["GEN", "GEN"]`, `sigma2 = 0.05` fixed.
- **Reference:** BLUPs (intercept, slope) from the mixed model equations with the whole inverse marginal covariance: cluster `c07`
  (-0.165067, -0.099029), `c03` (0.452740, 0.003167), `c00` (0.222839, 0.040562).
- **metafopy:** `c07` (-0.155422, -0.068466), `c03` (0.422449, 0.016976), `c00` (0.210216, 0.030857); the largest difference is 0.0306 against a
  largest BLUP of 0.453.
- **metafor:** R metafor 5.2-1 prints metafopy's values for `c00` and `c01` to six decimals (`measurements.md`, 2026-11-19). metafopy ports
  `ranef.rma.mv` line for line.
- **Cause:** `ranef.rma.mv` takes only the within-cluster block of the inverse marginal covariance, which drops the links another term (here `g`,
  which crosses the clusters) makes between clusters. The two forms agree when no other term links the clusters; the mixed model equations and the
  marginal form with the whole inverse agree with each other to 8.6e-16. metafor's own verbose message for this branch reads "Computation of BLUPs
  not currently available for struct="GEN"".

## Decisions

### D1: ranef with GEN, which BLUPs does metafopy return?

- **What is decided:** whether `ranef` for a `GEN` fit with a term that crosses the clusters keeps metafor's within-cluster-block BLUPs or returns the
  mixed-model-equation BLUPs.
- **Background:** `ranef` for `GEN` is a port of `ranef.rma.mv`, whose own message says the BLUPs are not currently available for this structure, so
  the values are not a documented result. The nested, `R`-scaled, `UN` and two-term `ID` fits all agree with the equations to 1e-10 or better
  (`measurements.md`, 2026-11-19); only a second term that links the clusters shows the difference.
- **What it affects:** the `ranef` code for `rma_mv` fits and its R parity cases for such fits; metafopy would then depart from R metafor 5.2-1 for
  this case only, and `docs/differences.md` would gain a section.
- **Options:**
  - *Keep metafor (Recommended):* the case's `reason` gains "accepted as a difference from the reference on <date>", it stays a strict expected
    failure, and no code changes. metafopy keeps matching R metafor.
  - *Follow the reference:* a `TODO.md` item changes `ranef` to use the whole inverse, flips the case, marks the R parity fixtures, and adds a
    `known.toml` entry and a `docs/differences.md` section.
  - *Investigate:* a `TODO.md` item records the figures above, and the case stays an expected failure.
- **Recommendation:** keep metafor: metafopy ports metafor, and the code path is one metafor itself flags as unavailable, so no documented behaviour
  is changed; the difference stays recorded in the case's reason either way.

## Answers

Left empty for the operator's verdict: the operator session publishes the review page, copies the answer here with the date, and commits it.

## Sources needed

Read 2026-11-19 from the papers folder `/Users/simonb/src/metafopy papers/` (text extracted with `pdftotext -layout` into a scratch directory outside
the repository; nothing saved in the repository).

| Source | Equations wanted | Status |
|---|---|---|
| Viechtbauer and Cheung (2010), Res Synth Methods 1:112-125, `Outlier and influence diagnostics for meta-analysis.pdf` | Cook's distance, DFBETAS, externally studentised residual | Read: equations 15 and 16, 20 and 21, 23 |
| Christensen, Pearson and Johnson (1992), Technometrics 34:38-45, `Case-Deletion Diagnostics for Mixed Models.pdf` | set deletion, generalised Cook's distance | Read: section 2, Propositions 1 to 3 and the distance `C_beta(i)` |
| Henderson (1975), Biometrics 31:423-447, `Best Linear Unbiased Estimation and Prediction under a Selection Model.pdf` | the mixed model equations and their inverse | Read: equation 5 and the text after equation 8 |
| Searle, Casella and McCulloch (1992), Variance Components, `Variance Components.pdf` | chapter 7, the mixed model equations | Read: section 7.6, equations 52 and 53 |
| Robinson (1991), Stat Sci 6:15-32, doi:10.1214/ss/1177011926 | the same equations | Not read: Project Euclid refused the automated download on 2026-11-19. It is not a "supply" paper and the two sources above give the equations, so no case cites it |

## Definition checks

- **Cook's distance:** Viechtbauer and Cheung equation 20 is `(b - b(-i))' X'W X (b - b(-i))`, with `X'W X` the inverse of the full-data variance of the
  coefficients; the reference uses `vb^-1` of the full fit. Their footnote says the form has no division by `p` because it is calibrated against a
  chi-square distribution. Christensen, Pearson and Johnson define `C_beta(i) = (b - b[i])' (X'V^-1 X) (b - b[i]) / p`, which divides by `p` (the F
  calibration of Cook 1977). The reference and metafor follow Viechtbauer and Cheung; the two papers differ only by the factor `p`, a calibration
  convention and not a disagreement about the influence. Not recorded as a mismatch case.
- **DFBETAS:** equation 23 divides by the square root of the diagonal of `(X'W(-i)X)^-1`, with the weights at the deleted-case component; the reference
  uses `vb*`, the GLS variance on all rows at the components `c_(S)`. Same definition.
- **Externally studentised residual:** equations 15 and 16, `(y_i - mu_i(-i)) / sqrt(v_i + tau2(-i) + Var[mu_i(-i)])`; the reference's `C = M[S, S] + X_S
  vb_(S) X_S'` is the set form. Same definition.
- **Set deletion:** Christensen et al. give one-step approximations of the components, from the full-data fit; the reference refits, the definition
  metafor's `reestimate=True` uses, so their one-step values are not what is compared.
- **Mixed model equations:** Henderson equation 5 and Searle et al. equations 52 and 53 are the reference's `[X'R^-1X, X'R^-1Z; Z'R^-1X, Z'R^-1Z +
  G^-1]` system with `R = V`; Henderson states `Var(u-hat - u) = C22 sigma^2`, the lower block of the inverse, as the reference takes the variances.
