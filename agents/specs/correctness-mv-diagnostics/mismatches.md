# Correctness tests for rma_mv diagnostics: mismatches and sources needed

**Task list:** `agents/tasks/correctness-mv-diagnostics/tasks-1.md`
**Spec:** `agents/specs/correctness-mv-diagnostics/spec.md`

The mismatch items are added by Task 5 of the task list.

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
