# Correctness tests for the rma_mv diagnostics: measurements

Measured for the [spec](spec.md) before it was written. The scripts lived in the session's scratch directory, outside the repository; each figure is
the largest relative difference `max |a - b| / max(1, |b|)` over every value the function returns.

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
## Deletion diagnostics against brute-force refits

Measured on 2026-11-19 at commit `93aace4`, macOS (development machine), metafopy against a reference that refits each deletion with
`correctness.formulas.mv_fit` (re-estimated components) or `correctness.formulas.gls` at the full-data components (`reestimate=False`), and computes
Cook's distance, DFBETAS, the deleted residuals and the cluster `X2` from their definitions (spec, "Definitions"). Fits are the existing cases of
`src/tests/correctness/cases/rma_mv.toml`, or that case with its `mods` removed ("no mods").

| Fit | Deletion | `reestimate` | Cook | DFBETAS | rstudent `z` | `X2` | reference time | metafopy time |
|---|---|---|---|---|---|---|---|---|
| `mv.nested.kon` no mods (`~ 1 \| district/school`, REML) | each row | True | 5.0e-11 | 1.4e-10 | 2.4e-10 | - | 2.0 s | 1.4 s |
| same | each row | False | 3.2e-16 | 2.0e-15 | 4.4e-16 | - | 0.0 s | 0.4 s |
| same | `district` | True | 1.3e-10 | 2.1e-10 | 2.2e-10 | 7.3e-10 | 0.2 s | 0.2 s |
| same | `district` | False | 9.6e-16 | 1.7e-15 | 4.0e-16 | 7.9e-16 | 0.0 s | 0.1 s |
| `mv.nested.kon` (`mods = ~ year`) | each row | True | 3.5e-09 | 1.4e-08 | 2.2e-08 | - | 5.9 s | 1.7 s |
| same | each row | False | 1.3e-11 | 2.9e-11 | 1.0e-11 | - | 0.0 s | 0.7 s |
| same | `district` | True | 1.2e-08 | 2.3e-08 | 1.6e-08 | 1.7e-08 | 1.2 s | 0.4 s |
| same | `district` | False | 2.4e-11 | 2.4e-11 | 6.0e-12 | 2.7e-12 | 0.0 s | 0.1 s |
| `mv.method.reml.kon` (`~ 1 \| district`) | each row | True | 2.5e-11 | 1.7e-10 | 6.5e-10 | - | 0.7 s | 0.9 s |
| same | each row | False | 2.7e-15 | 1.3e-14 | 3.6e-15 | - | 0.0 s | 0.3 s |
| `mv.struct.un.berkey` (`~ outcome \| trial`, UN) | each row | True | 4.8e-08 | 3.4e-08 | 1.4e-08 | - | 0.2 s | 0.6 s |
| same | each row | False | 6.2e-15 | 3.8e-15 | 2.1e-15 | - | 0.0 s | 0.1 s |
| same | `trial` | True | 3.2e-08 | 3.0e-08 | 1.7e-08 | 2.8e-08 | 0.1 s | 0.3 s |
| same | `trial` | False | 8.0e-15 | 4.2e-15 | 1.7e-15 | 2.6e-15 | 0.0 s | 0.1 s |

Every reference refit converged (largest gradient norm 5.6e-07). With `reestimate=True` the differences are those of two optimisers, below the
harness's `MV_COMPONENT_TOLERANCE` (1e-6); with `reestimate=False` they are rounding, below `MV_TIGHT_TOLERANCE` (1e-10) except that the uncentred
`year` moderator (values near 2000) costs about two digits (up to 2.9e-11), still inside it.

## ranef against Henderson's mixed model equations

Same date and commit. The reference solves the mixed model equations with `R = V` and `G` the block-diagonal covariance of all random effects at
metafopy's components, and takes the prediction error variances from the inverse of the coefficient matrix.

| Fit | Terms | BLUPs | `se` | beta from the equations |
|---|---|---|---|---|
| `mv.nested.kon` no mods | `district`, `district/school` | 1.7e-15 | 1.8e-16 | 1.5e-15 |
| `mv.nested.kon` (`~ year`) | same | 8.0e-11 | 5.5e-11 | 1.1e-09 |
| `mv.method.reml.kon` | `district` | 5.6e-15 | 4.7e-15 | 2.6e-15 |
| `mv.struct.un.berkey` | `~ outcome \| trial` (UN) | 1.3e-15 | 4.2e-17 | 1.3e-15 |
| `mv.two_terms.berkey` | `~ outcome \| trial`, `~ outcome \| author` (ID, ID) | 1.7e-15 | 5.6e-17 | 1.1e-15 |
| `mv.rscale.cor.moura` | `study.id`, `species.id` with the tree's `R` | 4.6e-15 | 3.8e-15 | 2.4e-15 |
| `mv.rscale.none.moura` | same, `Rscale = "none"` | 3.0e-15 | 3.0e-15 | 2.8e-16 |
| `mv.struct.gen.slopes` (`~ x \| cl`, GEN) | the coefficient BLUPs (no `se`) | 8.3e-16 | - | 7.5e-16 |

The `year` fit loses digits in the equations' own solve (beta 1.1e-09 from metafopy's, which `mv_model` confirms to 1e-10 by the marginal GLS), so
the `ranef` cases use fits without an uncentred moderator.

## GEN BLUPs with a second term that crosses the clusters: a disagreement

`mv.struct.gen.slopes`'s data with a factor `g` (`g0`, `g1`, `g2` by row number modulo 3) and `random = [~ 1 | g, ~ x | cl]`, `struct = "GEN"`,
`sigma2 = 0.05` fixed. The fitted `G` is `[[0.05497, -0.00937], [-0.00937, 0.04328]]`.

- Henderson's equations and the marginal form `G Z' M^-1 (y - X b)` with the whole `M^-1` agree with each other to 8.6e-16.
- metafopy's `ranef` differs from them by up to 0.0306 (absolute; the largest BLUP is 0.453). Cluster `c00`: metafopy (0.210216, 0.030857),
  reference (0.222839, 0.040562).
- R metafor 5.2-1 (`.audit/Rlib`) prints the same values as metafopy for `c00` and `c01` to six decimals: metafopy ports `ranef.rma.mv` line for
  line. Both use the within-cluster block `W[incl, incl]` of `M^-1` and drop its off-diagonal blocks, which are zero only when no other term links
  the clusters; metafor's own verbose message for this branch reads "Computation of BLUPs not currently available for struct="GEN"".

## Reduction to rma_uni

`dat.bcg` (log risk ratios, `mods = ~ ablat`), `rma_mv(random = ~ 1 | id)` with one row per `id` against `rma_uni`, REML:

- with `rma_uni`'s default `threshold` the two `tau2` differ by 8.8e-05 relative (0.0763547 against 0.0763480), and the diagnostics by up to 6.2e-05;
- with `control = {threshold: 1e-12, maxiter: 1000}` on `rma_uni` the `tau2` agree to 1e-11 and Cook's distance, `rstudent` `z` and the `ranef`
  BLUPs and `se` to 6.7e-12 at most.
<!-- doc-counts: accept-end -->
