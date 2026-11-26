# Correctness tests: rma_mv diagnostics

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-19

This spec adds correctness cases for the `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef` to the harness in
`src/tests/correctness/`: independent references computed from the definitions (explicit leave-out refits, Henderson's mixed model equations),
property cases, and the mismatch review where metafopy disagrees.

**Approved** by the operator on 2026-11-19 on the review page of `agents/reviews/correctness-mv-diagnostics-spec.md` (no comment); its decision D1
sends the `rma_uni` diagnostics to a separate `TODO.md` item. Version 1.1 records the approval; the content is otherwise unchanged from 1.0.

## Motivation

The `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef` (`src/metafopy/methods/mv_diagnostics.py`, `mv_ranef.py`) are checked
only against R metafor, by the JSON parity fixtures: a metafor bug in them passes every test. The [correctness-mv-robust spec](../correctness-mv-robust/spec.md)
left them out ("a later spec can add the `rma_mv` diagnostics") and the `TODO.md` item "Correctness coverage the extension specs left for later"
(found 2026-11-15) records that nothing covers them. **The defining publication
outranks metafor** (`claude/rules/metafopy.md`): where the reference and metafopy disagree, a `mismatch` case records it and metafopy's output
changes only after the user confirms it. The location-scale items of the same `TODO.md` item are not covered here.

Everything the [correctness-tests spec](../correctness-tests/spec.md) settles is reused unchanged: the case format, the kinds of evidence and their
tolerances, the gate (each item needs a `published` or `derived` case or an `[[exempt]]` entry), the `formulas.py` import ban, strict xfail for
`mismatch` cases and their review. Dated figures quoted below are in [measurements.md](measurements.md) (measured 2026-11-19).

## Module location

- `src/tests/correctness/formulas.py` - `mv_deletion` and `mv_blups` (API below); `numpy` and `scipy` only, no metafopy import.
- `src/tests/correctness/test_derived.py` - the checks `mv_deletion` and `mv_ranef`; `test_properties.py` - the property checks.
- `src/tests/correctness/cases/rma_mv.toml` - the new cases and `[[exempt]]` entries.
- `src/tests/correctness/test_coverage.py` - the new items.
- `agents/specs/correctness-mv-diagnostics/mismatches.md` - the review document of mismatches and sources needed.

No change to `src/metafopy`.

## Definitions and references

For the rma.uni model, Viechtbauer and Cheung (2010) define Cook's distance, DFBETAS and the externally standardised residual by deleting a study and
refitting, as metafor's help pages describe them (the paper itself is to be read in step 4). **No publication defines the `rma_mv` versions.** metafor
generalises the rma.uni formulas by putting the marginal covariance `M = V + sum of the random-effects matrices` in place of `diag(v_i + tau2)` and
deleting a row or a whole cluster; Christensen, Pearson and Johnson (1992) is the standard source for case-set deletion in mixed models (to be read in step
4). The independent reference is therefore our own formula from these definitions, computed by brute force: for each deleted set `S` (one row, or the rows
of one cluster) with kept rows `K`:

- `reestimate=True`: refit on `K` with `formulas.mv_fit` (the components `c_(S)`); `reestimate=False`: `c_(S)` = the full-data components.
- `b_(S)`, `vb_(S)`: GLS on the rows `K` with `M(c_(S))[K, K]`; `b`, `vb`: the full fit (GLS at metafopy's components, as `mv_model` checks).
- Cook's distance: `(b - b_(S))' vb^-1 (b - b_(S))` over the `btt` coefficients (`vb` of the full fit, Viechtbauer and Cheung's form with `M`).
- DFBETAS: `(b_j - b_(S)j) / sqrt(vb*_jj)`, `vb*` = GLS variance on all rows with the components `c_(S)`.
- rstudent: deleted residual `e = y_S - X_S b_(S)`, its covariance `C = M(c_(S))[S, S] + X_S vb_(S) X_S'`, `z = e / sqrt(diag C)`; with `cluster`,
  `X2 = e' C^-1 e` (the help page's cluster-level multivariate residual, chi-square with `k_i` df).

`reestimate=False` is the approximation metafor's `residuals.rma` help page documents ("variance/correlation components ... not re-estimated"),
coded from that documentation under the rule decided on 2026-11-13 for options that only software documentation defines (correctness-mv-robust,
Decisions).

`ranef` returns the BLUPs of the random effects and their prediction error variances, defined by Henderson's mixed model equations (Henderson 1975;
Robinson 1991, Searle, Casella and McCulloch 1992). metafopy and metafor compute the marginal form `D Z' M^-1 (y - X b)` and `D - D Z' P Z D`; the
reference solves the mixed model equations `[X'R^-1X, X'R^-1Z; Z'R^-1X, Z'R^-1Z + G^-1] [b; u] = [X'R^-1y; Z'R^-1y]` with `R = V`, `Z` and `G` of all
terms at once, and takes the variances from the lower block of the inverse coefficient matrix: a different algorithm for the same definition. The help
page also cites Kackar and Harville (1981); neither side adds a correction for estimated components, so the cases compare at the components given.

Every case is `derived`. The sources, each to be read before its case's locator is written:

- `cooks_distance`, `dfbetas`, `rstudent` (the rma.uni definitions): Viechtbauer W, Cheung MW-L (2010). Outlier and influence diagnostics for
  meta-analysis. Res Synth Methods 1:112-125. doi:10.1002/jrsm.11. Paywalled, not in the folder: supply.
- The same, deleting sets of cases in mixed models: Christensen R, Pearson LM, Johnson W (1992). Case-deletion diagnostics for mixed models.
  Technometrics 34:38-45. doi:10.2307/1269550. Supply.
- `rstudent`'s `X2` and `reestimate=False`: metafor 5.2-1 help pages `residuals.rma` and `influence.rma.mv` (documentation, no DOI), read 2026-11-19.
- `ranef`: Henderson CR (1975). Best linear unbiased estimation and prediction under a selection model. Biometrics 31:423. doi:10.2307/2529430
  (supply); Robinson GK (1991). That BLUP is a good thing: the estimation of random effects. Stat Sci 6:15-32. doi:10.1214/ss/1177011926 (open,
  Project Euclid); Searle SR, Casella G, McCulloch CE (1992). Variance Components. Wiley. doi:10.1002/9780470316856, chapter 7 (supply).

DOIs were checked against Crossref on 2026-11-19. "Folder" is the papers folder `/Users/simonb/src/metafopy papers` (outside the repository). No paper
above was readable on 2026-11-19 (Robinson's open copy did not load through the fetch tool; the help pages were read), so each case's `source.locator`
names the equation once the source is read, as the earlier specs record values. No paper found prints an `rma_mv` diagnostic on a bundled dataset (a text
search of the folder copies of Konstantopoulos 2011 and Berkey et al. 1998 on 2026-11-19 found none), so there are no `published` cases; the examples of
`influence.rma.mv` print metafor's own output, which is parity, not evidence.

## API

```python
# formulas.py
@dataclasses.dataclass(frozen=True)
class MVDeletion:
    """Per deleted set: Cook's distance, DFBETAS (rows by set), deleted residuals, their standard errors, z, X2, and each refit's convergence."""

    cook: FloatArray
    dfbetas: FloatArray
    resid: FloatArray
    se: FloatArray
    z: FloatArray
    X2: FloatArray
    converged: tuple[bool, ...]
    gradient_norms: FloatArray
    messages: tuple[str, ...] = ()  # the optimiser message of each refit


def mv_deletion(
    yi: FloatArray,
    V: FloatArray,
    X: FloatArray,
    terms: list[RandomTerm],
    method: str,
    variances: list[FloatArray],
    correlations: list[FloatArray],
    groups: list[IntArray],
    reestimate: bool,
    btt: list[int] | None = None,
) -> MVDeletion:
    """Leave-one-set-out diagnostics by explicit refits (the definitions above); each term is cut to the kept rows."""


@dataclasses.dataclass(frozen=True)
class MVBlups:
    """beta and, per term in the order of ``terms``, the BLUPs of the observed levels and their prediction error variances."""

    beta: FloatArray
    blups: tuple[FloatArray, ...]
    pev: tuple[FloatArray, ...]


def mv_blups(yi: FloatArray, V: FloatArray, X: FloatArray, terms: list[RandomTerm], variances: list[FloatArray], correlations: list[FloatArray]) -> MVBlups:
    """Henderson's mixed model equations with R = V; a sigma2 term's G is sigma2 I (sigma2 R with a known R), a structured term's G is
    I_outer kron struct_matrix, a GEN or GDIAG term's Z is its design within each outer level."""


# test_derived.py
# auditme: accept[spec-api-signature-drift] the check in test_derived.py shares its name with the reference in formulas.py; its signature is built in Task 2
def mv_deletion(case: correctness.cases.Case, result: typing.Any) -> None:
    """The case's ``then`` is ``rstudent`` (or ``cooks_distance`` with ``btt``) with ``cluster`` and ``reestimate``; the check refits the case
    (``refit``) and calls ``cooks_distance``, ``dfbetas`` and ``rstudent`` with the same arguments, against ``formulas.mv_deletion``."""


def mv_ranef(case: correctness.cases.Case, result: typing.Any) -> None:
    """``ranef(fit)`` against ``formulas.mv_blups`` at metafopy's components, rows matched by their labels (``level`` or ``inner | outer``)."""
```

`mv_setup` in `test_derived.py` already rebuilds `y`, `V`, `X`, the terms and metafopy's components; both checks use it. The groups come from
`pandas.factorize(frame[cluster], sort=True)`, which is also the order of metafopy's cluster output.

## Coverage list

`coverage_items()` gains `mv_diagnostics_items()`, with two constants in `test_coverage.py` (no list in the code names them, as with `LS_PREDICT`):
`MV_DIAGNOSTICS = ('cooks_distance', 'dfbetas', 'rstudent')` and `MV_DELETIONS = ('cluster', 'fixed', 'obs')`.

- `rma_mv.<f>.<d>` for each `f` in `MV_DIAGNOSTICS` and `d` in `MV_DELETIONS` (`obs`: each row, `reestimate=True`; `cluster`: whole clusters,
  `reestimate=True`, with `X2` for `rstudent`; `fixed`: `reestimate=False`), and `rma_mv.cooks_distance.btt`;
- `rma_mv.ranef.sigma2` (`~ 1 | f` and nested), `rma_mv.ranef.R` (a known correlation matrix), `rma_mv.ranef.G` (`~ inner | outer`),
  `rma_mv.ranef.H` (a second structured term), `rma_mv.ranef.GEN` (random coefficients), `rma_mv.ranef.pi.z` (bounds `pred +- z se`);
- `rma_mv.ranef.pi.t` with `[[exempt]]` reason "defined only in metafor's code; R parity covers it" (the t quantile with `k - p` or
  `nlevels - p` degrees of freedom of `term_crit` is in no documentation).

Out of scope: `rstandard`, `hatvalues`, `influence` and `weights` of `rma_mv`; `ranef`'s `expand`, `vcov` and `transf` (presentation only); the
parallel options (the same refits). The `rma_uni` diagnostics have no coverage item either (checked 2026-11-19: no case in `src/tests/correctness`
names them); they are the `TODO.md` item "No correctness case covers the rma_uni diagnostics" (decision D1 of the review, 2026-11-19).

## Cases

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
Tolerances are the harness's existing `rma_mv` ones: with `reestimate=True` `MV_COMPONENT_TOLERANCE` (1e-6), the agreement of two optimisers;
`reestimate=False` and `ranef` `MV_TIGHT_TOLERANCE` (1e-10); properties `PROPERTY_TOLERANCE` (1e-8). The measured differences are at most 4.8e-8,
2.9e-11 and 6.7e-12 respectively (measurements.md). Fits use no uncentred moderator in `ranef` cases (the `~ year` fit loses digits in the mixed
model equations' own solve).
<!-- doc-counts: accept-end -->

| Case | Fit | `then` | Covers |
|---|---|---|---|
| `mv.diag.obs.kon` | `dat.konstantopoulos2011`, `~ 1 \| district/school`, REML | `rstudent`, each row | `*.obs` |
| `mv.diag.cluster.kon` | same | `rstudent(cluster="district")` | `*.cluster` |
| `mv.diag.fixed.kon` | same | `rstudent(reestimate=False)` | `*.fixed` |
| `mv.diag.ml.kon` | `~ 1 \| district`, ML | `rstudent`, each row | `*.obs` |
| `mv.diag.{obs,cluster,fixed}.berkey` | `dat.berkey1998`, `vblocks`, `~ 0 + outcome`, `~ outcome \| trial`, UN | as above, `trial` | as above |
| `mv.diag.btt.berkey` | same | `cooks_distance(btt=[1])` | `rma_mv.cooks_distance.btt` |
| `mv.ranef.nested.kon` | as `mv.diag.obs.kon`, `test="z"` | - | `ranef.sigma2`, `ranef.pi.z` |
| `mv.ranef.r.moura` | as `mv.rscale.cor.moura` | - | `ranef.R` |
| `mv.ranef.un.berkey`, `mv.ranef.two_terms.berkey` | as `mv.struct.un.berkey`, `mv.two_terms.berkey` | - | `ranef.G`, `ranef.H` |
| `mv.ranef.gen.slopes` | as `mv.struct.gen.slopes` | - | `ranef.GEN` |
| `mv.ranef.gen.crossed` | `mv.struct.gen.slopes`'s data plus `g` (row number mod 3), `[~ 1 \| g, ~ x \| cl]`, GEN, `sigma2=0.05` | - | `ranef.GEN` |

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
`mv.ranef.gen.crossed` is expected to be a `mismatch`: on 2026-11-19 metafopy and R metafor 5.2-1 agree with each other and differ from the BLUP
by up to 0.031, because `ranef` for GEN uses only the within-cluster block of `M^-1` (measurements.md). It goes to `mismatches.md` like any other.

Property cases (`test_properties.py`): on `dat.bcg` (`~ ablat`), `rma_mv(random = ~ 1 | id)` with one row per `id` gives `rma_uni`'s Cook's
distance, DFBETAS, `rstudent` and `ranef` when `rma_uni` has `control={'threshold': 1e-12, 'maxiter': 1000}` (with the default threshold the two
`tau2` differ by 8.8e-5 relative, measurements.md); `cluster` set to a column with one value per row gives the per-row results; a one-row cluster's
`X2` equals its `z` squared; `reestimate=True` and `False` agree when every component is fixed by the call.
<!-- doc-counts: accept-end -->

## Error handling

- A reference refit that does not converge (`mv_fit`'s `success` false or gradient norm above `1e-6`) fails the case naming the case, the deleted
  set and the optimiser message; it is never compared. The case's data is changed only by a reviewed edit, never by loosening the tolerance.
- Where metafopy returns NaN for a set (its refit failed) and the reference converges, the check fails naming the set: that is a disagreement and
  goes to review as a `mismatch`, not an exemption.
- `mv_blups` raises `ValueError` naming the term when a `G` is singular (a variance at 0); the `ranef` cases use fits with positive components.
- `mv_deletion` raises `ValueError` when a deletion leaves `X[K]` rank deficient (metafor returns NaN there); no case deletes such a set.
- The existing loader errors (malformed case, unknown check, duplicate id, `mismatch` without `reason`) apply unchanged.

## Implementation

1. `formulas.mv_deletion` and `formulas.mv_blups`, with harness tests in `test_coverage.py`: on a small inline dataset, `mv_deletion` with
   `reestimate=False` against the closed-form rma.uni deletion formulas, and `mv_blups` against the marginal form `G Z' M^-1 (y - X b)`.
2. The checks in `test_derived.py` and `test_properties.py`; the items and the exemption in `test_coverage.py`, failing until step 3.
3. The cases above in `rma_mv.toml`, each with its `source`. None is `slow`: no measured fit took more than a few seconds with its reference (measurements.md).
4. Sources: list each "supply" row of the reference table in `mismatches.md` under "Sources needed", with the equations wanted (Viechtbauer and
   Cheung's Cook's distance, DFBETAS and studentised residual; Christensen et al.'s set deletion; the mixed model equations and their inverse).
   **Work stops** until the operator supplies or drops each one; then fill each case's locator and check the definitions above against them. A
   definition that differs from the source (a factor `1/p` in Cook's distance, another variance in DFBETAS) changes the reference, and any resulting
   disagreement with metafopy becomes a `mismatch` case.
5. Review: every `mismatch` case gets an item in `mismatches.md` (source, reference, metafopy and metafor values, options *follow the reference*,
   *keep metafor*, *investigate*) with its review page, per the shared `reviews.md` rule. **Work stops** until each has an answer.
6. Close: file each *follow the reference* answer as a `TODO.md` item (fix, flip the case, mark the parity fixtures, `known.toml`,
   `docs/differences.md`); remove this area from the `TODO.md` item; set this spec's status in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pandas`, `pytest` (existing); no new dependency, and no R at test time.
- The papers marked "supply", from the operator.
