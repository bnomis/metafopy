# Correctness tests: rma_uni diagnostics

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-20

This spec adds correctness cases for the `rma_uni` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef` to the harness in
`src/tests/correctness/`: values printed by the defining publication, independent references computed from its equations (explicit leave-one-out
refits, Henderson's mixed model equations), property cases, and the mismatch review where metafopy disagrees. Decisions D1 and D2 add the other
`influence` columns, `rstandard` and `blup`.

**Approved** by the operator on 2026-11-20 on the review page of `agents/reviews/correctness-uni-diagnostics-spec.md` (no comment), with the
recommended option of each decision. Version 1.1 records the approval and applies the decisions (see Decisions).

## Motivation

<!-- doc-counts: accept-start found on 2026-11-19 - a dated record, not a claim about the present -->
The `rma_uni` diagnostics (`src/metafopy/methods/influence.py`, `ranef.py`, `blup.py`) are checked only against R metafor, by the JSON parity
fixtures: a metafor bug in them passes every test. The `TODO.md` item "No correctness case covers the rma_uni diagnostics" (found 2026-11-19 while
writing the [rma_mv diagnostics spec](../correctness-mv-diagnostics/spec.md)) records that no case in `src/tests/correctness` names them; decision D1
of `agents/reviews/correctness-mv-diagnostics-spec.md` (answered 2026-11-19) kept them out of that spec and filed the item.
<!-- doc-counts: accept-end -->

**The defining publication outranks metafor** (`claude/rules/metafopy.md`): where the reference and metafopy disagree, a `mismatch` case records it
and metafopy's output changes only after the user confirms it. Everything the [correctness-tests spec](../correctness-tests/spec.md) settles is
reused unchanged: the case format, the kinds of evidence and their tolerances, the gate (each item needs a `published` or `derived` case or an
`[[exempt]]` entry), the `formulas.py` import ban, strict xfail for `mismatch` cases and their review. Dated figures quoted below are in
[measurements.md](measurements.md) (measured 2026-11-20).

## Module location

- `src/tests/correctness/formulas.py` - `uni_deletion` and `uni_blups` (API below); `numpy` and `scipy` only, no metafopy import. `ranef` reuses
  `mv_blups`.
- `src/tests/correctness/test_derived.py` - the checks `uni_deletion`, `uni_ranef` and `uni_blup`; `test_published.py` - `influence_printed` and
  `blup_printed`;
  `test_properties.py` - `uni_mean_shift`.
- `src/tests/correctness/cases/rma_uni_diagnostics.toml` (new) - the cases and `[[exempt]]` entries; `properties.toml` - the property cases.
- `src/tests/correctness/test_coverage.py` - the new items.
- `agents/specs/correctness-uni-diagnostics/mismatches.md` - the review document of mismatches.

No change to `src/metafopy`.

## Definitions and references

Viechtbauer and Cheung (2010) define every deletion diagnostic of the rma.uni model by deleting study `i`, refitting, and comparing. With `W =
diag(1 / (v + tau2))`, `b`, `vb = (X'WX)^-1` of the full fit, and `tau2(-i)`, `b(-i)`, `vb(-i)` of the refit without `i`:

- Cook's distance, eq 20: `D_i = (b - b(-i))' vb^-1 (b - b(-i))`, over the `btt` coefficients; no division by `p` (the footnote of section 6: the
  chi-square calibration of eq 22).
- DFBETAS, eq 23: `(b_j - b(-i)_j) / sqrt([(X' W(-i) X)^-1]_jj)` with `W(-i) = diag(1 / (v + tau2(-i)))` over all `k` studies.
- Externally studentised residual, eqs 15 and 16: `t_i = (y_i - x_i b(-i)) / sqrt(v_i + tau2(-i) + x_i vb(-i) x_i')`; the deleted residual and
  its standard error are the numerator and the denominator.
- A fixed `tau2` (the call's `tau2=`) is kept in every refit, so `tau2(-i) = tau2`; EE fits have `tau2 = 0` throughout.

The reference is coded from these equations by brute force, refitting with `formulas.rma_tau2` (the estimators the harness already checks against
their publications) and weighted least squares. The paper also defines DFFITS (eq 19), COVRATIO (eq 25), `R_i` from `tau2(-i)` (eq 27), the hat
values (eq 13) and the internally studentised residual (eq 14, metafopy's `rstandard`); decision D1 brings each into scope, coded from its
equation with the same refits. `QE(-i)` is Cochran's `Q` of the refit.

`ranef` returns the BLUPs `u_i = lambda_i (y_i - x_i b)`, `lambda_i = tau2 / (tau2 + v_i)`, and their prediction error variances `lambda_i v_i +
lambda_i^2 x_i vb x_i'` (the marginal form metafor computes). The reference solves Henderson's mixed model equations with `R = V`, `Z = I`, `G =
tau2 I`, which is `formulas.mv_blups` with one `~ 1 | study` term: a different algorithm for the same definition, already checked for `rma_mv`.
The intervals are `pred +- crit se`, `crit` the normal quantile, or with `test` `t`, `knha`, `hksj` or `adhoc` the t quantile on `k - p` degrees of
freedom (the `ranef` help page of metafor 5.2-1, read 2026-11-20, coded from the documentation under the rule decided on 2026-11-13 in
[correctness-mv-robust](../correctness-mv-robust/spec.md), Decisions). For EE fits the help page says the BLUPs are 0.

`blup` (decision D2) returns `x_i b + u_i`, the shrunken estimate of each study's true effect (Raudenbush 2009, eqs 16.23-16.25), with its prediction
error variance from the same mixed model equations (`formulas.uni_blups`, a thin wrapper of `mv_blups` that adds `x_i b` and the covariance of `b`
and `u_i` from the inverse coefficient matrix). The book's eq 16.25 omits the uncertainty of `b`, so its printed values are compared only for the
estimates and the range of the estimates, never the `se`.

Sources ("folder" is `/Users/simonb/src/metafopy papers`, outside the repository; DOIs checked against Crossref on 2026-11-20):

| Source | Used for | Folder | Locator |
|---|---|---|---|
| Viechtbauer W, Cheung MW-L (2010). Outlier and influence diagnostics for meta-analysis. Res Synth Methods 1:112-125. doi:10.1002/jrsm.11 | the deletion diagnostics, printed values, the mean-shift model | yes, read 2026-11-20 | eqs 9, 13-27; section 5.1, Table I; section 5.2, Table II |
| Henderson CR (1975). Best linear unbiased estimation and prediction under a selection model. Biometrics 31:423-447. doi:10.2307/2529430 | `ranef` | yes, read 2026-11-19 (rma_mv spec) | eq 5, the text after eq 8 |
| Searle SR, Casella G, McCulloch CE (1992). Variance Components. Wiley. doi:10.1002/9780470316856 | `ranef` | yes, read 2026-11-19 (rma_mv spec) | section 7.6, eqs 52, 53 |
| Raudenbush SW (2009). Analyzing effect sizes: random-effects models. Handbook of Research Synthesis and Meta-Analysis, 2nd ed., 295-315 (no DOI) | printed BLUPs (decision D2) | yes, read 2026-11-20 | Table 16.2; section 16.5.6.1, eqs 16.23-16.25 |
| Raudenbush SW, Bryk AS (1985). Empirical Bayes meta-analysis. J Educ Stat 10:75-98. doi:10.3102/10769986010002075 | cited by metafor's `blup` help | no: not needed (no case depends on it) | - |
| metafor 5.2-1 help pages `influence.rma.uni`, `residuals.rma`, `ranef` (no DOI) | t intervals, EE BLUPs | read 2026-11-20 | Details |

Raudenbush and Bryk (1985) is not needed: Henderson and Searle et al. define the equations and Raudenbush (2009) prints the values, so no case
depends on it, and the list does not stop for it. Robinson (1991) and Kackar and Harville (1981), also cited by the help page, add
nothing the cases compare (no correction for an estimated `tau2` is made by either side).

**Published values.** The paper's first example is `dat.bcg` with `mods = ~ ablat_c + year_c` (latitude centred at 33, year at 1966) and the DL
estimator of its eq 9. It prints `tau2` 0.0790, for studies 4, 7 and 13 the hat value, `t_i`, `D_i` and `R_i`, the DFBETAS of study 4 and the
mean-shift dummy of study 4; metafopy reproduces each within half its last digit and agrees with R metafor 5.2-1 (measurements.md). The second
example (Table II) prints no per-study diagnostic; its rows give a derived case with three moderators. The third (Table III, not
bundled) prints only approximate statements ("approximately 25% more efficient"), so it is not used.

## API

```python
# formulas.py
@dataclasses.dataclass(frozen=True)
class UniDeletion:
    """Per study: Cook's distance, DFBETAS (rows by study), the deleted residual, its standard error and z, tau2(-i), QE(-i), DFFITS, COVRATIO,
    the hat value and the internally studentised residual (eqs 13, 14, 19, 25 and 27)."""

    cook: FloatArray
    dfbetas: FloatArray
    resid: FloatArray
    se: FloatArray
    z: FloatArray
    tau2_del: FloatArray
    qe_del: FloatArray
    dffits: FloatArray
    cov_r: FloatArray
    hat: FloatArray
    rstandard: FloatArray


def uni_deletion(
    yi: FloatArray,
    vi: FloatArray,
    method: str,
    mods: str | None = None,
    data: dict[str, FloatArray] | None = None,
    tau2: float | None = None,
    btt: list[int] | None = None,
) -> UniDeletion:
    """Leave-one-out diagnostics by explicit refits, Viechtbauer and Cheung (2010) eqs 15, 16, 20 and 23: each refit's tau2 from ``rma_tau2`` on
    the kept studies (or the fixed ``tau2``), ``b(-i)`` and ``vb(-i)`` by weighted least squares; ``data`` is cut to the kept rows. Raises
    ``ValueError`` naming the study when a deletion leaves the model matrix rank deficient."""


@dataclasses.dataclass(frozen=True)
class UniBlups:
    """Per study: the BLUP of the true effect ``x_i b + u_i`` and its prediction error standard error."""

    pred: FloatArray
    se: FloatArray


def uni_blups(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: float) -> UniBlups:
    """Henderson's mixed model equations with ``R = diag(vi)``, ``Z = I``, ``G = tau2 I`` (through ``mv_blups``); the variance of ``x_i b + u_i``
    from the inverse coefficient matrix, its ``b`` block, ``u`` block and their covariance."""


# test_derived.py
# auditme: accept[spec-api-signature-drift] the check in test_derived.py shares its name with the reference formulas.uni_deletion; the signature above is the reference's
def uni_deletion(case: correctness.cases.Case, result: typing.Any) -> None:
    """``cooks_distance`` (with the ``btt`` of the case's ``then``), ``dfbetas``, ``rstudent``, ``rstandard`` and every ``influence`` column of
    the case's fit (``refit``) against ``formulas.uni_deletion``, at ``DERIVED_TOLERANCE``; a NaN from metafopy where the reference has a value
    fails naming the study."""


def uni_ranef(case: correctness.cases.Case, result: typing.Any) -> None:
    """``ranef(fit)``: ``pred`` and ``se`` against ``formulas.mv_blups`` with one ``~ 1 | study`` term at metafopy's ``tau2``, and the interval
    bounds with the normal or t quantile; for EE fits every value is 0."""


def uni_blup(case: correctness.cases.Case, result: typing.Any) -> None:
    """``blup(fit)``: ``pred``, ``se`` and the interval bounds against ``formulas.uni_blups`` at metafopy's ``tau2``, with the normal or t
    quantile as ``ranef``."""


# test_published.py
def influence_printed(case: correctness.cases.Case, result: typing.Any) -> None:
    """``influence(fit)`` values a source prints: ``hat[i]``, ``rstudent[i]``, ``cook.d[i]``, ``R[i]`` (100 (tau2 - tau2.del) / tau2),
    ``dfbetas[i,j]`` and ``tau2``, through ``check_values``."""


def blup_printed(case: correctness.cases.Case, result: typing.Any) -> None:
    """``blup(fit)`` values a source prints: ``pred[i]``, ``min(pred)`` and ``max(pred)``, through ``check_values``."""


# test_properties.py
def uni_mean_shift(case: correctness.cases.Case, result: typing.Any) -> None:
    """Viechtbauer and Cheung section 3 (eqs 17, 18): for each study, the z statistic of a dummy for it added to ``mods`` equals its ``rstudent``
    ``z``. Holds for DL, REML and EE; not for ML (measurements.md), so an ML case is refused with ``ValueError``."""
```

## Coverage list

`coverage_items()` gains `uni_diagnostics_items()`, with one new constant in `test_coverage.py`, `UNI_FITS = ('ee', 'me', 're', 'tau2_fixed')`
(no list in the code names them), and the existing `MV_DIAGNOSTICS` for the function names:

- `rma_uni.<f>.<fit>` for each `f` in `MV_DIAGNOSTICS` and `fit` in `UNI_FITS` (`re`: estimated `tau2`, no moderators; `me`: estimated `tau2` with
  moderators; `ee`: an equal-effects fit; `tau2_fixed`: `tau2` given by the call), and `rma_uni.cooks_distance.btt`;
- `rma_uni.ranef.re`, `rma_uni.ranef.me`, `rma_uni.ranef.ee`, `rma_uni.ranef.pi.z`, `rma_uni.ranef.pi.t`;
- decision D1: `rma_uni.influence.<column>` for each column in a new constant `UNI_INFLUENCE = ('cov_r', 'dffits', 'hat', 'qe_del', 'tau2_del')`,
  and `rma_uni.rstandard.marginal`;
- decision D2: `rma_uni.blup.re` and `rma_uni.blup.me`;
- `[[exempt]]`, reason "defined only in metafor's code; R parity covers it": `rma_uni.<f>.knha` for each `f` (the `s2w` scaling, decision D3),
  `rma_uni.<f>.unweighted` and `rma_uni.<f>.weights` (`weighted=False` and user `weights`, which the paper does not define).

Out of scope: `hatvalues` (the same numbers as the `hat` column), `leave1out`, `cumul` and `weights` (decision D1 kept them out); `ranef`'s `transf` and `na_action` (presentation only) and `progbar`, which change no number.

## Cases

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
Tolerances: `published` half the last printed digit; `derived` `DERIVED_TOLERANCE` (1e-10); properties `PROPERTY_TOLERANCE` (1e-8). REML and ML cases
pass `control = {threshold = 1e-13, tol = 1e-13, maxiter = 1000, tau2_max = 1e6}` as `inference.toml` does: with the default `threshold` metafopy's
refits stop up to 1.7e-3 from the root (ML cov.r), with it the largest difference is 1.6e-11. DL and EE agree to 3.7e-14 with the defaults.
<!-- doc-counts: accept-end -->

| Case | Kind | Fit | `then` | Covers |
|---|---|---|---|---|
| `uni.diag.published.bcg` | published | Table I inline (counts, `ablat_c`, `year_c`), `escalc` RR, DL, `~ ablat_c + year_c` | `influence` | `*.me`, `influence.hat`, `influence.tau2_del` |
| `uni.diag.dl.bcg` | derived | the same fit | `rstudent` | `*.me` |
| `uni.diag.btt.bcg` | derived | the same fit | `cooks_distance(btt=[1, 2])` | `cooks_distance.btt` |
| `uni.diag.dl.wtl` | derived | Table II inline, DL, `~ length + meta + college` | `rstudent` | `*.me` |
| `uni.diag.reml.bcg` | derived | `dat.bcg`, REML, no mods | `rstudent` | `*.re` |
| `uni.diag.reml.raudenbush` | derived | `dat.raudenbush1985`, REML, no mods | `rstudent` | `*.re` |
| `uni.diag.ml.bcg.ablat` | derived | `dat.bcg`, ML, `~ ablat` | `rstudent` | `*.me` |
| `uni.diag.ee.bcg.ablat` | derived | `dat.bcg`, EE, `~ ablat` | `rstudent` | `*.ee` |
| `uni.diag.fixed.bcg.ablat` | derived | `dat.bcg`, `tau2 = 0.05`, `~ ablat` | `rstudent` | `*.tau2_fixed` |
| `uni.ranef.reml.raudenbush` | derived | `dat.raudenbush1985`, REML, `test = "z"` | `ranef` | `ranef.re`, `ranef.pi.z` |
| `uni.ranef.reml.bcg.ablat` | derived | `dat.bcg`, REML, `~ ablat`, `test = "t"` | `ranef` | `ranef.me`, `ranef.pi.t` |
| `uni.ranef.ee.bcg.ablat` | derived | `dat.bcg`, EE, `~ ablat` | `ranef` | `ranef.ee` |
| `uni.blup.published.raudenbush` | published | `dat.raudenbush1985`, REML | `blup` | `blup.re` |
| `uni.blup.reml.raudenbush` | derived | the same fit | `blup` | `blup.re` |
| `uni.blup.reml.bcg.ablat` | derived | `dat.bcg`, REML, `~ ablat` | `blup` | `blup.me` |

Every derived `uni.diag.*` case also checks every `influence` column and `rstandard` (decision D1), so they cover `rma_uni.influence.<column>` and
`rma_uni.rstandard.marginal`. The published `blup` case holds `pred[3]` "0.21", `min(pred)` "-0.03" and `max(pred)` "0.25" (Table 16.2 and section
16.5.6.1; metafopy 0.2144, -0.0293 and 0.2485, measurements.md); the book's mixed-model range is not used (Decisions).

The published case's `expect` holds the printed values of measurements.md (study 4 is index 3): `tau2` "0.0790", `hat`, `rstudent`, `cook.d` and
`R` of studies 4, 7 and 13, and `dfbetas[3,0]` to `dfbetas[3,2]`; its locator names section 5.1 and Table I. No case is expected to be a `mismatch`:
every reference measured agrees with metafopy to 1.6e-11 or better.

Property cases (`properties.toml`, check `test_properties.uni_mean_shift`): `property.uni.mean-shift.dl.bcg` (the paper's fit) and
`property.uni.mean-shift.reml.bcg` (`~ ablat`, tight control); measured differences 1.5e-14 and 3.9e-15. The existing
`property.mv.diagnostics-equal-uni` case already ties the `rma_mv` and `rma_uni` diagnostics together.

## Error handling

- `formulas.uni_deletion` raises `ValueError` naming the study when its deletion leaves `X` rank deficient (metafor returns NaN); no case deletes
  such a study. `rma_tau2` has no failure mode on these fits (closed forms, or `brentq` on a bracketed root).
- Where metafopy returns NaN for a study (its refit failed) and the reference has a value, the check fails naming the study: that is a disagreement
  and goes to review as a `mismatch`, not an exemption.
- `uni_mean_shift` raises `ValueError` for an ML case, whose identity does not hold.
- The existing loader errors (malformed case, unknown check, duplicate id, `mismatch` without `reason`, a printed name the check does not know)
  apply unchanged.

## Implementation

1. `formulas.uni_deletion`, with a harness test in `test_coverage.py`: on a small inline dataset, an intercept-only fit with
   `tau2` fixed against eq 24 for DFBETAS and the intercept-only form of eq 20, `(mu - mu(-i))^2 sum(w)`, which the refits must reproduce.
2. The checks in `test_derived.py`, `test_published.py` and `test_properties.py`; the items and the exemptions in `test_coverage.py`, failing
   until step 3.
3. The cases above, each with its `source` (citation, DOI, locator, `read`). None is `slow`: no measured fit took 0.2 s with its reference.
4. Sources: every source a case cites is in the folder; check each case's `source.locator` against the paper it names. No work stops here.
5. Review: every `mismatch` case gets an item in `mismatches.md` (source, reference, metafopy and metafor values, options *follow the reference*,
   *keep metafor*, *investigate*) with its review page, per the shared `reviews.md` rule. **Work stops** until each has an answer. If there is none,
   `mismatches.md` says so under its automatic checks and the step needs no answer.
6. Close: file each *follow the reference* answer as a `TODO.md` item (fix, flip the case, mark the parity fixtures, `known.toml`,
   `docs/differences.md`); move the `TODO.md` item "No correctness case covers the rma_uni diagnostics" to `DONE.md`; set this spec's status to
   implemented in `agents/specs/index.md`.

## External dependencies

- `numpy`, `scipy`, `pandas`, `pytest` (existing); no new dependency, and no R at test time.
- The papers in the folder named above; nothing to supply.

## Decisions

Answered by the operator on 2026-11-20 on the review page of `agents/reviews/correctness-uni-diagnostics-spec.md`, each with no comment:

- **D1, the other diagnostics the paper defines: include.** DFFITS, COVRATIO, `tau2.del`, `QE.del`, the hat values and `rstandard` are checked
  against the paper's equations with the same refits; `leave1out` and `cumul` stay out.
- **D2, `blup`: include**, with a `published` case from Raudenbush (2009) for the estimates and their range. The book's mixed-model range (-0.07,
  0.41) is not reproduced by its own rounded coefficients (-0.064) and is not used; its posterior variance omits the uncertainty of `b`, so the `se`
  is derived only.
- **D3, the Knapp-Hartung scaling: exempt**, "defined only in metafor's code; R parity covers it"; `test="knha"` diagnostics get no case.
