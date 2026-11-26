# Identity-link location-scale fits reach the higher maximum

**Version:** 1.1
**Status:** approved
**Date:** 2026-11-26
**Review:** `agents/reviews/identity-link-location-scale-fits-reach-the-higher-maximum-spec.md` (page https://claude.ai/artifact/BKmKMVHKKBrZjvFvSJMWhS); answered 2026-11-26: approved, with every choice under "Choices made, not asked".

With `link='identity'` the likelihood of a location-scale model (`rma_uni(..., scale=)`) is maximised over `Z alpha >= 0`, and its maximum can lie
on a face where `tau2_i = 0` for some rows. metafopy, like metafor 5.2-1, fits from one start (`start_alpha`), runs the optimiser (`constrOptim`'s
stand-in) and then `rma_ls_opt.polish_identity`, so it reports whichever maximum that start leads to. The maximum likelihood estimator of
Viechtbauer and Lopez-Lopez (2022, doi:10.1002/jrsm.1562, after equation 8) is the point of highest likelihood within the constraint. This spec
adds derived starts after the identity polish - one on each face `z_i' alpha = 0` and one reflection per scale slope - each polished by
`polish_identity` alone, and keeps the best when its log-likelihood is higher and it is a regular maximum on its face. It extends
[Location-scale fits reach the higher maximum](../location-scale-fits-reach-the-higher-maximum/spec.md) (implemented 2026-11-25) to the identity
link, behind the same `control['restarts']`. On 2026-11-25 the operator chose that metafopy reports the higher maximum, departing from R; this spec
lists every output that changes, so approving it is the confirmation `claude/rules/metafopy.md` asks for.

## Motivation

The `TODO.md` item "Identity-link location-scale fits can stop below a higher maximum on the face `tau2 = 0`", as filed:

<!-- doc-counts: accept-start found on 2026-11-25 - a dated record, not a claim about the present -->
> Found on 2026-11-25 at `9bb0947` while prototyping `agents/specs/location-scale-fits-reach-the-higher-maximum/spec.md` (macOS arm64): on the data of
> `ls.permutest.location.exact.ML.z.bcglat` fitted with `link='identity'`, 83 of the 1287 permutations of the location model have a higher maximum
> than metafopy's single-start fit reaches, most of them on the face `tau2 = 0` of one `lat` group (a valid boundary estimate under this link). No
> start kind tried (slope reflection, a start just inside that face, a start mapped from the log-link fit) recovered more than 12 of the 83.
<!-- doc-counts: accept-end -->

Measured on 2026-11-26 on macOS arm64 at `f0f6d44`, on the same data (`dat.bcg` RR, `mods = ~ lat` permuted, `scale = ~ lat`, ML; scripts in the
session scratchpad, not kept; the figures are the record):

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
- The true maximum, from a 41 x 41 grid over `(tau2_lat0, tau2_lat1) >= 0` with every grid maximum polished (L-BFGS-B, bounds at 0): 85 of the
  1287 default fits are below it by more than 1e-6, 81 of those with the maximum on a face `tau2_g = 0`; no default end point lies on a face. The
  1287 likelihoods have one (740), two (468) or three (79) distinct local maxima. The 85 include the 83 of the item.
- Why the earlier starts failed: a refit through the optimiser from just inside a face maximum keeps it on 84 of the 85, but from a start on the face
  (`tau2_g = 0` exactly) on 4, because the barrier optimiser needs an interior start; and from a start beside the face with the other group's `tau2`
  at the default fit's value on 44. On permutation 679 the refit from 1e-3 inside the face maximum drifts back to the lower interior maximum.
- Polishing with `polish_identity` alone (no optimiser) from the face and reflection starts below: all 85 reach the grid maximum (84 by a face
  start, 1 by the reflection), no other permutation changes, and none ends above the grid maximum. A face start with an entry printed `-0.` (a negative rounding residue) evaluated to a
  log-likelihood of `-inf` until near-zero entries were set to exactly 0 (that version recovered 71).
- R metafor 5.2-1 from its default start gives metafopy's values (permutation 18: ll `-12.62494606`, z of `latp` `0.2490`; 68: `-12.58248234`;
  679: `-10.46818403`; 914: `-12.64295234`); with `control = list(alpha.init = ...)` 1e-3 inside the grid maximum it reaches it on 18 (ll
  `-11.42937473`, z `-3.5966`), 68 and 914, but not on 679.
- The test reference `formulas.ls_reference_fit` (SLSQP from `ls_starts` and `ls_identity_starts`) reaches the grid maximum on 73 of the 85,
  permutation 18 among them.
- At the 85 maxima the information projected on the face (`formulas.ls_face_basis`) is positive definite, face dimension 1 or 2, min/max
  eigenvalue ratio at least `5.6e-3`.
<!-- doc-counts: accept-end -->

## Module location

`src/metafopy/models/rma_ls_fit.py` (the restart and the guard), `src/metafopy/models/rma_ls_result.py` (`LSControl.restarts` docstring),
`src/tests/test_rma_ls_restarts.py`, `src/tests/correctness/cases/location_scale.toml`, `docs/differences.md`.

## API

```python
# src/metafopy/models/rma_ls_fit.py
def identity_restart_starts(alpha: FloatArray, Z: FloatArray) -> list[FloatArray]:
    """Starts for the identity link from the end point ``alpha``, each with ``Z alpha >= 0``.

    One per distinct non-zero row ``r`` of ``Z``: the nearest ``alpha`` (in the metric ``Z'Z``) with ``r' alpha = 0``, kept only when it is feasible
    on every row. One per non-constant column ``j`` when ``Z`` has a constant column: the slope reflected about the column's midrange
    (``alpha[j] -> -alpha[j]``, the intercept gains ``2 alpha[j] c_j``), which swaps the ``tau2`` at the column's two ends; kept when feasible.
    """


def is_face_maximum(problem: LSProblem, alpha: FloatArray) -> bool:
    """The analytic information ``problem.ls_information`` at ``alpha``, projected on the null space of the active rows (``z_i' alpha`` within
    ``rma_ls_opt.ACTIVE_TOL`` of zero, relative to the largest), is finite and has min/max eigenvalue ratio above ``RESTART_RATIO``. A face of
    dimension 0 (the active rows pin every coefficient) passes."""


def identity_restart_fit(problem: LSProblem, par_opt: FloatArray, alpha_arg: FloatArray) -> FloatArray:
    """Polish from each of :func:`identity_restart_starts` with ``rma_ls_opt.polish_identity`` and return the end point of highest log-likelihood
    when it beats ``par_opt`` by more than ``RESTART_GAIN`` and :func:`is_face_maximum` holds there; otherwise ``par_opt``."""
```

`LSControl.restarts` keeps its name and default (`True`); its docstring and `docs/differences.md` say that it now covers both links.

## Implementation

### When it runs

In `fit_rma_ls`, right after the existing `polish_identity` of the end point, and only when all of these hold: `link == 'identity'`;
`con.restarts`; `optbeta` is false; no user `weights`; no scale coefficient is fixed (`alpha_arg` all NaN); `Z` has a non-constant column.
Bounds on `alpha` are already refused for this link. The conditions are those of the log-link restart, so fits with a fixed coefficient
(`profile`, `confint`) and every refit with `restarts=False` keep the single start; `permutest`, `vif(sim=True)` and the leave-one-out methods
carry the setting through `refit_ls` as they do for the log link.

### The starts

For a distinct non-zero row `r` of `Z` (rows compared exactly, on the `Z` that `fit_rma_ls` passes to `polish_identity`, standardised as metafor does), with `G = pinv(Z'Z)`:
`a = alpha - G r (r' alpha) / (r' G r)`. Entries with absolute value below `1e-12` times `max(1, max|a|)` are set to 0, so that a dummy face is
exactly `tau2 = 0` (a negative rounding residue, printed `-0.`, evaluates to `-inf`, measured above). When `Z` has a constant column and the smallest `Z a` is negative
by no more than `ACTIVE_TOL * max(1, max|Z a|)`, that deficit is added to the intercept; a start still infeasible is dropped. For a binary column
the face starts are the two `tau2_g = 0` starts with the other group's `tau2` unchanged; for a continuous column only the rows at the ends of the
column give feasible starts (measured 2026-11-26: 90 of 278 face starts feasible across the test suite's identity fits).

### The restart

Each start goes to `polish_identity` with the fit's own `loglik`, `ls_score` and `ls_information` and every coefficient free; no optimiser runs,
because the barrier stand-in needs an interior start and drifts off a face maximum (permutation 679 above). `polish_identity` already keeps
`Z alpha >= 0` and never ends below its start. A start whose polish raises `EstimationError` or `LinAlgError`, or ends with a non-finite
log-likelihood, is skipped. The best end point replaces the fit's when its log-likelihood is higher by more than `RESTART_GAIN` (1e-6, the
log-link value: restarts reaching the same maximum agree to rounding, and the smallest gain among the 85 was `2.6e-3`) and `is_face_maximum` holds
with `RESTART_RATIO` (1e-5, the log-link value; the 85 maxima have at least `5.6e-3`). The Hessian, `tau2`, `beta` and everything after are then
computed at it as now.

### The test reference

`formulas.ls_reference_fit` is not changed: it reaches the higher maximum on permutation 18, which the new case uses, and no correctness case fits
the identity link on the bcglat permutations.

### Tests and records

- `src/tests/test_rma_ls_restarts.py`: `identity_restart_starts` on a binary `Z` gives the two `tau2_g = 0` starts and the swap; on a continuous
  `Z` only feasible starts; a start's near-zero entries are exactly 0; `is_face_maximum` passes a face maximum and refuses a singular face; a
  `restarts=False` identity fit equals the single-start fit.
- A new correctness case on permutation 18 of the bcglat data with `link='identity'`: `rma_uni` reports the higher maximum (ll `-11.42937466`,
  `tau2` of `lat = 0` equal to 0) against `ls_reference_fit`, and `control={'restarts': False}` gives R's single start (ll `-12.62494606`, z of
  `latp` `0.2490`), from an R fixture generated by `src/tests/r/generate_location_scale.R`.
- `docs/differences.md`, under "Location-scale fits with link="identity"": the restart, when it runs, the R behaviour it departs from and
  `restarts=False`.
- A scoped audit run, `bin/metafor-audit.sh --only '*/ls_*'`, before and after; every new difference is a restarted identity fit, recorded in
  `src/audit/data/known.toml` with this spec as its reason, or the task stops.

## Output changes

Measured with the prototype on 2026-11-26 at `f0f6d44`:

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
- The location-scale tests and correctness cases (701 tests): no fit changes and every test passes. 47 identity polishes there had every
  coefficient free, 26 of them with a continuous `Z` of 9 distinct rows.
- On the bcglat permutations with the identity link: the 85 fits above move to the higher maximum; the location `z` can change sign (permutation
  18: `0.2490` to `-3.5966` in R at the two maxima). No test or correctness case runs this permutest.
<!-- doc-counts: accept-end -->

## Error handling

A start that is infeasible is dropped; a polish that raises `metafopy.errors.EstimationError` or `numpy.linalg.LinAlgError`, or ends non-finite,
is skipped without a warning: the first fit stands. No new error reaches the caller. `control['restarts']` keeps its existing check (a non-bool
raises `metafopy.errors.InputError`).

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `control['restarts']` | `bool` | `True` | Run the derived restarts of a location-scale fit, for the log and now the identity link; `False` gives metafor's single start |

## Cost

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
The prototype on the bcglat identity fits: 39.4 ms to 97.2 ms per `rma_uni` fit (three extra polishes for a binary `Z`), so the exact permutest of
1287 refits from about 51 s to about 125 s.
<!-- doc-counts: accept-end -->

## External dependencies

None new: numpy, already used.

## Decisions

### Choices made, not asked

- **The identity link follows the operator's choice of 2026-11-25** (metafopy reports the higher maximum) and shares `control['restarts']`, so one
  setting returns both links to R's single start.
- **Polish only, no optimiser refit**: the optimiser loses face maxima (above); `polish_identity` is the step that already finishes every
  identity fit.
- **The reflection start is kept** although it recovered 1 of the 85 (permutation 914, an interior maximum that no face start reaches).
- **The test reference is left as it is**: no case needs more of it; its 12 misses on the bcglat permutations matter only to a permutest case,
  and none exists for this link.
- **The same conditions as the log-link restart** (every coefficient free, no `optbeta`, no user weights): fixed-coefficient refits were left out
  of the log-link spec for its guard's sake, and the same holds on a face of dimension 1.
