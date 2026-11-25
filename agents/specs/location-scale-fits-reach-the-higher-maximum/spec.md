# Location-scale fits reach the higher maximum

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-25
**Review:** `agents/reviews/location-scale-fits-reach-the-higher-maximum-spec.md` (page https://claude.ai/artifact/MsBFkxwxF87RoVgnkLtSKv); answered 2026-11-25: approved, with every choice under Decisions.

The likelihood of a log-link location-scale model (`rma_uni(..., scale=)`) can have more than one interior maximum. metafopy, like metafor
5.2-1, fits from one start (`start_alpha`, a regression of the transformed squared residuals of a HE fit) and reports whichever maximum that start
leads to. The maximum likelihood estimator of Viechtbauer and Lopez-Lopez (2022, doi:10.1002/jrsm.1562, section 2.5) is the point of highest
likelihood, so where the start leads to the lower of two maxima, metafopy and R both report a fit that is not the estimate the paper defines.
This spec adds one derived restart per scale slope after the usual fit, and keeps the restart's end point only when its log-likelihood is higher
and it is a regular interior maximum. The operator chose on 2026-11-25 that metafopy reports the higher maximum (the `/todo-do` interview for the
item below), departing from R; this spec lists every output that changes, so approving it is the confirmation `claude/rules/metafopy.md` asks for.

**Approved** by the operator on 2026-11-25 on the review page (no comment), with every output change listed below. Version 1.1 records the
approval; the spec's content is unchanged from 1.0.

## Motivation

The `TODO.md` item "Location-scale fits can stop at a lower local maximum than the test reference", as filed:

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
> Found on 2026-11-24 while measuring `ls.permutest.location.exact.ML.z.bcglat` on macOS arm64 at `282a6b9`
> (`agents/specs/location-scale-fits-stable-across-platforms/measurements.md`). On 8 of the 1210 regular permutations (indices 68, 264, 433, 651,
> 805, 979, 1110, 1278 in `distinct_label_orders` order) metafopy's `z` differs from the test reference's by more than 1e-4. On permutation 68
> metafopy and R both stop at log-likelihood `-12.58248229` (alpha `[-0.778, -0.951]`); the test's reference fit reaches the higher
> `-12.57323737` (alpha `[-3.042, 2.526]`). Permutations 264 and 433 likewise.
>
> Note 2026-11-24: permutation 433 alone puts metafopy one over the reference bound of `ls.permutest.location.exact.ML.z.bcglat`, now a
> strict-xfail mismatch naming this item; remove its `status = "mismatch"` once this item is fixed.
<!-- doc-counts: accept-end -->

Measured on 2026-11-25 on macOS arm64 at `9bb0947` (scripts in the session scratchpad, not kept; the figures are the record):

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
- The test reference `formulas.ls_reference_fit`, rewritten by Task 3 of the
  [stability spec](../location-scale-fits-stable-across-platforms/spec.md), now agrees with metafopy on six of the eight permutations and reaches
  the higher maximum only on 433 (ll `-11.96789514` against `-12.33206543`) and 1278 (`-12.17150179` against `-12.65213972`): it stops once two of
  its starts agree, and on the other six both stop low.
- R metafor 5.2-1 gives metafopy's values on 433 and 1278 from its default start, and the higher maximum with `control = list(alpha.init = ...)` at
  the reference's point (433: z `-2.5047` and `1.9024`; 1278: z `-0.3574` and `1.7704`).
- Refitting all 1287 permutations from 15 extra starts: 8 permutations (the eight above) have a higher interior maximum, each the "group swap"
  (alpha near `[-3.5, 3.2]`, the slope of the opposite sign); 80 have a higher value on the ridge where `tau2` tends to 0 in one `lat` group (alpha
  near `[-27, 27]`, no finite maximum), and 68 default fits already lie on that ridge.
- A prototype of this spec (one restart per slope, the guard below) changes exactly the eight permutations; the case's `lat` count moves from
  308 to 306 of 1287 (p `0.23932` to `0.23776`) and the intercept's from 794 to 793, both inside the reference bounds (239 to 307, 782 to 850).
<!-- doc-counts: accept-end -->

The 80 ridge permutations are not part of this spec: their likelihood rises towards a boundary, so no start gives a finite maximum, and the guard
leaves them as R does.

## Module location

`src/metafopy/models/rma_ls_fit.py` (the restart, the guard, `LSControl.restarts`), `src/metafopy/models/rma_ls_result.py` (`LSControl`),
`src/tests/correctness/formulas.py` (the reference fit's starts), `src/tests/correctness/cases/location_scale.toml`, `src/tests/test_rma_ls.py`,
`docs/differences.md`.

## API

```python
# src/metafopy/models/rma_ls_result.py
@dataclasses.dataclass(frozen=True)
class LSControl:
    ...
    restarts: bool = True
    """Refit a log-link model from one derived start per scale slope and keep a higher regular maximum (metafopy only; ``False`` is metafor's
    single start)."""


# src/metafopy/models/rma_ls_fit.py
RESTART_STEP = 4.0  # a restart moves each slope towards and past zero by this many standard deviations of its column of Z, inverted
RESTART_GAIN = 1e-6  # a restart replaces the fit only when its log-likelihood is higher by more than this
RESTART_RATIO = 1e-5  # ... and its information on the standardised Z has min/max eigenvalue ratio above this


def restart_starts(alpha: FloatArray, Z: FloatArray) -> list[FloatArray]:
    """One start per non-constant column ``j`` of ``Z``: ``alpha`` with ``alpha[j] - sign(alpha[j]) * RESTART_STEP / sd(Z[:, j])`` (``sign(0)`` is 1)."""


def is_regular_maximum(problem: LSProblem, alpha: FloatArray) -> bool:
    """The analytic information ``problem.ls_information`` at ``alpha``, taken on the standardised ``Z`` (each non-constant column centred and
    scaled, as ``formulas.ls_standardise``), is finite and has min/max eigenvalue ratio above ``RESTART_RATIO``."""
```

`control={'restarts': False}` is accepted by `rma_uni` and is carried into every refit (`refit_ls`), so `permutest`, `profile` and the
leave-one-out methods of such a fit stay at metafor's single start.

## Implementation

### When it runs

After `fit_rma_ls` has its end point (the existing BFGS, Nelder-Mead and `polish_log_fit` steps), and only when all of these hold: `link == 'log'`;
`con.restarts`; `optbeta` is false; no scale coefficient is fixed (`alpha_arg` all NaN); `alpha_min` and `alpha_max` are infinite; no user
`weights`; `Z` has a non-constant column. Otherwise the fit is unchanged.

Fits with a fixed coefficient (profile and `confint` refits) are left out because one free coefficient leaves the guard nothing to check (the
prototype changed 100 such refits with an eigenvalue ratio of 1). Bounded `alpha`, `optbeta` and user weights were not measured.

### The restart

For each start of `restart_starts(alpha_hat, Z)`, run the same pipeline as the first fit from that `alpha_init` (the `mapinvfun` transform, the
optimiser of the call, `polish_log_fit` when it applies), without the Hessian. A start whose fit raises `EstimationError` or gives a non-finite
log-likelihood is skipped. The best end point replaces `alpha_hat` when its log-likelihood exceeds the first fit's by more than `RESTART_GAIN` and
`is_regular_maximum` holds; the Hessian, `tau2`, `beta` and everything after are then computed at it as now.

Why these values (measured 2026-11-25 at `9bb0947`, over the 1287 permutations and the 10697 location-scale fits the test suite makes):

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
- `RESTART_STEP`: steps 3, 4 and 6 recover all 8 higher maxima, step 2 recovers 7; reflecting the slope about the column's midrange recovers 6 and
  shifting the intercept by 2 or 4 recovers at most 2.
- `RESTART_RATIO`: the information ratio on the standardised `Z` is `9.1e-3` to `0.93` at 30727 interior end points and below `2.7e-8` at 2735
  ridge end points; `1e-5` lies about three orders from each side. The reference's `LS_DEGENERATE` (1e-8) would let 25 ridge points through.
- `RESTART_GAIN`: restarts reaching the same maximum agree to `5e-15`; the smallest gain between distinct maxima was `9.2e-3`.
<!-- doc-counts: accept-end -->

### The test reference

`formulas.ls_reference_fit` gains the same derived restarts from its own end point (in the reference's own code, not imported from metafopy) and
`ls_is_maximum` on the standardised `Z` with the threshold `1e-5`, so that it reaches the higher maximum on all eight permutations. Without it the
reference's lower bound of the `lat` count stays one hit high (permutation 979 counts there, not at the higher maximum).

### Tests and records

- `ls.permutest.location.exact.ML.z.bcglat` loses `status = "mismatch"` and its `reason` (the strict xfail would otherwise fail on the pass).
- A new correctness case on permutation 433 of the same data checks that `rma_uni` reports the higher maximum (ll `-11.96789514`, z `1.9024`)
  against `ls_reference_fit`, and that `control={'restarts': False}` gives R's (ll `-12.33206543`, z `-2.5047`).
- The R-parity tests whose fits change (below) pass `control={'restarts': False}`, so they keep checking the port of metafor's procedure.
- `docs/differences.md`, under "Optimiser stop points": the restart, when it runs, the R behaviour it departs from and `restarts=False`.
- A scoped audit run, `bin/metafor-audit.sh --only 'construct/ls*'` (and the permutest scripts), before and after; every new difference is a
  restarted fit, recorded in `src/audit/data/known.toml` with this spec as its reason, or the task stops.

## Output changes

Measured with the prototype on 2026-11-25 at `9bb0947`:

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
- Main fits (`rma_uni` calls of the tests and the correctness cases): none change.
- Permuted refits with every coefficient free: 43 change, all from a second interior maximum - 8 in the bcglat case, 23 in
  `test_rma_ls.py::test_permutest_exact_iter_and_exact`, 8 across `test_permutest[t|both|btt|less|alpha]`, 3 in the `test_ls_vif` permutation
  simulation and 1 in `test_audit8_crashes::test_ls_fit_on_a_ridge_is_fast`; REML fits among them.
- p-values that move: bcglat `lat` 0.23932 to 0.23776 (intercept 0.61694 to 0.61616); `test_permutest[less]` pval 0.98 to 0.96, `[px2]` 0.12 to
  0.16, `[btt]` pval_alpha 0.65 to 0.675, `[t]` pval_alpha 0.52 to 0.54. Nine tests fail without the `restarts=False` above, and the bcglat case passes.
<!-- doc-counts: accept-end -->

## Error handling

A restart that raises `metafopy.errors.EstimationError` or `numpy.linalg.LinAlgError`, or gives a non-finite log-likelihood, is skipped without a
warning: the first fit stands. No new error reaches the caller from a restart. `make_ls_control` checks the new entry: a `restarts` that is not a
bool raises `metafopy.errors.InputError` (the existing entries are not type-checked; this one is, because a truthy string such as `'FALSE'` would
otherwise turn the restarts on).

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `control['restarts']` | `bool` | `True` | Run the derived restarts of a log-link location-scale fit; `False` gives metafor's single start |

## Cost

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
The prototype (taking the information from the test formulas, which is slower than `LSProblem.ls_information`): one bcglat fit 4.7 ms to 13.0 ms;
the exact permutest of 1287 refits 7.5 s to 18.6 s.
<!-- doc-counts: accept-end -->

## External dependencies

None new: numpy and scipy, already used.

## Decisions

### Choices made, not asked

- **Default on, with `restarts=False` as the way back to R** - follows from the operator's choice that metafopy reports the higher maximum;
  reversible by changing the default.
- **R-parity tests pass `restarts=False`** rather than recording each changed permutation as a departure: they test the port of metafor's
  procedure, and the new correctness case tests the departure.
- **The identity link is left out and filed as its own `TODO.md` item.** On the bcglat data 83 of 1287 identity-link permutations have a higher
  maximum, most on the face `tau2 = 0` of one group (a valid boundary estimate), and no start kind tried recovered more than 12 of them
  (measured 2026-11-25). It needs its own start design and a guard on the active face.
- **The 80 ridge permutations stay as R leaves them**: they have no finite maximum, and the guard rejects every ridge end point.
