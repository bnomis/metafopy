# rma_mv REML stop

**Version:** 1.2
**Status:** implemented
**Date:** 2026-11-17
**Corrected:** 2026-11-21 (version 1.2): statements that no longer matched the code, from audit review-1 (`agents/audit-reviews/review-1.md`, finding group F6)

`rma_mv` with the default optimiser maximises the (restricted) log-likelihood with SciPy's BFGS on the analytic score and accepts a stop that BFGS
itself reports as failed ("precision loss") whenever the largest gradient entry is under `SLACK_GRADIENT` (1e-4). Where the line search loses precision
early, the fit is accepted up to 7 orders of magnitude further from the optimum than where it stops on the development Mac, and the stop varies between
platforms and between runs. This spec adds a guarded Newton polish on the analytic score after the BFGS stop, so that every platform finishes at a
rounding-size score, and narrows the test `property.mv.reml-score.kon` to that. The change is to the library, so its review is the user's
confirmation that metafopy's output changes (`claude/rules/metafopy.md`); every output that moves is listed below with its measured size.

The source is the `TODO.md` item "The rma_mv REML fit stops short of the optimum on Linux and Windows" (filed 2026-11-17). It follows the
[Cross-platform numerics](../cross-platform-numerics/spec.md) spec, whose Task 8 found it, and touches the code tested by
[Correctness: rma_mv and robust](../correctness-mv-robust/spec.md).

## Motivation

### The `TODO.md` item, as filed

<!-- doc-counts: accept-start found on 2026-11-17 - a dated record, not a claim about the present -->
> Found on 2026-11-17 by Task 8 of `agents/tasks/cross-platform-numerics/tasks-1.md` (probe run 44002318304 of commit `f57ab45`; figures in
> `agents/specs/cross-platform-numerics/measurements.md`, "The rma_mv REML score"). The REML score of `sigma2` in `property.mv.reml-score.kon`, relative to
> its scale `tr(P Z Z')`, is 1e-11 on the development Mac and the containers, 1.3e-07 and 3.8e-08 on CI Linux, and up to 1.1e-06 on CI Windows; the BFGS
> gradient is 1.3e-10 (Mac), 6.0e-07 (Linux) and 1.8e-05 (Windows). The stop also varies from run to run on one platform.
>
> Cause: `scipy.optimize.minimize` BFGS ends on "precision loss" with `success` false on every platform, and `mv_opt.maximise` accepts the stop because the
> gradient is under `SLACK_GRADIENT` (1e-4). Where the line search loses precision early, the fit is accepted up to 4 to 7 orders of magnitude from the
> optimum. The test asserts the measured range (2e-6 of the scale) meanwhile.

The CI probe of commit `f57ab45` (`measurements.md`, "The CI probe run of commit `f57ab45`"): CI Linux 8 iterations, gradient 6.0e-07, score
[1.3e-05, 3.7e-05]; CI Mac 11 iterations, 1.3e-10, [1.4e-09, -8.3e-09]; CI Windows 8 iterations, 1.8e-05, [5.1e-05, 1.1e-03]; the log-likelihood
-7.2404667768 on all three to 1e-12. The operator answered on 2026-11-17: assert the measured range now (tolerance 2e-6 of the scale), file the fix.
<!-- doc-counts: accept-end -->

### Measured on 2026-11-17 while writing this spec

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->
Development Mac, main at `46b8361`, numpy and scipy from `uv.lock`. The polish was prototyped outside the repository as a wrapper around
`mv_opt.maximise` (scripts `mvpolish.py`, `kon.py`, `parity.py`, `rscore.py` in the session's scratch directory, not kept); it is the algorithm of
Implementation below with the step cap of the recommended option. An early BFGS stop like CI's was reproduced by loosening `gtol` or `maxiter`:
`gtol=1e-5` gives CI Linux's figures exactly (gradient 6.0e-07, relative score 1.28e-07 and 3.80e-08).

**`property.mv.reml-score.kon`** (`dat.konstantopoulos2011`, `~ 1 | district/school`, `mods = ~ year`, REML). Relative score = score / `tr(P Z Z')`:

| Stop | Before: gradient, relative score | After: Newton steps, gradient, relative score |
|---|---|---|
| default control | 1.8e-10; 9.2e-12, -1.2e-11 | 1; 3.6e-11; -7.8e-12, -2.0e-13 |
| case's `MV_CONTROL` (`gtol` 1e-12) | 1.3e-10; 1.3e-11, -8.6e-12 | 1; 2.5e-11; -8.6e-12, -2.0e-13 |
| `gtol=1e-5` (as CI Linux) | 6.0e-07; 1.3e-07, 3.8e-08 | 1; 4.2e-12; 2.9e-12, 6.3e-14 |
| `gtol=1e-4` (as CI Windows) | 2.5e-05; 6.4e-07, 1.6e-06 | 1; 8.6e-12; -5.9e-13, -6.2e-13 |
| `maxiter=4` | 3.8e-06; -1.0e-06, -1.5e-07 | 1; 2.0e-11; 5.8e-14, 1.0e-13 |

After the polish every stop ends at `sigma2` [0.072265591338 or ...339, 0.032650195992], against [0.072265591337, 0.032650195993] before on the Mac
(the stops above: relative change at most 2.8e-11 on the default and `MV_CONTROL` fits, 1.1e-6 from the `maxiter=4` stop). `beta[1]` 5.3118182720e-03 to
5.3118182724e-03 after (8e-11 relative spread), `se[1]` 9.4357939553e-03 to ...9556e-03; the log-likelihood moves by at most 3e-11, which is rounding.
R metafor 5.2-1's own estimates (`src/tests/reference/phase4/rma_mv.json`, `kon_mods`, the same model with `year` centred) have a relative score of
8.0e-09 and 1.6e-09: R's `nlminb` (finite-difference gradient, `rel.tol` 1e-10) stops further from the optimum than the Mac's BFGS and the polish.

**Every fit of `src/tests/test_rma_mv.py` `CASES`, polish on against off** (largest relative change of `sigma2`, `tau2`, `rho`, `gamma2`, `phi`,
`beta`, `se`; relative to `max(|value|, 1e-8)`):

- with the step cap: the largest change is 7.1e-08 (`ass_mods`, `ass_mods_t`: `sigma2`), then 5.7e-08 (`sim_gen`: `rho`); every other fit changes by
  at most 2.7e-08, most by 1e-09 or less. The distance to R's fixture is unchanged to within those amounts (`ass_mods` 7.1e-08 before, 5.0e-08 after;
  `kon_mods` 4.4e-09 before, 8.9e-09 after). The log-likelihood changes by at most 2e-14, except `kon_weights` (+7.4e-10, a weighted fit, whose
  objective is not the likelihood maximised). Fits with nothing to estimate (`kon_fe`, `kon_fix_all`, ...) are untouched. `kon_spexp`, `sim_R_known`,
  `sim_R_cov0`, `sim_R_none`, `sim_R_sparse` and `sim_gdiag` are left unchanged by the cap (their first Newton step is 0.06 to 1.0 in the transformed
  parameters: a ridge or a boundary); `sim_s_and_g` and `sim_two_inner_outer` by the positive-definiteness guard.
- without the cap, those fits move: `kon_spexp` `rho` 0.0559587 to 0.0393538 (R 0.0559200; log-likelihood +2.8e-09), `sim_R_*` `sigma2[0]` from 4.2e-09
  (`cov0`: 8.0e-09) to 3.8e-12 (`cov0`: 7.3e-12) (R 4.3e-10 and 8.8e-10; log-likelihood +5.5e-08 and +6.1e-08, `beta` nearer R's), `sim_gdiag` `tau2[2]`
  2.3e-10 to 4.2e-12 (R 1.2e-10; +2.7e-09). The `kon_spexp` change fails that fixture's `FLAT` tolerance (2e-3).

The cost is one Hessian of the score per Newton step (two score evaluations per free parameter): fits of `CASES` took 0.00 to 0.10 s before and
after, the largest increase `sim_na_un` 0.05 to 0.10 s.

Tests on the development Mac with the capped prototype loaded (`pytest -p mvpolish`, every `src/tests` file that names `rma_mv`, `RMAMV` or
`mv_opt`, `-m "not slow"`): 4098 passed, 13 skipped, 1 xfailed, none failed (13:28 to 13:41 UTC). In the arm64 Linux container
(`bin/linux-tests.sh`, a local clone of `46b8361` with the prototype loaded from `src/tests/conftest.py` and `property.mv.reml-score.kon` at
`tolerance = 1e-10`; `test_properties.py`, `test_rma_mv.py`, `test_rma_mv_extras.py`): 390 passed, 11 skipped (skips that main has too: printed
digits on flat fits, coefficient labels). The containers' unpolished stop was already at 1e-11 (`measurements.md`), so that run shows no regression on
Linux, not the CI fix: CI Linux and Windows are checked after merge.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/models/mv_opt.py`: new `polish_score` and `score_jacobian`, the `POLISH_*` constants, a call in `maximise`.
- Tests: `src/tests/test_mv_opt.py` (new), `src/tests/correctness/cases/properties.toml`, `src/tests/correctness/test_properties.py` (comment).
- Records: `docs/differences-notes.md` ("Optimisers and flat likelihoods"), `bin/platform-probe.py` (the `mv-reml-score` group records the polish).

## API

```python
POLISH_GRADIENT = 1e-10  # stop when the largest |gradient| entry of -ll is at most this
POLISH_MAX_STEP = 1e-3  # a Newton step with a larger entry (transformed scale) means a ridge or a boundary: stop there
POLISH_STEPS = 8


def score_jacobian(problem: metafopy.models.mv_lik.Problem, par: FloatArray, free: numpy.typing.NDArray[numpy.intp], reml: bool) -> FloatArray:
    """Hessian of ``-ll`` in the free entries of the transformed parameters: central differences of
    :func:`metafopy.models.mv_lik.gradient` (step ``1e-5 * max(1, |par_i|)``), symmetrised. ``len(free) x len(free)``."""


def polish_score(problem: metafopy.models.mv_lik.Problem, par: FloatArray, free: numpy.typing.NDArray[numpy.intp], reml: bool) -> tuple[FloatArray, float]:
    """Newton steps on the score from ``par`` (the BFGS stop) until the largest gradient entry is at most ``POLISH_GRADIENT``; returns the parameters
    and ``-ll`` there. A step is taken only while the Hessian is positive definite, no entry of the step exceeds ``POLISH_MAX_STEP``, ``-ll`` does not
    rise by more than ``metafopy.core.stationarity.REL_TOL * (1 + |-ll|)`` and the largest gradient entry falls; otherwise ``par`` as reached so far is
    returned. At most ``POLISH_STEPS`` steps."""
```

`maximise` keeps its signature and `OptResult`; `iterations` stays BFGS's count.

## Implementation

`maximise`, default optimiser only (`choice is None`): after the BFGS run and the existing L-BFGS-B fallback and error check, and only when
`not nonsmooth` and `free` is not empty, `x, val = polish_score(problem, expand(x), free, reml)` (the free entries of its result). The polish runs on
every such fit, whether BFGS reported success or not (see the choice "When the polish runs").

`polish_score`, per step: `g = mv_lik.gradient(...)`; stop if `max|g| <= POLISH_GRADIENT`; `H = score_jacobian(...)`; `numpy.linalg.cholesky(H)`,
stop on `LinAlgError`; `step = -H^-1 g` by the Cholesky factor; stop if `max|step| > POLISH_MAX_STEP`; evaluate `-ll` and the gradient at
`par + step`; stop if `-ll` is not finite, rose by more than `REL_TOL * (1 + |-ll|)`, or the largest gradient entry did not fall; else accept.
The objective test is loose on purpose: near the optimum `-ll` changes by less than its own rounding (the `kon` fits differ by 3e-11 in `-ll`, and a
test at 1e-13 rejected every useful step in the prototype), so the gradient decrease is what accepts a step, as the score is what the tests check.

The Hessian is the Jacobian of the analytic score, which `mv_lik.gradient` gives exactly for `sigma2` and by central differences of `E` for `tau2`,
`rho`, `gamma2` and `phi`; the prototype reached the target on those too (`ber_*`, `sim_cs`, `sim_splin`) or stopped on "the gradient did not fall"
(`sim_car`, `sim_gen`, `sim_spgau`, `sim_na_un`, left at gradient 1e-10 to 9e-10). Where `mv_lik.gradient` uses `numeric_gradient` (`nearpd`, or
some coefficients fixed by `beta`), the guards keep any step that does not lower the gradient from being taken.

Choices made, not decisions: the polish evaluations are not passed to `log` (the `verbose` trace keeps R's `nlminb` iterations only, and R has no
polish) and do not count towards `mfmaxit` (metafor's limit on the optimiser's evaluations); the other optimisers of `control['optimizer']` are not
polished (they port R's optimisers, whose stop R reports). `SLACK_GRADIENT` is unchanged: a stop between 1e-4 and the polish target that the polish
cannot improve is accepted as before.

### Tests

- `src/tests/test_mv_opt.py` (new): on the `kon` model of `property.mv.reml-score.kon`, a fit with `control={'gtol': 1e-4}` ends with the largest
  log-`sigma2` gradient at most `POLISH_GRADIENT` and `sigma2` within 1e-9 relative of the default fit (measured: 2.8e-11); confirm it fails with the
  polish call removed (gradient 2.5e-05). `polish_score` at an exact optimum takes no step; on a boundary fit (`sim_R_known` of `test_rma_mv.py`) it
  returns the input unchanged (the cap); `score_jacobian` agrees with a central difference of `negative_loglik` to 1e-5 relative on the `kon` fit.
- `src/tests/correctness/cases/properties.toml`: `property.mv.reml-score.kon` `tolerance` 2e-6 to 1e-10 of the scale (the measured largest after the
  polish, 8.6e-12 on the development Mac, times 10, rounded up: the margin rule of the cross-platform numerics spec). Its comment and the comment in
  `mv_score_is_zero` say what the polish does and give the measurement with its date. Confirm the case fails at 1e-10 without the polish under
  `gtol=1e-5` (relative 1.3e-07). `property.mv.ml-score.kon` keeps its absolute 1e-6.
- R parity: `src/tests/test_rma_mv.py`, `test_rma_mv_extras.py`, `test_mv_diagnostics.py`, `test_mv_inputs.py` and the other test files that import
  `rma_mv` pass unchanged (see Measured above); no fixture is regenerated and none is marked differing. Any that fails is a stop for the operator, not
  a tolerance change.
- `bin/platform-probe.py` group `mv-reml-score` also records the polish's Hessian count (`polish.hessians`, the number of `score_jacobian` calls, an upper bound on the Newton steps) and final gradient, so the CI probe can confirm it; its test in
  `src/tests/test_platform_probe.py` still passes.

### Records

- `docs/differences-notes.md`, "Optimisers and flat likelihoods", the `rma_mv` BFGS line: the stop is now finished by Newton steps on the score, the
  test asserts 1e-10 of the scale, R's own stop is 8e-09 of it (measured 2026-11-17).
- The `mv_opt` module docstring says that a Newton polish follows BFGS, and why (the slack accepts stops that vary by platform).
- The audit: `bin/metafor-audit.sh --only '*/mv_*'` (the `rma_mv` fixtures `mv_gen`, `mv_ks`, `mv_ks_mods`, `mv_slab`, `mv_sp`, `mv_wide`) in the
  background, with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` in a worktree; a new finding is reported to the operator.
- `TODO.md`: the item moves to `DONE.md` once CI is green (after merge, by the operator).

## Error handling

`polish_score` raises nothing of its own: every failure (Hessian not positive definite, a step over the cap, a non-finite `-ll` or gradient, no fall
in the gradient) ends the polish and keeps the best point so far, which is at least the BFGS stop. `maximise` keeps its errors
(`metafopy.errors.EstimationError` for no convergence or `mfmaxit`, `metafopy.errors.NotSupportedError`), raised before the polish runs.
`numpy.linalg.LinAlgError` from the Cholesky factorisation is caught inside `polish_score` only.

## Settings

None: the polish has no `control` entry. `POLISH_GRADIENT`, `POLISH_MAX_STEP` and `POLISH_STEPS` are module constants, as `SLACK_GRADIENT` is.
`control['gtol']` and `control['maxiter']` keep their meaning for BFGS.

## External dependencies

NumPy (`numpy.linalg.cholesky`, `solve`), already used. No new dependency.

## Acceptance

- The local checks pass: `ruff check`, `ruff format --check`, `mypy`, and the tests above plus every test file that imports `metafopy.models.mv_opt`
  or `rma_mv`, on the development Mac.
- `bin/linux-tests.sh -- src/tests/correctness/test_properties.py src/tests/test_rma_mv.py src/tests/test_mv_opt.py` passes (arm64).
- CI green on Linux, macOS and Windows, with `property.mv.reml-score.kon` at 1e-10, is checked after merge by the operator; a failure there is filed
  with the probe's figures, not met by widening the tolerance.

## Choices made

Made while writing the spec and reported, not asked: it follows from the project's R-parity rule and is reversible by amending this spec.

### When the polish runs, and how far it may move

**What it settles:** whether the Newton polish is capped to small steps, which leaves fits on a ridge or a boundary where BFGS stopped, or may walk
along a ridge or towards a boundary.
**Background:** measured 2026-11-17 above. A fit near an interior optimum needs one Newton step of at most 3e-6 (the CI-like `kon` stops) or 2.7e-05
(`ber_un_fixtau2`) in the transformed parameters. `kon_spexp`, `sim_R_*` and `sim_gdiag` take steps of 0.06 to 1.0: on a `rho` ridge (`kon_spexp`, log-likelihood +2.8e-09,
below `nlminb`'s `rel.tol` of 1e-10 relative to 31.9) or a variance heading to 0 on the log scale (`sim_R_*`, `sim_gdiag`), where each step lowers the
log-variance by about 1. `docs/differences-notes.md` already records `kon_spexp` and `sim_s_and_g` as flat fits that land "a hair away" from R.
**Affects:** which parity outputs move; whether `kon_spexp`'s `FLAT` entry in `test_rma_mv.py` must be marked differing; the property test is fixed
either way.
**Options:**
- *Capped* (`POLISH_MAX_STEP` 1e-3): outputs move by at most 7.1e-08 relative on every parity fixture; no fixture fails; ridge and boundary fits keep
  today's estimates, with their gradient under the slack as now.
- *Uncapped*: those fits move to a higher log-likelihood (`kon_spexp` `rho` 0.05596 to 0.03935, against R's 0.05592; `sim_R_*` `sigma2[0]` 4e-09 to
  4e-12 and `beta` nearer R's; `sim_gdiag` `tau2[2]` 2e-10 to 4e-12); `kon_spexp` fails its fixture and is marked differing, with a
  `docs/differences.md` entry and a check of the `kon_spexp` log-likelihood profile over `rho` before the change is accepted.
**Chosen:** *capped*. The defect is a stop accepted short of an interior optimum, which one small step fixes on every platform; the ridge and
boundary moves gain less than `nlminb`'s own stopping tolerance, move one fit away from R, and belong to the flat-likelihood question already recorded,
not to this item.

## Review

Approved by the operator on 2026-11-17 on the review page of `agents/reviews/order-todo-items-1-10-specs.md` (no comment): this is the
confirmation `claude/rules/metafopy.md` asks for of the output changes listed above. Version 1.1 records it.
