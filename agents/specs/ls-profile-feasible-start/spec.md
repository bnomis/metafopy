# A feasible start for identity-link profile refits that have none

**Version:** 1.1
**Status:** proposed
**Date:** 2026-11-19

The profile likelihood of a scale coefficient of a location-scale model is the log-likelihood maximised over the other scale coefficients with that
coefficient fixed (Viechtbauer and Lopez-Lopez, 2022, doi:10.1002/jrsm.1562, section 2.5.3). metafopy's `profile_fit` refits from the starts of
`profile_starts` and keeps the best; under the identity link (`tau2_i = Z_i alpha`) a start that leaves a `tau2_i` negative is dropped. This spec adds
one start of the largest minimum `tau2_i` when every start but the default has been dropped, so that the refit has a feasible place to begin. It changes
one metafopy `confint` output, listed below, so approving the spec is the confirmation `claude/rules/metafopy.md` asks for.

**Approved** by the operator on 2026-11-19 on the review page of `agents/reviews/ls-profile-feasible-start-spec.md` (no comment), with the output
change listed below. Version 1.1 records the approval; the spec's content is unchanged from 1.0.

## Motivation

The `TODO.md` item "metafopy's profile CI misses the identity-link lower bound of the ls scale intercept", as filed:

<!-- doc-counts: accept-start found on 2026-11-19 - a dated record, not a claim about the present -->
> Found on 2026-11-19 by `agents/tasks/ls-profile-identity-start/tasks-1.md` (Task 2): on the ML identity-link `dat.bcg` fit
> (`ls.fit.ML.identity.bcg`, `mods = scale = ~ ablat`, `escalc` RR), `confint(result, alpha=0, tol=1e-12)` gives `nan` for the lower bound of alpha.1
> (the scale intercept), with the warnings "Error when trying to invert the Hessian" and "Cannot obtain lower bound of profile likelihood CI due to
> convergence problems". R metafor 5.2-1 also gives `NA` with the same two warnings. The reference `ls_profile_bounds` gives -0.214322988803, checked
> against a brute-force profile (-9.060018239 on a 40001-point grid, crossing level -9.060018228).
<!-- doc-counts: accept-end -->

Earlier work on the same code: [ls profile starts](../ls-profile-starts/spec.md) (2026-11-18) added the starts beyond the line (`LINE_WEIGHTS` 1.5 and
2.0); `agents/tasks/ls-profile-identity-start/tasks-1.md` (2026-11-19) gave the reference `formulas.ls_profile_ll` in
`src/tests/correctness/formulas.py` feasible starts, `formulas.ls_profile_identity_starts`, whose linear-programming start this spec ports.

### What the investigation found

All figures are in [measurements.md](measurements.md) (measured 2026-11-19 at `f1fdc9b`).

- **Cause.** With `alpha.1` below its estimate -0.02666, the fit's own `alpha` and every point of the line start leave a `tau2_i` negative, so
  `profile_starts` keeps only the default start (`None`), and that refit fails at every value from the search limit -0.2932 to -0.03. `profile_fit`
  returns `None`, the profile is NaN on the whole lower interval, and `confint` warns and gives NaN. The feasible `alpha.2` at `alpha.1 = v < 0` is
  `alpha.2 >= -v / min(ablat)` (0.0165 at the bound), above the fit's 0.00205.
- **A feasible start fixes it.** The start that maximises the smallest `tau2_i` with `alpha[pos]` at the fixed value (a linear programme, its
  minimum capped at the mean fitted `tau2_i` so it is bounded), and points a quarter, a half and three quarters of the way from it to the fit's own
  `alpha` that are feasible, give the bound -0.21432298822, the reference's -0.214322988803 to 6e-10, with no warning from `confint`.
- **Only that output moves.** Over `src/tests/test_rma_ls.py` and `src/tests/correctness`, adding the start changes one result: the strict
  expected failure `ls.confint.scale.ML.identity.bcg.alpha1_lb` passes. Every other test, including the R parity fixtures of the identity-link fits,
  is unchanged.
- **When to add it.** Adding it to every identity-link refit gives the same results and raises `confint` of the ML fit from 0.39 to 1.14 s and of the
  REML fit from 1.92 to 4.50 s. Adding it only when no start but the default is left gives 0.49 and 1.89 s, and leaves every refit that has a
  feasible start as it is today. This spec takes the second.

### Out of scope, as the product already settles them

- The upper bound of alpha.1 stays `>0.2399`, the end of the interval searched (R metafor `>0.2401`, the reference NaN); the profile is above the
  cut-off there, so the sign is the documented outcome.
- The alpha.2 bounds (-0.0136, 0.0165) already equal the reference; R metafor's `<-0.0082`, `>0.0082` come from its narrower search interval, the
  difference `docs/differences.md` records under "Profile-likelihood bounds of the scale coefficients".

## Module location

- `src/metafopy/methods/ls_profile.py`: a new function `feasible_start` and its use in `profile_starts`.
- `src/tests/correctness/cases/location_scale.toml`: `ls.confint.scale.ML.identity.bcg.alpha1_lb` loses its `status` and `reason`, and its bound
  joins `ls.confint.scale.ML.identity.bcg`.
- `src/tests/test_rma_ls.py`: regression tests. `docs/differences.md`, `CHANGELOG.md`: records. `TODO.md` and `DONE.md`: the item moves when it ships.

## API

```python
def feasible_start(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> FloatArray | None:
    """A start of the identity-link refit at ``alpha[pos] = val`` whose smallest ``tau2_i`` is as large as it can be, or ``None`` if none is positive.

    Solves the linear programme: maximise ``t`` over the other coefficients ``a`` subject to ``Z[:, free] a + Z[:, pos] val >= t`` for every study and
    ``t <= mean(Z @ x.alpha)`` (1 where that mean is not positive), with ``scipy.optimize.linprog``. Returns ``x.alpha`` with ``alpha[pos] = val`` and
    the solution's ``a``, or ``None`` when the programme fails or its ``t`` is not positive.
    """
```

`profile_starts(x, pos, val)` keeps its signature. Under the identity link, when the filtered list holds no start but `None` and `pos` has other
coefficients, it appends `feasible_start(x, pos, val)` and the points `start + w (own - start)` for `w` in `FEASIBLE_WEIGHTS = (0.25, 0.5, 0.75)`
(`own` the fit's `alpha` with `alpha[pos] = val`) whose `Z @ point` is nowhere negative. Its docstring says so.

## Error handling

No new exception. A linear programme that fails, or whose `t` is not positive (no `alpha` with every `tau2_i > 0` at that value), adds no start, and
`profile_fit` returns `None` as today; `confint` then warns "Cannot obtain lower bound ..." (or upper) and gives NaN, as now.

## Implementation

1. `feasible_start` builds `c = [0, ..., 0, -1]`, `A_ub = [-Z[:, free], 1]`, `b_ub = Z[:, pos] val` and bounds `(None, None)` for each coefficient and
   `(None, cap)` for `t`, as `formulas.ls_profile_identity_starts` does; `cap` keeps the programme bounded. It is written in metafopy's own code, not
   imported from the tests.
2. `profile_starts` calls it only in the case above, so a refit with a feasible line start is unchanged, and the extra cost falls only on values
   where every refit would otherwise fail.
3. Tests in `src/tests/test_rma_ls.py`: `feasible_start` at `alpha.1 = -0.2143` of the ML identity `dat.bcg` fit gives a point with every
   `tau2_i > 0`; `profile_starts` there includes it, and at a value where a line start is feasible (`alpha.1 = 0.1`) returns the same list as before;
   `confint(x, alpha=0, tol=1e-12)` gives the lower bound -0.214322988803 within 1e-8 and no "Cannot obtain lower bound" warning.
4. The correctness case: delete `ls.confint.scale.ML.identity.bcg.alpha1_lb` and add `"alpha.1.lower"` to the `bounds` of
   `ls.confint.scale.ML.identity.bcg`, so that case checks every bound but alpha.1 upper (which has no crossing).
5. Records: a `CHANGELOG.md` "Unreleased" entry (behaviour change: the ML identity-link `dat.bcg` lower bound of `alpha.1` is -0.2143 where it was
   NaN with a warning); a sentence in `docs/differences.md` "Profile-likelihood bounds of the scale coefficients" that R metafor gives NA there because
   its refit from the default start fails.

## Settings

None. `FEASIBLE_WEIGHTS` is a module constant, as `LINE_WEIGHTS` is.

## External dependencies

`scipy.optimize.linprog` (scipy is already a dependency; `src/tests/correctness/formulas.py` uses the same function).
