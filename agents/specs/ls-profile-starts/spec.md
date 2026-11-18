# Profile starts of location-scale models that reach the highest branch

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-18

The profile likelihood of a scale coefficient of a location-scale model is the log-likelihood maximised over the other scale coefficients with that
coefficient fixed (Viechtbauer and Lopez-Lopez, 2022, doi:10.1002/jrsm.1562, section 2.5.3). metafopy's `profile_fit` refits from the starts of
`profile_starts` and keeps the best, and `confint` and `profile` of a location-scale fit both use it. This spec adds two starts beyond the far end of the
line of starts, so that the refit reaches the branch of highest log-likelihood where none of the present starts lies near it. It changes metafopy's
`confint` and `profile` output; every change is listed below with its size, so approving the spec is the confirmation `claude/rules/metafopy.md` asks for.

**Approved** by the operator on 2026-11-18 on the review page of `agents/reviews/order-todo-items-1-5-specs.md` (no comment), with every output
change listed below. Version 1.1 records the approval; the spec's content is unchanged from 1.0.

## Motivation

The `TODO.md` item "metafopy's profile of the ls scale coefficient leaves the highest branch before the crossing", as filed:

<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->
> Found on 2026-11-18 by `agents/tasks/ls-confint-scale-branch/tasks-1.md` (Tasks 1 and 2) on `ls.confint.scale.ML.bcg` in
> `src/tests/correctness/cases/location_scale.toml`: the profile of `alpha.2` (the `ablat` coefficient of the scale model) has a continuous branch and a
> `tau2 = 0` plateau (3.868641), and the continuous branch is the maximum until it crosses the cut-off at 1.38120753375 (the fixed reference and R's best
> refits agree). metafopy's own profile jumps to the plateau earlier: 3.789459 at a = 1.36 and the plateau 3.868641 from 1.37, so `confint` gives the
> upper bound 1.3598 with no `>` sign, a jump in the profile and not the crossing 1.3812.
<!-- doc-counts: accept-end -->

Earlier work on the same code: `agents/tasks/ls-profile-crossings/tasks-1.md` (2026-11-16) added the line of starts (`LINE_WEIGHTS`) and removed the
search cap; `agents/tasks/ls-confint-scale-branch/tasks-1.md` (2026-11-18) widened the starts of the reference `formulas.ls_profile_ll` in
`src/tests/correctness/formulas.py` (the line from its start to the adjusted start plus, for the log link, shifts of -130 to 10 in steps of 10).

### What the investigation found

All figures are in [measurements.md](measurements.md) (measured 2026-11-18 at `dcfb11c`, R metafor 5.2-1).

- **Cause.** `profile_starts` gives the default start, the fit's own `alpha` with `alpha[pos] = val`, and points a quarter, half, three quarters and
  all of the way to the fitted line start, which keeps `Z alpha` as near `Z alpha_hat` as it can. For the ML `dat.bcg` fit at a = 1.36 these starts have
  `alpha.1` from -4.96 to -49.2, while the continuous branch has its optimum at `alpha.1` -74.3. From every one of them `nlminb` ends on the flat
  `tau2 = 0` plateau, where `alpha.1` runs to minus infinity; refits from a fixed `alpha.1` reach the branch reliably only from -75 and -70. The
  optimum lies beyond the line because the line weighs every study equally, while the profile optimum keeps `tau2_i` right for the studies that carry
  the information. Near 1.317 one start in the span still lands on the branch; from 1.36 none does.
- **Two starts beyond the line fix it.** Weights 1.5 and 2.0 along the same line (the 1.5 start has `alpha.1` about -71) give 3.789459 at 1.36,
  3.813933 at 1.37 and 3.838488 at 1.38, Task 1's grid values, and `confint` gives 1.3812075, the reference's 1.38120753377. Over a = 1.2 to 1.6 in
  steps of 0.005 the ML and REML profiles are nowhere below the reference by more than 1e-6.
- **They also fix identity-link fits the item did not name.** There every line start that leaves a `tau2_i` negative is dropped; on the identity fits
  of `dat.bcg` and on `FITS.bd_identity_ni` all were dropped at some grid values, so the profile was NaN there and `confint` showed a `<` bound where
  the profile is 0.5 to 3.7 below the cut-off. The new starts are feasible there: the bounds become crossings, checked to 4e-11 by a brute-force
  profile (a one-dimensional search over the feasible interval) and by a multi-start SLSQP for the three-coefficient fit.
- **Alternatives measured.** The reference's grid of shifts gives the same log-link values but about four times the run time and leaves the identity
  fits as they are. Continuation from the estimate in 4 steps fixes the log-link case, not the identity ones; in 8 steps it falls onto the plateau
  again.
- **R.** R metafor 5.2-1 gives `>0.4245` for this bound: it searches only up to ten times the absolute estimate, an accepted difference
  (`docs/differences.md`, "Profile-likelihood bounds of the scale coefficients"), and its refits from the default start sit on the plateau at every a.
  The change does not alter that difference; no R parity fixture moves.

## Module location

- `src/metafopy/methods/ls_profile.py`: `LINE_WEIGHTS` and the docstring of `profile_starts`.
- `src/tests/correctness/cases/location_scale.toml`: `ls.confint.scale.ML.bcg` loses its `status` and `reason`.
- `src/tests/test_rma_ls.py`: regression tests. `docs/differences.md`, `CHANGELOG.md`: records. `TODO.md` and `DONE.md`: the item moves when it ships.

## API

No signature changes. The constant and the docstring of `profile_starts` become:

```python
LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def profile_starts(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> list[FloatArray | None]:
    """The starting values of the refits at ``alpha[pos] = val``: the default start (``None``), then points from the fit's ``alpha`` along the line start.

    The fitted line start sets the other coefficients by least squares so that ``Z alpha`` stays as near ``Z alpha_hat`` as it can with ``alpha[pos]``
    at ``val``. Far from the estimate the optimiser stops at the first start whenever the likelihood is nearly flat there, so the starts between the
    two (a quarter, a half and three quarters of the way) are tried as well, and two beyond the line start (one and a half and twice the way): the
    profile optimum can lie past the line, where every start inside it ends on a ``tau2 = 0`` plateau. A start that leaves a ``tau2_i`` negative under
    the identity link is left out.
    """
```

## Error handling

No new error. A refit that fails from a new start is skipped as now (`refit_ls` returns `None`); `profile_fit` still returns `None` only when every
start fails, and `profile_lrt` then raises `metafopy.errors.EstimationError` as now.

## Implementation

1. In `ls_profile.py`, set `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)` and replace the docstring of `profile_starts` with the one above. The
   loop `starts.extend(own + weight * (line - own) for weight in LINE_WEIGHTS)` and the identity-link filter stay; the new starts come last, so a tie
   keeps the fit found now (`profile_fit` keeps a later fit only if its log-likelihood is strictly higher). This is the prototype measured.
2. In `location_scale.toml`, remove `status` and `reason` from `ls.confint.scale.ML.bcg`.
3. In `test_rma_ls.py`, add `test_profile_keeps_the_continuous_branch_of_alpha2_on_bcg`: on `ls(bcg(), '~ ablat', '~ ablat', method='ML')`,
   `2 (res.logLik() - profile_fit(res, 1, a)[0])` is 3.789459, 3.813933 and 3.838488 at a = 1.36, 1.37, 1.38 (abs 1e-5), and
   `confint(res, alpha=1, tol=1e-12)` gives `ci.ub` 1.3812075 (abs 1e-6) with no `ub_sign`. Add
   `test_confint_identity_lower_bound_is_a_crossing`: on `FITS['bcg_year_identity']`, `confint(res, alpha=1, tol=1e-12)` gives `ci.lb` -0.1356308
   (abs 1e-6) with no `lb_sign` (the brute-force profile crosses there). Run each against the present starts and see it fail first.
4. `docs/differences.md`, "Profile-likelihood bounds of the scale coefficients": add that on the same fit metafopy's upper bound of `alpha.2` is
   1.3812 where R prints `>0.4245`. `CHANGELOG.md` under "Unreleased": a behaviour-change entry naming the starts and the outputs that move below.
5. Run the tests listed in measurements.md ("Tests and run time"); the expected result is every test passing, `ls.confint.scale.ML.bcg` included.

### Every output that moves

Measured 2026-11-18 at `dcfb11c` by the sweep of every location-scale fit in the correctness cases and in `FITS` of `test_rma_ls.py` (measurements.md has
the table):

- ML `dat.bcg`, log link, `alpha.2` upper bound: 1.3598266 to 1.3812075 (+0.0214), in every fit of that model (`ls.fit`, `ls.test.*`, `bcg_ablat_ml`).
- Identity-link lower bounds that were a `<` at a point not crossed, now crossings: ML `dat.bcg` `alpha.2` -0.0082 `<` to -0.0136 (the
  `I(ablat/10)` form -0.0820 `<` to -0.1356); REML `dat.bcg` `alpha.1` 0 `<` to -0.2686 and `alpha.2` NaN to -0.0288; `bd_identity_ni` `alpha.2`
  0 `<` to -0.0049. Their profiles gain the values that were NaN (2, 9 and 4 of 20 points), equal to the brute-force profile to 1e-6.
- Two log-link profile points rise to a higher refit: `published.vl2022.table1.wald` `alpha.1` at -4.849 by 0.155, `table3.ML.mixed` at -2.882 by
  0.019; their bounds do not move.
- Optimiser noise on `bd_3_unweighted` and `bd_3_weights`: bounds by at most 2.6e-8, profile values by at most 4.4e-7, all upwards.

### Run time

Each refit point tries two more starts. `pytest src/tests/test_rma_ls.py`: 34.5 s now, 38.8 s with the change; the
`ls.confint.scale.*` cases are unchanged (about 45 s, mostly the reference); the test files listed in measurements.md take 825 s and 839 s (measured 2026-11-18).

## Settings

None: `LINE_WEIGHTS` is a module constant, as now, and no `control` entry is added.

## External dependencies

None new.

## Acceptance

- `ls.confint.scale.ML.bcg` passes with no `status`: metafopy's upper bound of `alpha.2` is within `LS_PROFILE` of the reference's 1.38120753377.
- The two new tests pass, and each fails against `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)`.
- The tests listed in measurements.md pass; no R parity fixture is marked as differing; `ruff check`, `ruff format --check`, `mypy` and the
  pre-commit hooks pass.
- `TODO.md`'s item moves to `DONE.md` with the commit.

## Not in scope

The reference `formulas.ls_profile_ll` cannot fit the identity link at some feasible values (it gives -1e10 at `alpha.2 = -0.00615` of the ML
identity `dat.bcg` fit, where the brute-force profile is -8.045852), because its SLSQP starts are infeasible there. No `ls.confint.scale.*` case uses
the identity link, so no case is affected; it is reported to the operator, not changed here.

## Choices made

Made while writing the spec and reported, not asked; each is reversible by amending this spec.

### Which starts to add

**What it settles:** the smallest set of starts that keeps the highest branch. **Background:** measured 2026-11-18 (measurements.md). Weight 1.5 alone
does the work on the `dat.bcg` fits; 1.25 alone and 2.0 alone do not. **Options:** *(1.5, 2.0)*: fixes every case measured, `test_rma_ls.py` +12%;
*(1.25, 1.5, 1.75, 2.0, 2.5, 3.0)*: the same values, more margin, +32%; *the reference's shifts*: about four times the run time, log link only;
*continuation from the estimate*: not stateless across the parallel grid, and fails at 8 steps. **Chosen:** *(1.5, 2.0)*, with 2.0 kept beside 1.5
for a little margin at a cost of one refit; a fit whose optimum lies further out than twice the line is not known in any case.

### The identity-link moves are part of this change

**What it settles:** whether the starts apply to the identity link too, though the item named only the log link. **Background:** the same starts make
the identity-link profiles defined where they were NaN, and every bound that moves becomes a checked crossing, where the old one was a `<` the profile
does not reach. **Chosen:** apply them to both links, as `LINE_WEIGHTS` is now; limiting them to the log link would keep bounds known to be wrong.
