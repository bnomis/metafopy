# Location-scale fits stable across platforms

**Version:** 1.2
**Status:** implemented
**Date:** 2026-11-24
**Review:** `agents/reviews/location-scale-fits-stable-across-platforms-spec.md` (page https://claude.ai/artifact/CmGGzkSKzMSKFE7pnXqeVY); answered 2026-11-24: approved, D1 a, D2 a, D3 a.

CI fails tests that pass on the development Mac, all on location-scale fits of `dat.bcg`: a documentation example, two `vif` correctness cases and
one exact permutation test. Each failure is a computation whose result depends on rounding, so the platform (and even the run) decides the outcome.
This spec makes each one deterministic: metafopy's starting values for a study of leverage 1, metafopy's log-link optimiser stop, and the test's
reference fit of a log-link model. Two of the changes are to the library, so the review of this spec is the user's confirmation that metafopy's
output changes (`claude/rules/metafopy.md`). The source is the first `TODO.md` item, "CI fails three numerical tests that pass on a local macOS
run". It follows [Cross-platform numerics](../cross-platform-numerics/spec.md) and [rma_mv REML stop](../rma-mv-reml-stop/spec.md), and changes
the reference of [Correctness: location-scale extensions](../correctness-ls-extensions/spec.md). Figures are in [measurements.md](measurements.md).

## Motivation

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
From CI runs 45270718992 (`93fea02`) and 45335133105 (`00f7fda`, which changes only `TODO.md`), Python 3.14.8, numpy 2.5.4, scipy 1.18.1 on
ubuntu-26.04, macos-26 and windows-2025-vs2026. Test order and coverage are not the cause (the `TODO.md` note of 2026-11-24). The cases:

1. `src/tests/test_docs_examples.py::test_page_examples_run[differences.md]` (all runners, both runs): the example at `docs/differences.md`
   line 1442 (`mods = '~ ablat * alloc'`, `scale = '~ ablat'`, three NA studies) raises R's own "No missing values allowed in 'alpha_init'.".
   With k = 10 and five location columns, study 4 has hat value 1, so its HE residual in `start_alpha` (`src/metafopy/models/rma_ls_fit.py`) is 0
   in exact arithmetic. metafopy computes `2.3e-14` locally, just over the `100 * DBL_EPS` (`2.2e-14`) snap that both it and R's
   `rstandard.rma.uni` apply; R computes `-7.4e-15`, sets it to 0, and `log(0)` makes `alpha.init` NaN. The documented difference ("the NA comes
   from the dropped column") gives the wrong cause, and the fit it documents is the end point of one start among several (log-likelihood `0.0858`,
   Hessian not invertible, against `-4.1456` from another start).
2. `src/tests/correctness/test_derived.py::test_derived[ls.vif.reestimate.ML.bcg2]` (and `ls.vif.REML.bcg2` on ubuntu in the second run):
   `LinAlgError: Singular matrix` in the test's `ls_reference_fit` (`src/tests/correctness/formulas.py`). With `year` not centred, one BFGS step
   from some starts gives `tau2 = exp(-2000) = 0` and a score of exactly 0, which is kept; the interior optimum's score is `7.6e-07` to `1.0e-05`
   (information condition `1e11`), around `LS_SCORE_TOLERANCE` (`1e-6`). Where rounding puts it over, only degenerate points are kept.
3. `test_derived[ls.vif.REML.bcg2]`: metafopy's location vif `4.723135` (ubuntu, first run) and `4.723162` (windows, second run) against
   the reference `4.723144` (tolerance `1e-6`). metafopy's BFGS and Nelder-Mead stop at the objective's rounding, and the REML likelihood is
   flat enough that the location vif moves by up to `2.2e-6` relative when `yi` is multiplied by `1 + 1e-15 N(0, 1)`. R's own stop is
   `4.7231321` (`2.6e-6` off), with the same spread. The ML case is not sensitive.
4. `test_derived[ls.permutest.location.exact.ML.z.bcglat]` (windows, second run): metafopy's p `794/1287` (the same locally) against the
   reference's upper bound `793/1287`. Not a tie (metafopy and metafor compare `|z| >= |z_obs| - comptol` alike): 77 permuted models have no
   finite maximum, and the reference counts as unfitted only those whose near-zero information `numpy.linalg.inv` refuses: 12 locally, 7 on windows.

Found on the way, out of scope: on 8 regular permutations of case 4 metafopy and R stop at the same lower local maximum, the reference at a higher
one (`measurements.md`). The first task files it with `/todo-add`.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/models/rma_ls_fit.py`: `marginal_residuals` (case 1); the call of the log-link polish after `rma_ls_opt.minimise` (case 3);
  `LSProblem` gains the log-link score and information.
- `src/metafopy/models/rma_ls_opt.py`: new `polish_log` (case 3).
- `src/tests/correctness/formulas.py`: `ls_reference_fit` (cases 2 and 4).
- Tests: `src/tests/test_rma_ls.py`, `src/tests/test_rma_ls_opt.py` (new), `src/tests/correctness/test_formulas_ls.py`.
- Records: `docs/differences.md`, `src/audit/data/known.toml`, `docs/differences-notes.md`, this directory's `measurements.md`.

## API

```python
# src/metafopy/models/rma_ls_fit.py
LEVERAGE_ONE = 1e-10  # a study with 1 - h_ii below this has a zero residual in exact arithmetic


def marginal_residuals(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Raw residuals and their standard errors (metafor's ``rstandard``); a residual under ``100 * DBL_EPS``, or of a study with
    ``1 - h_ii < LEVERAGE_ONE``, is 0."""


class LSProblem:
    def log_score(self, alpha: FloatArray) -> FloatArray:
        """Score of the scale coefficients under the log link: ``-1/2 Z' ((d - e^2) tau2)``, ``tau2 = exp(Z alpha)``."""

    def log_information(self, alpha: FloatArray) -> FloatArray:
        """Observed information (minus the Hessian of the log-likelihood) under the log link, with ``D_j = diag(tau2_i z_ij)``:
        ``-1/2 tr(T D_l T D_j) + 1/2 tr(T D_jl) + y'P D_l P D_j P y - 1/2 y'P D_jl P y`` (``T`` = ``W`` for ML, ``P`` for REML)."""


# src/metafopy/models/rma_ls_opt.py
POLISH_LOG_SCORE = 1e-10  # stop when the largest |score| entry is at most this
POLISH_LOG_MAX_STEP = 1e-3  # a larger step entry (standardised Z) means a ridge or a boundary: stop there
POLISH_LOG_STEPS = 8


def polish_log(start: FloatArray, loglik: Objective, score: Gradient, information: Gradient, free: numpy.typing.NDArray[numpy.bool_]) -> FloatArray:
    """Newton steps on the free scale coefficients from ``start`` (the BFGS and Nelder-Mead end point) until the largest score entry is at
    most ``POLISH_LOG_SCORE``. A step is taken only while the information is positive definite, no entry exceeds ``POLISH_LOG_MAX_STEP``, the
    log-likelihood does not fall by more than ``metafopy.core.stationarity.REL_TOL * (1 + |ll|)`` and the largest score entry falls; otherwise the
    point reached so far is returned. At most ``POLISH_LOG_STEPS`` steps."""


# src/tests/correctness/formulas.py
LS_MAXIMUM_RATIO = 1e-5  # a log-link end point whose information, on the standardised Z, has min/max eigenvalue ratio at or below this has no finite maximum


def ls_is_maximum(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> bool:
    """The information on the standardised ``Z`` (``ls_standardise``) at ``alpha`` has min/max eigenvalue ratio above ``LS_MAXIMUM_RATIO`` (an interior maximum, not an underflowed ``tau2``)."""
```

`Gradient` is `collections.abc.Callable[[FloatArray], FloatArray]`. `ls_reference_fit` keeps its signature and return keys.

## Implementation

### Case 1: a study of leverage 1 (decision 1, option a)

`marginal_residuals` sets `ei[1 - numpy.diag(H) < LEVERAGE_ONE] = 0.0` after R's snap. `start_alpha` then gives NaN, and `fit_rma_ls` raises
R's "No missing values allowed in 'alpha_init'." on every platform. `marginal_residuals` is called only by `start_alpha`.

- Records: remove the `docs/differences.md` section "A location-scale fit with an interaction whose column R drops" (the compare block at line
  1442 with it), and the `known.toml` entry `construct/ls_na/mods={"$formula":"~ ablat * alloc"}` (`accepts-where-R-refuses`). This replaces
  decision D2 (a) of `agents/reviews/order-todo-items-1-10-specs.md` (2026-11-17), which was made on the wrong cause.
- Test (`test_rma_ls.py`): the example raises `InputError` with that message, unperturbed and with `yi * (1 + 1e-15 N(0, 1))` for
  `numpy.random.default_rng(1)` draws 1 to 9. A test fixture, not a ported procedure, so `numpy.random` is allowed. Before the fix (run the test on
  `main`) the unperturbed input fits locally and the nine perturbed ones raise. That reproduces the CI behaviour on the Mac.
- Audit: `bin/metafor-audit.sh --only 'construct/ls_na*'` must report that both sides refuse. Then `bin/metafor-audit.sh --recall`: a miss of
  recall item `a5-ls-na-factor-scale` is a stop for the operator.

### Case 2: the reference fit of a log-link model (choices made below)

For `link == 'log'`, `ls_reference_fit` centres and scales each non-constant column of `Z` (`ddof=1`), giving `Zs = Z T^-1`. It runs its starts,
Newton polish and score test on `Zs`, and maps the result back: `alpha = T^-1 alpha_s`, `va = T^-1 va_s T^-T`. `tau2`, `ll`, `beta` and `vb` do
not change under the map. A start is kept only when `ls_is_maximum` holds at its polished end point. The final Nelder-Mead polish is skipped for the log link: Newton steps on the analytic score already reach rounding there, and
the prototype's Nelder-Mead moved the end point off it. The identity link is unchanged.

- Local reproduction (`test_formulas_ls.py`): with `LS_SCORE_TOLERANCE` monkeypatched to `5e-7`, `ls_reference_fit` on the `bcg2` data succeeds
  for ML and REML. On `main` the same test raises `LinAlgError` (measured). With the tolerance at `1e-9` it succeeds too, and the location vif
  equals the `1e-6` result to `1e-8` relative. `ls_is_maximum` is false at the degenerate point that the REML data's third start reaches in
  the raw `Z` (alpha `[0.89967, -0.01675, -1.00986]`, score exactly 0) and true at the reference optimum.

### Case 3: metafopy's log-link stop (decision 2, option a)

In `fit_rma_ls`, after `rma_ls_opt.minimise` with `link == 'log'` and the default optimiser (`nlminb`, run as BFGS), `polish_log` runs on the free
coefficients. Its inputs are the analytic `log_score` and `log_information` of the problem (`Z` is already standardised there when `scaleZ` holds),
mapped through `mapfun` where `alpha_min` or `alpha_max` is set. It does not run with `optbeta=True`. It runs inside the permutation tests and
other refits too: they call the same fit. The polish evaluations are not logged by `verbose` and do not count towards `maxiter`, as for `rma_mv`.
The `rma_ls_opt` module docstring and `docs/differences-notes.md` (the location-scale optimiser line) say so, with this spec's measurements.

- Local reproduction (`test_rma_ls_opt.py`): `ls.vif.REML.bcg2`'s model with `yi * (1 + 1e-15 N(0, 1))`, five draws: the location vif lies within
  `1e-8` relative of the unperturbed fit's and of the reference's. On `main` the draws spread to `2.2e-6` relative (measured), so the test fails
  there. `polish_log` at a degenerate fit (`tau2` heading to 0) returns the start unchanged.
- Outputs that move: every log-link location-scale fit, by the distance between the BFGS stop and the optimum. `test_rma_ls.py` and the other R
  parity tests pass at their committed tolerances (`OPT`), and no fixture is regenerated. The task records the largest change per fixture in
  `measurements.md`, and the exact p of `ls.permutest.location.exact.ML.z.bcglat` before and after (794/1287 before).

### Case 4: the exact permutation reference (decision 3, option a)

Nothing beyond case 2: a permuted model with no finite maximum keeps no start, `ls_reference_fit` raises `RuntimeError`, and
`ls_exact_permutation_p` counts it in `failed`. Expected locally: 77 failed, bounds 774 to 851 of 1287. That is the classification of
`measurements.md`; the task re-measures it with the change. A different count is recorded there with the reason before the case is run.
metafopy's 794 lies inside. `ls.permutest.scale.exact.ML.z.bcglat` must pass unchanged in verdict.

### Closing

1. On the development Mac: `pytest src/tests/correctness/test_derived.py -k "ls."` (slow cases included), `src/tests/test_docs_examples.py`, the
   new tests and every test file importing `metafopy.models.rma_ls_fit`, `rma_ls_opt`, `metafopy.methods.ls_*` or `correctness.formulas`
   (`grep -rl` at the time); `ruff check`, `ruff format --check`, `mypy`. Optionally `bin/linux-tests.sh` on the same files.
2. The operator merges the branch to `main` and pushes (`/ship`); the task does not push.
3. After the push, `gh run list --branch main --limit 3` and `gh run view <id> --log-failed` for the CI run of the merge commit. If any test fails
   on any runner, the task stops as blocked, naming each failing test id, runner and value; it widens no tolerance.
4. `TODO.md`: the operator moves the item to `DONE.md` once CI is green.

## Error handling

- Library: no new exception class. Case 1 raises the existing `metafopy.errors.InputError("No missing values allowed in 'alpha_init'.")`.
  `polish_log` raises nothing of its own: a non-positive-definite information (`numpy.linalg.LinAlgError` from its Cholesky factor, caught
  inside), a capped step, a non-finite log-likelihood or score, or no fall in the score ends it at the best point so far.
- Test reference: `ls_reference_fit` raises `RuntimeError` when no start is kept, as before; callers already turn it into an `AssertionError`
  (`ls_reference`), a skip with a warning (`ls_vif_reestimate`) or a failed permutation (`ls_exact_permutation_p`).
- A stop for the operator, not a fix in the task: any output outside the case's own moves past its committed tolerance (an R parity fixture, a
  correctness case other than the four, a printed example); the recall check misses; or the counts of case 4 differ from `measurements.md`.
- Any change that would need a new `docs/differences.md` or `known.toml` entry is written there, with the reason and date, in the same task.

## Settings

No `control` entry. Module constants: `LEVERAGE_ONE` (`float`, `1e-10`), `POLISH_LOG_SCORE` (`float`, `1e-10`), `POLISH_LOG_MAX_STEP`
(`float`, `1e-3`), `POLISH_LOG_STEPS` (`int`, `8`) in the library; `LS_MAXIMUM_RATIO` (`float`, `1e-5`) in `formulas.py`. `LS_SCORE_TOLERANCE`
stays `1e-6`; `LS_RELATIVE` (the location vif tolerance) stays `1e-6`.

## External dependencies

NumPy and SciPy, already used. No new dependency. R metafor 5.2-1 in `.audit/Rlib` for the audit step.

## Decisions

### Choices made, not asked

- **Reference fit in standardised coordinates, with a positive guard (case 2).** The likelihood does not depend on how `Z` is parameterised, so
  fitting in centred and scaled coordinates changes no value the test checks. It removes the `1e11` condition number that leaves the optimum's
  score at the tolerance. The guard asks for evidence that a point is a maximum (a positive definite information), not only for a small score,
  which an underflowed `tau2` gives for free. `LS_DEGENERATE` sits in a gap measured on case 4 (regular ratios at least `4.3e-3`, degenerate at
  most `1.1e-13`). Reversible by reverting the test file.
- **Case 4 follows from case 2** and needs no change of its own; the bounds widen from 12 to 77 permutations, which decision 3 asks about.

### 1. A study of leverage 1 in the scale starting values

**What is decided:** whether metafopy refuses this model with R's error on every platform, or fits it on every platform.
**Background:** R stops because the zero residual of a leverage-1 study makes `log(ei^2)` `-Inf` in its start regression. On 2026-11-17 (D2 (a)
of `agents/reviews/order-todo-items-1-10-specs.md`) the user chose "metafopy does not reproduce a failure of R's optimiser setup", on the stated
cause that the NA came from a dropped location column. That cause was wrong: metafopy fitted only because its residual is `2.3e-14` on the Mac,
`1e-15` over the snap, and the fit it reached is one of several.
**Affects:** `docs/differences.md`, `known.toml`, `test_docs_examples.py`; any location-scale fit with a leverage-1 study under the log link.
**Options:**
- *a. Refuse with R's error* (`LEVERAGE_ONE`): deterministic, matches R, and removes a documented difference and its `known.toml` entry.
- *b. Fit, leaving such studies out of the start regression*: deterministic start; the end point (log-likelihood `-4.1456`, beta
  `2.924, -0.084, -2.262, -0.536, 0.044`) differs from the one documented today. The differences entry is rewritten with the right cause, and a
  start metafor does not have is invented.
**Recommended:** a. The model gives no information on the scale of that study, and the end point under b depends on the start metafopy invents.

### 2. metafopy's log-link optimiser stop

**What is decided:** whether metafopy's log-link fit is finished on the analytic score, or the test is widened to the stop's spread.
**Background:** BFGS with a numerical gradient and a Nelder-Mead polish stop at the objective's rounding (`-ll` to `1e-10` relative). On the flat
REML likelihood of `ls.vif.REML.bcg2` that leaves the location vif `2e-6` to `4e-6` relative from the optimum on CI. R's `nlminb` is `2.6e-6`
off. Newton steps on the analytic score and observed information from metafopy's stop reach `1e-10` (measured with the test formulas,
2026-11-24). The precedent is [rma_mv REML stop](../rma-mv-reml-stop/spec.md) (2026-11-17, capped Newton polish).
**Affects:** every log-link location-scale output, by about `1e-6` relative (within the parity tolerance `OPT`); permutation and profile refits;
the run time of each fit, by one analytic information per Newton step.
**Options:**
- *a. Capped Newton polish* (`polish_log`): metafopy ends at the same point on every platform; outputs move by the stop's error; ridge
  and boundary fits stay where they are, held by the cap.
- *b. Widen the location vif tolerance of `ls_vif` to `4e-5`* (10 times the largest measured deviation, the margin rule of
  [Cross-platform numerics](../cross-platform-numerics/spec.md)): no library change; metafopy's output stays platform-dependent at `1e-6`.
**Recommended:** a. It fixes the cause, which also reaches the other flat location-scale fits recorded in `known.toml` (`ls_year`, `ls_na`).

### 3. How strict the exact permutation reference stays

**What is decided:** whether the exact permutation cases check only the widened bounds, or also each regular permutation.
**Background:** 77 of the 1287 permuted models of case 4 have no finite maximum, so their `z` depends on where an optimiser stops. After case 2
the bounds are 774 to 851 (0.601 to 0.661); metafopy gives 794. metafopy's `zval_perm` comes in the reference's permutation order, but on 8 regular
permutations metafopy and R stop at a lower local maximum than the reference.
**Affects:** `ls_permutest_exact` in `test_derived.py`, both exact permutation cases.
**Options:**
- *a. Bounds only*: a weaker but deterministic check; the local-maximum finding is filed separately.
- *b. Bounds plus per-permutation agreement* of the `|z| >= |z_obs| - comptol` verdict on the regular permutations: stronger; it passes only
  once the 8 permutations are recorded as a mismatch (reference against metafopy and R) in `agents/specs/correctness-ls-extensions/mismatches.md`.
**Recommended:** a, with the finding filed: b makes this item depend on a separate question (which local maximum metafopy should report).

Version 1.2, 2026-11-24: the recall step of Case 1 is dropped. `bin/metafor-audit.sh --recall` audits metafopy as of each earlier audit's
commit (`--metafopy-src`), so a library change cannot change its result; the `--only 'construct/ls_na*'` run stays. The CI check of the
Closing happens after the operator's `/ship`, outside the task list, since the merge gate needs every task done.
