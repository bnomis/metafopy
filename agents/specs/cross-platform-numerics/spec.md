# Cross-platform numerics

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-16

Numerical tests that pass on the development Mac fail in CI on Linux, macOS and Windows. This spec defines, per failure group, how the cause is confirmed,
which verdicts are allowed, how Linux is reproduced locally and Windows checked through CI, and when the work is done: the CI test jobs green on all
three platforms and the local suite still passing. It changes test code, test tolerances, a container script, a probe script and a CI workflow; it changes
no result of `src/metafopy/` unless the operator confirms a disagreement at an operator stop.

The source is the `TODO.md` item "Numerical tests whose results or tolerances hold only on the development machine" (filed 2026-11-16). It follows the
[Correctness tests](../correctness-tests/spec.md), [Correctness GLMM](../correctness-glmm/spec.md),
[Correctness: location-scale models](../correctness-location-scale/spec.md), [Correctness: rma_mv and robust](../correctness-mv-robust/spec.md),
[Correctness: publication bias](../correctness-publication-bias/spec.md) and [CM.EL information](../cm-el-information/spec.md) specs, whose tests fail.

## Motivation

### The `TODO.md` item, dates and figures as filed

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
> Found on 2026-11-16 by a TODO review in CI run 43832306599 on `ff7eb2b` (Python 3.14.8; Linux `ubuntu-26.04`, macOS `macos-26`, Windows
> `windows-2025-vs2026`), while the final checkpoint of `agents/tasks/order-numerical-fixes.md` passed on the development Mac the same day. [...]
>
> - `ls.*.bcg` correctness cases: on Windows the ML reference fit ends at another optimum (`ls.fitstats.ML.bcg` ll -7.5393 against -9.4736;
>   `ls.predict.fitted.ML.z.bcg` "expected a nonnegative input, got -1.5653430049987447e+22"); on Linux every REML case says "the reference fit did not
>   converge" (`ls_reference_fit`: "no start converged (log link, REML): Desired error not necessarily achieved due to precision loss.").
> - `ls.fit.ML.identity.bcg` (Windows, Linux) and `ls.fit.REML.identity.bcg` (Linux): `alpha` off by 4e-8 to 3e-6 at tolerance 1e-6.
> - `selmodel.decreasing.bangertdrowns2004` on all three: the reference beta is -4.537 (Linux), -2248.16 (macOS) or the optimiser stops with "Positive
>   directional derivative for linesearch" (Windows); metafopy gives 0.10569.
> - `glmm.fit.or.um.rs.ml.jackson2018.pneumonia` and `glmm.cor.um.rs.ml.bcg` (Windows, Linux): "UM.RS: the score at the optimum is 1.05e-06 to 3.02e-06,
>   not below 1e-06".
> - `test_nchg.py::test_ml_se_matches_spec_measurement_at_tight_optimum`: `pneumonia` (Windows, 0.8881022865 against 0.88810229648 at 8.9e-09) and
>   `collins1985a` (Linux, 0.1363854789 against 0.13638547727 at 1.4e-09).
> - `property.mv.reml-score.kon` (Linux): REML score 1.3354e-05, theory 0.
> - `test_stepcon.py::test_inference[rau_stepcon_mods]` (Windows, Linux); `test_ls_vif.py::test_sim_permutes_the_data_variables_as_r` (all three);
>   `test_audit4_numerics.py::test_permutest_ls_boundary_fits_match_r` (all three); `test_rma_ls.py::test_identity_fit_is_the_maximum[bcg_year_identity]`
>   and `test_selmodel.py::test_print[hack_halfnorm_less]` (macOS).
>
> Cause: not confirmed. Suspected: the optimisers' stopping points and the floating-point results of the platforms' BLAS and libm builds differ, and the
> tests compare against a reference fit or an absolute tolerance measured on one machine. Some tests were added in the numerical fixes order on
> 2026-11-15 and 2026-11-16 (the CM.EL tight-optimum check, the identity-link polish).
<!-- doc-counts: accept-end -->

### Checked on 2026-11-16 while writing this spec

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
- CI on `main` has failed on every completed run since 2026-11-01 20:57 (`2fc8c6d`); the last pass was run 41344495853 on `2abba09`. Runs 43832306599
  (`ff7eb2b`) and 43883708179 (`dae8748`) failed the same tests (the `FAILED` lines of `gh run view <id> --log-failed` are identical): Linux 29, macOS 7,
  Windows 33, of which the harness items below account for one on Linux and macOS and eight on Windows. The `lint` job passed. Run 43928389985
  (`0622fd5`) was in progress at 18:28 UTC. The failures are therefore reproducible per platform, not run-to-run noise.
- Every job installs numpy 2.5.4 and scipy 1.18.1 from `uv.lock` (Python 3.14.8 from uv). The development Mac runs Python 3.14.7 (`.python-version`),
  the same numpy and scipy built against Apple Accelerate, on macOS 27 (Darwin 27.2.0); the CI Mac runs `macos-26`; the Linux and Windows wheels use
  OpenBLAS.
- Where metafopy and a reference disagree, metafopy's printed value is the same on every platform and the reference moves: `ls.fit.ML.identity.bcg`
  metafopy alpha `[-0.02665867, 0.00205067]` on Linux and Windows, reference `[-0.02665863, 0.00205066]` (Linux) and `[-0.02665849, 0.00205065]`
  (Windows); `ls.fit.ML.log.bcg` on Windows, metafopy alpha `[-4.95749055, 0.03859326]` with ll -7.5393, the reference `[-6.94692843, -2.45754824]` with
  the lower ll -9.4736.
- Second run details: `test_permutest_ls_boundary_fits_match_r` gives `pval_alpha[0]` 0.11905 (Linux), 0.12195 (macOS), 0.15 (Windows) for R's 0.10638;
  `rau_stepcon_mods` `ci_lb[0]` is -0.0017827 (Linux) and -0.0017917 (Windows) for R's -0.0017894; `bcg_year_identity` on the CI Mac gets a reference
  alpha of `[0.20251, -0.03682]` against metafopy's `[-0.02666, 0.02051]`; `hack_halfnorm_less` prints `ci.lb` 0.0049 against R's 0.0124 at a noisy-Hessian
  tolerance of 0.25 relative and 5e-3 absolute; `test_sim_permutes_the_data_variables_as_r` finds no NaN draw where it asserts exactly one (`[seed]` on all
  three, `[stream]` on macOS and Windows).
<!-- doc-counts: accept-end -->

## Scope

In scope: every failure group in [Failure groups](#failure-groups), the Linux container script, the probe and its workflow, and the records.

Out of scope, each handled by its own task list that runs before this spec's list:

- `TODO.md` "A harness test runs Rscript with no skip where R is missing" (`test_r_takes_the_cluster_port_it_is_given` in
  `src/tests/test_harness_engine.py`).
- `TODO.md` "Audit harness tests assume POSIX and fail on Windows" (tests in `src/tests/test_harness_engine.py` and `src/tests/test_harness_scripts.py`).

Also out of scope: the `lint` job, the R fixtures workflow `.github/workflows/r-fixtures.yml`, and the gap between `.python-version` (3.14.7) and CI's
Python (3.14.8). The container uses CI's version so that this gap cannot hide a difference.

**Prerequisite** (checked by the first task): both headings above are gone from `TODO.md` and present in `DONE.md`, and on the branch being built
`pytest src/tests/test_harness_engine.py src/tests/test_harness_scripts.py` passes. If not, the list stops: the acceptance below cannot be met while those
tests fail.

## Module location

- `bin/linux-tests.sh` and `bin/linux-tests.Dockerfile`: the Linux container (new).
- `bin/platform-probe.py` and `bin/platform-probe.sh` (wrapper): the probe (new).
- `.github/workflows/platform-probe.yml`: runs the probe on the three CI platforms and uploads its output (new).
- Tests changed as the verdicts require: `src/tests/correctness/formulas.py` (`ls_reference_fit`, `ls_starts`), `src/tests/correctness/glmm_formulas.py`
  (`maximise`, `GRADIENT_TOLERANCE`), `src/tests/correctness/test_derived.py` (`selmodel_maximum`, `LS_RELATIVE`), `src/tests/correctness/test_properties.py`
  (`mv_score_is_zero`), `src/tests/correctness/cases/*.toml` (case `tolerance`), `src/tests/test_nchg.py`, `src/tests/test_stepcon.py`,
  `src/tests/test_ls_vif.py`, `src/tests/test_audit4_numerics.py`, `src/tests/test_rma_ls.py`, `src/tests/test_selmodel.py`, `src/tests/refutil.py`.
- Records: `agents/specs/cross-platform-numerics/measurements.md` (new, dated), `docs/differences-notes.md` ("Optimisers and flat likelihoods").

## API

```python
# bin/platform-probe.py
@dataclasses.dataclass(frozen=True)
class GroupMeasurement:
    """What one failure group's tests compare, measured in this environment: named arrays, optimiser messages, and the error if the group raised."""

    group: str
    values: dict[str, list[float]]
    messages: list[str]
    error: str | None


def environment() -> dict[str, str]:
    """``sys.platform``, ``platform.machine()``, the Python version and build, numpy and scipy versions, the BLAS and LAPACK libraries from
    ``numpy.show_config(mode='dicts')``, ``platform.libc_ver()`` on Linux and the OS release: the identity of the numbers that follow."""


def measure_groups(groups: collections.abc.Sequence[str] | None = None) -> dict[str, GroupMeasurement]:
    """Measure each group (all when ``None``) with the same helpers its tests call (``correctness.formulas``, ``correctness.glmm_formulas``, the test
    modules' fixtures); a group that raises is recorded with its error and the others still run.

    Raises:
        KeyError: a name in ``groups`` is not a group.
    """


def main(argv: list[str] | None = None) -> int:
    """``platform-probe.py [--group NAME ...] --out PATH``: write ``{'environment': ..., 'groups': ...}`` as JSON; exit 0, or 1 if any group raised."""
```

```text
bin/linux-tests.sh [--arch arm64|amd64] [--probe OUT_DIR] [-- PYTEST_ARGS...]
    Build (or reuse) the image from bin/linux-tests.Dockerfile for linux/<arch>, copy the working tree into it, run `uv sync --locked --all-extras`, then
    pytest with PYTEST_ARGS (default: the group test files, see Settings) or, with --probe, bin/platform-probe.py writing OUT_DIR/probe-linux-<arch>.json.
    Exit status: pytest's or the probe's; 2 when Docker is not running or the image cannot be built.
```

## Failure groups

Platforms are those of run 43883708179 (2026-11-16): L Linux, M macOS (CI), W Windows. "Compared against" is what the test treats as the truth. The
mechanism is read from the test code on 2026-11-16; the cause is still to be confirmed.

| Group (probe name) | Tests | Fails on | Compared against | Mechanism in the test |
|---|---|---|---|---|
| `ls-bcg-reference` | `ls.*.bcg` log-link cases of `cases/location_scale.toml`, `property.ls.fit.REML.log.bcg` | REML: L; ML: W | `formulas.ls_reference_fit` | BFGS with `gtol` 1e-9 from `ls_starts`; a start whose result is not `success` (including "precision loss") is dropped, then the best kept one is polished by Nelder-Mead. On L no REML start counts as converged; on W the kept ML start is a worse stationary point |
| `ls-identity-alpha` | `ls.fit.ML.identity.bcg` (L, W), `ls.fit.REML.identity.bcg` (L) | L, W | `ls_reference_fit`, identity link | SLSQP (`ftol` 1e-14) then Nelder-Mead; `ls_agree` allows `LS_RELATIVE` (1e-6) of the largest absolute `alpha`, so 2.7e-8 for the ML fit |
| `selmodel-decreasing` | `selmodel.decreasing.bangertdrowns2004` | L, M, W | `test_derived.selmodel_maximum` | one SLSQP run (`ftol` 1e-15) from metafopy's start, `beta` unbounded, `delta` floored at 1e-6 with decreasing constraints; it runs away (beta -4.5, -2248) or stops |
| `glmm-um-rs-score` | `glmm.fit.or.um.rs.ml.jackson2018.pneumonia`, `glmm.cor.um.rs.ml.bcg` | L, W | `glmm_formulas.fit_problem` | the reference's own score (Richardson central differences at steps 5e-5 and 1e-4 of a quadrature likelihood) must be below the absolute `GRADIENT_TOLERANCE` 1e-6 |
| `cm-el-tight-se` | `test_nchg.py::test_ml_se_matches_spec_measurement_at_tight_optimum` | `collins1985a` L, `pneumonia` W | the figures of the CM.EL information spec (2026-11-15) | Nelder-Mead (`xatol` 1e-11) to the optimum, then `rel=1e-8` on the standard error |
| `mv-reml-score` | `property.mv.reml-score.kon` | L | theory: score 0 | metafopy's own `rma_mv` fit with `gtol` 1e-12; the REML score of `sigma2` must be within the case `tolerance` 1e-6, absolute |
| `stepcon-rau-mods` | `test_stepcon.py::test_inference[rau_stepcon_mods]` | L, W | R fixture | `LOOSE` 5e-4 relative and `ATOL` 1e-6 on `ci_lb` near zero (-0.00179) of a fit flat in `delta` |
| `ls-vif-sim` | `test_ls_vif.py::test_sim_permutes_the_data_variables_as_r` | `[seed]` L, M, W; `[stream]` M, W | metafopy's dev-Mac behaviour | asserts exactly one NaN draw (a ridge refit whose Hessian is NaN on the dev Mac); CI finds none |
| `permutest-ls-boundary` | `test_audit4_numerics.py::test_permutest_ls_boundary_fits_match_r` | L, M, W | R fixture | `pval_alpha`, `QSp` at 0.1 relative (0.25 on `win32`); their denominators count permuted fits with a defined standard error, which differ by platform |
| `ls-identity-maximum` | `test_rma_ls.py::test_identity_fit_is_the_maximum[bcg_year_identity]` | M | `ls_reference_fit`, identity link | metafopy's `alpha` must equal the reference's at 1e-6; on the CI Mac the reference ends at another point (`[0.2025, -0.0368]`) |
| `selmodel-print-halfnorm` | `test_selmodel.py::test_print[hack_halfnorm_less]` | M | R fixture | printed text at `NOISY_HESSIAN` 0.25 relative and `NOISY_ATOL` 5e-3; `delta` at its bound (0.0001), so the Hessian and `ci.lb` move |

### Confirming the cause, per group

A cause is confirmed by figures from at least the development Mac, the Linux container and the three CI platforms (the probe), recorded in
`measurements.md` with the date and `environment()` of each. Per group:

- `ls-bcg-reference`, `ls-identity-alpha`, `ls-identity-maximum`: for every start, the optimiser message, `-fun`, the analytic `ls_score` (identity link:
  the score along the free directions and the sign of the multipliers of the active rows of `Z alpha >= 0`), and metafopy's `alpha` and log-likelihood.
  Confirmed as a reference defect when metafopy's point has a log-likelihood at least the reference's minus `LS_LOGLIK` and a score at least as small,
  on every platform. If instead the reference finds a higher log-likelihood anywhere, metafopy is not at the maximum: operator stop.
- `selmodel-decreasing`: the log-likelihood at each platform's reference end point and at metafopy's point, and along the ray from metafopy's point to the
  runaway point. A reference defect when no end point beats metafopy's by more than `SELMODEL_LL_TOLERANCE`; if one does (an unbounded or higher
  likelihood), operator stop.
- `glmm-um-rs-score`: the score at the reference optimum by each step pair, and its spread over steps 2e-5 to 4e-4 (the finite-difference noise floor),
  beside the analytic score where `glmm_formulas` has one. Confirmed as noise when the measured score is within the noise floor on each platform.
- `cm-el-tight-se`: the Nelder-Mead end point, its `neg_loglik` and `marginal_information`-based score, and the standard error. Confirmed when the
  standard error's spread across platforms follows the spread of the end point.
- `mv-reml-score`: metafopy's optimiser message and gradient in its own parameterisation, the REML score in `sigma2`, and `tr(P Z Z')` (its scale). If
  metafopy's optimiser reports convergence and the score is a rounding-size fraction of its scale, the tolerance is wrong; if not, operator stop.
- `stepcon-rau-mods`, `selmodel-print-halfnorm`: the compared quantities and their standard errors on each platform. Confirmed when the spread is a small
  fraction of the standard error.
- `ls-vif-sim`, `permutest-ls-boundary`: which draws or permutations are NaN or boundary fits, and their `alpha` at the stop, on each platform. Confirmed
  when the only difference is the count of boundary stops along a ridge (the operator accepted that stop as an optimiser difference on 2026-11-04).

## Verdicts

Each group gets one verdict, written in `measurements.md` with its figures:

1. **Fix metafopy.** Allowed only after an operator stop that shows the figures and the reference, and the operator's confirmation (project rule: the
   defining publication first, R metafor second; metafopy's output changes only after the user confirms each disagreement). Goes through its own spec if it
   changes `src/metafopy/`; this list stops there.
2. **Make the reference fit robust.** For a reference in test code (`formulas`, `glmm_formulas`, `selmodel_maximum`): more starts, keeping a result whose
   score passes a stated check whatever the optimiser's message, a final polish (Newton on the analytic score, or projected for the identity link), and a
   final check that the score is small. The fit must still never call metafopy. It is right when it gives the same end point on every platform, to
   well inside the case tolerance.
3. **Set a tolerance from the measured spread across platforms.** Allowed when the spread is far below the effect the test exists to detect. The new value
   follows the margin rule of the choice "Tolerance margin"; its comment names the measured spread, the environments and the date.
4. **Mark the case platform-dependent, with a reason.** Allowed only for a quantity that depends on an optimiser stop along a flat ridge or at a bound, where
   verdicts 2 and 3 cannot hold it. How it is marked is the choice "Platform-dependent marks"; the reason names the measurement and the entry in
   `docs/differences-notes.md`.

Preference: 2 over 3 over 4 when the reference is what moves; 3 or 4 for a quantity of metafopy that moves with the platform. A tolerance is never widened to
cover a metafopy value that a robust reference shows is wrong.

## Implementation

### Linux reproduction

`bin/linux-tests.Dockerfile`: `FROM ubuntu:26.04` (CI's image family, so glibc's libm matches the runner's release) with the uv binary copied from
`ghcr.io/astral-sh/uv:<version>`, both pinned to exact tags; the task records the image digests in `measurements.md`. `bin/linux-tests.sh`:

- checks `docker info` first and exits 2 with "Docker is not running: start Docker Desktop" if it fails;
- streams the working tree (`git ls-files -co --exclude-standard`) into the container by `tar`, so the host `.venv`, `.audit` and untracked data never
  enter it and nothing in the checkout is written;
- keeps uv's cache and the container's virtual environment in named Docker volumes per architecture (`metafopy-uv-cache`, `metafopy-venv-<arch>`);
- runs `uv sync --locked --all-extras --python 3.14.8`, then `uv run pytest --mpl <args>` or the probe; it never re-locks.

Linux on the development Mac runs `linux/arm64` natively. CI's Linux is x86_64, where OpenBLAS chooses other kernels, so arm64 agreement is checked, not
assumed: the first measuring task runs the group tests in the arm64 container and compares the failing list with the Linux list of run 43883708179. If
they differ, it repeats with `--arch amd64` (emulated, slower) and records which reproduces; CI stays the arbiter either way. I chose this order because
arm64 is the fast loop and the comparison tells at once whether it is a faithful one; it is reversible by passing `--arch`.

### Windows and the CI Mac

Windows exists only in CI, and the CI Mac differs from the development Mac, so both are measured through `.github/workflows/platform-probe.yml`: the
`test` job's matrix (`ubuntu-26.04`, `macos-26`, `windows-2025-vs2026`, Python 3.14.8, `uv sync --locked --all-extras`), running
`uv run python bin/platform-probe.py --out probe-${{ runner.os }}.json` and uploading it with `actions/upload-artifact` pinned to its newest release tag
(checked with `gh api repos/actions/upload-artifact/releases/latest`, per the security rule on action pins). The probe is a separate workflow so that it
runs whatever the tests do. Its outputs are fetched with `gh run download <id>` into a directory outside the repository (`$TMPDIR/metafopy-platform-probe/`);
only the figures copied into `measurements.md` are committed. How the workflow is triggered during the work is the decision "CI runs before the merge".

### Order of work

1. Prerequisites (see [Scope](#scope)); `docker info`; `gh auth status`.
2. Container script; the arm64 (then amd64 if needed) comparison with CI's Linux list; record.
3. The probe, the probe workflow; measure on the development Mac, in the container and in CI; write `measurements.md`.
4. Per group, in the table's order: confirm the cause, choose the verdict, change the test; run the group's tests locally and in the container. A group
   whose confirmation points at metafopy is an operator stop; the other groups carry on.
5. CI on all three platforms (before or after the merge, per the decision); a group still failing returns to step 4 with the new figures.
6. Records: `docs/differences-notes.md` for every platform-dependent mark; `measurements.md` complete; the `TODO.md` item moves to `DONE.md` after the
   merge, by the usual `todo-do` flow, not by this list.

### Tests of the new code

- `bin/platform-probe.py`: a test in `src/tests/test_platform_probe.py` that `measure_groups(['cm-el-tight-se'])` returns a `GroupMeasurement` with finite values
  and no error, that an unknown group raises `KeyError`, and that `main` writes JSON with `environment` and `groups` keys (to `tmp_path`).
- Each robust reference fit: a test that a start known to stop with "precision loss" (forced by `maxiter` or a perturbed start) is still accepted when
  its score passes, and that a start at a worse stationary point is not chosen over a better one; each test is run once against the old code and seen to
  fail, as the shared testing rule requires.
- `bin/linux-tests.sh` is checked by running it (`--help`, the Docker-not-running message with the daemon stopped, a group run); no pytest test.

## Error handling

- Docker not running or the image not building: `linux-tests.sh` exits 2 with the message above; no fallback to the host.
- `uv sync --locked` failing in the container (lock out of date for Linux): the script exits with uv's status and message; the task stops and reports,
  it never runs `uv lock` (project rule: `uv.lock` is not edited without instruction).
- A probe group raising: recorded in its `GroupMeasurement.error`; the other groups run; `main` exits 1.
- A CI probe artifact missing for a platform: the measuring task stops and names the run id and job; the group is not judged on fewer platforms.
- A robust reference that still finds no point with a small score: it raises `RuntimeError` naming the case, every start's message and score, as
  `ls_reference_fit` does now.

## Settings

| Setting | Type | Default | Description |
|---|---|---|---|
| `linux-tests.sh --arch` | `arm64 \| amd64` | `arm64` | Container architecture (`linux/<arch>`) |
| `linux-tests.sh --probe` | path | unset | Run the probe instead of pytest, writing `probe-linux-<arch>.json` there |
| `linux-tests.sh` pytest arguments | list | the test files of the groups: `src/tests/correctness/test_derived.py`, `src/tests/correctness/test_properties.py`, `src/tests/test_nchg.py`, `src/tests/test_stepcon.py`, `src/tests/test_ls_vif.py`, `src/tests/test_audit4_numerics.py`, `src/tests/test_rma_ls.py`, `src/tests/test_selmodel.py` | Anything after `--` replaces them |
| Container Python | `str` | `3.14.8` | CI's version, from `.github/workflows/ci.yml` |
| `platform-probe.py --group` | list | all groups | Groups to measure |
| `platform-probe.py --out` | path | required | JSON output, outside the repository |

## External dependencies

Docker Desktop (installed; the operator starts its daemon before the list runs, a prerequisite because the container needs it and no other Linux is
available locally), the `ubuntu:26.04` and `ghcr.io/astral-sh/uv` images, `gh` (authenticated, for logs, runs and artifacts), `actions/upload-artifact`.
NumPy and SciPy as now; no new Python dependency.

## Acceptance

- The three `test` jobs of `.github/workflows/ci.yml` pass on the CI run of the merge commit on `main` (`gh run view <id> --json jobs`), and `lint` still
  passes.
- Locally, on the development Mac, every test directory passes, run one at a time in the background at the final checkpoint (the order's checkpoint
  command; about 30 minutes as of 2026-11-16).
- The group tests pass in the Linux container on the architecture that reproduced CI.
- `git diff main -- src/metafopy` is empty, unless an operator stop confirmed a change.
- Every changed tolerance and every platform-dependent mark has a comment with the measured spread, the environments and the date, and `measurements.md`
  holds each group's figures and verdict.

## Decisions

Answered on 2026-11-16 in `agents/reviews/order-ci-platform-failures-specs.md`: D1 *Draft pull request*, D2 *keep*; the spec was approved
there with its choices. Version 1.1 records the answers.

### CI runs before the merge

**What is decided:** whether the task list may push its worktree branch and open a draft pull request, so that CI and the probe run on Linux, macOS and
Windows before the merge to `main`.
**Background:** CI triggers on a push to `main` and on `pull_request` only (`.github/workflows/ci.yml`, checked 2026-11-16), so a pushed branch alone runs
nothing. Windows can only be observed in CI, and the CI Mac fails tests the development Mac passes. The repository is public: a pull request is visible.
The shared rule is to push only when the user asks; this decision is that ask, or its refusal.
**Affects:** whether acceptance can be proved before the merge; how many merges to `main` the work takes; outward-facing actions by an unattended queue.
**Options:**
- *Draft pull request*: the list pushes its branch, opens a draft PR, reads CI and probe results with `gh`, and loops until green; the PR is closed
  without merging after the usual local merge, and the remote branch deleted. CI runs about 40 minutes per push on Windows (2026-11-16).
- *After the merge only*: the list merges with figures from the development Mac and the container; the probe workflow also triggers on `workflow_dispatch`
  and runs on `main` after the merge; a group still failing on Windows or the CI Mac becomes another round with another merge. `main` stays red meanwhile
  (it is already red).
- *Operator pushes*: the list stops when it needs CI; the operator pushes and opens the PR; the list resumes and reads the results.
**Answer (2026-11-16): *Draft pull request*.** Recommended: Windows and the CI Mac decide whether the work is done, and without them each round costs a merge to `main`;
a draft PR is the narrowest outward action that gives those results.

### The probe workflow after the work

**What is decided:** whether `.github/workflows/platform-probe.yml` stays once CI is green.
**Background:** it is needed during the work; afterwards it would re-measure the groups when wheels or runner images change.
**Affects:** CI minutes on pull requests (Dependabot opens some); a future cross-platform failure starts with figures instead of a container session.
**Options:** *keep*, triggered by `workflow_dispatch` and by `pull_request` with a `paths` filter (`uv.lock`, `src/metafopy/**`, `src/tests/correctness/**`,
`bin/platform-probe.py`); *delete* it in the last task.
**Answer (2026-11-16): *keep*, with the filter.** Recommended: a dependency update is exactly when the spread can move, and the filter keeps other pull requests free of it.

## Choices made

Made while writing the spec and reported, not asked: each follows from the project's practice and is reversible by amending this spec.

### Tolerance margin

**What it settles:** how a tolerance set under verdict 3 relates to the measured spread.
**Background:** the spread is measured on five environments (development Mac, container, three CI platforms). The CI Mac and the development Mac share
`sys.platform`, so a tolerance keyed on the platform cannot tell them apart; the existing `p_rtol = 0.25 if sys.platform == 'win32' else 0.1` in
`test_audit4_numerics.py` already fails on the CI Mac.
**Affects:** every verdict-3 group; how often a runner image or wheel update breaks CI again.
**Options:** *one value, 10 times the largest measured deviation*, for all platforms; *one value, 2 times*; *per-platform values* keyed on `sys.platform`.
In each case, a value that would exceed the effect the test exists to detect is refused and the group goes to verdict 2 or 4.
**Chosen:** *one value, 10 times*: it absorbs an image update without hiding a real error at the scales involved (1e-9 to 1e-6), and one value
behaves the same on both Macs.

### Platform-dependent marks

**What it settles:** how a verdict-4 case is written.
**Background:** the precedent is `refutil.linux_only` and `macos_only` (`skipif` on `sys.platform`) for last-bit libm results. A strict `xfail` keyed on
`sys.platform` fails where the platform passes (the development Mac against the CI Mac); a non-strict `xfail` passes either way and checks nothing.
**Affects:** `ls-vif-sim`, `permutest-ls-boundary` and any other group judged platform-dependent.
**Options:** *assert the observed range*: the test accepts every outcome measured on every platform (for example 0 or 1 NaN draws; a `pval_alpha` whose
denominator lies in the measured range) and still checks the rest exactly; *skip by platform* with `skipif` and the reason; *non-strict xfail*.
**Chosen:** *assert the observed range*, with *skip by platform* only where no range can be stated: the test keeps running everywhere and still
catches a change outside what the platforms do.

## Operator stops

- A group whose confirmation shows metafopy below a reference's maximum or off a theoretical value (most likely candidates from the mechanisms:
  `mv-reml-score`, `ls-identity-maximum`, `selmodel-decreasing`): the list stops with the figures from every environment and the options (fix through a
  spec, or record a correctness mismatch). The other groups carry on first.
- The prerequisites unmet (the harness task lists not merged, Docker not running, `gh` not authenticated).
- Under the decision "CI runs before the merge" answered *Operator pushes*, each CI round.
