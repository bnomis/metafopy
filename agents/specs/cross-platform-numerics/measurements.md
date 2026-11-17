# Cross-platform numerics: measurements

**Spec:** `agents/specs/cross-platform-numerics/spec.md`

Each section is a dated record of what was measured, not a claim about the present.

## Linux container (Task 2 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-16 - a dated record, not a claim about the present -->

Images, read on 2026-11-16 (`docker buildx imagetools inspect`, `docker image inspect`):

| Image | Digest or id |
|---|---|
| `ubuntu:26.04` (index digest) | `sha256:d66b366dac61b575bee2aef07e9c26e8070096fbe84a3d4d3eb0da069b5d76db` |
| `ghcr.io/astral-sh/uv:0.13.8` (index digest; newest release on 2026-11-12, no resolver quarantine in `pyproject.toml`) | `sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85` |
| `metafopy-linux-tests:arm64` (built id) | `sha256:4e800974134e331f98e65345fecfb70cf8b3ae7dcdb08789eaa572d288569fbc` |
| `metafopy-linux-tests:amd64` (built id) | `sha256:889741624b33684c0139fe26aaffe2c5c5ab83f510214f9deacf1b570fcc7049` |

Group test files run with `bin/linux-tests.sh` (Python 3.14.8, `uv sync --locked --all-extras`), compared with the `ubuntu-26.04` job's `FAILED` lines of
CI run 43883708179 (`test_r_takes_the_cluster_port_it_is_given` is one the two earlier harness lists fixed, so it is left out of the comparison):

| Environment | Result | Failing tests not in CI's Linux list | CI's Linux failures that passed |
|---|---|---|---|
| arm64 container | 9 failed, 1420 passed | `glmm.fit.or.um.rs.fe.nielweise2007.beta`, `mv.confint.pl.tau2.berkey`, `test_ls_vif ... [stream]`, `test_identity_fit_bd_is_a_stationary_point` | all `ls.*REML*` cases, `property.mv.reml-score.kon`, `ls.fit.ML.identity.bcg`, `ls.heterogeneity.bcg`, `test_stepcon [rau_stepcon_mods]`, `test_ls_vif [seed]` |
| amd64 container (emulated) | 7 failed, 1422 passed | `test_ls_vif ... [stream]`, `test_ml_se_..._at_tight_optimum[nielweise2007]`, `test_identity_fit_is_the_maximum[bcg_year_identity]` | all `ls.*REML*` cases, `property.mv.reml-score.kon`, `ls.fit.ML.identity.bcg`, `ls.heterogeneity.bcg`, `test_permutest_ls_boundary_fits_match_r`, `test_stepcon [rau_stepcon_mods]`, `[collins1985a]` of the tight-optimum test |

Common to CI's Linux, arm64 and amd64: `glmm.cor.um.rs.ml.bcg`, `glmm.fit.or.um.rs.ml.jackson2018.pneumonia`, `selmodel.decreasing.bangertdrowns2004`, and a
`test_ls_vif` and a `test_nchg` tight-optimum case.

**Verdict: neither architecture reproduces CI's Linux list.** Both reproduce the GLMM UM.RS, decreasing selection model, vif simulation and tight-optimum
groups, in different cases; the `ls` REML group, `mv-reml-score`, `stepcon-rau-mods` and (on arm64 only) `permutest-ls-boundary` of CI's list did not
fail in either container. arm64 is the faster loop (about 14 minutes against about 17); CI stays the arbiter. The five-environment probe (Tasks 3 and 4)
decides each group.

`bin/linux-tests.sh -- src/tests/test_nchg.py` on arm64: runs to completion, 1 failed (`test_ml_se_matches_spec_measurement_at_tight_optimum
[collins1985a]`), 32 passed.

<!-- doc-counts: accept-end -->

Found while building the script: GNU tar in the emulated amd64 container cannot create nested directories ("Function not implemented"), so the image
installs `bsdtar` (`libarchive-tools`) and the script extracts with it.

Correction to the section above, found on 2026-11-16 while running the probe: `bin/linux-tests.sh` then ran `uv run` without naming the interpreter, so uv
re-selected the project's `.python-version` (3.14.7) and rebuilt the container's virtual environment before running pytest. The group runs above were
therefore on Python 3.14.7, not CI's 3.14.8. The script now sets `UV_PYTHON` to 3.14.8 for `uv run` too; the probe run below shows 3.14.8. The group
comparison above is not repeated: the probe's figures decide each group.

## The probe on the development Mac and in the arm64 container (Task 3 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-16 - a dated record, not a claim about the present -->

`bin/platform-probe.py` run on 2026-11-16 (outputs in `$TMPDIR/metafopy-platform-probe/probe-mac.json` and `probe-linux-arm64.json`, outside the
repository). Every group ran in both without error. `environment()`:

| | Development Mac | Linux arm64 container |
|---|---|---|
| `sys.platform`, machine | `darwin`, `arm64` | `linux`, `aarch64` |
| OS | macOS 27.2.1 | Ubuntu 26.04.1 LTS, glibc 2.43 |
| Python | 3.14.7 (Clang 21.0.0) | 3.14.8 (Clang 23.1.3) |
| numpy, scipy | 2.5.4, 1.18.1 | 2.5.4, 1.18.1 |
| BLAS, LAPACK | Accelerate | scipy-openblas 0.3.34.106.0 |

The Mac's Python is the main checkout's venv (3.14.7), not CI's 3.14.8; the figures of the other three environments (CI Mac, Windows, Linux) come in
Task 4. Figures below are the probe's, to the digits shown.

| Group | Development Mac | Linux arm64 container |
|---|---|---|
| `ls-bcg-reference` (BFGS starts, message) | ML: all three start `success`; REML start0 "precision loss" (score 2.3e-06), start2 "precision loss" at another point (score 3.7), start1 success | ML start1 "precision loss" (score 2e-08); REML start0 success, start2 "precision loss" but at the same point as start1 (score 9e-09). Both: metafopy's `ll` -7.53929025 (ML), -8.087277925 (REML) equals the reference's; metafopy's score 5e-09 to 8e-09 (ML), 1e-09 to 6e-09 (REML) |
| `ls-identity-alpha` (SLSQP) | ML reference `alpha` [-0.026658667, 0.002050667], `ll` -7.139288818, metafopy `ll` -7.139288823; REML starts 0, 2, 3 "Positive directional derivative" at [0.339135521, -0.0061661] (`ll` -7.988765268), start1 success at [0.01337, 0.001627] (`ll` -8.0808) | ML the same point to 1e-9; REML starts 0 and 2 succeed at [0.339135521, -0.0061661] and start4 ends at [0.01337, 0.001627]; reference `alpha` [0.33913552, -0.0061661], `ll` -7.988765268, equal to metafopy's. The scores of the free directions stay 742 (ML) and 98 (REML) on both: the optimum lies on the boundary `Z alpha = 0` |
| `selmodel-decreasing` | reference converged: `ll` -17.421673311 against metafopy's -17.421673313, `ll` along the ray from metafopy's point to the end point constant to 1e-9 | reference ran away: end point beta -2.8667, tau2 1.3122, delta [1, 1e-06, 1e-06], `ll` -400.155; along the ray from metafopy's point `ll` falls from -17.42 to -56.2, -97.9, -135.4, -400.2. The reference's maximum is metafopy's point on both; no end point beats it |
| `glmm-um-rs-score` (largest absolute score, steps 2e-5, 5e-5, 1e-4, 2e-4, 4e-4, Richardson) | pneumonia 1.4e-06 to 1.9e-06, Richardson 1.4e-06; bcg 4.7e-07 to 8.6e-06, Richardson 1.7e-06 | pneumonia 1.1e-06 to 1.8e-06, Richardson 1.7e-06; bcg 1.1e-06 to 1.1e-05, Richardson 1.1e-06. `loglik` equal to 1e-9 in both |
| `cm-el-tight-se` | `se` at the tight optimum 0.136385477, 0.35104033, 0.888102302 (expected 0.13638547727, 0.35104032933, 0.88810229648); Nelder-Mead stops after 80, 63, 64 iterations | 0.136385479, 0.35104033, 0.888102302; stops after 66, 59, 62 iterations; end points differ from the Mac's in the eighth decimal (collins1985a `alpha` 0.174406202 against 0.174406193) |
| `mv-reml-score` | `sigma2` [0.072265591, 0.032650196]; score [1e-09, -8e-09] on a scale `tr(PZ)` of [104.47, 962.61] | `sigma2` the same; score [2e-09, -5e-09] on the same scale |
| `stepcon-rau-mods` | `ci_lb` [-0.001791281, -0.021703414] (R: [-0.0017894, -0.021703647]), `se` [0.068166661, 0.004925723], `delta` [1, 0.637551407] | `ci_lb` [-0.001790637, -0.021703416], `se` [0.068166654, 0.004925723], `delta` [1, 0.637554816]; `vd` 0.257365565 against 0.257367814 |
| `ls-vif-sim` (NaN draws of the first 50) | seed: none in beta, draw 42 in alpha; stream: none in beta, draw 3 in alpha | seed: none in beta, draw 0 in alpha; stream: none in either |
| `permutest-ls-boundary` | `pval_alpha` [0.1136, 0.5227, 0.2727] (R: [0.1064, 0.4894, 0.2553]), `QSp` 0.6818 (R: 0.6383); undefined permutations 4, 21, 24, 25, 40, 46 | `pval_alpha` [0.125, 0.55, 0.3], `QSp` 0.7; undefined permutations 4, 7, 21, 25, 32, 36 |
| `ls-identity-maximum` | metafopy and reference `alpha` [-0.02665867, 0.020506669], `ll` -7.139288823 and -7.139288818; starts differ in which one SLSQP calls successful | the same point and `ll` to 1e-9 |
| `selmodel-print-halfnorm` | `delta` 0.000137796 (at its bound), `se` 0.105032128, `ci_lb` 0.011259798, `vd` 1.120669121 | `delta` 0.000137796, `se` 0.105720076, `ci_lb` 0.009911444, `vd` 1.13981677 |

What these two environments already show, before CI's three:

- `ls-bcg-reference`: which BFGS starts report "precision loss" differs between the Mac and Linux, and no environment's metafopy point is beaten: the
  reference's log-likelihood equals metafopy's wherever the reference returns. A reference defect candidate (the dropped starts), to be confirmed on the
  CI platforms.
- `selmodel-decreasing`: on Linux the reference's single SLSQP run leaves metafopy's point for a runaway of log-likelihood -400; nothing beats metafopy's
  point. A reference defect candidate.
- `glmm-um-rs-score`: the score at the optimum is 1e-06 to 1e-05 with a spread across step sizes of the same size, in both environments.
- `permutest-ls-boundary`, `ls-vif-sim`: which draws are boundary fits or NaN differs between the Mac and Linux, as the spec's table says.
- `selmodel-print-halfnorm`: `se` differs by 0.7 per cent between the two with `delta` at its bound.
- `mv-reml-score`: the score is 1e-09 to 1e-08 on a scale of 100 to 1000 (about 1e-11 relative) in both: rounding size.

<!-- doc-counts: accept-end -->

## The probe on the three CI platforms (Task 4 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-16 - a dated record, not a claim about the present -->

`bin/platform-probe.py` run by `.github/workflows/platform-probe.yml` on the head commit of draft pull request 74 (branch
`worktree-cross-platform-numerics-tasks-1`): probe run 43972461382, started on 2026-11-16 and read on 2026-11-17; the same commit's CI run is
43972460854 (its `test` jobs fail on all three platforms, as before). Every group ran on all three platforms; Windows' `selmodel-decreasing` measurement
recorded an assertion in its messages (the reference's SLSQP did not converge; its end point is below). `environment()`:

| | CI Linux | CI Mac | CI Windows |
|---|---|---|---|
| `sys.platform`, machine | `linux`, `x86_64` | `darwin`, `arm64` | `win32`, `AMD64` |
| OS | Ubuntu 26.04.1 LTS, glibc 2.43 | macOS 26.7.2 | Windows Server 2025 (10.0.26100) |
| Python, numpy, scipy | 3.14.8, 2.5.4, 1.18.1 | 3.14.8, 2.5.4, 1.18.1 | 3.14.8, 2.5.4, 1.18.1 |
| BLAS, LAPACK | scipy-openblas 0.3.34.106.0 | Accelerate | scipy-openblas 0.3.34.106.0 |

Figures below are the probe's, to the digits shown, in the order CI Linux / CI Mac / CI Windows.

| Group | Figures |
|---|---|
| `ls-bcg-reference` | metafopy's `ll` -7.53929 (ML) and -8.08728 (REML) on all three, equal to the reference's wherever it returns. BFGS "precision loss" starts: ML none / start1 / starts 0 and 1; REML start0 / start0 / starts 0 and 2. Windows REML start2 stops at another point (`ll` -8.35439, score 3.75), and Windows' reference ML result is that of start2, `ll` -9.47361 (alpha [-6.94693, -2.45755]) **below metafopy's -7.53929**: the reference's returned fit is a worse stationary point there |
| `ls-identity-alpha` | metafopy equal on all three (ML `ll` -7.13929, REML -7.98877). REML reference: Linux ends at [0.0133747, 0.00162732] (`ll` -8.08078), Mac and Windows at [0.339136, -0.0061661] (`ll` -7.98877); the successful starts differ per platform; ML start2 on Linux fails with "Singular matrix E in LSQ subproblem" (`ll` infinite), on Mac reaches the point [0.202508, -0.00368196] (`ll` -7.89187) |
| `selmodel-decreasing` | metafopy's point and `ll` -17.4217 equal on all three. The reference's SLSQP: Linux converges to metafopy's point; **Mac runs away** (beta -2248.16, tau2 2683050, `ll` -378.22); **Windows ends in "Positive directional derivative for linesearch"** at beta -3059.28, tau2 4973450 (`ll` -393.029). `ll` along the ray from metafopy's point falls to about -365 to -393 on the Mac and Windows; no end point beats metafopy's. A reference defect on the Mac and Windows |
| `glmm-um-rs-score` | pneumonia Richardson score 7.5e-07 / 1.4e-06 / 1.9e-06; bcg 1.4e-06 / 1.7e-06 / 1.3e-06; fixed-step scores 4.7e-07 to 1.0e-05 across steps and platforms; `loglik` equal on all three |
| `cm-el-tight-se` | `se` at the tight optimum equal to the expected values to the digits shown on all three; Nelder-Mead stops after 68 / 80 / 78 (collins1985a), 59 / 63 / 65 (nielweise2007), 64 / 64 / 63 (pneumonia) iterations; the end points agree to the digits shown |
| `mv-reml-score` | `sigma2` [0.0722656, 0.0326501 or 0.0326502] on all three; score [5.1e-05, 1.1e-03] on Linux, [1.4e-09, -8.3e-09] on the Mac, [3.0e-07, -3.0e-08] on Windows, on a scale `tr(PZ)` of [104.475, 962.61]: relative 4.9e-07 and 1.1e-06 on Linux, 1.3e-11 and -8.6e-12 on the Mac, 2.9e-09 and -3.1e-11 on Windows |
| `stepcon-rau-mods` | `se` [0.0681665, 0.00492571] / [0.0681667, 0.00492573] / [0.0681664, 0.00492571] (R: [0.0681675, 0.00492576]); `ci_lb` first element -0.00179123 / -0.001791 / -0.00179168 (R: -0.0017894); `delta` second element 0.637549 / 0.637554 / 0.637544 (R: 0.637582); `tau2` 3.2e-10 / 7.6e-09 / 9.0e-09; `vd` last element 0.257361 / 0.257368 / 0.257355 |
| `ls-vif-sim` (NaN draws of the first 50) | seed: beta none on all; alpha draw 21 on Linux, none on the Mac and Windows; stream: none on all |
| `permutest-ls-boundary` | `pval_alpha` [0.125, 0.525, 0.3] / [0.121951, 0.512195, 0.268293] / [0.15, 0.55, 0.3] (R: [0.106383, 0.489362, 0.255319]); `QSp` 0.675 / 0.634146 / 0.675 (R: 0.638298); the undefined permutations differ per platform; the boundary permutations are the same on all three; a permutation's largest absolute alpha reaches 521.1 on the Mac (draw 40) and 443.2 on Windows (draw 14) |
| `ls-identity-maximum` | metafopy's point and `ll` -7.13929 equal on all three; the reference ends at metafopy's point on Linux and Windows and at [0.202511, -0.0368202] (`ll` -7.89187, **below metafopy's**) on the Mac, where starts 0 and 1 report "Positive directional derivative" |
| `selmodel-print-halfnorm` | `delta` 0.000137796 (at its bound) on all three; `se` 0.104977 / 0.108272 / 0.103257; `ci_lb` 0.0113676 / 0.00490912 / 0.01474; `vd` 1.11914 / 1.21194 / 1.07183 |

What the five environments show together (the development Mac and the arm64 container are in the section above):

- The reference's returned fit moves with the platform in `ls-bcg-reference` (ML, Windows), `ls-identity-maximum` (Mac) and `selmodel-decreasing` (Mac and
  Windows), and in each of them it lies below metafopy's log-likelihood: reference defect candidates, not metafopy's.
- `mv-reml-score` is the outlier on CI's Linux only: the score is 1e-06 relative there against 1e-11 to 1e-09 elsewhere.
- `selmodel-print-halfnorm`: `se` spreads over 0.1033 to 0.1083 across the CI platforms with `delta` at its bound.

<!-- doc-counts: accept-end -->

## Verdicts: the location-scale reference fits (Task 5 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Groups `ls-bcg-reference`, `ls-identity-alpha` and `ls-identity-maximum`, judged on the figures of the five environments recorded above (the probe runs of
2026-11-16).

**Cause, confirmed as a reference defect in all three.** In every environment metafopy's log-likelihood equals or exceeds the reference's wherever the
reference returns (ML -7.53929 and REML -8.08728 for the log link; ML -7.13929 and REML -7.98877 for the identity link), and metafopy's score is
1e-09 to 8e-09 on the log link. Nothing beats metafopy's point, so there is no operator stop and `src/metafopy` is unchanged. What moves with the platform
is the reference:

- `ls-bcg-reference`: which BFGS starts report "precision loss" differs per platform and the old code dropped them; on CI Windows the kept ML start was a
  stationary-looking stop at `ll` -9.47361, below metafopy's -7.53929, and on CI Linux no REML start counted as converged.
- `ls-identity-alpha`: the REML likelihood has two optima ([0.339136, -0.0061661], `ll` -7.98877, and [0.0133747, 0.00162732], `ll` -8.08078); the old
  code kept the lowest `-fun` among the starts SLSQP called successful, and on CI Linux that was the lower optimum.
- `ls-identity-maximum`: on the CI Mac the starts that reached metafopy's point reported "Positive directional derivative" and were dropped, leaving the
  stop at [0.202511, -0.0368202] (`ll` -7.89187).

**Verdict 2 (make the reference fit robust)** for the three groups, in `ls_reference_fit` and `ls_starts` of `src/tests/correctness/formulas.py`:

- five starts (`LS_START_FACTORS`) instead of three; the optimiser's message is no longer used to drop a start;
- log link: every start is polished by `ls_newton_polish` (Newton on the analytic score and information, a step accepted only if the log-likelihood does not
  fall) and kept when its largest absolute score is at most `LS_SCORE_TOLERANCE` (1e-6); the search stops once two kept starts agree to `LS_AGREEMENT`
  (1e-9), which keeps the cost of the many permutation refits below the old cost;
- identity link: the score does not vanish on the boundary `Z alpha = 0` (742 and 98 in the free directions at the optimum, above), so every feasible finite
  end point is kept and the highest log-likelihood wins;
- the best point is polished by Nelder-Mead as before; `RuntimeError` when no start is kept.

Tests (`src/tests/correctness/test_formulas_ls.py`): a BFGS start reported as "precision loss" is still accepted when its score is small (ML and REML);
a worse optimum that the optimiser calls successful is not chosen over the better one (identity REML, `ablat` design). Each was run against the old code
on 2026-11-17 and failed (`RuntimeError: no start converged` twice; `ll` -8.08078 against -7.98877).

Runs of `test_derived.py`, `test_rma_ls.py` and `test_formulas_ls.py` with `-k "ls or identity or bcg"` on 2026-11-17 with the new reference: development
Mac 480 passed, 1 xfailed; arm64 container 478 passed, 2 failed (below). The CI platforms are judged in Task 11. A first version that ran every start for every
permutation refit made `ls.permutest.location.exact.ML.z.bcglat` fail in the container (11 of 1287 permutations without a reference fit); the log-link search
now stops once two kept starts agree, and the case passes.

Found while running the group tests in the arm64 container, not caused by this change and outside these three groups: `glmm.cor.um.rs.ml.bcg` (the
`glmm-um-rs-score` group, Task 7) and `test_identity_fit_bd_is_a_stationary_point` (`src/tests/test_rma_ls.py`, which asserts metafopy's own score below
1e-6 and got 4.0e-06; it also failed in the arm64 container run of Task 2 and is not in CI's Linux list).

<!-- doc-counts: accept-end -->

## Verdict: the decreasing selection model's reference maximum (Task 6 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Group `selmodel-decreasing` (`selmodel.decreasing.bangertdrowns2004`), judged on the figures of the five environments recorded above.

**Cause, confirmed as a reference defect.** metafopy's point has `ll` -17.4217 on all three CI platforms. The reference's single SLSQP run converges to it on
Linux and runs away on the Mac (beta -2248.16, `ll` -378.22) and Windows (beta -3059.28, `ll` -393.029); along the ray from metafopy's point the log-likelihood
falls everywhere. No end point beats metafopy's, so there is no operator stop and `src/metafopy` is unchanged.

Measured on 2026-11-17 in the arm64 container, the single-start runs of the new reference: from metafopy's start (`tau2` 0.048) the maximum, `ll` -17.421673;
from `tau2` scaled by 0.1 an end point at `ll` -368.34 and by 10 at -28.26, both called "Optimization terminated successfully". So the optimiser's message and
a second `tau2` scale are not enough; starts that move the selection parameters inside the decreasing constraints (0.8, 0.5, 0.2 powers) reach the maximum
there.

**Verdict 2 (make the reference fit robust)**, in `selmodel_maximum` and `selmodel_best_start` of `src/tests/correctness/test_derived.py`:

- several starts (metafopy's, `tau2` scaled by `SELMODEL_START_SCALES`, and for a decreasing model the selection parameters at `SELMODEL_SHRINKS`); the
  message is not used, a run counts when finite, off the bounds, not below its start and not at the iteration limit; the highest log-likelihood wins;
- `beta` and `tau2` bounded far beyond any maximum (`SELMODEL_BETA_SPAN`, `SELMODEL_TAU2_SPAN`), and trial steps clipped at the `delta` floor.

Test `test_selmodel_maximum_ignores_a_run_that_leaves_the_coefficient_bounds` makes the first run a reported success at the runaway point; it failed against
the old code (`ll` -427.64 against -17.42) and passes. `pytest src/tests/correctness/test_derived.py -k selmodel` passes on the development Mac and in the
arm64 container (`bin/linux-tests.sh`). The CI platforms are judged in Task 11.

<!-- doc-counts: accept-end -->

## Verdicts: the GLMM UM.RS score and the CM.EL tight-optimum check (Task 7 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Groups `glmm-um-rs-score` and `cm-el-tight-se`, judged on the figures of the five environments recorded above. `src/metafopy` is unchanged.

**`glmm-um-rs-score`: verdict 3.** The log-likelihood at the reference optimum is equal to 1e-9 on all five environments, so the optimum is the same point. The
score `fit_problem` checks (Richardson at steps 5e-5 and 1e-4) is 7.5e-07 to 1.9e-06 (pneumonia) and 1.1e-06 to 1.7e-06 (bcg) across the five environments,
and the score at fixed steps from 2e-5 to 4e-4 spans 4.7e-07 to 1.1e-05: the score at the optimum is the finite-difference noise floor of the quadrature
likelihood, the same size as its spread over step sizes. `GRADIENT_TOLERANCE` in `src/tests/correctness/glmm_formulas.py` goes from 1e-6 to 2e-5 (10 times
the largest Richardson score, 1.9e-06). That is below `GLMM_GRADIENT_TOLERANCE` 1e-4 in `test_derived.py`, which still checks the score again, and far below
the effect the fits detect (a mis-specified likelihood moves the score by order 1).

**`cm-el-tight-se`: verdict 3.** The Nelder-Mead end points agree between environments only in the eighth decimal (collins1985a `alpha` 0.174406193 on
the Mac, 0.174406202 in the arm64 container, 0.174406198 on CI Linux), and the standard error follows them. Largest relative deviation of the standard error
at the tight optimum from the specification's figure: development Mac 6.5e-09, arm64 container 1.2e-08, CI Linux 2.2e-08, CI Mac 6.5e-09, CI Windows
1.1e-08. `rel` in `test_ml_se_matches_spec_measurement_at_tight_optimum` (`src/tests/test_nchg.py`) goes from 1e-8 to 2.5e-7 (10 times the largest, 2.2e-08,
rounded up); the same test's check at the default estimate uses 1e-5, so the tight-optimum check still detects an information error well below that.

Tests: `pytest src/tests/test_nchg.py src/tests/correctness/test_derived.py -k "glmm or nchg"` passes on the development Mac; in the arm64 container
(`bin/linux-tests.sh`) it passes except `glmm.fit.or.um.rs.fe.nielweise2007.beta` (beta[0] -1.22065947 against -1.22065944, tolerance 1e-8), an arm64-only
failure already recorded for Task 2's run, absent from CI's Linux list and from the group table, so not this task's. The CI platforms are judged in Task 11.

<!-- doc-counts: accept-end -->

## The rma_mv REML score: figures so far (Task 8 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Group `mv-reml-score` (`property.mv.reml-score.kon`, absolute `tolerance` 1e-6), measured on 2026-11-17 with `bin/platform-probe.py`, which now also records
each `scipy.optimize.minimize` call's message, success flag, iteration count and largest absolute gradient in metafopy's own parameterisation.

| Environment | BFGS message | Largest absolute gradient | Score of `sigma2` | Scale `tr(PZ)` | Score relative to scale |
|---|---|---|---|---|---|
| Development Mac | "precision loss", `success` false, 11 iterations | 1.3e-10 | [1.4e-09, -8.3e-09] | [104.475, 962.612] | 1.3e-11, -8.6e-12 |
| arm64 container | the same | 8.6e-11 | [1.8e-09, -5.3e-09] | the same | 1.7e-11, -5.5e-12 |
| amd64 container (emulated) | the same | 1.3e-10 | [1.8e-09, -8.1e-09] | the same | 1.7e-11, -8.4e-12 |
| CI Linux, CI Mac, CI Windows | not recorded by the earlier probe | not recorded | see the CI table above: Linux [5.1e-05, 1.1e-03] | the same | Linux 4.9e-07, 1.1e-06 |

The BFGS never reports success here: `mv_opt.maximise` accepts the stop because the gradient is under `SLACK_GRADIENT` (1e-4), so "reports
convergence" holds through that path only. Three local environments end at a gradient of 1e-10; the emulated amd64 container does not reproduce CI Linux's
score of 1e-3. If CI Linux's log-sigma2 gradient is about `sigma2 * score / 2` it is near 2e-5 (derived, not measured): between the three local figures and
the slack, so the cause (an earlier BFGS stop on Linux's libm and BLAS, accepted by the slack) is not confirmed.

### The CI probe run of commit `f57ab45` (probe run 44002318304, read on 2026-11-17)

| Environment | BFGS message | Iterations | Largest absolute gradient | Score of `sigma2` | Score relative to scale |
|---|---|---|---|---|---|
| CI Linux | "precision loss", `success` false | 8 | 6.0e-07 | [1.3e-05, 3.7e-05] | 1.3e-07, 3.8e-08 |
| CI Mac | the same | 11 | 1.3e-10 | [1.4e-09, -8.3e-09] | 1.3e-11, -8.6e-12 |
| CI Windows | the same | 8 | 1.8e-05 | [5.1e-05, 1.1e-03] | 4.9e-07, 1.1e-06 |

The log-likelihood agrees across the three to 1e-12 (-7.2404667768). The earlier guess (Linux's gradient near 2e-5) was wrong: Linux's gradient is
6.0e-07 and it is Windows that stops at 1.8e-05. In the earlier probe run the same two platforms gave scores [5.1e-05, 1.1e-03] on Linux and
[3.0e-07, -3.0e-08] on Windows, so the stop varies from run to run on one platform, not only between platforms.

**Verdict: blocked (operator stop).** Every platform's optimiser ends on "precision loss" with `success` false, and `mv_opt.maximise` accepts the stop only
because the gradient is under `SLACK_GRADIENT` (1e-4). CI Linux and CI Windows end 4 to 7 orders of magnitude further from the optimum than the Mac
(relative score up to 1.1e-06 against 1e-11), which is a tolerance-sized stop, not a rounding-size one. The task's condition for verdict 3 (convergence
reported, score a rounding-size fraction of the scale on every platform) is not met, so metafopy is off a theoretical value on two platforms and the
spec sends this to the operator. The options for the operator: (a) set the case's `tolerance` relative to the scale (about 2e-06 of 962.6, i.e. 2e-03)
as verdict 3 anyway; (b) tighten the optimiser's stop in `src/metafopy` (changes the library, outside this list's scope); (c) assert the measured range
as verdict 4.

**Answered 2026-11-17 (operator): option (c), range now, and file the fix.** `property.mv.reml-score.kon` asserts the REML score of each `sigma2`
within 2e-06 of its scale `tr(P Z Z')` (the largest measured, 1.1e-06 on CI Windows, rounded up), verdict 4; the ML case stays absolute. The fix
(tightening the stop in `src/metafopy`) is filed in `TODO.md`; `src/metafopy` is unchanged here.

<!-- doc-counts: accept-end -->

## Verdicts: the stepcon interval and the halfnorm print (Task 9 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Judged on the figures of the five environments recorded above. `src/metafopy` is unchanged.

**`stepcon-rau-mods`: verdict 3.** `ci_lb[0]` is -0.00179123 (CI Linux), -0.001791 (CI Mac), -0.00179168 (CI Windows), -0.001791281 (development Mac) and
-0.001790637 (arm64 container) against R's -0.0017894: the largest deviation is 2.4e-06, on a standard error of 0.068 (3.5e-05 of it). The other compared
quantities stay inside their relative tolerances (`se[0]` 0.0681664 to 0.0681667 against R's 0.0681675; `delta[1]` 0.637544 to 0.637555 against 0.637582).
The old tolerance (5e-4 relative of 0.00179, plus 1e-6 absolute) allowed 1.9e-06. `LOOSE_ATOL` in `src/tests/test_stepcon.py` gives that case an absolute
tolerance of 2.5e-05 (10 times the largest deviation, rounded up) on the Wald quantities; it is 3.7e-04 of the standard error, so the test still detects
an interval error. `pytest src/tests/test_stepcon.py -k inference` passes on the development Mac.

**`selmodel-print-halfnorm`: operator stop.** `delta` sits at its bound (0.000137796) on all environments and the Hessian moves with the platform: `se`
0.104977 (CI Linux), 0.108272 (CI Mac), 0.103257 (CI Windows), 0.105032 (development Mac), 0.105720 (arm64 container) against R's 0.104459 (a spread of
0.005, 5 per cent of the standard error); `ci_lb` 0.0113676, 0.00490912, 0.01474, 0.011260, 0.009911 against R's 0.0123831. The largest deviation is 0.0075
(CI Mac). The printed `ci.lb` is compared at the larger of `NOISY_HESSIAN` 0.25 relative (0.0031) and `NOISY_ATOL` 5e-3, so 0.0075 fails. Verdict 3's margin
(10 times the largest deviation) gives an absolute tolerance of 0.075, which is 72 per cent of the standard error and six times R's `ci.lb`: the spec
refuses a tolerance larger than the effect the test exists to detect. Options for the operator: (a) assert the observed range, `NOISY_ATOL` 0.01 (1.3 times
the largest deviation, against the spec's 10 times); (b) `NOISY_ATOL` 0.075 as the margin rule gives; (c) skip the case's print comparison on `darwin`
and `win32`. Not changed.

<!-- doc-counts: accept-end -->

## Verdicts: the ridge stops of the vif simulation and the permutation boundary fits (Task 10 of `tasks-1.md`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->

Figures are from the probe sections above (development Mac, arm64 container, CI Linux, CI Mac, CI Windows). Verdict 4 (platform-dependent, asserted as a
range) for both groups: the only difference between environments is the count of boundary or NaN refits along a ridge, the stop the operator accepted
on 2026-11-04.

- `ls-vif-sim`: NaN draws of the alpha part's first 50 are 0 or 1 in every environment (seed 42: 1 on the development Mac, arm64 container and CI Linux, 0
  on the CI Mac and Windows; stream: 1 on the development Mac, 0 elsewhere); the beta part has none. The test asserts at most one NaN draw and keeps its
  comparison of the other draws with R.
- `permutest-ls-boundary`: `pval_alpha` and `QSp` are counts over 40 (5/40, 21/40, 12/40; 6/40 on Windows), 41 (CI Mac) or 44 (development Mac)
  permutations with a defined standard error, against 47 in R. The test asserts each is a count over 40 to 47 permutations (replacing the 0.1 and
  `win32` 0.25 relative tolerances) and still compares `pval`, `QMp` and the interior `alpha_perm` and `zval_alpha_perm` with R.
- `git diff main -- src/metafopy` is empty.

<!-- doc-counts: accept-end -->
