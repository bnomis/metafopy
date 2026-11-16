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
