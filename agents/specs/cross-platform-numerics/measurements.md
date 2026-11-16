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
