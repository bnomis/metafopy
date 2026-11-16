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
