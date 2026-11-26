# Identity-link location-scale fits reach the higher maximum: measurements

Dated figures for the [spec](spec.md). Each task adds a section; none rewrites an earlier one.

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
## Task 1: baselines on the unchanged code

Measured on 2026-11-26 at commit `8cd2ba6`, macOS (development machine). Every fit is `rma_uni` with `mods = ~ lat`, `scale = ~ grp`
(the bcglat data of `ls.permutest.location.exact.ML.z.bcglat`, with the permuted `lat` as its own column and `grp` the original `lat`),
`method = "ML"`, `link = "identity"`, `test = "z"`. Permutation indices are 0-based in `correctness.formulas.distinct_label_orders` order.

| Permutation | metafopy ll | R default-start ll (spec) | `alpha` | location z (`lat`) | `tau2` of the `grp` = 0 / 1 groups (`ls_reference_fit`) | `ls_reference_fit` ll |
|---|---|---|---|---|---|---|
| 18 | -12.62494604 | -12.62494606 | 0.26763, 0.02893 | 0.2489 | -1.0e-10 / 0.86439 | -11.42937464 |
| 68 | -12.58248229 | -12.58248234 | 0.45933, -0.28186 | 0.8992 | 0.04772 / 0.59684 | -12.57323737 |
| 679 | -10.46818400 | -10.46818403 | 0.73960, -0.72466 | 3.8725 | 0.76851 / -1.0e-10 | -10.46391648 |
| 914 | -12.64295219 | -12.64295234 | 0.15902, 0.25299 | 0.3913 | 0.49193 / 0.13619 | -12.64031710 |

metafopy's log-likelihoods match R's to 1e-6 for all four. The `tau2` columns are those of `ls_reference_fit` (the independent
`scipy.optimize` fit), not of metafopy; the `alpha` and z columns are metafopy's. metafopy's own `tau2` (identity link: `alpha[0]` for the `grp` = 0 group, `alpha[0] + alpha[1]` for `grp` = 1): permutation 18,
0.26763 / 0.29656; 68, 0.45933 / 0.17747; 679, 0.73960 / 0.01494; 914, 0.15902 / 0.41200. Permutation 18's reference `tau2` of about -1e-10 is the
boundary `Z alpha = 0`.

Log-likelihood of every one of the 1287 permutation fits: JSON outside the repository, `/tmp/idlink-scratch/baseline-8cd2ba6.json` (key `lls`;
also the four rows above under `detail` and the timings). Range -12.6488 to -3.7836.

Time: median of 20 identity fits of permutation 18, 80.9 ms; one loop over the 1287 permutation fits, 47.5 s. (The spec measured 39.4 ms per
identity fit at `f0f6d44` on 2026-11-26; this machine was slower on this run, so later tasks compare against these figures, not that one.)

Scoped audit on the unchanged code: `bin/metafor-audit.sh --only '*/ls_*'` with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib`
(the worktree has no `.audit/Rlib` of its own, so the harness's `R_LIB` default does not resolve there), run directory
`.audit/runs/20261126-104210` in the worktree (ignored by git), 1078 s for the options step. Its report lists 0 causes: 0 unaccepted
differences. 14647 findings matched known differences and 9896 numeric differences below the reporting threshold were not listed. The run
covered the matrix, options and construct steps (280, 3076 and 848 cases), so the later comparison is of this report as a whole.
<!-- doc-counts: accept-end -->

<!-- doc-counts: accept-start measured on 2026-11-26 - a dated record, not a claim about the present -->
## Task 3: the restart in `fit_rma_ls`

Measured on 2026-11-26 on top of commit `28b10e7` (Task 3's change uncommitted), same machine and fits as Task 1.

| Permutation | ll before | ll after | `alpha` after | location z (`lat`) after |
|---|---|---|---|---|
| 18 | -12.62494604 | -11.42937466 | 0.0, 0.86439 (`tau2` 0 / 0.86439) | -3.5966 |
| 68 | -12.58248229 | -12.57323737 | 0.04772, 0.54912 | -1.2410 |
| 679 | -10.46818400 | -10.46391648 | 0.76851, -0.76851 | 4.8125 |
| 914 | -12.64295219 | -12.64031716 | 0.49166, -0.35534 | -1.1012 |

Over the 1287 permutation fits (JSON, outside the repository: `/tmp/idlink-scratch/task3.json`), 85 log-likelihoods changed by more than
1e-6 against Task 1's JSON, and none went down (the smallest change is 0.0).

Time: median of 20 identity fits of permutation 18, 202.6 ms (Task 1: 80.9 ms); one loop over the 1287 permutation fits, 131.5 s (Task 1: 47.5 s).

Tests: `pytest src/tests/test_rma_ls.py src/tests/test_ls_vif.py src/tests/test_audit8_crashes.py` 213 passed; `pytest
src/tests/correctness/test_derived.py -k "ls."` 198 passed, 1 xfailed (the xfail is the recorded 2026-11-20 difference). No result changed.
<!-- doc-counts: accept-end -->
