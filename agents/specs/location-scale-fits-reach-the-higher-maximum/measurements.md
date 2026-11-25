# Location-scale fits reach the higher maximum: measurements

Dated figures for the [spec](spec.md). Each task adds a section; none rewrites an earlier one.

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
## Task 1: baselines on the unchanged code

Measured on 2026-11-25 at commit `23d7f89`, macOS (development machine). Case: `ls.permutest.location.exact.ML.z.bcglat` in
`src/tests/correctness/cases/location_scale.toml` (`dat.bcg` log risk ratios, `lat` in both parts, ML, log link, `test = "z"`). The case still has
`status = "mismatch"` and names the `TODO.md` item "Location-scale fits can stop at a lower local maximum than the test reference".

### The eight permutations with a higher interior maximum

Rows of `X` permuted in `correctness.formulas.distinct_label_orders` order, indices from 0. `ll` is the log-likelihood, `z` the location z of `lat`.
metafopy is `rma_uni` on the permuted `lat`; the reference is `formulas.ls_reference_fit` with `z = beta / sqrt(diag(vb))`.

| Permutation | metafopy `ll` | metafopy `z` | reference `ll` | reference `z` |
|---|---|---|---|---|
| 68 | -12.47880001 | 0.311137 | -12.58248229 | 0.899234 |
| 264 | -12.64299284 | -0.127192 | -12.25193541 | -1.962194 |
| 433 | -12.18550364 | -0.279416 | -11.96789514 | 1.902418 |
| 651 | -12.59845763 | 0.275161 | -12.62803581 | 0.426883 |
| 805 | -12.60271821 | -0.139755 | -12.62596814 | -0.794745 |
| 979 | -12.63705281 | -0.171199 | -11.72866962 | -3.194397 |
| 1110 | -12.61397039 | 0.284891 | -12.61556080 | 0.577271 |
| 1278 | -12.63122400 | -0.149547 | -12.17150179 | 1.770378 |

### The exact permutest

- metafopy: 1287 permutations; hits for `lat` 308 and for the intercept 794 (`pval` 0.23931624 for `lat` and 0.61693862 for the intercept).
- Reference (`formulas.ls_exact_permutation_p`, location part): hit bounds for the intercept 782 to 850 and for `lat` 239 to 307; permutations
  without a reference fit: 68 of 1287.

### Times

- Median of 20 `rma_uni` fits of the bcglat model: 7.6 ms.
- One exact `permutest` (1287 refits): 8.5 s.
- The reference's exact enumeration (1287 refits): 6.6 s.

### The scoped audit

`bin/metafor-audit.sh --only '*/ls_*'` at `23d7f89`, with `METAFOPY_AUDIT_RLIB` set to the main checkout's pinned R library
(`/Users/simonb/src/metafopy/.audit/Rlib`, because the worktree's own `.audit/Rlib` does not exist). Run directory (ignored by git):
`.audit/runs/20261125-213640/` in this worktree. Result: 0 findings make 0 causes, so 0 unaccepted differences; 14655 findings matched
`data/known.toml` and 9894 numeric differences below the reporting threshold were not listed. The run took about 18 minutes, nearly all in the
options step.
<!-- doc-counts: accept-end -->

<!-- doc-counts: accept-start measured on 2026-11-25 - a dated record, not a claim about the present -->
## Task 2: the test reference after the restarts

Measured on 2026-11-25 at the Task 2 working tree (on top of `69c0d1d`), macOS (development machine).

- `formulas.ls_reference_fit` now reaches the highest regular interior maximum on all eight permutations. Log-likelihoods, found independently as
  the best of 40 random starts per permutation (ridge end points with `|alpha| > 10` dropped): 68 `-12.57323737`, 264 `-11.64011395`,
  433 `-11.96789514`, 651 `-12.60144281`, 805 `-12.13242026`, 979 `-11.67846163`, 1110 `-12.58432080`, 1278 `-12.17150179`. Before the change the
  reference stopped low on 68, 264, 651, 805, 979 and 1110 (and was already right on 433 and 1278).
- The reference's exact bounds for `ls.permutest.location.exact.ML.z.bcglat` (`ls_exact_permutation_p`, location part): `lat` hits 238 to 306
  of 1287 (`pval` 0.18492618 to 0.23776224), intercept 781 to 849 (`pval` 0.60683761 to 0.65967366); permutations without a fit 68. Task 1 had
  239 to 307 and 782 to 850, also 68.
- metafopy's 308 `lat` hits are still above the upper bound 306, so the case stays a strict xfail until Task 4.
- Every other location-scale correctness case keeps its verdict (`pytest src/tests/correctness/test_derived.py -k "ls."`: all pass but two xfails,
  the bcglat case and `ls.profile.ML.identity.bcg`).
<!-- doc-counts: accept-end -->
