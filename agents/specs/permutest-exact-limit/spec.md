# permutest: refuse an exact test whose permutation matrix R cannot hold

**Version:** 1.1
**Status:** approved
**Date:** 2026-11-18

`permutest(fit, exact=True)` refuses a test only when its arrays of permutation statistics would pass `MAX_PERMUTATION_BYTES` (16 GiB). R's exact
test also builds the matrix of every permutation before its first refit, which the guard does not count, so a case such as `uni_mods` passes the
guard, then runs for days in metafopy, while R cannot hold it. This spec makes the guard count that matrix, so metafopy refuses what R's exact test
cannot hold on the same 16 GiB budget, and records the outcome in the audit ledger.

### Motivation (the `TODO.md` item, dates and figures as filed)

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
Measured on 2026-11-18 by Task 2 of `agents/tasks/audit-ledger-gaps/tasks-1.md` (metafopy `dd58553`; full audit run
`.audit/runs/20261118-042436`, case `options/permutest/uni_mods/exact=true`, kind `r-killed`, "R passed the memory limit of 4 GiB (killed)"): for
`rma(yi, vi, mods = ~ ablat, data = es_bcg)`, `permutest(fit, exact='i')` reports 389188800 iterations and 15567552000 bytes of arrays, just under
`MAX_PERMUTATION_BYTES` (17179869184) in `src/metafopy/methods/permutest.py`, so metafopy does not refuse. `permutest(fit, exact=True)` was still running
after about 29 minutes (cut off by the background limit) at about 275 MB resident at 20 minutes.
<!-- doc-counts: accept-end -->

Cause (from the item): the guard bounds the size of the arrays, not the run time, so a case R cannot hold in 4 GiB passes the guard and runs for a
very long time.

### What the investigation found (measured 2026-11-18 at metafopy `dcfb11c`, Apple M3 Pro with 36 GB, R 4.6.2, metafor 5.2-1 from `.audit/Rlib`)

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
- **When R stops with "Number of iterations requested too large.":** only when an allocation fails. `R/permutest.rma.uni.r` wraps each of
  `zval.perm`, `beta.perm` (`matrix(NA_real_, nrow=X.iter, ncol=x$p)`, or `rep(NA_real_, X.iter)` without moderators) and `QM.perm`
  (`rep(NA_real_, X.iter)`) in `try()` and stops with that message on a `try-error` (lines 155-168 and 298-311; `R/permutest.rma.ls.r` lines
  187-200, 332-345 and 554-567 the same). There is no rule on the count itself; the only other stop is "Too many iterations required for an exact
  permutation test." when the count is `Inf` (`NA` from overflow). An allocation fails when it would pass R's vector heap limit
  (`mem.maxVSize()`): on macOS the larger of 16 GB and the physical memory (R 3.5.0 NEWS, `R_MAX_VSIZE`; 36864 MB here), unlimited on Linux by
  default. With `R_MAX_VSIZE=4Gb`, R refuses `uni_mods` with that message in 0.047 s.
- **How R counts exact iterations:** `2^k` without moderators; otherwise the unique row permutations of the sorted model matrix,
  `round(prod((max(t)+1):k) / prod(factorial(t[-which.max(t)])))` for the run lengths `t` of identical rows (lines 49-86).
  `unique_perms_count` in `permutest.py` computes the same as an exact integer (`inf` past 1.7e308). For `uni_mods` the 13 `ablat` values run
  2,1,1,1,2,2,2,1,1, so 13!/2!^4 = 389188800, and `permutest(fit, exact='i')` returns `389188800` in both (R: `389188800`, a double; metafopy an
  `int`). The location-scale fit `ls_bcg` gives `[389188800.0, 25740.0]` in both.
- **Why `uni_mods` passes R's check and then dies:** the arrays take 389188800 x (2 x 2 + 1) x 8 = 15567552000 bytes (14.5 GiB), under the 36864 MB
  heap limit, so they are allocated; `NA_real_` fills them, so the resident size passes the harness's 4 GiB (`audit.config.MEMORY_LIMIT`, watched by
  `audit.engine.watch`) while the first matrix is written, and R is killed. Without the harness R would go on to `.genuperms(X.indices)`, which
  builds every permutation as a double matrix of `k` columns: 389188800 x 13 x 8 = 40475635200 bytes (37.7 GiB). Arrays and matrix together are
  56043187200 bytes, over the 36864 MB limit, so R would stop there with its vector-memory error after its recursion (5.4 us a row: 453600 rows of
  the first 10 labels took 2.44 s). Had it the memory, its refits (61.26 s for the 25740 of `uni_factor` in the audit run, 2.38 ms each) would take
  about 10.7 days. Without moderators R builds the same kind of matrix with `expand.grid` (`signmat`, `2^k` x `k` doubles).
- **metafopy** generates the permutations one at a time (`multiset_perms`, 0.40 us each) and allocates the arrays with `numpy.empty`, so memory
  grows only as rows are written. Its refit loop takes 693 us a permutation on `uni_mods` (first 2000 permutations), 75 hours for all of them, and
  the result would hold the 15.6 GB arrays.

| Audit case (`options/permutest/<case>/exact=true`) | Exact iterations | R in run `20261118-042436` | metafopy in that run | Current guard bytes | Proposed guard bytes |
|---|---|---|---|---|---|
| `uni_reml`, `uni_counts` | 8192 | ok | ok | 196608 | 1048576 |
| `uni_na` | 2048 | ok | ok | 49152 | 229376 |
| `trimfill_bcg` | 131072 | time limit | ok (35.7 s) | 3145728 | 20971520 |
| `uni_factor` | 25740 | ok | ok | 1441440 | 4118400 |
| `uni_mods_na` | 907200 | time limit | time limit | 36288000 | 108864000 |
| `ls_na` (location, scale) | 3628800, 907200 | time limit | time limit | 145152000 | 435456000 |
| `ls_bcg` (location, scale) | 389188800, 25740 | killed at 4 GiB | time limit | 15567552000 | **56043187200: refused** |
| `uni_mods` | 389188800 | killed at 4 GiB | time limit | 15567552000 | **56043187200: refused** |
| `uni_mods_year`, `ls_year` | 6227020800 | refused | refused | 348713164800 | 996323328000 |
| `ls_bangert` | 2.75e57 | refused | refused | refused | refused |

- **Tests:** every test file naming `permutest` (800 passed, 3 skipped, 85 s) was run with a scratch plugin recording each `permutation_arrays`
  call. The largest counts were 6227020800 (the two existing refusal tests, refused under both guards) and then 1000, so no test is newly refused.
- **The publications** (Follmann and Proschan 1999; Good 2009) define the exact test over every permutation and set no limit; the limit is a
  resource rule, and R's is the machine's memory.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/methods/permutest.py`: `permutation_arrays`, the `MAX_PERMUTATION_BYTES` comment, `run_permutations`, the `Raises` of
  `permutest_rma_uni`.
- `src/metafopy/methods/ls_permutest.py`: `beta_permutations`, `alpha_permutations`, the `Raises` of `permutest_rma_ls`.
- Tests and records: `src/tests/test_permutest_vif_plots.py`, `src/tests/test_permutest_ls_plots.py`, `src/audit/data/known.toml`,
  `docs/differences.md`.

## API

One keyword is added to an internal helper; no public signature changes.

```python
def permutation_arrays(n_iter: int, width: int, perm_cols: int = 0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Arrays for the test statistics, the coefficients (``n_iter`` x ``width`` each) and the omnibus statistic (``n_iter``) of each permutation.

    They are not filled, so memory is only taken as permutations are stored: the caller writes every row, NaN for a refit that fails.
    ``perm_cols`` is the number of columns of the matrix of every permutation that R's exact test builds before its first refit (``k`` for an
    exact test, 0 for random permutations); metafopy never builds it, but counts it so that it refuses what R cannot hold.

    Raises:
        metafopy.errors.InputError: the arrays and R's permutation matrix would take more than ``MAX_PERMUTATION_BYTES``
            (R: "Number of iterations requested too large.").
    """
```

## Error handling

The same `metafopy.errors.InputError('Number of iterations requested too large.')` as R and as now, raised by `permutation_arrays` when
`n_iter * (2 * width + 1 + perm_cols) * 8 > MAX_PERMUTATION_BYTES`, before any refit. `exact='i'` is unaffected: it returns the count before any
allocation, as R does. A user who meets the error is pointed by the docs entry to `exact=False` with a larger `iter`.

## Implementation

1. `permutation_arrays` gains `perm_cols` and uses the bound above. Rewrite the comment on `MAX_PERMUTATION_BYTES`: R's exact test holds three
   arrays of permutation statistics and the matrix of every permutation (`.genuperms`, or `expand.grid` of the signs without moderators, `k`
   doubles per permutation); how much R can hold depends on the machine (on macOS its vector heap stops at the larger of 16 GB and the physical
   memory, R 3.5.0 NEWS), so metafopy counts the same bytes against a fixed 16 GiB on every platform.
2. `run_permutations` passes `perm_cols=k if exact else 0`; `beta_permutations` and `alpha_permutations` in `ls_permutest.py` pass
   `perm_cols=k if exact else 0` too (R's location-scale test builds the same matrices: `signmat`, `.genuperms(X.indices)`,
   `.genuperms(Z.indices)`).
3. `Raises` of `permutest_rma_uni` and `permutest_rma_ls`: "an exact test needs more iterations than the largest float, or more memory than R's
   exact test can hold in 16 GiB (its arrays and its matrix of every permutation; R stops with an error as well)".

The 16 GiB value stays: it is R's smallest default heap on macOS, the platform the R fixtures come from. For a model with `k = 13` and two
coefficients the bound is 119304647 exact iterations; with no moderators and `k = 13` it is 8192 at most anyway.

## Decisions

### D1. Refuse `uni_mods`, or make it finish

**What is decided:** what `permutest(fit, exact=True)` does on an exact test like `uni_mods` (389188800 permutations), which R cannot hold.

**Background:** see the investigation. The audit cannot compare either way: R is killed at 4 GiB, and with more memory R stops in `.genuperms`, so no
R output exists. The rule since 2026-11-11 (`claude/rules/metafopy.md`) is that metafopy's output changes only after the user confirms it; this
review is that confirmation.

**What it affects:** `permutest.py`, `ls_permutest.py`, two test files, `known.toml`, `docs/differences.md`. No R parity fixture and no
correctness case changes under any option.

| Option | Outputs that change | What happens next |
|---|---|---|
| **(a) Count R's permutation matrix in the byte guard (recommended)** | `uni_mods` and `ls_bcg` with `exact=True`: `InputError` in microseconds, where both now run for days. Any exact test with `n_iter * (2 width + 1 + k) * 8` over 16 GiB (119304647 iterations for `k = 13`, two coefficients). No audit or test case besides those two. | The change above, two tests, the ledger entry, a docs entry. Runs up to the bound may still take a day (100 million refits at 693 us); R would take longer. |
| (b) Finish faster: batched refits | None refused. A measured prototype (scratch, 2026-11-18) refits a batch of permutations at once with REML Fisher scoring in numpy: the general form (no `k` x `k` matrices) takes 29.8 us a permutation at batches of 1000, 3.2 h for 389188800, plus 4.6 us (0.5 h) to generate and build the model matrices; a closed form for one moderator takes 3.6 us (0.39 h). Both agree with `refit_arrays` to 9.1e-14 in `zval` on 2000 random permutations. | A second implementation of the `rma_uni` refit (every method, `tau2` fixed, weights, `test`, failed refits, the location-scale model) to keep in step with the first. The result still holds 15.6 GB of arrays, over the audit's 4 GiB and its 120 s case limit, and R gives nothing to compare, so the audit case stays open. |
| (c) A fixed iteration cap, 10^7 | Every exact test over 10^7 iterations: `uni_mods`, `ls_bcg`, and tests R completes on most machines (for example 5e7 iterations at `k = 13` need 7.2 GB in R). | A rule R does not have, chosen for run time (10^7 refits at 693 us is about 2 h); its value is arbitrary and refuses work R does. |
| (d) Change nothing | None. | The case stays listed as an R kill, metafopy keeps running for 75 h on it, and the ledger can only say metafopy times out. |

**Recommendation: (a).** It is the root cause the item names: the guard already means "refuse what R cannot hold", but counts only part of what R
holds. It refuses exactly the two audit cases R cannot run and nothing R can, keeps R's message, needs no second refit path, and is reversible.

**Decided: (a)**, by the operator on 2026-11-18 on the review page of `agents/reviews/order-todo-items-1-5-specs.md` (no comment). Version 1.1
records the answer; the options (b), (c) and (d) stay above as the record of what was weighed.

## Choices made

- **The audit records the case in the ledger, not by giving R a heap limit.** The `TODO.md` item names the ledger entry. Setting `R_MAX_VSIZE` to
  `MEMORY_LIMIT` in `audit.rdriver.environment` would make R refuse too, so the case would compare error with error, but it would also turn an
  allocation failure inside R's `try()` of a refit (anywhere in the audit) into a failed refit instead of a kill, which needs its own full-audit
  check. Not proposed here.
- **16 GiB stays the budget** (see Implementation), and the error message stays R's.

## Tests and records

For option (a).

1. `src/tests/test_permutest_vif_plots.py`: `test_an_exact_test_whose_permutation_matrix_r_cannot_hold_is_refused`: on
   `rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg())`, `permutest(fit, exact='i') == 389188800` and `permutest(fit, exact=True)` raises
   `InputError` matching "Number of iterations requested too large". `test_permutation_arrays_count_r_permutation_matrix`: with
   `MAX_PERMUTATION_BYTES` monkeypatched to 10000, `permutation_arrays(100, 2)` returns (4000 bytes) and `permutation_arrays(100, 2, perm_cols=13)`
   raises (14400 bytes). Before the change the first test does not raise within `timeout 60` (check by hand); the second fails with `TypeError`.
2. `src/tests/test_permutest_ls_plots.py`: the same refusal for `rma_uni('yi', 'vi', mods='~ ablat', scale='~ alloc', data=refutil.bcg())`.
3. `src/audit/data/known.toml`: the `r-killed` entry of `options/permutest/ls_bcg/exact=true` gets `match = ["options/permutest/ls_bcg/exact=true",
   "options/permutest/uni_mods/exact=true"]` and a `reason` saying: both exact tests need 389188800 permutations; R allocates 14.5 GiB of arrays and
   passes the harness's 4 GiB while filling them, and would next need 37.7 GiB for its matrix of every permutation; metafopy refuses with R's
   "Number of iterations requested too large." (checked on the implementation date, R 4.6.2, metafor 5.2-1).
4. `docs/differences.md`, "Stricter than R": `### permutest(exact=TRUE) beyond what R can hold`, in the format of the `fsn()` entry: R side
   `permutest(rma(yi, vi, mods = ~ ablat, data = dat), exact = TRUE)` noting it is killed or stops for memory (and refuses at once under
   `R_MAX_VSIZE=4Gb`), Python side `# docs: skip` with the `InputError`; a `*Why:*` with the byte count and the date; and the advice to use
   `exact=False` with a larger `iter`.
5. Move the `TODO.md` item to `DONE.md` with the decision and its date.
6. Run `pytest src/tests/test_permutest_vif_plots.py src/tests/test_permutest_ls_plots.py src/tests/test_rma_ls.py src/tests/test_diagnostics.py
   src/tests/test_docs_examples.py`, then `ruff format`, `ruff check`, `mypy`; then `bin/metafor-audit.sh` and `bin/metafor-audit.sh --recall`,
   and check that the report no longer lists `options/permutest/uni_mods/exact=true` and that the changed entry is not stale.

## Acceptance

- `permutest(..., exact=True)` on `uni_mods` and `ls_bcg` raises the `InputError` above without running a refit; every other test passes unchanged.
- The audit report lists neither case and no stale entry; `--recall` passes.
- `ruff format --check`, `ruff check` and `mypy` pass.

## External dependencies

None new. R 4.6.2 with metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib` for the docs example and the audit.
