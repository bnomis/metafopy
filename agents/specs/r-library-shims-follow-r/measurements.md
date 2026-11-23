# Measurements: R-library shims follow R

<!-- doc-counts: accept-start measured on 2026-11-23 at 3324124 - a dated record, not a claim about the present; never regenerate -->
Measured on 2026-11-23 at `3324124`, 05:42-05:50 UTC, with the pinned R 4.6.2 and metafor 5.2-1 (`src/audit/data/r-versions.json`,
library `.audit/Rlib`) on the metafor source the scripts step reads (`.audit/source/metafor`, version 5.2-1 in its `DESCRIPTION`). It is a
dated record: never edit it.

## How each column was measured

- **R and shim**: the snippet's R value in `src/tests/reference/audit/snippets.json`, and the shim's value or error from
  `auditsnippets.run_in_python` on the same snippet (the strict `xfail` cases of `DISAGREE` in `src/tests/test_harness_rlib.py`).
- **Root cause**: the line of `src/audit/rlib.py` at `3324124`.
- **Callers**: `grep -rn '\b<shim>\b' src/audit src/tests bin`. Every shim is reached only through the `BASE` table (`rlib.py:922`), which
  `rcall` (`rlib.py:1466`) consults for an R function name before metafopy; `r_class` is also called by `r_inherits` (`rlib.py:847`), and
  `recycled` by the `+` and `-` operators and by `null_safe`, which wraps `* / ^ %% %/% == != < > <= >=` (`rlib.py:1260-1299`).
  `audit.rlib` is imported under `src/audit` only by `scripts.py` and `transpile.py`, so the shims serve the scripts step alone; no other step
  (`matrix`, `options`, `construct`, ...) and no recall item outside `scripts/*` passes through them.
- **Static reach**: [reach-static.R](reach-static.R) walks the parse tree of every script the scripts step runs (the `\examples` of each Rd
  page through `tools::Rd2ex`, as `src/audit/r/runscripts.R` lists them, and each `tests/testthat/test_*.R`): 191 scripts. It counts the
  call sites of the R function (all forms), then those in the case's form where the form is visible in the text (named `cbind` arguments,
  `dimnames =`, `seq(to =, length.out =)` without `from`, a `list()` mixing named and unnamed arguments).
- **Dynamic reach**: [reach-dynamic.R](reach-dynamic.R) runs the same 191 scripts in R as `runscripts.R` runs them (same stubs, seed and
  120 s limit per unit), with a wrapper for each R function in the stubs environment that records each script-level call in the case's form
  (`class()` of an object without a class attribute, `length(NULL)`, `is.na()` of a data frame, an operator on two operands longer than one
  and of different lengths, ...). The wrappers were checked on a script that makes every form once: each form was recorded.
  metafor's own code does not see them, so the count is of the calls the transpiled Python makes through the shims. All 191 scripts ran to
  the end.
- **Python reach**: for each dynamic hit, whether the Python side reaches the shim's failing branch, checked with `audit.rlib.rcall`.

## Cases

| Case | Shim | R vs shim | Root cause | Reach | Risk if reached | Recommendation |
|---|---|---|---|---|---|---|
| `as_dataframe_matrix` | `r_as_dataframe` | `as.data.frame(matrix(1:4, 2))`: R columns `V1`, `V2`; shim columns `0`, `1` | `rlib.py:554` `pandas.DataFrame(x)` names columns by position and ignores a `Named` matrix's names | 27 call sites in 11 scripts; 0 units pass a matrix (all pass data frames or results) | `missing-field` findings `$V1`, `$V2` (invented) | fix to R: small, and the wrong names invent findings |
| `as_numeric_text` | `r_as_numeric` | `as.numeric('3.5')`: R `3.5`; shim raises `TypeError: iteration over a 0-d array` | `rlib.py:495` `as_array` makes a str a 0-d array, which line 496 iterates | 12 sites in 6 scripts; 4 units pass one string (`as.numeric(R.Version()$major)` and `$minor` in `test/analysis_example_viechtbauer2007a`), 4 pass longer character vectors; Python: 0, since `R.Version()` raises `UnsupportedError` first | a `crash` finding blamed on metafopy (invented), and the unit's targets stale | fix to R: a one-line change; reached as soon as `R.Version` gets a shim |
| `cbind_named` | `r_cbind` | `cbind(a = 1:2, b = 3:4)`: R column names `a`, `b`; shim none | `rlib.py:374` `numpy.column_stack` drops the names `entries` holds (only the data-frame and deparsed path of line 372 keeps them) | 66 sites in 19 scripts, 0 with a named argument; 1 unit in R (`do.call(cbind, unclass(tmp)[1:4])` in `test/analysis_example_lipsey2001`); Python: 0, `do.call` raises `UnsupportedError` | `label-diff` (invented); `x[, "a"]` on the result fails | fix to R: small |
| `class_numeric` | `r_class` | `class(1)`: R `numeric`; shim `float` | `rlib.py:842` the Python type name stands in for R's class | 4 sites in 2 scripts (one is `class(conf2) <- ...`, a replacement); 3 units in R, all on data frames (`rd/aggregate.escalc`); 0 on numbers | `value-diff` (invented); `if (class(x) == ...)` takes the other branch | fix to R, with `class_matrix` and `inherits_number` (same cause) |
| `class_matrix` | `r_class` | `class(matrix(1:4, 2))`: R `matrix`, `array`; shim `ndarray` | `rlib.py:842`, as above | as above: 0 on a matrix | `shape-diff` (invented) | fix to R |
| `inherits_number` | `r_inherits` | `inherits(1, 'numeric')`: R `TRUE`; shim `False` | `rlib.py:848` compares the Python type name; a fit (`RMAResult`) never inherits `rma` either | 0 sites, 0 units | wrong branch of an `if`, both ways | fix to R, through the fixed `r_class` |
| `isna_data_frame` | `r_isna` | `is.na(data.frame(a = c(1, NA), b = c(NA, 2)))`: R a logical matrix with column names `a`, `b`; shim a plain array without names (the TODO item says "a frame with other column names"; measured, it is an unnamed array) | `rlib.py:292` `x.isna().to_numpy()` drops the column names | 66 sites in 10 scripts; 0 units on a data frame | `label-diff` (invented) | fix to R: small, as `r_as_matrix` already builds the names |
| `length_null` | `r_length` | `length(NULL)`: R `0`; shim `1` | `rlib.py:239` `numpy.asarray(None).size` is 1 | 48 sites in 8 scripts; 0 units on `NULL` | `value-diff` (invented); `if (length(x))` takes the other branch, which can hide a later difference | fix to R: a one-line change |
| `list_mixed` | `r_list` | `list(1, b = 'x')`: R an unnamed first element and `b`; shim a dict keyed `'1'` and `'b'` | `rlib.py:181` names positional elements `'1'`, `'2'`, ...: a dict cannot hold an unnamed element | 211 sites in 49 scripts, 0 mixing named and unnamed; 0 units | `missing-field` `[0]` (invented); `x[[1]]` looks up the key `1` | narrow the contract: no script reaches it, and R's mixed list needs a new list type that every indexing shim reads (decision D1) |
| `matrix_dimnames` | `r_matrix` | `matrix(1:4, 2, dimnames = list(...))`: R row and column names; shim none | `rlib.py:330` `**kw` takes `dimnames` and line 341 returns a plain array | 34 sites in 7 scripts, 0 with `dimnames`; 0 units | `label-diff` (invented) | fix to R: small, `Named` holds the names |
| `recycled_operands` | `recycled` | `c(1, 2, 3, 4) + c(10, 20)`: R `11 22 13 24`; numpy raises `operands could not be broadcast` | `rlib.py:1260` only a `pandas.Series` is turned into an array; two arrays of different lengths are left to numpy | 3130 sites of `+ - * / ^` in the scripts; 0 units with two operands longer than one and of different lengths (any of the operators `recycled` serves), 0 with a matrix and such a vector | a `crash` (invented) and stale targets that skip later units (hiding differences); a square matrix times a vector of its side is silently row-wise in numpy where R is column-wise (checked: `matrix(1:4, 2) * c(10, 20)` is `10 40 30 80` in R, `10 20 60 80` through the shim) | fix to R, including the matrix case, since numpy gives a wrong value without an error there |
| `seq_length_out_with_to` | `r_seq` | `seq(to = 10, length.out = 4)`: R `7 8 9 10`; shim `1 4 7 10` | `rlib.py:210` `from` defaults to 1 when `to` and `length.out` are given; line 215 spreads from 1 to `to` | 38 sites in 12 scripts, 0 with `to` and `length.out` without `from`; 0 units | silent `value-diff` (invented) | fix to R: small |
| `tail_list` | `r_tail` | `tail(list(a = 1, b = 2, c = 3), 2)`: R elements `b`, `c`; shim raises `IndexError` | `rlib.py:600` `numpy.asarray(dict)` is a 0-d array; `r_head` has the same defect at line 593 (checked: `r_head({'a': 1, 'b': 2}, 1)` raises the same `IndexError`) | 3 sites in 2 scripts; 1 unit on a list (`tail(rstandard(res), 4)` in `test/misc_list_rma`, a `list.rma`), which Python takes through the dataclass branch of line 598, not this one; 0 on a plain list | `crash` (invented) | fix to R, `r_head` with it |

## Prototype

Each fix proposed in the spec was tried in memory on 2026-11-23, patched over `audit.rlib` from a scratch script (no file under `src`
changed), against the fixture: twelve of the thirteen cases then match R and every `MATCHING` snippet still does. `list_mixed` stays
different, as a dict cannot hold it.

## Recall

`.audit/runs/recall-20261123-061000/recall.md` (the last run): 176 of 176 items of audits 4 to 6 reproduced, `a6-glmm-um-rs` listed apart as
oracle moved. The recall items that the scripts step satisfies (`scripts/misc_rma_ls*`, `scripts/*zeros*`, `scripts/rma.mh/*`,
`scripts/labbe/*`, `scripts/matreg/*`) are matched in that run by findings on metafopy calls (`rma`, `profile`, `rma.mh`, `labbe`,
`matreg`, `rma.mv`), none of which passes through a changed branch; `a6-plot-selmodel-after-qq` (`scripts/selmodel/*` with kind `crash`, or
`matrix/plot/sel_bcg`) is matched by the `matrix/plot/sel_bcg` crash, which the shims do not reach. With no script reaching any case, a fix should
leave the recall result unchanged; the recall run is what confirms it, since the measurement is of R's types and Python's can differ.
<!-- doc-counts: accept-end -->
