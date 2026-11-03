# R fused arithmetic

**Version:** 1.0
**Status:** implemented
**Date:** 2026-11-03

metafopy reproduces R's `dbinom()` to the last bit, because `gosh()` draws
the sizes of its random subsets with `sample(prob = dbinom(...))` and R's
sampler sorts the probabilities by a heapsort that is not stable: a
different last bit reorders sizes whose probabilities are equal, and every
subset drawn after it differs. Which last bit R gives depends on how R was
compiled: clang on arm64 fuses `a*b + c` into one fused multiply-add (one
rounding), while R on x86_64 rounds twice. This spec makes metafopy's port
follow the platform it runs on, tests both variants everywhere, and adds R's
platform and its arithmetic to the audit's R oracle pin.

## Motivation

From the `TODO.md` item "gosh gives different subsets and results from R on
every fit" (found 2026-11-02 by the whole audit
`.audit/runs/20261102-223017`; every `gosh` fit differs from R in `$incl`
and every result). Traced on 2026-11-03:

- Exact subsets (all subsets fitted) already match R. Random subsets do not:
  after `set.seed(1234)`, `gosh(rma(yi, vi, data = dat), subsets = 20)` on
  bcg (`measure = "RR"`) gives other subsets, sizes and estimates.
- `sample(2:13, 8, TRUE, prob = dbinom(2:13, 13, 0.5))` gives R
  `6 5 5 5 9 5 6 7`, metafopy `6 8 8 8 9 8 6 7`; sampling without weights
  matches.
- `dbinom(c(4, 10, 12), 13, 0.5)`: R 4.6.2 gives `0x1.657fffffffffdp-4`,
  `0x1.1dffffffffffep-5`, `0x1.9fffffffffff4p-10`; metafopy gives
  `0x1.657fffffffffap-4`, `0x1.1e00000000000p-5`, `0x1.a000000000001p-10`.
- The pinned R is Homebrew R 4.6.2, `aarch64-apple-darwin27.0.0`, built by
  clang with `CFLAGS = -g -O2`, which contracts `a*b + c` within one
  expression. R's `src/nmath/dbinom.c`, `bd0.c` and `stirlerr.c` are the same
  in the R 4.5 and R 4.6 branches except for a comment.
- With `math.fma` at the three places below, Python gives R's bits for those
  three values, and the `gosh` call above gives R's `incl`, `k` and
  estimates exactly. The fused `pow1p` forms give R's bits for
  `dbinom(0, 2:4, 0.3)` and `dbinom(2:4, 2:4, 0.7)`; the unfused ones differ.
- The existing fixture `src/tests/reference/phase12/audit3.json`
  (`gosh.dbinom_k`, made 2026-10-15 with R 4.5.1 on Linux) holds the
  unfused bits (`0x1.1ep-5` for x = 10 of 13). Its tests, and the seeded
  `gosh` tests built on the same R, are skipped off Linux by
  `refutil.linux_only`, whose reason ("the last bit of R's dbinom() depends
  on the C library") names the wrong cause.

Decisions taken with the operator on 2026-11-03:

1. **Follow the platform**: fused arithmetic on arm64 (`arm64`,
   `aarch64`), unfused elsewhere; tests check both variants on every
   platform.
2. **The R oracle pin records R's platform and arithmetic**; `check-r`
   refuses a mismatch, and `--allow-r-drift` overrides it, as for package
   versions ([R oracle pin](../r-oracle-pin/spec.md)).

Amended 2026-11-03, after Task 1 of `agents/tasks/r-fused-arithmetic/tasks-1.md`
stopped: the last bit also depends on the C library. With correctly rounded
`log`, `exp` and `log1p`, the unfused port still misses 7 of the 844 values
of `audit3.json`, and moving one `log1p`, `log` or `exp` result by 1 ulp
reproduces each of them: glibc's functions are not correctly rounded, and
Apple's libm differs from them at a few arguments. So `refutil.linux_only`'s
old reason was half right. Decided with the operator the same day: each
fixture is compared to the last bit on the C library that made it (glibc on
Linux for `audit3.json`, Apple's libm on macOS for `dbinom_fused.json`) and
with a relative tolerance of 1e-12 elsewhere (one ulp of a `log` grows to
about 500 ulp of the result at k = 400); the seeded `gosh` tests, which
depend on the exact order of tied weights, run only on their fixture's
platform. This replaces "tests check both variants on every platform" in
decision 1.

## Module location

- `src/metafopy/core/dbinom.py` - `FUSED`, `fused` arguments, the fused forms
- `src/metafopy/methods/gosh.py`, `src/metafopy/core/rng.py` - docstrings
- `src/tests/refutil.py`, `src/tests/test_diagnostics.py`,
  `src/tests/test_audit3_misc.py` - tests of both variants
- `src/tests/r/generate_dbinom_fused.R` (new),
  `src/tests/reference/phase12/dbinom_fused.json` (new) - the fused fixture
- `src/audit/r/common.R`, `src/audit/r/versions.R`, `src/audit/r/setup.R`,
  `src/audit/steps.py`, `src/audit/data/r-versions.json`,
  `src/audit/README.md` - the pin
- `src/tests/test_harness_compare.py` - tests of the pin

## API

```python
# metafopy/core/dbinom.py

FUSED: bool  # platform_fuses() at import; tests patch it


def platform_fuses() -> bool:
    """Whether R built for this machine fuses a*b+c into one rounding: True on arm64/aarch64 (platform.machine()), else False."""


def bd0(x: float, np: float, fused: bool) -> float: ...
def pow1p(x: float, y: float, fused: bool) -> float: ...
def dbinom_raw(x: int, n: int, p: float, q: float, fused: bool) -> float: ...


def dbinom(x: numpy.typing.ArrayLike, n: int, p: float, fused: bool | None = None) -> FloatArray:
    """R's dbinom(x, n, p) for integer x and n, to the last bit of R built with (fused=True) or without (fused=False) fused
    multiply-add; None uses FUSED."""
```

```python
# audit/steps.py: check_r also compares the pin's R_fma with metafopy.core.dbinom.FUSED
```

## Implementation

### The fused forms

clang's default `-ffp-contract=on` fuses only within one C expression. In
the code `dbinom()` reaches for integer arguments, that is:

| Function | Unfused (today) | Fused |
|---|---|---|
| `bd0`, `x > np` branch | `x * (lg - 1.0) + np` | `math.fma(x, lg - 1.0, np)` |
| `bd0`, `x <= np` branch | `x * lg + np - x` | `math.fma(x, lg, np) - x` |
| `dbinom_raw` | `exp(lc - 0.5 * lf)` | `exp(math.fma(-0.5, lf, lc))` |
| `pow1p`, `y == 2` | `x * (x + 2.0) + 1.0` | `math.fma(x, x + 2.0, 1.0)` |
| `pow1p`, `y == 3` | `x * (x * (x + 3.0) + 3.0) + 1.0` | `math.fma(x, math.fma(x, x + 3.0, 3.0), 1.0)` |
| `pow1p`, `y == 4` | `x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0` | `math.fma(x, math.fma(x, math.fma(x, x + 4.0, 6.0), 4.0), 1.0)` |

The Taylor branch of `bd0`, `stirlerr` (table and divisions only) and `lf`
(sums only) have no product added to anything and are unchanged. `fused` is
passed down from `dbinom`; `FUSED = platform_fuses()` is read once at import.
`gosh` keeps calling `dbinom(sizes, k, 0.5)` and gains no argument.

### Fixtures and tests

- `generate_dbinom_fused.R` writes, from the pinned R,
  `dbinom(0:k, k, p)` as `%a` for k in 2, 5, 8, 13, 16, 24, 28, 37, 90,
  210, 400 and p in 0.5, 0.3, 0.7 (0.3 and 0.7 reach `pow1p` through
  `x == 0` and `x == n`), and one seeded `gosh` reference: `set.seed(99)`,
  `gosh(r13, subsets = 100)` in the shape of `REF['gosh_random_global']`;
  plus `R.version.string`, `R.version$platform` and the fusion probe below.
- `test_dbinom_bits_match_r` runs everywhere with `fused=False` against
  `audit3.json`, and a new test with `fused=True` against
  `dbinom_fused.json`. A helper `refutil.assert_bits(got, expected, exact)`
  compares to the last bit when `exact` is true and with `rtol=1e-12`
  otherwise; `exact` is `sys.platform.startswith('linux')` for the unfused
  fixture and `sys.platform == 'darwin'` for the fused one.
- The seeded `gosh` tests patch `metafopy.core.dbinom.FUSED`: `False` for
  the Linux fixtures (`test_gosh_seeded`,
  `test_gosh_random_subsets_global_seed`), marked `refutil.linux_only`
  with the reason "glibc's log, log1p and exp set the last bit of R's
  dbinom()"; `True` for the new fused `gosh` reference, marked by a new
  `refutil.macos_only` ("Apple's libm sets the last bit of R's dbinom()").
- `platform_fuses` is tested by patching `platform.machine`.

### The pin

`common.R` gains `r_identity()`, returning `R` (version), `R_platform`
(`R.version$platform`) and `R_fma`: `"fused"` when
`sprintf("%a", dbinom(10, 13, 0.5)) == "0x1.1dffffffffffep-5"`, `"unfused"`
when it is `"0x1.1ep-5"`, else `"unknown"`. `versions.R` and `setup.R` write
these entries in place of `vers$R`; underscores keep them apart from package
names, which cannot contain one. `setup.R` drops all three from the packages
to install. `library_drift` compares them as it compares packages, so
`check_r` refuses a different platform or arithmetic with no new code there.
`check_r` also refuses when the pin's `R_fma` disagrees with
`metafopy.core.dbinom.FUSED` (R and Python built for different machines,
such as Python under Rosetta), naming both; `--allow-r-drift` turns that
into a note too. `data/r-versions.json` gains `R_platform =
"aarch64-apple-darwin27.0.0"` and `R_fma = "fused"`.

### Documentation

The `gosh` docstring's `seed` entry and `set_seed`'s docstring say the
random subsets match R built for the same kind of machine (fused on arm64,
not on x86_64). `src/audit/README.md` names the two new pin entries.

### Rerun and the rest of the item

Run `bin/metafor-audit.sh --only gosh` and compare its causes with the
item's. Each cause that remains (the item also lists `$res$factor.alloc.*`
columns where Python gives none, `profile` bounds where R gives none, and
`vif` and `dfbetas` lines) gets a dated note on the item, which then stays
open for them; if none remain, the item moves to `DONE.md`.

## Error handling

- `check_r`: a pin without `R_platform` or `R_fma` is drift (the entry is
  missing on one side); a different value, or `R_fma` against Python's
  `FUSED`, raises `RuntimeError` naming both values and the three ways on
  (reinstall, re-pin, `--allow-r-drift`).
- `r_identity()` gives `R_fma = "unknown"` for a value neither variant
  produces; `check_r` refuses it unless allowed.
- `dbinom` keeps its `InputError` for invalid arguments.

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `metafopy.core.dbinom.FUSED` | `bool` | `platform_fuses()` | Which R arithmetic `dbinom` reproduces; internal, patched by tests |
| `R_platform`, `R_fma` in `src/audit/data/r-versions.json` | `str` | the 2026-11-03 R | The pinned R build |

## Out of scope

- `norm_rand` uses SciPy's `ndtri`, not a port of R's `qnorm`: after
  `set.seed(1234)`, 1347 of R's first 2000 `rnorm` draws differ from it by
  about one unit in the last place (2026-11-03). Nothing sorts or branches
  on those bits, and the audit compares draws with a tolerance.
- The ports of `ptukey`/`qtukey` (`core/tukey.py`) and `dnorm`
  (`transf.py`) give values compared with a tolerance.

## External dependencies

None new; `math.fma` is in the standard library from Python 3.13.
