# R oracle pin

**Version:** 1.0
**Status:** implemented
**Date:** 2026-11-03

The audit harness compares metafopy with R metafor, so R and every package
it loads are the oracle. Only metafor, metadat and gsl are pinned today, and
the check that would catch a moved library never runs. This spec pins the
whole R library, makes `check-r` refuse a run on a different one unless told
otherwise, installs the pins on a new machine, and records recall items whose
difference the pinned R no longer shows. It then brings the ledger, the
universe and the recall corpus up to the pinned oracle.

## Motivation

From the `TODO.md` item "The audit harness misses one recall item and carries
stale ledger entries", found on 2026-11-02 and traced on 2026-11-03:

- `docs/differences.md` records outputs captured on 2026-10-21 with R 4.3.3,
  lme4 2.0.6 and emmeans 2.0.4. The runs of 2026-11-01
  (`.audit/runs/20261101-204233`) and 2026-11-02
  (`.audit/runs/20261102-223017`) used R 4.6.2 and lme4 2.1-0.
- `src/audit/r/setup.R` pins metafor 5.2-1, metadat 1.6-0 and gsl 2.1-9 and
  installs every other package at the newest version R accepts.
  `audit.steps.check_r` compares the library with
  `src/audit/data/r-versions.json` only when that file exists; it has never
  been committed, so nothing was compared.
- Recall item `a6-glmm-um-rs` (`construct/glmm_rs`, audit 6: se 0.18214
  where R gives 0.18147) is no longer reproduced: on 2026-11-02 R gave
  0.182140 and Python 0.182139, a drift of 7.5e-6.
- Ledger entries in `src/audit/data/known.toml` went stale with it:
  `*/glmm_rs*` value-diff, `options/selmodel/*` print-diff on `delta.1`,
  `matrix/dfbetas/mv_wide` (`element 17: R None`) and
  `matrix/rstudent/mv_wide` (`line 19`), whose cause now shows at element 53
  and line 3, and `construct/glmm_cm/*` refused-where-R-works. The
  `glmm_cm` Hessian warning now shows as `warning-extra`, where the ledger has
  `warning-missing`. The universe differs from `data/universe.json`.

The shared versions rule: record what shapes an output, check it wherever
saved work meets the current environment, refuse a mismatch naming both
sides and how to proceed, allow an explicit override that the output records,
and pin each such dependency.

Decisions taken with the operator on 2026-11-03:

1. **Pin the current library**: R 4.6.2 and every package as the 2026-11-02
   run recorded them, not the older lme4 2.0.6.
2. **`check-r` refuses a different library**, naming each difference and how
   to proceed; `--allow-r-drift` runs anyway and the report says so.
3. **A recall item whose difference the pinned R no longer shows is marked
   `oracle_moved`**, kept as a record and counted apart from reproduced and
   missed items.
4. **`setup.R` installs every package at its pinned version.**

## Module location

- `src/audit/data/r-versions.json` (new) - the pin
- `src/audit/steps.py` - `check_r`, `Context.allow_r_drift`
- `src/audit/cli.py` - `--allow-r-drift`
- `src/audit/r/setup.R` - install from the pin
- `src/audit/recall.py`, `src/audit/data/recall.toml` - `oracle_moved`
- `src/audit/data/known.toml`, `src/audit/data/universe.json` - brought up to
  the pin
- `src/audit/README.md`, `docs/differences.md` - the pinned versions
- `src/tests/test_harness_errors.py`, `src/tests/test_harness_accounts.py` -
  tests

## API

```python
# audit/steps.py


@dataclasses.dataclass
class Context:
    ...
    allow_r_drift: bool = False


def library_drift(lock: Json, installed: Json) -> dict[str, tuple[str | None, str | None]]:
    """Every package (and ``R``) whose installed version differs from the pin, as name -> (pinned, installed); a package missing on either
    side has None there."""


def check_r(ctx: Context) -> Outcome:
    """Check the installed R library against ``data/r-versions.json``. A missing pin file or any drift raises RuntimeError naming each
    difference and how to proceed, unless ``ctx.allow_r_drift``, when the drift becomes a note in the report."""


# audit/recall.py


@dataclasses.dataclass
class Item:
    ...
    oracle_moved: str  # '' when the pinned R still shows the item; otherwise the date, the R version and what R now gives
```

`bin/metafor-audit.sh --allow-r-drift` sets `Context.allow_r_drift`, and
`--recall` passes the flag on to the runs it starts.

## Implementation

### The pin

`src/audit/data/r-versions.json` is the `r-versions.json` of
`.audit/runs/20261102-223017`, as `versions.R` wrote it: every non-base
package in R's library paths and `R` itself. `check_r` compares the whole of
it with what `versions.R` writes for the current run (`library_drift`); the
metafor check stays, so a run with the wrong metafor still names it first.

A drift error lists each package as `name pinned -> installed`, then says how
to proceed: reinstall with `Rscript src/audit/r/setup.R <library>
<versions.json>`; re-pin by copying a run's `r-versions.json` over the pin
after reviewing what the new versions change; or run once with
`--allow-r-drift`. With the flag, the same list goes into the report's notes
as `R library differs from the pin (--allow-r-drift): ...`.

### Setup

`setup.R` reads the pin. For each package whose installed version differs
from it, it calls `remotes::install_version(p, version, upgrade = "never",
dependencies = FALSE)`, in the order of the pin file; metafor, metadat and gsl
stay first. A package that fails to install is printed under `NOT INSTALLED`
with its pinned version, and the following `check-r` names it. `R` itself is
not installed: a different R is reported by `check-r`.

### Recall items the oracle moved

`recall.toml` items take an optional `oracle_moved` string. `recall.write`
leaves such items out of reproduced and missed, and lists them in a section
`Oracle moved (n)` with the text, so the totals read `176 of 176 items
reproduced; 1 not shown by the pinned R`. `a6-glmm-um-rs` gets
`oracle_moved = "2026-11-02, R 4.6.2 and lme4 2.1-0: R gives se 0.182140,
Python 0.182139"`. The header comment of `recall.toml` describes the field.

### Bringing the data up to the pin

With the pin in place, run the whole audit, then:

- remove each stale ledger entry whose cause R no longer shows
  (`*/glmm_rs*` value-diff and its print-diff companion if it is stale too,
  `options/selmodel/*` `delta.1` print-diff, the `glmm_cm`
  refused-where-R-works entry), and move the `mv_wide` entries' `detail` to
  where the same cause now shows; turn the `glmm_cm` `warning-missing` entry
  into the `warning-extra` it now is, with its reason;
- review the universe difference and write it with `--update-universe`;
- run `--recall` and confirm every item is reproduced or `oracle_moved`;
- in `docs/differences.md`, state the pinned R and lme4 versions where it
  records the old ones, and change each entry whose ledger entry changed.

The 2026-11-02 run that filed `TODO.md` items 2 to 8 used the library being
pinned, so those items stand; the rerun confirms it, and any cause that
vanishes or appears is noted on its item, dated.

## Error handling

- No pin file: `RuntimeError("no data/r-versions.json: pin the R library
  first ...")` naming the re-pin step; `--allow-r-drift` does not skip it.
- Drift: `RuntimeError` listing every difference and the three ways to
  proceed, as above.
- `setup.R` failing to install a pinned version: listed, not fatal; the next
  `check-r` refuses the run.
- An `oracle_moved` item that the harness does reproduce again is reported
  in its section as `reproduced again`, so the field can be removed.

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `--allow-r-drift` | flag | off | Run on an R library that differs from the pin; the report records the differences |
| `src/audit/data/r-versions.json` | file | the 2026-11-02 library | The pinned R version and package versions |

## Tests

- `library_drift`: equal libraries give nothing; a changed, a missing and an
  extra package each appear with None on the missing side.
- `check_r` with `audit.rdriver.run` stubbed: no pin raises; drift raises
  naming the package and the three ways on; with `allow_r_drift` it returns a
  note instead.
- `recall.write`: an `oracle_moved` item is in neither count and is listed in
  its section; one that matches a finding is marked `reproduced again`.
- A real `bin/metafor-audit.sh` run and `--recall` run after the data is
  brought up to the pin, as above.

## External dependencies

None new. `remotes` (already installed by `setup.R`) installs the pinned
versions.
