# The metafor audit harness

One command compares metafopy with R metafor everywhere, so that an audit finds everything in one go and two runs on the same code find the same things:

```
source bin/activate.sh
bin/metafor-audit.sh                  # every step, ends by writing wrong-or-missing.md
bin/metafor-audit.sh --recall         # does the harness reproduce what the earlier audits found by hand?
bin/metafor-audit.sh --list           # the steps
bin/metafor-audit.sh --rerender .audit/runs/<run>   # the report of an earlier run again, with today's data/known.toml (seconds, no R)
bin/metafor-audit.sh --only 'options/vcov/*'  --steps options    # a slice, for working on one thing
```

The harness lives in `src/audit` and is not part of the shipped package (`pyproject.toml` builds `src/metafopy` only). The report goes to
`.audit/runs/<timestamp>/wrong-or-missing.md` (one line per cause, the most widespread causes of each kind), `wrong-or-missing-full.md` (every cause) and the raw findings in `findings.json`. A full run is slow, so run it in the background; R shards that stop answering are killed after twice the case timeout and the case is reported as a timeout.

## Why each audit used to find different things

An audit was a hand-written sample of an unbounded space: which functions to call, with which arguments, on which models, and what to compare.
Each round drew a new sample, the probe scripts were thrown away, and nothing measured how much of the space a round covered or how many of the
real differences it found. This is the capture-recapture problem: two independent samples that overlap partly say how many differences there are
altogether ([Petersson et al. 2004](https://doi.org/10.1016/S0164-1212(03)00090-6)), and rounds that each found things the others missed say
that the samples were far from the whole. `hettest()` stayed unnoticed for four rounds because the list of functions came from memory.

The harness removes the sampling.

1. **The universe comes from R, not from memory.** `r/universe.R` dumps every exported function and S3 method of metafor with its formals, the choices of
   `match.arg`, `is.element`, `%in%` and `switch` arguments, the arguments read from `...` (`ddd$x`, `.chkdots`), the settings read with
   `getOption` (`na.action`), the fit classes each function accepts (`.chkclass`), the datasets and the fit class of every fixture. A new metafor
   function or argument appears in that dump and nowhere else is needed.
2. **Cases are generated, never written.** For every fixture (a fitted model or another result of metafor) and every generic with a method for its
   class, one case; for every swept fixture, one case per value of every argument; for every fitting function, the same on its own arguments.
3. **Both languages run the same case.** A case is a JSON call spec (`src/audit/r/common.R` documents the value tags). R and Python both run it,
   and the two value trees are compared, not their printouts alone
   ([differential testing](https://doi.org/10.1145/1295014.1295038), with R metafor as the oracle).
4. **Nothing is silently skipped.** An argument without values, or a function without a recipe, is reported under *Not covered*, and
   `src/tests/test_harness_accounts.py` fails on it. What is accepted as different is written down with its reason in `data/known.toml`, and an entry
   that matches nothing in a full run is reported as stale.
5. **The harness is checked against the audits.** `--recall` runs it on the commits the earlier audits used and reports which of their items it
   reproduced (`data/recall.toml`). An item it misses is a hole in the harness, fixed by widening the domains, never by a note in a report.

## The steps

| Step | What it does |
|---|---|
| `check-r` | Checks the pinned oracle (metafor 5.2-1, R library recorded in `data/r-versions.json` when present). |
| `universe` | Enumerates metafor from R; reports when `data/universe.json` (the committed snapshot) is out of date. |
| `matrix` | Every generic on every fit class and on every other result class (escalc frames, `list.rma`, `anova.rma`, `matreg`, ...), plus `print`. |
| `options` | One argument at a time, every value, on every swept fixture; the settings read with `getOption`. |
| `construct` | The fixtures themselves (values and warnings), then one argument at a time on every fitting function. |
| `standalone` | Functions that take no model (`transf.*`, `conv.*`, `fmt*`, `vcalc`, ...). |
| `datasets` | Every dataset of metafor and metadat loads in Python and equals R's. |
| `scripts` | Every help-page example and every `tests/testthat` file of metafor, run in R expression by expression and translated to Python (`transpile.py`, `rlib.py`). Units that cannot be translated are counted, not dropped; a variable whose assignment did not run in Python is removed, so that later units are counted as unsupported and not compared with its old value. `options(na.action=)` is passed as `na_action` to the functions that take it. `x$a <- v` has the value `v`, as in R. |
| `api` | Formals of every function and method against the Python callable for the same class. |

## Reading a finding

`kind` says what differs: `missing-function`, `missing-arg`, `missing-field` (R's result has an element metafopy lacks), `crash`, `timeout`,
`refused-where-R-works`, `accepts-where-R-refuses`, `value-diff`, `shape-diff`, `type-diff`, `label-diff` (names, row and column names such as the coefficient labels), `print-diff`, `warning-missing`, `warning-extra`.
Numbers agree within `RTOL = 1e-6` (`ATOL = 1e-8`); up to `DRIFT_RTOL = 1e-4` a difference is counted as drift and not listed (optimiser stopping
rules). Integer vectors that differ by exactly one are indices (R counts from 1). Printed output is compared token by token, tolerating the last
digit, but the number of decimals must be the same (R prints `0` where Python printed `0.0000`). Printed lines held in a character vector (`capture.output(print(x))`) are compared without their trailing spaces.

## What is in `data/`

| File | Content |
|---|---|
| `fixtures.json` | The models and results the matrix runs on. Add a fixture and every generic that has a method for its class runs on it. |
| `domains.toml` | Values for arguments by name (`[args]`), per function (`[fns.NAME.args]`), arguments that need another one set with them (`[companions.ARG]`), settings (`[options]`), the graphics arguments passed through `...`, and the arguments that are not options with the reason (`[skip]`). |
| `recipes.toml` | How the functions that take no model are called, base arguments of generics, calls that must run first (`pre`, also per fit), per-fixture overrides (`by_fit`), several calls of one function on other inputs (`[standalone.LABEL]` with `fn =`, `when` for arguments that change with the swept one), and the functions that are waived with a reason (`[unprobed]`). |
| `known.toml` | Accepted differences: a glob on the case id, kind and path, with the reason. |
| `recall.toml` | The items of the earlier audits and the finding that reproduces each. |
| `universe.json` | The snapshot of the R universe; refresh with `--update-universe`. |

## Setting up R

`Rscript src/audit/r/setup.R <library> <versions.json>` installs metafor 5.2-1, metadat 1.6-0 and gsl 2.1-8 at their pinned versions and the other
packages (lme4, clubSandwich, emmeans, ...) at the newest version R accepts, and records what it installed. Point `METAFOPY_AUDIT_RLIB` at the
library. The source of metafor's tests and examples is downloaded from CRAN and checked against `data/metafor-source.sha256`.

## What the harness cannot see

Graphics: a plot is compared by what its function returns (limits, positions, values), not by the text or pixels drawn. Items of that kind are in
`data/recall.toml` with `reach = "none"` and the reason, and they count as missed in the totals.
