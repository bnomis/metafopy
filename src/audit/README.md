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

## What audit 7 could not see (audit 8)

The second run of the harness, on the same code plus the fixes of audit 7, found different things again. The cause was in the harness, not in the sampling:
a comparison that reports nothing can be wrong in two ways, a difference that is hidden and a difference that is invented, and audit 7 had both.

| Hole in the harness | Effect |
|---|---|
| `scripts`: a nested assignment (`dat$yi[2] <- NA`) was translated wrongly, a unit that failed or could not be translated left its variables stale, `options(na.action=)` did nothing, a negative `subset` was shifted | later units compared different data in R and Python: invented differences, and real ones hidden behind them |
| `pycall`/`apidiff`: `summary`, `cbind`, `rbind`, `aggregate`, `as.data.frame`, `as.matrix`, `head`, `tail` and `coef` of a summary resolved to no Python callable | reported as missing functions, so their results were never compared |
| `universe.R`: print methods counted as drawing functions; a loop index (`bldiag(i)`) counted as an argument | a graphics sweep over print methods, an argument that does not exist |
| `valuetree`: R's `digits` is a vector of 9, scalars were not broadcast, dicts were not read as named vectors, scale fields were labelled with the location coefficient names, data frame row and column names were not compared | 6523 `$digits` cases, wrong `label-diff`, label differences of data frames not seen |
| `runcases.R`: `update.rma` leaves `na.action` at `na.pass` for the later cases of a shard, fixtures were unseeded | `R length 13, Python 11` and `$statistic.boot` differences that only existed in some shards |
| index arguments (`include`, `label`, `alpha`, `tau2`, `out`, `grpvars`, `regplot(mod=)`, `forest(rows=)`) not shifted to 0-based | the wrong coefficient profiled, wrong rows plotted, in both directions |
| `known.toml`: globs such as `*/ls_*` or `Python has none` that matched more than the reason says | real differences hidden (the recall check found four items of audit 6 swallowed this way) |

Rules that follow, each checked by a test or by `--recall`:

- A shortcut that makes both sides print the same (an R-layout printer in the harness) hides the gap in the library; the harness prints the object the way the library does.
- A `known.toml` entry names the fixture and the path it is about. `path` can be a list, `match` can be a list. A reason that says what R does has been reproduced in R.
- A step that cannot translate a unit forgets the variables it would have defined (`stale` in `scripts.py`), it does not carry on with stale state.
- Cases are deterministic: options are reset before each case and fixtures are seeded, so two runs of the same code give the same findings.
- `--recall` is run after every change of the harness, not only after a change of the library: it caught the swallowed items above.

## The steps

| Step | What it does |
|---|---|
| `check-r` | Checks the pinned oracle (metafor 5.2-1, R library recorded in `data/r-versions.json` when present). |
| `universe` | Enumerates metafor from R; reports when `data/universe.json` (the committed snapshot) is out of date. |
| `matrix` | Every generic on every fit class and on every other result class (escalc frames, `list.rma`, `anova.rma`, `matreg`, ...), plus `print`. |
| `options` | One argument at a time, every value, on every swept fixture; the settings read with `getOption`. |
| `construct` | The fixtures themselves (values and warnings), then one argument at a time on every fitting function. |
| `emmeans` | The emmeans package on the reference grid of a metafor fit (`emmprep()`, `emmeans()`, `contrast()`, `pairs()`, `test()`, `confint()`, `joint_tests()`, `by` groups, `at`, `weights`): the printed summaries and the values, in R and in Python. The calls are in the `[emmeans.LABEL]` tables of `data/recipes.toml`. |
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
| `known.toml` | Accepted differences: a glob on the case id, kind and path, with the reason. A `timeout` entry of a case near the time limit says `load_dependent = true` and is not reported as stale when the case finished. |
| `recall.toml` | The items of the earlier audits and the finding that reproduces each. |
| `universe.json` | The snapshot of the R universe; refresh with `--update-universe`. |

## Setting up R

`Rscript src/audit/r/setup.R <library> <versions.json>` installs metafor 5.2-1, metadat 1.6-0 and gsl 2.1-8 at their pinned versions and the other
packages (lme4, clubSandwich, emmeans, ...) at the newest version R accepts, and records what it installed. Point `METAFOPY_AUDIT_RLIB` at the
library. The source of metafor's tests and examples is downloaded from CRAN and checked against `data/metafor-source.sha256`.

## What the harness cannot see

Graphics: a plot is compared by what its function returns (limits, positions, values) and by the text it draws (titles, axis labels, tick labels, texts,
legends; `drawn-text-diff`), not by the pixels. `runcases.R` and `runscripts.R` note the strings given to R's `text`, `mtext`, `axis`, `title`, `plot` and
`legend` (`r/drawn.R`); `pyrun.drawn_text` reads them off the matplotlib figures. Plotmath and mathtext formulas are left out of the comparison, except for
`plain(...)` and the p-value ranges of the funnel legend, which are compared as the text they draw. Colours, line types, positions and the order of drawing
are not compared; items of that kind are in `data/recall.toml` with `reach = "none"` and the reason, and they count as missed in the totals.

A function the fixtures call through the formula, such as the user's own function in `mods = ~ ablat + audit_userfn(ablat)` (fixture `uni_userfn`; the
function is defined in `r/common.R` and in `pycall.py`), is found in the caller's environment on both sides.

### Harness regressions found by the final recall of audit 8

- R printed bytes that are not UTF-8 when the shell had no locale; the outcome file is now read with `errors='replace'`, and the audit script should be run with `LC_ALL=C.UTF-8`.
- `cline` had been dropped from the graphics arguments swept for every plot function, so `profile(cline=)` was no longer tried (`a5-graphics-profile` was missed). It is now swept for `profile` only (`[fns.profile] graphics` in `data/domains.toml`).
- The `print(x, digits = <vector>)` shim for results that print through `str()` takes the first element only for a single number now; a vector is reported, and the one case that remains (`VIFResult`) is a known entry (`a6-digits-printouts` is reproduced again).

### A time limit that metafor catches (audit 10)

R's `setTimeLimit()` raises its error wherever R is when the limit passes. When that is inside a `try()` of metafor (`permutest()` wraps every refit in one and stores `NA` for a refit that fails), the case carries on and returns a value with one step marked as failed: audits 8 to 10 saw a single extra `NA` in `permutest(exact = TRUE)` at a different position each time. `runcases.R` and `runscripts.R` now report a case or unit that ran past the limit as an R error, and `pyrun.classify` turns it into a `timeout` finding.

### R calls slower than the case limit (audit 11)

An R shard that stops answering is killed and its case is reported as `timeout` (it used to be filed as metafopy accepting what R refuses), in the
*Timeouts* section of the report. Whether a slow case passes the limit depends on the machine and on the other shards, so its ledger entry carries
`load_dependent = true`, records how long the call takes in R on its own and what R returns outside the harness, and is not reported as stale in a run
where the case finished.
