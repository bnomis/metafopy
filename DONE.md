# Done

Finished work, newest last. Items arrive from `TODO.md` unchanged.

## The auditme checker reports unaccepted hits and no hook enforces it

The code does not pass the shared auditme checker, and nothing runs the checker
on commit, push or in CI, so hits accumulate unseen and each new checker rule
surfaces as a batch. The findings below are one piece of work: a pre-push hook
fails on any unaccepted hit, so the hooks can be adopted only once every hit is
fixed or accepted.

Fix direction: fix or accept every unaccepted hit, as the first finding below
describes (a hit the rule gets wrong is accepted with
`# auditme: accept[<rule>] <reason>` and raised in the auditme skill's
`requests.md`); then adopt the shared guards with `/project-setup` - the
auditme pre-push hook, the CI gate and its deploy key - so a new hit fails at
push. Re-run the checker with `python3 ~/.claude/skills/auditme/checks/audit.py .`
until it reports no unaccepted hit.

Spec: `agents/specs/auditme-clean/spec.md` (approved 2026-10-31); tasks: `agents/tasks/auditme-clean/tasks-1.md`.

### The auditme checker's 2026-10-31 rules report hits here

Found on 2026-10-31 by running the shared auditme checker at agents `108559d`
on this repository at `394c45fb`. Two checker changes shipped that day (agents
`c437b07`) report hits here:

- `mutable-module-level-exported` now counts the builtin constructor calls
  `list(...)`, `dict(...)` and `set(...)` as their literals:
  - `src/audit/rlib.py:46` `METAFOR_FUNCTIONS`, a `set(...)` read by one other
    file
- the new `raises-section-drift` rule (A4) compares each `Raises:` section with
  the domain errors reaching the caller: 100 hits in 81
  functions; all but two are names a section omits. Sampled hits are real:
  `fsn` (`src/metafopy/bias/fsn.py:159`) calls `check_class` with no `try` and
  omits its `ArgumentError`. List them with the re-run command below.

Cause: the checker gained the rules; the code predates them.

Fix direction: fix or accept each hit - wrap an exported constructor call in
`tuple(...)`, `frozenset(...)` or `types.MappingProxyType(...)`, and correct each
`Raises:` section from the body. A hit the rule gets wrong is accepted with
`# auditme: accept[<rule>] <reason>` on its `def` or assignment line, and the
pattern raised as a request in the auditme skill's `requests.md`.
Re-run with `pre-commit run auditme-all --hook-stage manual --all-files` once
the hooks are adopted, or `python3 ~/.claude/skills/auditme/checks/audit.py .`.

### Older checker rules also report unaccepted hits

Found on 2026-10-31 by `/todo-review`, running the checker at agents `108559d`
on this repository at `9d02a0a`. Besides the hits of the finding above, it
reported 615 unaccepted hits under 24 older rules, most of them
`literal-duplication` (281), `typing-deprecated-alias` (80),
`paired-sibling-helpers` (64), `non-alphabetical-literal` (42),
`non-alphabetical-dict-keys` (26) and `missing-raises-section` (25). They
include seven more `mutable-module-level-exported` hits on literals, such as
`STEPS` (`src/audit/steps.py:180`) and `SETTINGS`
(`src/metafopy/printing.py:99`), and seven `silent-broad-except` hits in
`src/audit/`. List them with the checker command in the Fix direction.

### No auditme hook or CI gate is installed

Found on 2026-10-31 by `/todo-review`. `.pre-commit-config.yaml` holds only
the ruff hooks: no auditme `pre-push` hook and no CI job runs the checker,
although the first finding above re-runs it "once the hooks are adopted" and
`agents/lessons.md` says to check auditme hits before opening a PR. The shared
git conventions ask for the project's whole-tree checks as a `pre-push` hook
and in CI.

Done: 2026-11-01 - agents/specs/auditme-clean/spec.md, agents/tasks/auditme-clean/tasks-1.md.

## Windows CI fails `test_import_metafopy_does_not_load_altair`

Found on 2026-11-01 in CI run 41299245958 on `c00e511`: lint, Ubuntu and
macOS passed, and Windows failed this one test with `assert 1 == 0`. It also
failed on Windows alone on `fe07de6`, so every push to `main` has shown red
since then. The log shows the child process dying in `import _overlapped`
with `OSError: [WinError 10106] The requested service provider could not be
loaded or initialized`.

Cause: the test in `src/tests/test_interactive.py` runs the child with
`env={'PYTHONPATH': ':'.join(sys.path)}`, which replaces the whole
environment. Without `SYSTEMROOT`, Winsock cannot start on Windows, so
asyncio's `_overlapped` fails to load. The `PYTHONPATH` it passes has no
effect anyway: `-I` makes the child ignore every `PYTHON*` variable, and it
finds metafopy through the virtual environment's install. The `':'` join
would also split Windows paths at their drive letters.

Fix direction: drop the `env` argument so the child inherits the parent's
environment, and keep `-I`. Check it on the Windows CI job.

Done: 2026-11-01 - `src/tests/test_interactive.py` runs the child without `env`, so it inherits `SYSTEMROOT` (`0687fd5`); every job of CI run 41309320319 on that commit passed, Windows included.

## The metafor audit cannot start R on macOS 27.2

Found on 2026-11-01 by the first run of `bin/metafor-audit.sh` on this machine
(macOS 27.2, R 4.6.2, run `.audit/runs/20261101-204233`): `check-r` and
`universe` passed, then `matrix` stopped with `subprocess.SubprocessError:
Exception occurred in preexec_fn` when it started its first R shard, so no
case was compared.

Cause: `audit.rdriver.start` passes `preexec_fn=limit_memory`, which calls
`audit.config.limit_memory` (`src/audit/config.py:27`), and that sets
`resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))` with
`MEMORY_LIMIT` at 4 GiB. On macOS 27.2 the call raises `ValueError: current
limit exceeds maximum limit` for every finite limit tried (4, 8 and 64 GiB,
and a finite soft limit with an unlimited hard one), checked in a plain
Python process the same day.

Fix direction: make the cap best-effort where the platform refuses it (catch
the `ValueError` on macOS, as Windows already skips it), or cap with a limit
macOS honours, so the shard still starts; keep the cap on Linux, where it
works. The R shards are already killed after twice the case timeout, so a
runaway shard is still stopped. Then rerun the audit item below.

Done: 2026-11-01 - `audit.config.limit_memory` returns when `setrlimit` raises `ValueError`; tests in `src/tests/test_harness_engine.py`. The rerun (`.audit/runs/20261101-211517`) started R and completed `matrix`, then stopped in `options` on a new defect, filed in `TODO.md`.

## The metafor audit's options step loses a Python worker on macOS

Found on 2026-11-01 by the second run of `bin/metafor-audit.sh` on this
machine (macOS 27.2, R 4.6.2, run `.audit/runs/20261101-211517`), the first
after the memory cap stopped failing: `check-r`, `universe` and `matrix`
(2503 cases, 3298 findings) completed, and every R shard of `options`
finished, but `options` then stopped with
`concurrent.futures.process.BrokenProcessPool: A process in the process pool
was terminated abruptly`. No Python shard wrote its `py-*.json`, and macOS
wrote no crash report for it.

Cause: unconfirmed. `audit.engine.run_cases` runs the Python side of each
shard in a `ProcessPoolExecutor` (`src/audit/engine.py:96`), and a worker that
dies takes the whole step with it. Suspected: a case that grows without
bound, now that macOS runs workers without the 4 GiB address-space cap (see
the macOS item in `DONE.md`), so the system kills the worker where Linux
would raise `MemoryError` inside it.

Fix direction: run the step alone (`bin/metafor-audit.sh --steps options`)
and watch the workers' memory to find the case; then decide between a memory
guard macOS honours (for example checking the resident size between cases)
and making a lost worker a reported finding for its shard rather than the end
of the step.

Done: 2026-11-02 - the cause was `permutest(exact=True)` on fits with moderators, which allocated arrays for every permutation (13! for `uni_mods_year`) and built them all as a list. `metafopy.methods.permutest.permutation_arrays` now refuses arrays over 16 GB with R's "Number of iterations requested too large.", and `multiset_perms` yields one permutation at a time; tests in `src/tests/test_permutest_vif_plots.py` and `src/tests/test_permutest_ls_plots.py`. The rerun `bin/metafor-audit.sh --steps options` (`.audit/runs/20261102-061046`) completed: 20029 cases, 16758 findings, and every exact permutest case has R's outcome. Hardening the harness against a lost worker is filed in `TODO.md`.

## A Python audit worker that dies stops the whole audit step

Found on 2026-11-02 while fixing the lost worker in `options` (see `DONE.md`):
`audit.engine.run_cases` runs the Python side of the shards in a
`ProcessPoolExecutor`, so one worker killed by the system (out of memory, a
crash in compiled code) raises `BrokenProcessPool` and ends the step, and no
shard writes its results. On macOS 27 the Python workers run without a memory
cap, because `setrlimit(RLIMIT_AS)` is refused there, so any case that grows
without bound kills its worker instead of raising `MemoryError`. The case
that did so (`permutest(exact=True)`) is fixed; nothing stops the next one.

Fix direction: watch each Python worker's resident size between cases, or
from a thread, and record a case that passes a limit as an error; and make
`run_cases` record a dead worker's current case as a finding for its shard,
then carry on with the rest of that shard, as the R side already does
(`r-N-attempt1.jsonl`).

Spec: agents/specs/audit-shard-survival/spec.md; tasks: agents/tasks/audit-shard-survival/tasks-1.md.

Done: 2026-11-02 - agents/specs/audit-shard-survival/spec.md, agents/tasks/audit-shard-survival/tasks-1.md.

## The audit's scripts step stops when one of its processes dies

Found on 2026-11-02 while writing `agents/specs/audit-shard-survival/spec.md`,
whose review chose to keep this step out of that spec:
`audit.scripts.run` runs the Python side of the scripts step in a
`concurrent.futures.ProcessPoolExecutor`, so one worker killed by the system
raises `BrokenProcessPool` and ends the step. Its R processes are started
once each with no restart, and a shard whose `r-N.json` is missing raises
`RuntimeError`. Neither side has a memory watch on macOS.

Cause: the scripts shards write no per-unit start lines (`runscripts.R`), so
the case-level restart of the cases path cannot be applied to them.

Fix direction: after the shard survival spec ships, make `runscripts.R` and
`audit.scripts.run_shard` write a start line before each unit, and run both
sides through that spec's `audit.engine.run_shard` and `watch`.

Note 2026-11-02: the shard survival spec shipped (merged at `eadb479`);
`audit.engine.run_shard` and `watch` are on `main`, so this item can start.
`audit.scripts` still runs its Python side in a `ProcessPoolExecutor`.

Spec: agents/specs/scripts-shard-survival/spec.md; tasks: agents/tasks/scripts-shard-survival/tasks-1.md.

Done: 2026-11-02 - agents/specs/scripts-shard-survival/spec.md, agents/tasks/scripts-shard-survival/tasks-1.md.

## Remove the auditme accept markers once the shared rules stop the false positives

Found on 2026-11-01 while clearing the auditme hits
(`agents/specs/auditme-clean/spec.md`): some hits were the rules getting the
code wrong, not the code. Each site is accepted with a marker and listed by
rule in `agents/specs/auditme-clean/requests.md`.

Cause: the shared auditme rules decide from the shape of a call or a
`raise` alone; the fix is in the agents repository, filed in its `TODO.md` as
"auditme rules report hits the code does not have".

Fix direction: once that item ships, bump the agents rev in
`.pre-commit-config.yaml`, remove each marker listed in `requests.md` that
the new rules no longer need, and delete its entry there.

Note 2026-11-02: the agents item shipped on 2026-11-01 (its `DONE.md`,
`agents/specs/auditme-false-positives/spec.md`), merged on its `origin/main`
at `827a373`. `.pre-commit-config.yaml` still pins `8b43d9b`, which predates
it, so this item can start.

Done: 2026-11-02 - `.pre-commit-config.yaml` pins agents `49dab9d`; the 12 markers its rules no longer need (every `passthrough-shim`, `missing-raises-section` and RUF022 `__all__` marker, and the `raises-section-drift` markers of `coef_summary` and `llplot`) are removed, and `agents/specs/auditme-clean/requests.md` keeps only the requests whose sites still hit; `pre-commit run auditme-all --hook-stage manual --all-files` passes.

## Run the metafor audit to find the next gaps against R metafor

Found on 2026-11-01: `TODO.md` held no open port work, and this checkout
has no earlier run under `.audit/runs/` to draw from.
`bin/metafor-audit.sh` compares metafopy with R metafor across every exported
function and argument (see `src/audit/README.md`).

Cause: new port work is found by the audit harness, and it has not been run
since the auditme work.

Fix direction: run `bin/metafor-audit.sh` in the background, then file each
cause in its `wrong-or-missing.md` here with `/todo-add`. Run
`bin/metafor-audit.sh --recall` too, and fix any miss in `src/audit/data/`.

<!-- doc-counts: accept-start note of 2026-11-02 - a dated record, not a claim about the present -->
Note 2026-11-02: `bin/metafor-audit.sh --steps options` completed into
`.audit/runs/20261102-061046` at `2fc8c6d`; its `wrong-or-missing.md` lists
81 causes, none yet filed here. The `matrix` step last ran in
`.audit/runs/20261101-211517`, and no recall run was found. Still to do: file
those causes, run the whole audit in one run, and run `--recall`.
<!-- doc-counts: accept-end -->

Note 2026-11-02: the shard survival queue ran the `options` step again into
`.audit/runs/20261102-143634` in its worktree, which was removed at the merge
(`eadb479`), so that report is gone; `20261102-061046` is still the latest
`options` run in this checkout. That run's one new cause unrelated to the
change was a `gosh` case whose R cluster could not open its port, taken as a
clash in that run; it is not filed unless the next run shows it again.

Note 2026-11-02: nothing now stops a whole-audit run: the `options` and
`scripts` steps both survive a dead process (merged at `eadb479` and
`7c11296`). The scripts shard survival queue ran `--steps scripts` into
`.audit/runs/20261102-210939` in its worktree, removed at the merge; its
report is gone, and its figures are in Task 6's notes in
`agents/tasks/scripts-shard-survival/tasks-1.md`: three causes (two wrong
values and one universe difference), not yet filed here.

Done: 2026-11-02 - the whole audit ran into `.audit/runs/20261102-223017` at `ffd5ffe` (143 causes, 1740 findings) and `--recall` into `.audit/runs/recall-20261102-230628` (176 of 177 reproduced); every cause is filed in `TODO.md` in one item per family, and the recall miss with the harness item.

## The audit harness misses one recall item and carries stale ledger entries

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). The recall check (`bin/metafor-audit.sh --recall`, `.audit/runs/recall-20261102-230628`) reproduced 176 of 177 items of audits 4 to 6; it missed `a6-glmm-um-rs`: `rma_glmm(measure='OR', model='UM.RS')` on bcg: se 0.18214 where R gives 0.18147. The audit also reported the universe and the ledger lines below. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The miss is either a probe the harness no longer runs or a ledger entry in `src/audit/data/known.toml` that now hides it; the stale ledger entries match nothing in this run; the universe changed since `src/audit/data/universe.json` was written.

Fix direction: fix the recall miss in the harness (`src/audit/data/`), not in a report, as `AGENTS.md` says; remove or narrow each stale ledger entry; review the universe change and refresh it with `bin/metafor-audit.sh --update-universe`. Rerun `--recall` to confirm.

Note 2026-11-03: traced to the R oracle moving unchecked: R 4.3.3 and lme4
2.0.6 when `docs/differences.md` was captured (2026-10-21), R 4.6.2 and lme4
2.1-0 in the runs of 2026-11-01 and 2026-11-02. `setup.R` installs every
package but metafor, metadat and gsl at its newest version, and `check_r`
compares the library only with a `data/r-versions.json` that was never
committed. On 2026-11-02 R gave `glmm_rs` se 0.182140 and Python 0.182139.

Spec: agents/specs/r-oracle-pin/spec.md; tasks: agents/tasks/r-oracle-pin/tasks-1.md.

Causes, from the run's `wrong-or-missing-full.md`:

- the R universe differs from data/universe.json; refresh it with `bin/metafor-audit.sh --update-universe` (1 cases; `universe`)
- Stale ledger entries: construct/glmm_cm/* refused-where-R-works 'EstimationError: Cannot compute var-cov matrix of the fixed effects.', */glmm_rs* value-diff, matrix/dfbetas/mv_wide value-diff '*element 17: R None*', matrix/rstudent/mv_wide print-diff "*line 19: R '18      NA     NA      NA'*", options/selmodel/* print-diff "*R 'delta.1 *', Python 'delta.1 *'"
<!-- doc-counts: accept-end -->

Done: 2026-11-03 - agents/specs/r-oracle-pin/spec.md, agents/tasks/r-oracle-pin/tasks-1.md.

## The audit reports R's cluster port clash as a difference from R

<!-- doc-counts: accept-start found on 2026-11-03 - a dated record, not a claim about the present -->
Found on 2026-11-03 by `/todo-review`. The `DONE.md` entry of the whole-audit item records, in its note of 2026-11-02, a `gosh` case in `.audit/runs/20261102-143634` whose R cluster could not open its port, "not filed unless the next run shows it again". The whole audit `.audit/runs/20261103-071849` (metafopy `f040b41`, R 4.6.2) showed it again, on `profile`: `R: creation of server socket failed: port 11347 cannot be opened` (noted on the location-scale item).
<!-- doc-counts: accept-end -->

Cause: nothing in `src/audit/` sets the port of R's `parallel` clusters, so each R process takes a random port when it starts one (R's default), and shards running side by side can take the same port. R's failure is then reported as a cause, though it is not a difference of metafopy.

Fix direction: give each R process the harness starts its own port range (`R_PARALLEL_PORT` in its environment, derived from the shard), with a test that two shards never share a port; then check that the next whole audit shows no socket failure.

Done: 2026-11-03 - `audit.rdriver.cluster_port` gives each R process a free port of its own in `R_PARALLEL_PORT` (branch `fix/r-cluster-port`); tests `test_each_r_process_gets_its_own_free_cluster_port` and `test_r_takes_the_cluster_port_it_is_given` in `src/tests/test_harness_engine.py`. The next whole audit confirms no socket failure.

## gosh gives different subsets and results from R on every fit

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). Every `gosh` fit differs from R, in which subsets are included and in every result of them. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. Every fit is affected, so the likely cause is the choice or order of subsets in `src/metafopy/methods/gosh.py` (random subsets when there are more than `subsets` of them), not one statistic.

Fix direction: compare the subset matrix `incl` with R's for one small fit first; the printed summaries follow from it.

Spec: agents/specs/r-fused-arithmetic/spec.md (implemented); tasks: agents/tasks/r-fused-arithmetic/tasks-1.md.

Causes, from the run's `wrong-or-missing-full.md`:

- `$incl` element 259: R False, Python True (86 cases; `gosh`, `print`)
- `$res$H2` element 9: R 2.01139580879891, Python 22.871620858684505 (relative 0.91) (86 cases; `gosh`, `print`)
- `$res$I2` element 9: R 50.2832811112825, Python 95.62776942579347 (relative 0.47) (86 cases; `gosh`, `print`)
- `$res$QE` element 9: R 10.0569790439946, Python 114.35810429342253 (relative 0.91) (86 cases; `gosh`, `print`)
- `$res$k` element 3: R 4, Python 5.0 (relative 0.2) (86 cases; `gosh`, `print`)
- `$res$estimate` element 6: R -0.0270396616679308, Python -0.5357504809424486 (relative 0.95) (50 cases; `gosh`, `print`)
- `$res$tau` element 0: R 0.384719728838867, Python 0.0 (relative 1) (60 cases; `gosh`)
- `$res$tau2` element 0: R 0.148009269757851, Python 0.0 (relative 1) (60 cases; `gosh`)
- `$res$intrcpt` element 0: R -0.408765362241168, Python 0.3991816712497156 (relative 2) (36 cases; `gosh`)
- `[]$ci.lb$intrcpt` element 3: R None, Python -1.3381058095114846 (33 cases; `profile`)
- `[]$ci.ub$intrcpt` element 3: R None, Python 0.07707642670447312 (32 cases; `profile`)
- `$fit` element 0: R False, Python True (24 cases; `gosh`)
- `$res$ablat` element 2: R 0.00233066315964768, Python -0.012250480672567066 (relative 1.2) (24 cases; `gosh`)
- `$res$factor.alloc.random` element 1: R -0.651139340705579, Python None (12 cases; `gosh`)
- `$res$factor.alloc.systematic` element 1: R -0.58522921765392, Python None (12 cases; `gosh`)
- `$res$year` element 0: R None, Python 0.014467269976428554 (12 cases; `gosh`)
- `$beta$sim` element 1152: R None, Python 1.0193929566782072 (2 cases; `vif`)
- `$log.year...1900...1000.` element 53: R None, Python 0.0633444763804744 (1 cases; `dfbetas`)
- `[]$beta$<name>` element 2: R None, Python 5.766776911948009 (1 cases; `profile`)
- `[]$beta$intrcpt` element 2: R None, Python -40.15864014777602 (1 cases; `profile`)
- `[]$ci.lb$<name>` element 2: R None, Python -13.618798384495301 (1 cases; `profile`)
- `[]$ci.ub$<name>` element 2: R None, Python 25.15235220839132 (1 cases; `profile`)
- `[]$ll` element 2: R None, Python -9.81822951023776 (1 cases; `profile`)
- line 4: R 'k         6.0000  2.0000  5.0000  6.0000  7.0000  10.0000', Python 'k         6.2500  2.0000  5.0000  6.0000   7.2500  10.0000' (36 cases; `gosh`, `print`)
- line 4: R 'k         6.1000  3.0000  4.7500  6.0000   8.0000  11.0000', Python 'k         7.5000  4.0000  5.7500  8.0000   9.2500  11.0000' (14 cases; `gosh`)
- line 2: R 'Model fits succeeded: 17', Python 'Model fits succeeded: 19' (12 cases; `gosh`)
- line 4: R 'k        6.7500  4.0000  5.0000  6.5000  8.0000 11.0000', Python 'k        6.7000  3.0000  5.7500  7.0000  8.0000 11.0000' (11 cases; `gosh`)
- line 4: R 'k         6.4000    3.0000   5.0000   6.0000  7.2500 11.0000', Python 'k         6.7500    4.0000   5.0000   6.5000  8.0000 11.00 (10 cases; `gosh`)
- line 3: R '1.0000', Python '1.00' (4 cases; `vif`)
- line 4: R 'k                        5.5000  3.0000  4.2500  5.5000  6.7500  8.0000', Python 'k                        6.2000  4.0000  5.0000 (1 cases; `gosh`)
- line 4: R 'k         5.5000    3.0000    4.2500   5.5000  6.7500   8.0000', Python 'k          6.2000    4.0000   5.0000   6.0000  7.0000 10 (1 cases; `gosh`)
- line 4: R 'k        7.2000  5.0000  6.0000  7.0000  8.7500 10.0000', Python 'k        6.8000  3.0000  6.0000  7.0000  8.0000  9.0000' (1 cases; `gosh`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. The `dfbetas` cause (`$log.year...1900...1000.`) is now matched by the ledger entry of `matrix/dfbetas/mv_wide`, which names element 53 where it named element 17, so a rerender no longer lists it.

Note 2026-11-03: `bin/metafor-audit.sh --only gosh` on the pinned R 4.6.2 with the fused `dbinom` (metafopy `5b09d65`, run `.audit/runs/20261103-204725`, local) lists no `$incl`, `$res` or summary-line (`k`, `Model fits succeeded`) difference for a `gosh` case: the causes that came from different random subsets have gone. The only `gosh` findings left are `matrix/gosh/uni_digits` (`$digits` R 2, Python 4.0, and its `print` line) and the `scripts/gosh/*` missing fields (`chksum*`, `verbose`), all matched by the ledger. Two ledger entries (`matrix/gosh/*` and `options/gosh/trimfill_bcg/*` in `src/audit/data/known.toml`) still give the reason "R 4.3.3"; they hide any `matrix/gosh` difference, so their reason is stale and the match should be narrowed to `uni_digits`. The slice did not run the `vif`, `dfbetas` or `profile` causes listed above; they stay open here until a whole audit lists them or not.

Note 2026-11-04: the two ledger entries are removed. Without them, `bin/metafor-audit.sh --only gosh` (run `.audit/runs/20261104-070314`, local) listed only `matrix/gosh/uni_digits`: `gosh` dropped the fit's `digits` (R keeps `x$digits`), now fixed in `gosh.assemble` with `test_gosh_keeps_the_digits_of_the_fit`, after which the slice lists no `gosh` finding (`.audit/runs/20261104-070917`). Every `gosh` cause above has gone. The slice `--only vif --only dfbetas --only profile` (`.audit/runs/20261104-071144`) still lists the `vif` and `profile` causes above (`[]$ci.lb$intrcpt` R None in 33 cases, `$beta$sim`, `[]$ll`, the `vif` print `1.0000` against `1.00`) and none for `dfbetas`; they are not `gosh` differences, and they keep this item open.

Done: 2026-11-04 - the gosh causes went with the fused `dbinom` (spec `agents/specs/r-fused-arithmetic/spec.md`, tasks `agents/tasks/r-fused-arithmetic/tasks-1.md`) and `c424565` (`gosh` keeps the fit's `digits`); `.audit/runs/20261104-070917` lists no `gosh` finding. Its `vif` and `profile` causes moved to the location-scale, selection-model and multivariate items, by the dated notes of 2026-11-04 there.

## Location-scale fits differ from R in estimates, permutation tests, profile bounds and print

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). Location-scale fits (`rma` with `scale`; the `ls_*` fits) differ in the last digits of their standard errors and tests, in `permutest` (permuted coefficients of the opposite sign), in `confint` scale bounds where R gives `NA`, in `vif`, in four optimizers metafopy refuses, and in the headings of their print. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced; several may share one. The findings point at `src/metafopy/models/rma_ls_fit.py` (convergence), `src/metafopy/methods/ls_permutest.py`, `src/metafopy/methods/ls_confint.py`, `src/metafopy/methods/ls_vif.py` and `src/metafopy/models/rma_ls_result.py` (print). The R process killed for memory is R's own limit on `permutest(exact=TRUE)`, not a metafopy fault.

Fix direction: start with the fit itself (the 211-case `QM` and 127-case `se` differences on `ls_year` and `ls_na`), since the tests, `confint` and `permutest` are built on it; then `permutest`'s signs, the `confint` bounds where R gives up, the optimizers, and the print headings. Split this item if the parts turn out to have separate causes.

Causes, from the run's `wrong-or-missing-full.md`:

- R: R passed the memory limit of 4 GiB (killed) (2 cases; `permutest`)
- EstimationError: Optimizer (Rcgmin) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (Rvmmin) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (lbfgsb3c) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- EstimationError: Optimizer (optimParallel) did not achieve convergence (ABNORMAL: ). (1 cases; `rma`)
- `$QM.perm` element 14: R 0.0200910239625718, Python 11.119548182959777 (relative 1) (50 cases; `permutest`)
- `$beta.perm$intrcpt` element 19: R -47.5589351508886, Python 45.83744578027836 (relative 2) (50 cases; `permutest`)
- `$beta.perm$year` element 19: R 0.0238767715964596, Python -0.023592544278131733 (relative 2) (50 cases; `permutest`)
- `$zval.perm$intrcpt` element 19: R -1.32883778217893, Python 1.9383745471523741 (relative 1.7) (50 cases; `permutest`)
- `$zval.perm$year` element 19: R 1.3115090918244, Python -1.9512703736744204 (relative 1.7) (50 cases; `permutest`)
- `$beta$prop` element 1: R 0.998998998998999, Python 0.998 (relative 0.001) (2 cases; `vif`)
- `$vcovmat` element 10: R 0.600577420109585, Python 0.6005034810249658 (relative 0.00012) (2 cases; `rma`)
- `$alpha$prop` element 1: R 0.698396793587174, Python 0.6717325227963525 (relative 0.038) (1 cases; `vif`)
- `$alpha$sim` element 1990: R 1.00000000064568, Python None (1 cases; `vif`)
- `$ci.lb.alpha` element 1: R -0.137767763314913, Python -0.13771275253750662 (relative 0.0004) (1 cases; `rma`)
- `$ci.ub.alpha` element 0: R 3.49331755351219, Python 3.4915535605161026 (relative 0.0005) (1 cases; `rma`)
- `$pval.alpha` element 0: R 0.626571724710284, Python 0.6264407445288733 (relative 0.00021) (1 cases; `rma`)
- `$se.alpha` element 1: R 0.0701795309043885, Python 0.07015146427298927 (relative 0.0004) (1 cases; `rma`)
- `$va` element 2: R -0.162084992095922, Python -0.1619553584200976 (relative 0.0008) (1 cases; `rma`)
- `$zval.alpha` element 1: R -0.00311216515101929, Python -0.0031133933120681464 (relative 0.00039) (1 cases; `rma`)
- line 12: R 'QM(df = 2) = 1.2830, p-val = 0.5265', Python 'QM(df = 2) = 1.2828, p-val = 0.5266' (211 cases; `permutest`, `print`, `rma`, `rma.glmm` and 2 more)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0622  -21.4105  0.5339  .', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0622  -21.4102   (127 cases; `print`, `rma`, `summary`, `update`)
- line 2: R 'alpha.1 -10.4383    NA -2.9930', Python 'alpha.1 -10.4383 -27.0874 -2.9930' (44 cases; `confint`)
- line 2: R 'alpha.1  -3.0288 -18.7856 0.4381', Python 'alpha.1  -3.0288 -25.1411 0.4381' (43 cases; `confint`)
- line 14: R 'intrcpt           -3.0288  2.1031  -1.4401  0.1498   -7.1508   1.0933', Python 'intrcpt           -3.0288  2.1032  -1.4401  0.14 (6 cases; `rma`)
- line 6: R '0.3083  0.8897', Python '0.3077  0.8898' (5 cases; `vif`)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646   8  0.0992  -23.3477  2.4711  .', Python 'intrcpt  -10.4383  5.5980  -1.8646   8  0.0992  -2 (5 cases; `rma`)
- line 12: R 'Test of Scale Coefficients (coefficients 2:3):', Python 'Model Results (Scale):' (3 cases; `rma`)
- line 6: R 'ablat   -0.0189  0.0017  -11.0003  <.0001  -0.0223  -0.0155  ***', Python '-0.0189  0.0017  -11.0003  <.0001  -0.0223  -0.0155  * (3 cases; `rma`)
- line 14: R 'intrcpt           -5.4749  27.8296  -0.1967  0.8440  -60.0200  49.0701', Python 'intrcpt           -5.4749  27.8241  -0.1968  0. (3 cases; `rma`)
- line 13: R 'I(ni/100)     -1.00  0.61  -1.64  0.05   -2.19   0.19  *', Python 'I(ni/100)     -1.00  0.61  -1.64  0.05   -2.19   0.19  .' (2 cases; `permutest`)
- line 2: R 'alpha.1    -3.03 -18.79  0.44', Python 'alpha.1    -3.03 -25.14  0.44' (2 cases; `confint`)
- line 2: R 'alpha.1   -10.44    NA -2.99', Python 'alpha.1   -10.44 -27.09 -2.99' (2 cases; `confint`)
- line 1: R "'log Lik.' -3.214979 (df=4)", Python "'log Lik.' -3.21498 (df=4)" (1 cases; `logLik`)
- line 2: R '-10.3388 5.5402 -21.1974 0.5199', Python '-10.3388 5.5401 -21.1971 0.5196' (1 cases; `predict`)
- line 2: R '-3.5286 1.7967 -7.0501 -0.0071', Python '-3.5286 1.7970 -7.0506 -0.0066' (1 cases; `predict`)
- line 3: R 'I(ni/100)  1.0094  0.28', Python 'I(ni/100)  1.0094  0.30' (1 cases; `vif`)
- line 2: R 'QS(df = 1) = 0.5476, p-val = 0.4593', Python 'QS(df = 1) = 0.5474, p-val = 0.4594' (1 cases; `anova`)
- line 4: R 'Test of Location Coefficients (coefficients 2:3):', Python 'Model Results (Location):' (1 cases; `rma`)
- line 15: R 'Test of Scale Coefficients (coefficients 2:3):', Python 'intrcpt' (1 cases; `rma`)
- line 6: R 'allocrandom      2.2723  0.50', Python 'allocrandom      2.2723  0.97' (1 cases; `vif`)
- line 6: R 'alpha.3   1.1903 <-10.7124    NA', Python 'alpha.3   1.1903 <-10.7124 >13.0929' (1 cases; `confint`)
- line 2: R 'alpha.1  -3.0288 -17.9949 -0.1956', Python 'alpha.1  -3.0288 -25.2283 -0.1956' (1 cases; `confint`)
- line 2: R 'alpha.1 -10.4383    NA    NA', Python 'alpha.1 -10.4383    NA -2.9930' (1 cases; `confint`)
- line 17: R 'imag         2.1262  1.1857   1.7932  0.0729  -0.1978   4.4502   .', Python 'imag         2.1258  1.1857   1.7929  0.0730  -0.19 (1 cases; `rma`)
- line 12: R 'intrcpt                                              0.0876  -1.2308  0.0846  .', Python 'intrcpt                                (1 cases; `rma`)
- line 14: R 'intrcpt           -3.4482  6.0407  -0.5708  0.5681  -15.2878   8.3913', Python 'intrcpt           -3.4482  6.0391  -0.5710  0.56 (1 cases; `rma`)
- line 8: R 'intrcpt       6.9574  27.4113   0.2538  0.7996  -46.7677  60.6825', Python 'intrcpt       6.9587  27.4109   0.2539  0.7996  -46.7 (1 cases; `rma`)
- line 3: R 'intrcpt     0.2303 0.0655  3.5166  <.1  0.1019  0.3586     NA     NA', Python 'intrcpt     0.2303 0.0655  3.5166 0.0004  0.1019   (1 cases; `vif`)
- line 15: R 'intrcpt     -2.3458  0.8353  -2.8082  0.0050  -3.9830  -0.7085  **', Python 'intrcpt     -2.3456  0.8354  -2.8079  0.0050  -3.98 (1 cases; `rma`)
- line 3: R 'intrcpt   0.2677 0.2247  1.1914  0.2 -0.1727  0.7080     NA     NA', Python 'intrcpt   0.2677 0.2247  1.1914 0.2335 -0.1727  0.70 (1 cases; `vif`)
- line 11: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0500   -21.4105  0.5339  *', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0500   -21.4102 (1 cases; `permutest`)
- line 11: R 'intrcpt  -10.4383  5.5982  -1.8646  0.1500   -21.4105  0.5339', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.1500   -21.4102  0 (1 cases; `permutest`)
- line 14: R 'intrcpt  -10.4383  5.5982  -1.8646  0.0622  -19.6465  -1.2301  .', Python 'intrcpt  -10.4383  5.5980  -1.8646  0.0622  -19.6462  (1 cases; `rma`)
- line 3: R 'intrcpt -35.8515 40.6744 -0.8814  0.4 -115.5718 43.8688     NA     NA', Python 'intrcpt -35.8509 40.6745 -0.8814 0.3781 -115.5715 (1 cases; `vif`)
- line 3: R 'intrcpt -81.2229 19.3071 -4.2069  <.1 -119.0641 -43.3817     NA     NA', Python 'intrcpt -81.2229 19.3071 -4.2069 <.0001 -119.064 (1 cases; `vif`)
- line 1: R 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub X.intrcpt X.I.ni.100. X.meta', Python 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub X.intr (1 cases; `predict`)
- line 1: R 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub Z.intrcpt Z.I.ni.100. Z.imag', Python 'pred     se   ci.lb  ci.ub   pi.lb  pi.ub Z.intr (1 cases; `predict`)
- line 2: R 'year  379.7309 223.8329 -58.9736 818.4354', Python '1 379.7302 223.8271 -58.9629 818.4233' (1 cases; `predict`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. One cause appeared that belongs to no item: `R: creation of server socket failed: port 11347 cannot be opened` (`profile`, R's own failure to open a port while shards ran; not a difference of metafopy).

Note 2026-11-04: the `vif` causes the gosh item carried belong here: the slice `.audit/runs/20261104-071144` (local) lists them on the `ls_bcg`, `ls_year`, `ls_na` and `ls_bangert` cases. It also lists `vif` print differences with `digits=2` not recorded above: R '1.0000', Python '1.00' (`ls_bcg`, `ls_na`); R '4.7231  4.7231', Python '4.72  4.72' (`ls_year`); R '1.0094  1.0094', Python '1.01  1.01' (`ls_bangert`); and on `options/vif/ls_bangert/sim=true` `$beta$sim` element 1285 (R 1.83095037872554, Python 1.0234467077414686) and `$beta$prop` (R 0.282, Python 0.296). The R port clash the note of 2026-11-03 names was filed and fixed in `a6059a2` (each R process gets its own port); see `DONE.md`.

Done: 2026-11-04 - re-checked on the slices `.audit/runs/20261104-092750` and `20261104-094213` (local). The fit, `permutest`, `confint` and optimiser differences are stop points or failures of R's `nlminb` on flat scale likelihoods, accepted by decision of the operator (2026-11-04) in `src/audit/data/known.toml` and `docs/differences.md` ("Refits of a location-scale model that R's optimiser leaves short"); the missing scale-test headings follow from the Hessian difference already accepted there. Fixed: the `vif` print of location-scale fits (`test_ls_vif.py`), the row name of a single location coefficient and the wrapping of a wide coefficient table (`test_rma_ls.py`). The `predict` row name moved to the printed-output item; the printed call of `update` is in that item already.

## Generalised linear mixed models differ from R in tests and a warning

<!-- doc-counts: accept-start found on 2026-11-02 - a dated record, not a claim about the present -->
Found on 2026-11-02 by the whole audit `.audit/runs/20261102-223017` at metafopy `ffd5ffe` (metafor 5.2-1). `rma.glmm` fits differ in their tests (`Wld` 268.5990 against 191.7124 for `model="CM.EL"`), in small values, and do not give R's Hessian warning for the saturated model. The run's report is local and not in
git, so its cause lines are copied below as it printed them.

Cause: not yet traced. The findings point at `src/metafopy/models/rma_glmm.py` and `src/metafopy/models/glmm_fit.py`.

Fix direction: start with the `Wld` difference, which is far beyond rounding.

Causes, from the run's `wrong-or-missing-full.md`:

- `$prob` element 0: R 0.00197382147421394, Python 0.0019733636664781558 (relative 0.00023) (1 cases; `predict`)
- line 11: R 'QM(df = 1) = 76.2183, p-val < .0001', Python 'QM(df = 1) = 76.2266, p-val < .0001' (5 cases; `rma`, `rma.glmm`)
- line 11: R '-0.7138  0.1691  -4.2219  <.0001  -1.0451  -0.3824  ***', Python '-0.7138  0.1690  -4.2225  <.0001  -1.0451  -0.3825  ***' (2 cases; `rma.glmm`)
- line 3: R 'tau^2 (estimated amount of residual heterogeneity):     0.2733', Python 'tau^2 (estimated amount of residual heterogeneity):      (2 cases; `rma.glmm`)
- line 5: R 'I^2 (residual heterogeneity / unaccounted variability): 61.79%', Python 'I^2 (residual heterogeneity / unaccounted variability):  (1 cases; `rma.glmm`)
- line 8: R 'Wld(df = 12) = 268.5990, p-val < .0001', Python 'Wld(df = 12) = 191.7124, p-val < .0001' (1 cases; `rma.glmm`)
- line 14: R 'intrcpt     -10.2984  18.9350  -0.5439  0.5865  -47.4102  26.8135', Python 'intrcpt     -10.3059  18.9351  -0.5443  0.5863  -47. (1 cases; `rma.glmm`)
- line 14: R 'intrcpt  -61.2823  29.0390  -2.1103  0.0348  -118.1976  -4.3670  *', Python 'intrcpt  -61.2807  29.0369  -2.1104  0.0348  -118.1 (1 cases; `rma.glmm`)
- Cannot invert the Hessian for the saturated model. (4 cases; `rma.glmm`)
<!-- doc-counts: accept-end -->

Note 2026-11-03: the whole audit `.audit/runs/20261103-071849` on the pinned R 4.6.2 and lme4 2.1-0 (metafopy `f040b41`) lists every cause below again; none vanished. `rma_glmm(model='UM.RS')` no longer differs from R in se, so its value-diff ledger entry was removed. The warning `Cannot invert the Hessian for the saturated model.` is now a `warning-extra` ledger entry for the `control` optimizer cases (R inverts, metafopy warns); the `warning-missing` entry for the other cases stays, and the refused-where-R-works entry for `glmm_cm` was removed.

Note 2026-11-04: re-checked on the slice `.audit/runs/20261104-105354` (local, metafopy `a7b8700`); every cause above is listed again except the Hessian warning, which the ledger matches. On `construct/glmm_bcg/mods=~ alloc` (UM.FS) R's log-likelihood is -70.4602835 and metafopy's -70.4652470: metafopy's fit stops short of R's maximum (tau2 0.27290 against 0.27334, QM 1.8844 against 1.8981), so these differences are a defect of metafopy's fit, not a stop point of R's optimiser. Start there.

Note 2026-11-04 (correcting the note above): metafopy's fit does not stop short. The log-likelihood compared there is metafor's `ll.QE - deviance(res.ML)/2`, not the objective glmer maximises. For `~ alloc` R's `glmer()` reports 'Model failed to converge with max|grad| = 0.0186874 (tol = 0.002)', and on glmer's Laplace objective metafopy's estimates are 4.4e-5 higher than R's. R also stops before converging for `~ year`, `~ 0 + alloc` and `nAGQ = 3`; where it converges (`~ I(ablat^2)`, `~ ablat + I(ablat^2) + year`) the maxima agree to 1e-8 and the `QM` difference comes from lme4's finite-difference Hessian at estimates that differ in the sixth digit. The `Wld` of `model = 'CM.EL'` is the noisy saturated Hessian already accepted for `glmm_cm`.

Done: 2026-11-04 - by decision of the operator (2026-11-04) the differences are accepted as R's glmer stopping before convergence or lme4's finite-difference Hessian: the print-diff entry of `construct/glmm_bcg/*` and the `$prob` entry of `options/predict/glmm_bcg/*` in `src/audit/data/known.toml`, next to the value-diff entry of `*/glmm_bcg*` that already accepted the values; `docs/differences.md` ("rma_glmm with UM.FS: z value in the fourth digit"). The re-rendered slice `.audit/runs/20261104-105354` lists no GLMM finding but the printed call of `update`, which the printed-output item holds.

## `I()` in a formula reads `|` as a bitwise OR where R's is a logical OR

<!-- doc-counts: accept-start found on 2026-11-04 - a dated record, not a claim about the present -->
Found on 2026-11-04 during the review of `TODO.md`, checking the location-scale causes of the small-numeric item. `rma('yi', 'vi', mods='~ablat+year', scale='~0+I(1|ablat)', data=dat)` on `dat.bcg` (`escalc(measure='RR')`) gives the scale column 45, 55, 43, 53, 13, ... (the bitwise OR of 1 and the integer `ablat`) and a log-likelihood of -8.4771; R's `1|ablat` is a logical OR, the column is 1 in every row, the model is `scale = ~ 1` and R's log-likelihood is -8.1069 (metafopy's `scale='~1'` gives -8.10687, as R). The audit case `construct/uni_mods_year/scale={"$formula":"~ 0 + I(1|ablat)"}` lists only a tiny `$ci.ub` difference for it, so the harness does not see this.
<!-- doc-counts: accept-end -->

Cause: `metafopy.formula.design_matrix` hands `I(...)` to formulaic, which evaluates it as Python; `r_power` rewrites R's `^` only, so R's logical `|`, `&` and `!` reach Python's bitwise operators (on integers they give numbers, on floats an error).

Fix direction: translate R's logical operators inside `I()` (and anywhere formulaic evaluates R code) to element-wise logical operations, next to `r_power`; then find why the audit case shows no difference (how the harness passes the data or the formula) and make it show one before the fix.

Note 2026-11-04: the audit case is `I(1/ablat)`; `label()` (`src/audit/domains.py:116`) shows `/` as `|` in case ids. No audit domain used `|`, `&` or `!` inside `I()`, so the harness had nothing to see. The fix adds a unit test, and domain values with R's logical operators, so the audit covers them.

Note 2026-11-04: the same cause has a second symptom, folded into this item by decision of the operator (2026-11-04): R's `model.matrix` codes a logical `I()` term as a factor with levels FALSE and TRUE (`~ I(ablat > 30)` gives the column `I(ablat > 30)TRUE`, `~ 0 + I(ablat > 30)` a FALSE and a TRUE column); metafopy gave one numeric column named `I(ablat > 30)`. `emmprep()` of a fit with `|` in `I()` failed (Python reads `ablat > 30 | year < 1970` as a chained comparison).

Done: 2026-11-04 - `metafopy.formula.r_logical_calls` translates R's `|`, `&` and `!` in a call into element-wise `r_or`, `r_and` and `r_not` (R's three-valued logic through pandas' nullable booleans); `design_matrix` evaluates such a call into a placeholder column named by the call's R text (`r_logical_terms`), `emm_grid` evaluates the translated formula (`r_logical_formula`), and the context's `I()` (`r_identity`) makes a logical value R's FALSE/TRUE factor. Tests: `src/tests/test_formula.py` (names and columns as R's `model.matrix`, missing values, the `scale` fit's log-likelihood -8.10687), `src/tests/test_emmeans.py::test_reference_grid_of_a_logical_term`. Audit domains `I(ablat > 30)`, `I(ablat > 30 | year < 1970)`, `I(!(ablat > 30) & year < 1970)` (mods) and `~ 0 + I(1 | ablat)` (scale); the slice `.audit/runs/20261104-175221` (local) lists no value or name difference for them, only the missing scale-test heading of `ls_bcg` that other formulas show too.

## Accepted location-scale differences that no ledger entry matches

Cause: the operator accepted on 2026-11-04 the differences of location-scale fits that come from R's `nlminb` stopping at another point of a flat scale likelihood, or from R's numerical Hessian at the same maximum; the ledger entries written then match the `ls_*` fixtures and some `uni_*/scale=` cases by their details, so these cases are still listed as new.

Fix direction: check each case below against R (log-likelihood and estimates: the same maximum with other standard errors is the Hessian, other estimates a stop point); add a ledger entry with that evidence to `src/audit/data/known.toml` for each that is, and file any that is neither as a defect; re-render a slice to see the findings move to `known`.

### The missing scale-test heading of `ls_bcg` variants (filed 2026-11-04)

<!-- doc-counts: accept-start found on 2026-11-04 - a dated record, not a claim about the present -->
Found on 2026-11-04 by the slices `.audit/runs/20261104-174649` and `20261104-175221` (local, metafopy `3ec15be` with the change of `e1b260f`). On `construct/ls_bcg/mods=` with `~ ablat + alloc`, `~ ablat - 1`, `~ poly(ablat, 2)`, `~ ablat:alloc`, `~ I(ablat > 30)` and `~ I(ablat > 30 | year < 1970)`, R prints 'Test of Scale Coefficients (coefficients 2:3):' where metafopy goes on to 'Model Results (Scale):'. The audit lists them as new: no entry of `src/audit/data/known.toml` matches them, although the `Done:` line of the location-scale item in `DONE.md` (2026-11-04) says "the missing scale-test headings follow from the Hessian difference already accepted there". On the fixture itself (`mods = ~ ablat`, `scale = ~ alloc`) R and metafopy agree (`QS` 0.53647, scale se 2.1032, 5.1596, 2.4847). With `mods = ~ ablat + alloc` the scale estimates run off along the flat likelihood and stop at different points (R's intercept -21.54, metafopy's -33.715); R's Hessian there still inverts (se 44590.91, `QS` 0.1679, p 0.9195), metafopy's does not (se and `QS` NaN).
<!-- doc-counts: accept-end -->

Cause: R's `nlminb` stops at another point of a flat scale likelihood, the case accepted by decision of the operator (2026-11-04); `print` of a location-scale fit leaves out the scale test when `QS` is NaN (`src/metafopy/models/rma_ls_print.py:115`), so the difference shows as a missing heading.

Fix direction: add the print-diff entry the `Done:` line promised to `src/audit/data/known.toml` (match `construct/ls_bcg/mods=*`, detail `line *: R 'Test of Scale Coefficients*', Python 'Model Results (Scale):'`), with this evidence as its reason; re-render a slice to see the findings move to `known`.

### Location-scale fits of `uni_*` fixtures (copied from the small-numeric item)

- `$H` element 3: R 0.978416389473902, Python 0.9791954341103023 (relative 0.0008) (1 cases; `rma`)
- line 15: R 'allocrandom      -22.3482  55186.2609  -0.0004  0.9997  -108185.4320', Python 'allocrandom      -27.1962  626049.2249   0.0000   (1 cases; `rma`)
- line 11: R 'intrcpt   -1.1536  2.3709  -0.4866  0.6266  -5.8005  3.4933', Python 'intrcpt   -1.1536  2.3700  -0.4867  0.6264  -5.7987  3.491 (1 cases; `rma`)
- `$ci.ub` element 1: R -0.00033139675441687, Python -0.00033145363275964915 (relative 0.00017) (2 cases; `rma`)

Note 2026-11-04: the construct run `.audit/runs/20261104-193457` (local, metafopy `548759d`; `--only 'scale=' --only 'construct/ls_'`) lists these again as new: on `uni_reml` with `scale = ~ ablat` also `$va`, `$se.alpha`, `$zval.alpha`, `$pval.alpha`, `$ci.lb.alpha` and `$ci.ub.alpha` (relative 0.0002 to 0.0008; log-likelihood -12.2023666 in both, from the small-numeric item's note); on `uni_mods_year` with `scale = ~ ablat` `$ci.ub` (R 0.00181749426921627, Python 0.0018172047295845102) and 'QM(df = 1) = 0.0795, p-val = 0.7780' against '0.0793, p-val = 0.7782', with `scale = ~ alloc` the row 'intrcpt -4.8337' against '-4.8336', with `scale = ~ 0 + I(1/ablat)` the row '-73.4722  35.3727' against '-73.4724  35.3725' (also on `ls_year`); and on `construct/ls_year/optbeta=true` R's 'Test of Location Coefficients (coefficients 2:3):' where metafopy goes on to 'Model Results (Location):'.

Done: 2026-11-04 - checked each case against R 4.6.2 (metafor 5.2-1): every fit reaches R's log-likelihood to 1e-7 (`ls_bcg` variants -7.0165323 to -9.8717849, `uni_reml` -12.2023666, `uni_mods_na` -2.8010620, `uni_mods_year` -8.0233988 and -7.8420579 and -7.9073957; `ls_year` with `optbeta = TRUE` -7.6022560 against -7.6022565), with the scale estimates at other points of the flat likelihood or other numerical Hessians; none is a defect. Ledger (`src/audit/data/known.toml`): a print-diff entry for the missing test headings of `construct/ls_bcg/mods=*` and `construct/ls_year/optbeta=true`, the value-diff entry of `uni_mods*/scale=*` extended to `uni_reml` and `$ci.ub`, the print-diff entry of named rows extended to `construct/uni_*/scale=*`; `docs/differences.md` ("Refits of a location-scale model that R's optimiser leaves short"). The re-rendered run `.audit/runs/20261104-193457` (local) lists no new finding but the one-row scale tables of the printed-output item.

## Numerical Hessians and failed refits of R outside the accepted fits

Cause: as for the location-scale and GLMM fits accepted on 2026-11-04, R and metafopy reach the same maximum, but R's finite-difference Hessian, taken at a point that differs in the last digits, gives other standard errors, or cannot be inverted for some refits; or R's optimiser fails a refit that metafopy completes. The evidence is in the notes of 2026-11-04 on the selection-model item (`selmodel(type='beta')` of `trimfill_bcg`: same log-likelihood -99.049066667 and `delta`; `profile()` of `sel_bcg`: 'Error when trying to invert the Hessian') and on the multivariate item (`profile()` of `mv_wide`: R's refit fails, `ll` NA).

Fix direction: the decision of 2026-11-04 named location-scale and GLMM fits only; ask the operator whether it extends to these selection-model and multivariate cases, then add ledger entries with this evidence to `src/audit/data/known.toml` and `docs/differences.md`, or trace them further if not.

### Selection models (copied from the selection-model item)

- `$pval.delta` element 1: R 9.65330124664416e-05, Python 9.648235818601725e-05 (relative 0.00052) (1 cases; `selmodel`)
- `$se` element 0: R 0.0695218232173794, Python 0.06949089045726904 (relative 0.00044) (1 cases; `selmodel`)
- `$zval` element 0: R -2.3598482640218, Python -2.360898712835864 (relative 0.00044) (1 cases; `selmodel`)
- line 5: R '-0.1641  0.0695  -2.3598  0.0183  -0.3003  -0.0278  *', Python '-0.1641  0.0695  -2.3609  0.0182  -0.3003  -0.0279  *' (3 cases; `rma`, `selmodel`)

Note 2026-11-04: the `profile` causes the gosh item carried belong here. The slice `--only vif --only dfbetas --only profile` (`.audit/runs/20261104-071144`, local) lists `[]$ci.lb$intrcpt` and `[]$ci.ub$intrcpt` element 3 with R None, Python a value, on `matrix/profile/sel_bcg`, `matrix/profile/sel_digits` and the `options/profile/sel_bcg/*` cases, and `[]$ci.lb$intrcpt` element 2 (R -0.402157171730945, Python -0.40209684929512357) and `[]$ci.ub$ablat` on `matrix/profile/sel_mods_bcg`.

### Multivariate profile (copied from the multivariate item)

Note 2026-11-04: the `profile` causes on `matrix/profile/mv_wide` that the gosh item carried belong here: the slice `.audit/runs/20261104-071144` (local) lists `[]$ll`, `[]$beta$intrcpt`, `[]$beta$log(year - 1900 + 1000)` and `[]$ci.lb`/`[]$ci.ub` of both coefficients, element 2, R None, Python a value - the same pattern as `rstudent` and `confint` here.

Done: 2026-11-04 - by decision of the operator (2026-11-04) the decision on location-scale and GLMM fits extends to these cases: R's numerical Hessian of `selmodel(type='beta')` of `trimfill_bcg` (same log-likelihood and `delta`; R's se 0.06952 with either of its optimisers, metafopy's 0.06949), R's Hessian that it cannot invert in the `profile()` refits of `sel_bcg` and `sel_digits` (and the 1e-4 interval differences of `sel_mods_bcg`), and R's failed `profile()` refit of `mv_wide`. Three entries in `src/audit/data/known.toml`, the section "Numerical Hessians of selection models and failed refits of rma_mv" in `docs/differences.md`. The re-rendered slice `.audit/runs/20261104-210502` (local, `--only selmodel --only profile`) lists no finding of these cases; the two left (the `negexp` print row of `uni_factor`) belong to the selection-model item.
