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
