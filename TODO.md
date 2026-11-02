# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


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
