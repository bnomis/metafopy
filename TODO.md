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
