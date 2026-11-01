# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


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
