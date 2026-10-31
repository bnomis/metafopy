# Review: Auditme clean spec

**Spec:** [spec.md](spec.md)
**Review page:** https://claude.ai/artifact/QoVrK6ztdQZsb1KLd3Twng
**Date:** 2026-10-31

## What is being reviewed and how

The spec `agents/specs/auditme-clean/spec.md` (status `proposed`) plans the
work behind the `TODO.md` item "The auditme checker reports unaccepted hits
and no hook enforces it": bring the repository to zero unaccepted hits from
the shared auditme checker, then switch on the push and CI gates. Approve or
flag each item below; a flagged item gets a comment saying what to change.
The task list (`/taskdev`) waits until every item has an answer.

## Automatic checks

Run on 2026-10-31:

| Check | Result |
|---|---|
| Every template section present (Module location, API, Error handling, Implementation, Settings, External dependencies) | pass |
| API holds a type-annotated code block | pass |
| Module location names paths under `src/` | pass |
| Spec under the size guide (about 200 lines) | pass |
| Plain ASCII punctuation | pass |
| `agents/specs/index.md` links the spec | pass |
| Hit counts in the Motivation match the checker run of 2026-10-31 at agents `108559d` on `9d02a0a` | pass |

## Items

1. **Motivation.** The dated findings carried over from `TODO.md`: 100
   `raises-section-drift` hits, the `METAFOR_FUNCTIONS` hit, 615 hits under
   24 older rules, and no auditme hook installed.
2. **`src/audit/` policy.** The harness is accepted by rule (your answer of
   2026-10-31), except its real defects: silent broad `except` blocks,
   `.get(k, {})` chains that break on an explicit null, and one inconsistent
   error translation. `METAFOR_FUNCTIONS` is filled at run time, so it is
   accepted rather than frozen as the `TODO.md` item proposed.
3. **`Raises:` sections** (`raises-section-drift`, `missing-raises-section`):
   fix each from the body and its callees; accept where a callee's error
   cannot fire on that call (as `fsn` calling `check_class` without `must`
   appears to be).
4. **Typing aliases**: replace `typing.Callable` and the like with
   `collections.abc` forms; never accepted.
5. **Imports** (`from-import`, `aliased-imports`, `in-function-import`): fix
   to the project's import rules unless a cycle forces the import.
6. **`literal-duplication`**: accept with the reasons the project already
   uses (repetition mirrors metafor or the R fixture scripts); extract a
   constant where the literal is the project's own.
7. **Sorting** (`non-alphabetical-literal`, `non-alphabetical-dict-keys`):
   sort, unless the order is metafor's or the R script's.
8. **`paired-sibling-helpers`**: share one helper in `src/metafopy/`; test
   modules keep their own loaders, as existing markers say.
9. **`passthrough-shim`**: call the target directly, unless the wrapper is a
   metafor-named public alias.
10. **`duplicate-call-sequence`**: accept by default; extract only where the
    shared steps have one meaning.
11. **`redundant-sink-caller-plus-callee`**: compute once, unless R computes
    twice and the order changes the digits.
12. **`mutable-module-level-exported`**: freeze constants (`PCH`, `STEPS`);
    accept what is mutated by design (`printing.SETTINGS`, your answer of
    2026-10-31, and the caches `YICHECKS`, `LAST_EVALUATED`).
13. **`assert-never-missing`**: add a final `else` that raises, unless the
    value is validated earlier.
14. **Remaining single-site rules** (`except-interpolation`,
    `reexport-in-non-init`, `duplicated-word`, `filesystem-envelope-gap`):
    judged per site.
15. **Order of work.** The `auditme-changed` commit hook first (your answer
    of 2026-10-31), then `src/audit/`, then `Raises:` sections one package
    per commit, mechanical rules, judged rules, the push and CI gates at zero
    hits, and finally the move to `DONE.md`.
16. **Error handling.** A fix that breaks a test is reverted and the hit
    accepted; a checker bump brings its new hits into the same change; the
    local-only `lockfile-drift` hit (from the global `uv.toml`
    `exclude-newer = "14 days"`) is left to clear, with a request to auditme
    if it blocks the gates; `uv.lock` is not rewritten.
17. **Settings.** The pinned agents revision, the `auditme-all` stages, and
    the `AGENTS_DEPLOY_KEY` secret.

## Decisions

None open. The three decisions this spec needed were answered in the
interview on 2026-10-31: `src/audit/` accepted by rule, mutable-by-design
containers accepted with a reason, and the commit hook added first. The
placement of the CI step in the existing `lint` job is a choice I made
because that job already installs the environment; it is reversible.

## Answers

Answered on the review page on 2026-10-31: every item (1 to 17) approved,
with no flags and no comments.
