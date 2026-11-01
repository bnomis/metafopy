# Auditme clean

**Version:** 1.1
**Status:** implemented
**Date:** 2026-10-31; amended 2026-11-01 (order of work, Implementation)
**Review document:** [review.md](review.md)
**Review page:** https://claude.ai/artifact/QoVrK6ztdQZsb1KLd3Twng

Bring the repository to zero unaccepted hits from the shared auditme checker,
and keep it there with the shared guards: a commit hook on changed files from
the start, and a whole-tree pre-push hook and CI gate once the count is zero.
This closes the `TODO.md` item "The auditme checker reports unaccepted hits and
no hook enforces it".

## Motivation

Found on 2026-10-31 by running the shared auditme checker at agents `108559d`
on this repository (at `394c45fb`, and again at `9d02a0a`):

- the `raises-section-drift` rule, new that day, reported 100 hits in 81
  functions under `src/metafopy/`: a `Raises:` section omits a domain error
  that reaches the caller, raised in the function itself or declared by a
  callee (`fsn` in `src/metafopy/bias/fsn.py:159` calls `check_class` with no
  `try` and omits its `ArgumentError`);
- `mutable-module-level-exported` began counting `set(...)` calls and flagged
  `METAFOR_FUNCTIONS` (`src/audit/rlib.py:46`);
- 615 further unaccepted hits under 24 older rules, the largest being
  `literal-duplication` (281), `typing-deprecated-alias` (80),
  `paired-sibling-helpers` (64), `non-alphabetical-literal` (42),
  `non-alphabetical-dict-keys` (26) and `missing-raises-section` (25);
- `.pre-commit-config.yaml` holds only the ruff hooks, so nothing ran the
  checker and the hits accumulated unseen.

## Module location

Repository-wide: `src/metafopy/`, `src/tests/`, `src/audit/`, plus
`.pre-commit-config.yaml` and `.github/workflows/ci.yml`. No module is added.

## API

No public function, class or signature changes. Docstrings change (`Raises:`
sections), and some internal module-level constants change type. The two code
shapes this spec writes:

```python
import types

# a constant read from another module and never mutated: freeze it
PCH: types.MappingProxyType[int, tuple[str, bool, bool]] = types.MappingProxyType(
    {
        1: ('o', False, False),
    }
)

# a container mutated by design: keep it, and accept the hit for that rule only
SETTINGS: dict[str, int] = {'width': 80}  # auditme: accept[mutable-module-level-exported] R's getOption('width'), set by users as in R
```

A `Raises:` section lists every domain error from `metafopy.errors` that can
reach the caller, in the project's existing form (full module path, then the
condition), with each condition read from the body and its callees. For `fsn`,
whose section today names `InputError` and `NotSupportedError`:

```python
def fsn(x, vi=None, *, ...) -> FsnResult:
    """...

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, ...
        metafopy.errors.NotSupportedError: `x` is a model other than an `rma_uni` result, ...
        metafopy.errors.ArgumentError: <when `check_class` raises it, read from its body>
        metafopy.errors.EstimationError: <when the inner `rma_uni` fit fails, read from the body>
    """
```

Where a callee declares an error that cannot fire on this call, the hit is
accepted, not copied into the section. `check_class` raises `ArgumentError`
only when given `must`, and `fsn` passes only `notav`, so that hit is likely
such a case; step 3 decides it from the code.

## Fix-or-accept policy

Accept markers are always scoped (`# auditme: accept[<rule>] <reason>`), on the
hit line or the line above; the reason says why the pattern is right here, not
that it pre-exists. A hit the rule gets wrong is accepted and the pattern is
raised in the auditme skill's `requests.md` (a shared-repository change, made
there, not in this repository).

**`src/audit/`** (the harness that runs R metafor as the oracle) is accepted by
rule, as the operator decided on 2026-10-31: each hit gets a scoped marker
whose reason is the one written for that rule in this harness. The exception
is a real defect, which is fixed:

- `silent-broad-except`: log or re-raise, or narrow the exception type; keep a
  broad handler only where a harness failure must become a reported finding,
  with a marker saying so;
- `get-container-default-then-method-call`: replace `.get(k, {})` chains with
  `.get(k) or {}`, or validate the YAML shape once on load;
- `inconsistent-error-translation`: translate `MissingFunctionError` one way.

`METAFOR_FUNCTIONS` is filled at run time (`src/audit/scripts.py`), so it is
accepted, not frozen as the `TODO.md` item proposed.

**`src/metafopy/` and `src/tests/`**, per rule:

| Rule | Default | Accept when |
|---|---|---|
| `raises-section-drift`, `missing-raises-section` | fix the section | the error cannot reach the caller (caught on every path) |
| `typing-deprecated-alias` | fix (`collections.abc.Callable`, ...) | never |
| `from-import`, `aliased-imports`, `in-function-import` | fix (project import rules) | an import cycle forces it |
| `literal-duplication` | accept, with the reasons already used in the project | the literal is the project's own constant: then extract it |
| `non-alphabetical-literal`, `non-alphabetical-dict-keys` | sort | the order is metafor's or the R fixture script's |
| `paired-sibling-helpers` | fix in `src/metafopy/` (one shared helper) | test modules each keep their own loader |
| `passthrough-shim` | call the target directly | the wrapper is a metafor-named public alias |
| `duplicate-call-sequence` | accept | the shared steps form a helper with one meaning |
| `redundant-sink-caller-plus-callee` | compute once | R computes it twice and the order changes the digits |
| `mutable-module-level-exported` | freeze (`tuple`, `frozenset`, `types.MappingProxyType`) | mutated by design: `printing.SETTINGS`, the caches `YICHECKS` and `LAST_EVALUATED` |
| `assert-never-missing` | add a final `else` that raises | the value is validated earlier, as existing markers say |
| `except-interpolation`, `reexport-in-non-init`, `duplicated-word`, `filesystem-envelope-gap` | judge each site | as the site requires |

A rule not listed, or added to the checker while this work runs, follows the
same test: fix unless the pattern is right here, and say why in the marker.

## Implementation

Work proceeds in this order; each step is one or more commits, and each
commit passes the `auditme-changed` hook.

**The hook's scope** (amended 2026-11-01). The hook reports every
unaccepted hit in a file changed on the branch since `main`, not only in the
commit's own files, and every hit whose `related` files (the other copies of
a duplicated literal, the callee of a `Raises:` drift) include such a file.
A file stays in scope for every later commit once it has changed. So a
commit cannot fix one rule in a file and leave its other rules for later:
it clears every rule in every file in scope, including the files its own
fixes pull in. Task 4 of the first task list stopped on this on 2026-11-01
(72 hits in scope); the operator chose to clear by package.

1. **Commit hook.** Run `/project-setup adopt`: add the
   `git@github.com:bnomis/agents.git` hook block with `auditme-changed`
   (commit stage) and `auditme-all` (manual stage), pinned at the shared
   repository's current `main`, then `pre-commit install --hook-type
   pre-commit --hook-type pre-push`. `project-setup` files a `TODO.md` item
   when hits remain; this spec's item already covers that, so add the dated
   count to it instead of filing a second one.
2. **`src/audit/`**: fix its real defects, then accept the rest by rule.
3. **By package, every rule**: `src/metafopy/methods/` (with the files its
   fixes pull in), then `models`, `plot`, `bias`, `effects` and `core`, the
   remaining modules of `src/metafopy/`, and `src/tests/`. Each step clears
   every hit the hook reports, by the policy table above. A `Raises:`
   section is corrected from the function body and its callees, not from
   the hit's wording; the API reference built by mkdocs shows the result.
4. **Sweep**: whatever the whole-tree check still reports.
5. **Gates.** With zero unaccepted hits, add `stages: [manual, pre-push]` to
   `auditme-all`, add the CI step from `project-setup`'s
   `templates/ci-audit-step.yml` to the `lint` job of `ci.yml`, and run its
   deploy-key steps (each confirmed by the operator).
6. **Close.** Move the `TODO.md` item to `DONE.md` (created by this step) with
   a `Done:` line naming this spec and its task list; set this spec's status
   to `implemented`.

Each step's Definition of Done: `ruff format`, `ruff check`, `mypy` and
`pytest` pass, and its commit passes the `auditme-changed` hook.

## Error handling

- **A fix changes behaviour.** A test that fails after a refactor (a frozen
  container, a removed shim, a computation done once) means the fix was
  wrong: revert it and accept the hit with the reason the test shows.
- **The checker changes during the work.** The hooks are pinned to one
  agents revision; a newer checker is adopted only by bumping the pin, and
  its new hits are the catch-up surface of that bump, in the same commit or
  the next.
- **`lockfile-drift`.** On 2026-10-31 it fires only locally: the operator's
  global `uv.toml` sets `exclude-newer = "14 days"`, and `wcwidth` 0.9.3 is
  newer than that window, while CI's `uv sync --locked` passes. It clears
  when the release ages past the window. If it still blocks step 5, raise
  the environment dependence in the auditme `requests.md` rather than
  rewriting `uv.lock`, which `AGENTS.md` forbids without instruction.
- **No access to the shared repository.** `project-setup` stops if
  `git ls-remote` fails; CI needs the `AGENTS_DEPLOY_KEY` secret before the
  gate is added.

## Settings

| Setting | Type | Default | Description |
|---|---|---|---|
| `.pre-commit-config.yaml` agents `rev` | commit hash | shared `main` at step 1 | the checker version every hook and CI runs |
| `auditme-all` `stages` | list | `[manual]`, then `[manual, pre-push]` at step 5 | when the whole-tree sweep runs |
| `AGENTS_DEPLOY_KEY` | GitHub secret | none until step 5 | read-only key CI uses to fetch the shared hooks |

## External dependencies

- The shared agents repository (`git@github.com:bnomis/agents.git`): the
  auditme checker and the `auditme-changed` / `auditme-all` hooks. Check it
  is reachable with `git ls-remote git@github.com:bnomis/agents.git
  refs/heads/main`.
- `pre-commit`, already used for the ruff hooks.
