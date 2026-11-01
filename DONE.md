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
