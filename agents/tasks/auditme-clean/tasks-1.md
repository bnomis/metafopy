# Tasks: Auditme clean

**Spec:** `agents/specs/auditme-clean/spec.md`
**Created:** 2026-10-31
**Version:** 1

The spec was approved on its review page on 2026-10-31 (every item, answers
in `agents/specs/auditme-clean/review.md`). Every command below runs after
`source bin/activate.sh`, per `AGENTS.md`.

From Task 1 on, every commit runs the `auditme-changed` hook. Its scope is
wider than the commit (spec, Implementation, amended 2026-11-01): every
unaccepted hit in a file changed on the branch since `main`, and every hit
whose `related` files include one. So "the task's commit passes its
pre-commit hooks" means every rule is clear in every such file, including
files a fix pulls in. List the remaining hits at any time with
`pre-commit run auditme-all --hook-stage manual --all-files`.

**Clearing a package** (Tasks 4 to 9, amended 2026-11-01). Fix or accept
each hit by the spec's policy table. The scope check, run before each
commit with the changes staged, is the hook itself:
`pre-commit run auditme-changed`. Clear what it reports and run it again
until it passes. The checker reads a marker only on the hit line or the
line above it, one per line; for a decorated `def` the marker goes between
the last decorator and the `def`. Where a line takes hits from two rules,
or the line above already holds a marker, remove one cause in code (derive
a tuple from an existing type alias, add an alias) rather than exceed 160
columns.

**Accept markers** are always scoped, `# auditme: accept[<rule>] <reason>`,
on the hit line or the line above, with a reason that says why the pattern
is right here (spec, "Fix-or-accept policy"). Where a rule gets a site
wrong, accept it and add a line to `agents/specs/auditme-clean/requests.md`
(rule, `path:line`, why the rule is wrong); the operator raises those in the
shared auditme skill's `requests.md`, which no task here edits.

**Tests.** The project has no affected-tests tool. Each task names its test
files; for a task whose sites are found while it runs, the test files are
those that import each changed module, found with
`grep -rlE 'metafopy\.<package>\.<module>\b|audit\.<module>\b' src/tests`,
plus the module's own `test_<module>.py` where it exists. Record the files
run in the commit body.

---

## Task 1: Add the auditme commit hook

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 1; Settings

**Technical Goal:** Every commit from here on is checked by the shared
auditme checker on its changed files, so the backlog cannot grow while it is
worked down.

**Subtasks:**

- [x] Read the shared repository's current `main`: `git ls-remote git@github.com:bnomis/agents.git refs/heads/main` (first field). If it fails, stop as `blocked`: the hooks cannot be pinned
- [x] Add to `.pre-commit-config.yaml`, after the ruff block, the repo block `git@github.com:bnomis/agents.git` at that `rev`, with hooks `auditme-changed` and `auditme-all` with `stages: [manual]` (not `pre-push` yet), under the comment `# The shared audit hooks, pinned to one commit of the checker.`
- [x] Run `pre-commit install --hook-type pre-commit --hook-type pre-push`
- [x] Run `pre-commit run auditme-all --hook-stage manual --all-files` and count the unaccepted hits by rule
- [x] Add one line after the `Spec:` line of the `TODO.md` item "The auditme checker reports unaccepted hits and no hook enforces it": `Note YYYY-MM-DD: with the hooks pinned at agents <rev>, the whole-tree sweep reports <N> unaccepted hits (<rule>: <n>, ...).` Never edit the dated findings

**Acceptance Criteria:**

- [x] `.pre-commit-config.yaml` pins both hooks at one full commit hash; `auditme-all` has `stages: [manual]` only
- [x] `pre-commit run auditme-changed --files TODO.md` runs the checker (passes or reports; it does not error on setup)
- [x] The `TODO.md` note records the date, the rev and the count by rule

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` runs to completion (its hit count is the note's)

**Notes:**

- If the pinned rev is newer than agents `108559d` (the spec's measurement), its new rules add hits; they follow the spec's policy like the rest (spec, "Error handling: The checker changes during the work")
- `project-setup adopt` would file a `TODO.md` item for the backlog; this item already covers it, so only the note is added

---

## Task 2: Fix the real defects in `src/audit/`

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Fix-or-accept policy (`src/audit/`)

**Technical Goal:** The harness's silent broad handlers, null-unsafe `.get`
chains and inconsistent error translation are fixed, not accepted, as the
operator decided on 2026-10-31.

**Subtasks:**

- [x] `silent-broad-except` (`src/audit/pyrun.py`, `rlib.py`, `valuetree.py`): narrow each handler to the exceptions the call can raise, or log and re-raise; where a harness failure must become a reported finding (as `pyrun.py:238` says), keep the handler, make it record the exception type and message in the finding, and accept with a marker saying so
- [x] `get-container-default-then-method-call` (`src/audit/domains.py`): replace `x.get(k, {}).get(...)` with `(x.get(k) or {}).get(...)`, or validate the YAML shape once where it is loaded
- [x] `inconsistent-error-translation` (`src/audit/pycall.py:274`): callers of `resolve_function` translate `MissingFunctionError` to one error type; choose the one the reports already show for a missing function
- [x] Add a test for each changed handler and for an explicit `null` in the domains YAML, in the harness test module that covers the file

**Acceptance Criteria:**

- [x] No `silent-broad-except`, `get-container-default-then-method-call` or `inconsistent-error-translation` hit in `src/audit/` is unaccepted
- [x] A domains YAML entry set to `null` no longer raises `AttributeError`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_*.py` passes, plus any other test file that imports a changed `audit` module

**Notes:**

- Depends on Task 1
- `bin/metafor-audit.sh --recall` checks the harness still reproduces earlier audits; it needs R. Run it if R is installed and record the result; otherwise say in the commit body that it was not run

---

## Task 3: Accept the remaining `src/audit/` hits by rule

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Fix-or-accept policy (`src/audit/`)

**Technical Goal:** Every other unaccepted hit in `src/audit/` carries a
scoped marker with one reason per rule, written for this harness.

**Subtasks:**

- [x] List the unaccepted `src/audit/` hits by rule from the whole-tree sweep
- [x] For each rule, write one reason that holds for every site in the harness (for example `literal-duplication`: "the harness spells out each R call it translates"), and add it as a scoped marker at each site
- [x] `METAFOR_FUNCTIONS` (`src/audit/rlib.py:46`): accept `mutable-module-level-exported` with "filled at run time from the R script's function list (`src/audit/scripts.py`)"
- [x] A site where no rule-wide reason is true is a defect: fix it like Task 2 and add a test

**Acceptance Criteria:**

- [x] The whole-tree sweep reports no unaccepted hit under `src/audit/`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_*.py` passes

**Notes:**

- Depends on Task 2

---

## Task 4: Clear every rule in `metafopy.methods` and the files it pulls in

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`Raises:` form); Fix-or-accept policy; Implementation step 3 (amended 2026-11-01)

**Technical Goal:** Each `Raises:` section in `src/metafopy/methods/` names
every domain error that can reach the caller, and every rule is clear in
every file the commit hook brings into scope (header, "Clearing a package").

**Subtasks:**

- [x] For each `raises-section-drift` and `missing-raises-section` hit in `src/metafopy/methods/`, read the function body and the callee the hit names, and decide whether the error can reach the caller on that call
- [x] Where it can, add it to the section in the project's form (`metafopy.errors.<Name>: <condition>`), with the condition read from the code, not the hit text
- [x] Where it cannot (the call never takes the raising path, as `check_class` without `must`), accept with a marker naming why, and add the pattern to `agents/specs/auditme-clean/requests.md`
- [x] Clear every other hit the hook reports, by the policy table: the `methods` files' other rules, the callers in `models/`, `plot/`, `bias/` and `core/` whose `Raises:` sections drift from the new entries, and the files those fixes pull in in turn

**Acceptance Criteria:**

- [x] No unaccepted `raises-section-drift` or `missing-raises-section` hit in `src/metafopy/methods/`
- [x] `pre-commit run auditme-changed` passes with the task's changes staged
- [x] No public function changes signature

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `mkdocs build --strict` passes (needs `uv pip install -r docs/requirements.txt`, README "Building the docs")
- [x] `pytest -m 'not slow'` on the test files that import each module whose code changed, and those named after them (list in the commit body)

**Notes:**

- Depends on Task 1
- Stopped 2026-11-01 because the hook's scope is wider than the commit (spec, Implementation, amended 2026-11-01); the operator chose to clear by package, and Tasks 5 to 9 were rewritten to match

---

## Task 5: Clear every rule in `metafopy.models`

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 3 (amended 2026-11-01); Fix-or-accept policy

**Technical Goal:** No unaccepted hit, under any rule, in
`src/metafopy/models/`, nor in any file the commit hook pulls in.

**Subtasks:**

- [x] List the hits: run `python3 ~/.claude/skills/auditme/checks/audit.py . > <scratch>/audit.jsonl` and take every unaccepted hit whose `path` is under `src/metafopy/models/`
- [x] Fix or accept each by the spec's policy table (see "Clearing a package" in the header)
- [x] Before committing, run the scope check in the header and clear what it adds, until it reports nothing
- [x] Commit; the hook must pass

**Acceptance Criteria:**

- [x] No unaccepted hit under `src/metafopy/models/`
- [x] No public function changes signature

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `mkdocs build --strict` passes
- [x] `pytest` on the test files that import each module whose code (not only docstrings or markers) changed (see the header)

**Notes:**

- Depends on Task 4

---

## Task 6: Clear every rule in `metafopy.plot`

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** as Task 5

**Technical Goal:** As Task 5, for `src/metafopy/plot/`.

**Subtasks:**

- [x] As Task 5's subtasks, for `src/metafopy/plot/`

**Acceptance Criteria:**

- [x] No unaccepted hit under `src/metafopy/plot/`
- [x] No public function changes signature

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `mkdocs build --strict` passes
- [x] `pytest` on the test files that import each module whose code changed

**Notes:**

- Depends on Task 5

---

## Task 7: Clear every rule in the rest of `src/metafopy/`

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** as Task 5

**Technical Goal:** As Task 5, for `bias/`, `effects/`, `core/` and every
other module of `src/metafopy/` (`datasets/`, `interactive/`, the top-level
modules), one package per commit.

**Subtasks:**

- [x] As Task 5's subtasks, package by package, committing each separately (the `fsn` hit at `src/metafopy/bias/fsn.py:159` is the spec's `Raises:` example, if it is still open)
- [x] `redundant-sink-caller-plus-callee` (`src/metafopy/effects/fivenum_estimators.py` and others): compute the value once and pass it, unless the numbers then differ from R's in the compared digits; then revert and accept saying so

**Acceptance Criteria:**

- [x] No unaccepted hit under `src/metafopy/`
- [x] No public function changes signature

**Definition of Done:**

- [x] Each commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `mkdocs build --strict` passes
- [x] `pytest` on the test files that import each module whose code changed

**Notes:**

- Depends on Task 6

---

## Task 8: Clear every rule in `src/tests/`

**Status:** todo
**Priority:** medium
**Complexity:** large
**Spec Reference:** as Task 5

**Technical Goal:** As Task 5, for `src/tests/`.

**Subtasks:**

- [ ] As Task 5's subtasks, for `src/tests/`; test modules keep their own fixture loaders (`paired-sibling-helpers`: accept with "each test module keeps its own fixture loader"), and literals that mirror the R fixture scripts are accepted with the reasons the project already uses

**Acceptance Criteria:**

- [ ] No unaccepted hit under `src/tests/`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest` on every test file the task edits

**Notes:**

- Depends on Task 7

---

## Task 9: Sweep the whole tree

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Implementation step 4 (amended 2026-11-01)

**Technical Goal:** The whole-tree check reports no unaccepted hit, except
`lockfile-drift` if it still fires (Task 10).

**Subtasks:**

- [ ] Run `pre-commit run auditme-all --hook-stage manual --all-files`; fix or accept every remaining hit by the policy table, including any rule not in the table (fix unless the pattern is right here, and say why in the marker)

**Acceptance Criteria:**

- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` reports nothing but `lockfile-drift`, if that

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`), or there is nothing to commit
- [ ] `mypy` passes
- [ ] `pytest` on the test files that import each module whose code changed

**Notes:**

- Depends on Task 8
- A test that fails after a refactor means the fix was wrong: revert it and accept with the reason the test shows (spec, "Error handling")

---

## Task 10: Switch on the push and CI gates

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 6; Error handling (`lockfile-drift`, shared-repository access); Settings

**Technical Goal:** With zero unaccepted hits, the whole-tree sweep runs on
every push and in CI.

**Subtasks:**

- [ ] Run `pre-commit run auditme-all --hook-stage manual --all-files`. If any hit other than `lockfile-drift` is unaccepted, stop as `blocked` and name it. If `lockfile-drift` fires, run `uv lock --check`: if it still fails only because of a global `exclude-newer`, stop as `blocked` for the operator to raise it with auditme (the spec forbids rewriting `uv.lock`)
- [ ] Set `stages: [manual, pre-push]` under `- id: auditme-all` in `.pre-commit-config.yaml`
- [ ] Add the steps of the shared `project-setup` template `ci-audit-step.yml` (install the deploy key, `pre-commit run auditme-all --hook-stage manual --all-files`, remove the key) to the `lint` job of `.github/workflows/ci.yml` after `uv run mypy`, with the prerequisites it lists: `uv tool install pre-commit` and `echo "$PWD/.venv/bin" >> "$GITHUB_PATH"` after `uv sync`
- [ ] Commit, then stop as `blocked` for an operator session to create the deploy key and secret (each step confirmed): `ssh-keygen -t ed25519 -N '' -f <scratch>/agents_deploy_key`, `gh repo deploy-key add <scratch>/agents_deploy_key.pub --repo bnomis/agents --title metafopy`, `gh secret set AGENTS_DEPLOY_KEY --repo <owner>/metafopy < <scratch>/agents_deploy_key`, then delete the scratch files. The key is never printed. The operator then sets this task to `done` and restarts the list

**Acceptance Criteria:**

- [ ] `git push` runs `auditme-all` (seen in the push output)
- [ ] The CI `lint` job runs the audit step and passes, once the secret exists

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` exits 0

**Notes:**

- Depends on Task 9
- The deploy key and secret change GitHub, so an unattended run never does them

---

## Task 11: Close the item and the spec

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Implementation step 7

**Technical Goal:** The finished work is recorded and the whole tree is
checked once before the branch merges.

**Subtasks:**

- [ ] Create `DONE.md` with the title `# Done` and the line "Finished work, newest last. Items arrive from `TODO.md` unchanged."
- [ ] Append the `TODO.md` item "The auditme checker reports unaccepted hits and no hook enforces it" to `DONE.md`, heading and body unchanged, then the line `Done: YYYY-MM-DD - agents/specs/auditme-clean/spec.md, agents/tasks/auditme-clean/tasks-1.md.`, and remove it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its entry in `agents/specs/index.md` to "Implemented: ..."
- [ ] Add `agents/specs/auditme-clean/requests.md`'s entries, if any, to the hand-over message for the operator

**Acceptance Criteria:**

- [ ] `TODO.md` no longer holds the item; `DONE.md` holds it unchanged with a `Done:` line

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` exits 0
- [ ] `mypy` passes
- [ ] `python3 ~/.claude/skills/todo-review/verify_bodies.py <baseline> TODO.md DONE.md` exits 0, where `<baseline>` is `TODO.md` as of the commit that added this list: `git show $(git log --diff-filter=A --format=%H -- agents/tasks/auditme-clean/tasks-1.md):TODO.md > <scratch>/todo-before.md`
- [ ] `pytest` on the test files that import any module this list changed: `git diff --name-only <base>..HEAD -- src` gives the modules, and the header's grep their importers

**Notes:**

- Depends on Task 10
- The last test run may approach the whole suite, since the list touches most packages; that is expected
