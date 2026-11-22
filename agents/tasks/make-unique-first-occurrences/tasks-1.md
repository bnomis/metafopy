# Tasks: make_unique keeps every first occurrence

**Source:** `TODO.md`, "`make_unique` renames a label that occurs once when it collides with a generated one"
**Created:** 2026-11-22
**Version:** 1

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript -e '<code>'`. Worktrees have no `.audit/Rlib`, so use the main
  checkout's.

The fix and its output change were approved on 2026-11-22 (item 3 of `agents/reviews/order-r-parity-fixes-from-property-tests-specs.md`, Answers section). That approval covers only the change the
item describes. Any other change to metafopy's output stops the task as `blocked`, naming the function, the input and both values
(`claude/rules/metafopy.md`).

---

## Task 1: Make `make_unique` avoid every input label, as R's `make.unique` does

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** `TODO.md`, "`make_unique` renames a label that occurs once when it collides with a generated one"; review item 3

**Technical Goal:** a label the caller gave keeps its first occurrence, and a generated suffix never takes it, as in R's `do_makeunique`.

**Subtasks:**

- [x] Get R's `make.unique(x)` and `make.names(x, unique = TRUE)` for these inputs, saved to `<scratch>/r.json` (`<scratch>` is outside the
      repository):
      `c("a","a","a.1")`, `c("a","a.1","a")`, `c("a","a","a","a.2")`, `c("b","b.1","b","b.1")`, `c("x","x","x.1","x.1")`, `c("1","1")` and
      `c("a b","a b","a.b")`
- [x] In `src/metafopy/metadata.py::make_unique`, keep each first occurrence. Seed the set of labels to avoid with every input label, then
      give each later duplicate `label.k` with the smallest `k` (counting on from that label's last `k`) not in the set, and add it. This is
      R's `do_makeunique` in `src/main/unique.c`
- [x] Compare `make_unique` and `metafopy.effects.datatools.make_names(..., unique=True)` with `<scratch>/r.json`. A difference stops the
      task as `blocked`, naming the input and both outputs
- [x] Remove the strict `xfail` marker of `src/tests/test_name_matching_properties.py::test_make_unique_keeps_first_occurrences`. Add the
      R cases above as an example test of both functions in `src/tests/test_name_matching_properties.py`
- [x] Run `src/tests/test_data.py` and `src/tests/test_datatools.py`. A test whose expected study labels (`slab`) change stops the task as
      `blocked`, naming the test and both label lists, unless the new labels equal R's for the same input

**Acceptance Criteria:**

- [x] Both functions give R's output on every case above, and the property test passes without `xfail`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/metadata.py src/tests/test_name_matching_properties.py` passes
- [x] `pytest src/tests/test_name_matching_properties.py src/tests/test_data.py src/tests/test_datatools.py` passes

**Notes:**

- `metafopy.effects.measures.make_unique_dot1` copies metafor's own `.make.unique`, which renames singletons too. It is a different function
  and stays as it is
- Callers: `src/metafopy/metadata.py:359` (study labels) and `src/metafopy/effects/datatools.py:82` (at `ea53581`)

---

## Task 2: Full check and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** `TODO.md`, "`make_unique` renames a label that occurs once when it collides with a generated one"

**Technical Goal:** the branch passes every check once, and the `TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the `TODO.md` item "`make_unique` renames a label that occurs once when it collides with a generated one" to the end of `DONE.md`: append its heading and body unchanged, then the line
      `Done: <date> - agents/tasks/make-unique-first-occurrences/tasks-1.md Task 1 (<commit>); approved in agents/reviews/order-r-parity-fixes-from-property-tests-specs.md item 3.`, and remove it from `TODO.md`
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/make-unique-first-occurrences/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/make-unique-first-occurrences/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged, and the final run passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/make-unique-first-occurrences/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 1
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on

---

## Task 3: Blocked finding - `make_names(unique=True)` orders changed names last, as R does

**Status:** done
**Priority:** high
**Complexity:** small

Found in Task 1 (2026-11-22). `make_unique` now equals R's `make.unique` on every Task 1 case. `metafopy.effects.datatools.make_names(..., unique=True)` differs from R's
`make.names(x, unique = TRUE)` on one input:

- input `["a b", "a b", "a.b"]`
- R: `a.b.1`, `a.b.2`, `a.b`
- metafopy: `a.b`, `a.b.1`, `a.b.2`

R's `make.names` with `unique = TRUE` runs `make.unique` over the names with those that `make.names` changed placed last, so an unchanged name keeps its label. Changing this alters
metafopy's output beyond what review item 3 approved, so it needs the operator's approval before Task 1 can finish (the Task 1 subtask that compares both functions with R).

**Operator answer (2026-11-22, in the operator session):** leave `make_names(unique=True)` as it is. Do not change it to R's order. The
`make_unique` fix finishes alone.

**Subtasks:**

- [x] File the difference in `TODO.md` with `/todo-add`. Place it after the item this list closes, with today's date, the input, R's output,
      metafopy's output and the cause (R's `make.names` body: `o <- order(names != names2); names2[o] <- make.unique(names2[o])`, checked
      in `Rscript` on 2026-11-22). Its Fix direction says that matching R changes an output and needs the operator's confirmation
- [x] Add a strict `xfail` test of that input in `src/tests/test_name_matching_properties.py`, asserting R's output, with a reason naming the
      `TODO.md` item
- [x] Task 1's subtask that compares both functions with R is then met: `make_unique` matches R on every case, and the one `make_names`
      difference is filed with its strict `xfail`. Tick it, and set Task 1 to `done`

**Acceptance Criteria:**

- [x] The `TODO.md` item and the strict `xfail` exist, and `pytest src/tests/test_name_matching_properties.py` passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/tests/test_name_matching_properties.py` passes
- [x] `pytest src/tests/test_name_matching_properties.py src/tests/test_datatools.py` passes
