# Tasks: Docs strict-mode links to docs/differences.md

**Source:** TODO.md, "The Docs workflow fails on three docstring links to docs/differences.md"
**Created:** 2026-11-17
**Version:** 1

Row 1 of the run order `agents/tasks/order-todo-items-1-10.md`; it waits for nothing and has no operator stop. The Docs workflow
(`.github/workflows/docs.yml`) builds the site with `mkdocs build --strict` on pull requests and `mkdocs gh-deploy --strict` on `main`, and strict mode
aborts on three docstring links that point at `../../../docs/differences.md#...`, a path relative to the source file. mkdocstrings renders each
docstring inside a page under `docs/reference/` (`reference/methods.md`, `reference/models.md`, `reference/bias.md`), and mkdocs resolves a relative
link from that page, so the target the site holds (`differences.md`, listed in `mkdocs.yml` under "Differences from metafor") is `../differences.md`.

The links, as found on 2026-11-17:

- `src/metafopy/methods/confint.py`, the `type` argument of `confint`: `#where-metafor-departs-from-its-reference`
- `src/metafopy/models/selmodel.py`, the truncation-point paragraph of the `selmodel` docstring: `#where-metafor-departs-from-its-reference`
- `src/metafopy/bias/fsn.py`, the Fisher-pooling paragraph of the `fsn` docstring: `#fsn-fisher-pooling-and-the-general-type`

Both anchors match headings in `docs/differences.md` ("Where metafor departs from its reference" and "fsn(): Fisher pooling and the General type").
Every other docstring that points at the page names it in plain text (``` ``docs/differences.md`` ```, often with the section's title); these three
are the only Markdown links from `src/metafopy` to a docs page.

The local strict build mirrors the workflow's own install and build steps, in a virtual environment outside the repository so the project's `.venv`
and `uv.lock` are untouched:

```bash
uv venv --python 3.14.8 "$TMPDIR/metafopy-docs-venv"
uv pip install --python "$TMPDIR/metafopy-docs-venv/bin/python" -e '.[interactive]' -r docs/requirements.txt
"$TMPDIR/metafopy-docs-venv/bin/mkdocs" build --strict --site-dir "$TMPDIR/metafopy-docs-site"
```

(`site/` is not in `.gitignore`, so the build writes outside the repository.) The build fetches the Python, NumPy and pandas inventories named in
`mkdocs.yml` and executes `docs/notebooks/showcase.ipynb`, so it needs the network. The project has no affected-tests tool, so each task names its test
files. Only docstrings and a new test change; no metafopy result changes.

---

## Task 1: Rewrite the three links and pass the strict build

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's cause and fix direction (links as the site resolves them from a reference page; `mkdocs build --strict` locally)

**Technical Goal:** the three docstrings link to `docs/differences.md` by the path mkdocs resolves from a page under `docs/reference/`, and the docs
build the Docs workflow runs passes in strict mode.

**Subtasks:**

- [x] Check the starting point: `grep -rnF '../../../docs/differences.md' src/metafopy` lists exactly the three links in this list's header. If
      none is found, they were fixed elsewhere: go straight to the build subtask. If a link outside the three is found, rewrite it the same way and
      name it in this task's Notes
- [x] In each of `src/metafopy/methods/confint.py`, `src/metafopy/models/selmodel.py` and `src/metafopy/bias/fsn.py`, change the link target from
      `../../../docs/differences.md#<anchor>` to `../differences.md#<anchor>`, keeping the anchor and the link text `docs/differences.md` (which
      reads correctly in an editor's hover too). Change nothing else in the docstrings
- [x] Build the docs as the header says (create the virtual environment, install with the workflow's `uv pip install -e '.[interactive]' -r
      docs/requirements.txt`, then `mkdocs build --strict --site-dir "$TMPDIR/metafopy-docs-site"`), saving the whole output to
      `$TMPDIR/metafopy-docs-build.log`. If `uv venv` or the install fails (no network, Python 3.14.8 unavailable), stop as `blocked` with its
      error
- [x] If the build still warns about any of the three links, rewrite that link as plain text in the form the other docstrings use (``` ``docs/differences.md`` ```,
      "<section title>"), rebuild, and say so in this task's Notes. If it fails on a warning unrelated to these links, stop as `blocked` with the
      warning lines: that is not this item
- [x] Confirm in the built site that each link works: `grep -o 'href="[^"]*differences/#[^"]*"'` over `$TMPDIR/metafopy-docs-site/reference/methods/index.html`,
      `models/index.html` and `bias/index.html` shows the anchor, and the anchor is an `id` in `$TMPDIR/metafopy-docs-site/differences/index.html`.
      Record the result, dated, in this task's Notes

**Acceptance Criteria:**

- [x] `grep -rnF '../../../docs/' src/metafopy` finds nothing
- [x] `mkdocs build --strict` as above exits 0 and its log has no `WARNING` line
- [x] Each of the three rendered links points at an anchor that exists on the built differences page (or was rewritten as plain text, noted)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Key files: `src/metafopy/methods/confint.py`, `src/metafopy/models/selmodel.py`, `src/metafopy/bias/fsn.py`, `mkdocs.yml`,
  `.github/workflows/docs.yml`, `docs/differences.md`
- 2026-11-17: `mkdocs build --strict` exits 0. The log has two lines containing the word WARNING, neither from a link: the mkdocs 2 notice
  and the notebook kernel's TCP-encryption message (IPKernelApp); no mkdocs `WARNING -` line. The three rendered links are
  `../../differences/#where-metafor-departs-from-its-reference` (methods, models) and `../../differences/#fsn-fisher-pooling-and-the-general-type`
  (bias); both ids exist in `differences/index.html`. No link needed the plain-text fallback.
- The worktree has no `.venv`; mypy and pytest ran from the main checkout's `.venv` with the worktree's `src` on `PYTHONPATH`
- The virtual environment, build log and site stay under `$TMPDIR`, outside the repository; nothing from them is committed

---

## Task 2: A test that fails on a docstring link the site cannot resolve

**Status:** todo
**Priority:** medium
**Complexity:** small
**Source Reference:** the item's fix direction (catch such a link before CI)

**Technical Goal:** a later docstring link of the same kind fails the normal test run, which needs no docs install and no network, instead of the
Docs workflow after the push.

**Subtasks:**

- [ ] Add `src/tests/test_docstring_links.py`. It reads every `.py` file under `src/metafopy`, finds each Markdown link `[text](target)` whose target
      is not a URL (no scheme such as `https:`, `http:` or `mailto:`) and not a bare `#anchor`, and asserts for each that the target's path, resolved
      from `docs/reference/` (where mkdocstrings renders every docstring), is a file under `docs/`. Where the target has an `#anchor` and the file is
      Markdown, it also asserts the anchor is one of the file's heading slugs, computed as mkdocs' `toc` extension does by default (text of the heading
      without the leading hashes, characters other than word characters, hyphens and spaces removed, lowercased, spaces to hyphens). A failure names
      the source file, line and target
- [ ] Follow the project's conventions (`claude/rules/`): full module imports, no leading-underscore names; locate the repository root as
      `src/tests/docsblocks.py` does (`pathlib.Path(__file__).resolve().parents[2]`). Add a one-line module docstring saying why the test exists
      (strict mkdocs builds abort on such links) and the rule it checks
- [ ] Check the test both ways: it passes on this branch; with one link temporarily changed back to `../../../docs/differences.md#...` (not
      committed) it fails and names that link; and with a wrong anchor it fails. Record both checks, dated, in this task's Notes

**Acceptance Criteria:**

- [ ] `pytest src/tests/test_docstring_links.py` passes on the branch and fails on each of the two broken forms above
- [ ] The test runs with the project's `.venv` alone (no mkdocs import, no network)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_docstring_links.py` passes

**Notes:**

- Depends on Task 1
- The test checks only links from `src/metafopy` docstrings, the kind this item found; links inside `docs/*.md` are already checked by the strict
  build on every pull request that touches `docs/**`
- If Task 1 fell back to plain text for all three links, the test still runs and passes with nothing to check; keep it as the guard

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/docs-strict-links/tasks-1.md; <outcome>` with the date of the commit, where `<outcome>` is "the three docstring
      links now resolve from the reference pages (../differences.md#...), mkdocs build --strict passes locally, and
      src/tests/test_docstring_links.py guards the form" (adjusted if Task 1 fell back to plain text for any link)
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/docs-strict-links/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/docs-strict-links/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/docs-strict-links/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: only docstrings and a test change
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
