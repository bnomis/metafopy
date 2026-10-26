# Tasks: Interactive charts

**Spec:** `agents/specs/interactive/spec.md`
**Created:** 2026-10-26
**Version:** 1

The package adapts the forest and funnel charts built by hand in
`docs/notebooks/showcase.ipynb`, and sits beside the matplotlib `forest()`
and `funnel()` in `src/metafopy/plot/`; Task 5 carries the deduplication
subtask that pairing calls for.

Every command below runs after `source bin/activate.sh`, per `AGENTS.md`.
This checkout had no `.venv` on 2026-10-26, and the tests need the
`interactive` extra, so Task 1 builds the environment with it.

---

## Task 1: `ChartData` and `chart_data()`

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Module location; API (`ChartData`, `chart_data`); Implementation; Error handling

**Technical Goal:** Create `metafopy.interactive` with the frames both charts draw from, computed only from public fit fields, so every number a chart shows is one the R fixtures already pin.

**Subtasks:**

- [x] Run `uv sync --extra interactive` and `pre-commit install` (the hooks were not installed on 2026-10-26)
- [x] Create `src/metafopy/interactive/__init__.py` with `__all__ = ['forest_chart', 'funnel_chart', 'chart_data', 'ChartData']`, importing Altair first and raising `ImportError` naming the extra (`pip install "metafopy[interactive] @ git+https://github.com/bnomis/metafopy"`) when it is missing
- [x] Do not import the package from `src/metafopy/__init__.py`
- [x] Write `data.py`: the frozen `ChartData` dataclass with the `studies` columns `slab, yi, vi, sei, ci_lb, ci_ub, weight, order, href, group, fill` and the `pooled` columns `label, estimate, ci_lb, ci_ub, pi_lb, pi_ub, k, tau2, I2`
- [x] Implement `chart_data()` for `RMAResult` and `RMAMVResult`: per-study intervals `yi +/- z(level) * sqrt(vi)` passed through `transf`; weights from `weights_()` as percentages; pooled rows from `predict(x, transf=transf)` and then one per `group_fits` entry in mapping order
- [x] Omit studies with missing `yi` or `vi`, indexing `slab`, `href` and `groups` before omission
- [x] Raise `metafopy.errors.InputError` for a wrong-length `slab`, `href` or `groups`, for `group_fits` without `groups`, and for a fit with moderators
- [x] Add a validator used by both chart functions: a hand-built `ChartData` missing a column raises `InputError` naming it
- [x] Write `src/tests/test_interactive.py` (skipped with `pytest.importorskip('altair')`) covering each of the above on `dat.bcg`

**Acceptance Criteria:**

- [x] On `dat.bcg` with `transf=numpy.exp`, the pooled row equals `predict(fit, transf=numpy.exp)` and `weight` equals `weights_()`
- [x] `import metafopy` still works without Altair installed (checked by asserting `'altair' not in sys.modules` after a fresh `import metafopy` in a subprocess)
- [x] Each `InputError` case is raised, with the message naming the offending argument or column

**Definition of Done:**

- [x] `pre-commit run --all-files` passes
- [x] `mypy` passes
- [x] `pytest src/tests/test_interactive.py -m 'not slow'` passes

**Notes:**

- Key files: `src/metafopy/interactive/`, `src/metafopy/models/mv_result.py` (`RMAMVResult`), `src/metafopy/methods/predict.py`
- If Altair or vl-convert lack type information, add them to `[[tool.mypy.overrides]]` `ignore_missing_imports` in `pyproject.toml`, as `formulaic` is
- No leading-underscore names and full-path imports (`claude/rules/metafopy.md`, `AGENTS.md`)

---

## Task 2: `forest_chart()`

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`forest_chart`); Implementation

**Technical Goal:** An interactive forest plot from a fit or a `ChartData`, with the showcase notebook's marks and interactions.

**Subtasks:**

- [x] Write `forest.py`: accept a fit (forwarding the remaining keywords to `chart_data()`) or a `ChartData`
- [x] Draw each study as a `mark_rule` from `ci_lb` to `ci_ub` and a `mark_square` sized by `weight`; each pooled row as a diamond with its prediction interval as a thin rule
- [x] Add a tooltip per mark (estimate, interval, weight), the `selection_point` on `slab`, and an `href` encoding on study marks, leaving `None` entries unlinked
- [x] Order rows by `order` (`'input'`, `'obs'`, `'weight'`); draw `refline` on the transformed scale, none when `None`
- [x] Inline the data with `altair.data_transformers.disable_max_rows()` scoped to the call, and use fixed mark colours, not a theme
- [x] Extend `src/tests/test_interactive.py`

**Acceptance Criteria:**

- [x] `forest_chart(fit).to_dict()` validates and contains no data URL
- [x] `href` values reach the study marks; `None` entries carry none
- [x] A multilevel fit (`dat.konstantopoulos2011`, `random='~ 1 | district/school'`) draws a forest
- [x] A forest with `group_fits` has one pooled row per group after the overall one
- [x] `vl_convert.vegalite_to_svg()` renders the chart without error
- [x] Altair's global data-transformer setting is unchanged after the call

**Definition of Done:**

- [x] `pre-commit run --all-files` passes
- [x] `mypy` passes
- [x] `pytest src/tests/test_interactive.py -m 'not slow'` passes

**Notes:**

- Depends on Task 1
- Model: the forest cells of `docs/notebooks/showcase.ipynb` (cell 12)

---

## Task 3: `funnel_chart()`

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`funnel_chart`); Error handling

**Technical Goal:** An interactive funnel plot of a single-level model, drawn from a fit, a `TrimfillResult` or a hand-built `ChartData`.

**Subtasks:**

- [ ] Write `funnel.py`: the pseudo-confidence region `estimate +/- z(level) * sei` with the y axis reversed, as metafor's `funnel()` draws it; `yaxis` `'sei'` or `'vi'`
- [ ] Draw a `TrimfillResult`'s imputed studies (`fill`) as open circles after the observed ones
- [ ] For a `ChartData`, draw studies on the untransformed `yi` scale against the first pooled row's estimate, with its `fill` column marking imputed studies; raise `InputError` when it has no pooled row
- [ ] Raise `NotSupportedError` for a multilevel fit
- [ ] Extend `src/tests/test_interactive.py`

**Acceptance Criteria:**

- [ ] `funnel_chart(fit).to_dict()` validates; `trimfill(fit)` adds open-circle marks for exactly its imputed studies
- [ ] A `ChartData` built by hand from a fit's numbers gives the same region as the fit itself
- [ ] A multilevel fit raises `NotSupportedError`; a `ChartData` with an empty `pooled` raises `InputError`
- [ ] `vl_convert.vegalite_to_svg()` renders the chart without error

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_interactive.py -m 'not slow'` passes

**Notes:**

- Depends on Task 1
- Model: the funnel cells of `docs/notebooks/showcase.ipynb` (cells 12 and 23)

---

## Task 4: The showcase notebook calls the library

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Module location (notebook rewritten)

**Technical Goal:** Replace the notebook's hand-built forest and funnel cells with calls to `metafopy.interactive`, keeping the estimator radio button and leave-one-out dropdown in the notebook, as the spec decides.

**Subtasks:**

- [ ] Rewrite the forest and funnel cells to call `forest_chart()`/`funnel_chart()` and compose the notebook's own linked controls around them
- [ ] Re-execute the notebook so its stored outputs match the code

**Acceptance Criteria:**

- [ ] The notebook imports `metafopy.interactive` and defines no forest or funnel mark of its own
- [ ] `pytest src/tests/test_showcase_notebook.py` passes (at least five `vega` HTML outputs)

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_interactive.py src/tests/test_showcase_notebook.py` passes (the notebook test is marked `slow`, so it is named here rather than skipped)

**Notes:**

- Depends on Tasks 2 and 3

---

## Task 5: Documentation, deduplication and closing the spec

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Documentation; Tests (docs examples)

**Technical Goal:** Document the package where users look for plots, check it for copies of what `plot/` already owns, and mark the spec implemented.

**Subtasks:**

- [ ] Add an "Interactive charts" section to `docs/guide/plots.md` with a runnable example (covered by `test_docs_examples.py`)
- [ ] Name `metafopy.interactive` in `docs/install.md`'s interactive section, and correct the notebook's last cell's `pip install "metafopy[interactive]"` to the git-URL form `docs/install.md` gives
- [ ] Add `docs/reference/interactive.md` beside `docs/reference/plots.md`, using the same mkdocstrings form, and list it under "Function reference" in `mkdocs.yml`
- [ ] Add a `CHANGELOG.md` entry
- [ ] Hoist anything this package duplicates from `src/metafopy/plot/` (`forest.py`, `funnel.py`, `rutils.py`) or the notebook to a single owner: grep the tree for each literal collection, constant and one-line helper the new modules define (the z quantile for `level`, interval arithmetic, default labels), and import rather than copy. Choose the owner from the existing import graph; prefer a module `interactive` already imports over a new edge
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to match

**Acceptance Criteria:**

- [ ] `pytest src/tests/test_docs_examples.py` runs the new `plots.md` example
- [ ] `mkdocs build --strict` succeeds with the new reference page
- [ ] No helper or constant in `src/metafopy/interactive/` has a copy elsewhere in `src/metafopy/`

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_interactive.py src/tests/test_docs_examples.py src/tests/test_showcase_notebook.py` passes (the last task's run: every test that imports `metafopy.interactive` or runs its examples)

**Notes:**

- Depends on Tasks 1-4
- `mkdocs build` needs `uv pip install -r docs/requirements.txt` (README "Building the docs")
- LitRev's `report/html.md` spec waits on this list: once it is merged and pushed, LitRev's `[tool.uv.sources]` pin moves to its commit

---
