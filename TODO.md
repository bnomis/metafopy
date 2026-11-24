# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## CI fails three numerical tests that pass on a local macOS run

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
Found on 2026-11-24 in CI run 45270718992 (commit `93fea02`, Python 3.14.8, numpy 2.5.4, scipy 1.18.1 on every runner, as on the local
`.venv`), the first run whose test jobs got past `uv sync --locked`: every earlier run since at least `58c7418` (2026-11-23) stopped there,
until `93fea02` declared `exclude-newer` in `pyproject.toml`. Failing on ubuntu-26.04, macos-26 and windows-2025-vs2026:

- `src/tests/correctness/test_derived.py::test_derived[ls.vif.reestimate.ML.bcg2]`: `numpy.linalg.LinAlgError: Singular matrix`;
- `src/tests/test_docs_examples.py::test_page_examples_run[differences.md]`: `differences.md line 1442: InputError: No missing values
  allowed in 'alpha_init'.`

Failing on ubuntu-26.04 only: `test_derived[ls.vif.REML.bcg2]`, location vif `4.723135238933293` against the reference
`4.723144269851787` (tolerance `1e-06`).

On 2026-11-24, on macOS arm64 at `93fea02`, all three pass run alone, and the four `ls.vif` cases of `test_derived.py` pass run together.
<!-- doc-counts: accept-end -->

Cause: unconfirmed. Library versions match, and the macos-26 runner is arm64 like the local run, so a BLAS or CPU difference is not the
first suspect. CI runs the whole suite in one process with `uv run pytest --mpl --cov --cov-report=term-missing`; state left by an earlier
test, or the coverage tracer, are the suspects. All three cases fit a location-scale model on `dat.bcg`. The `differences.md` example
(around line 1442) is the one where R stops with "No missing values allowed in 'alpha.init'" on
`rma(yi, vi, mods = ~ablat * alloc, scale = ~ablat)` and metafopy is documented to fit; on CI metafopy raises R's error, so its
computed `alpha` start is NaN there. A NaN or ill-conditioned start is the suspected common symptom.

Fix direction: reproduce first - run `uv run pytest --mpl --cov` over the whole suite locally, then bisect by module; then fix the cause in
`src/metafopy/`, or record a platform difference in `docs/differences.md` and `src/audit/data/known.toml` if it is one. Until then the
`test` jobs of CI stay red, which hides any new failure.
