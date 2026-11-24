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

Note 2026-11-24: causes found from CI run 45335133105 (commit `00f7fda`); test order and coverage are not them (`test_coverage.py` then
the `ls.vif` cases, and the whole of `test_derived.py`, pass locally with `--mpl --cov`, and the local `.venv` matches `uv sync --locked
--all-extras`). Four cases, all ill-conditioned location-scale fits on `dat.bcg`:
- `differences.md` line 1442: with the three NA studies dropped (k = 10, five location columns) one study has leverage 1, so its HE residual
  in `start_alpha` (`src/metafopy/models/rma_ls_fit.py:312-321`) is zero in exact arithmetic; locally it is `2.3e-14` and the fit goes on,
  on CI the start is NaN and metafopy raises R's own error. The documented difference is a local round-off artefact.
- `ls.vif.reestimate.ML.bcg2`: the singular matrix is in the test's reference fit (`ls_reference_fit`,
  `src/tests/correctness/formulas.py:3111`, `numpy.linalg.inv` of the information), with `year` uncentred in both parts.
- `ls.vif.REML.bcg2`: metafopy's location vif is `4.723135` (ubuntu) and `4.723162` (windows) against the reference `4.723144` on every
  runner, so metafopy's optimiser end point moves with the platform.
- `ls.permutest.location.exact.ML.z.bcglat` (windows only): p `0.61694` against a reference upper bound `0.61616`, a difference of
  exactly 1/1287, one permutation on the other side of a near-tie.

Spec: `agents/specs/location-scale-fits-stable-across-platforms/spec.md`; tasks: `agents/tasks/location-scale-fits-stable-across-platforms/tasks-1.md`.

Note 2026-11-24: the task list is done and the branch passes the full local run on macOS arm64 (log in the main checkout's `ralpher-logs/location-scale-fits-stable-across-platforms/tasks-1/final-tests.log`). This item moves to `DONE.md` once the CI run of the merge commit is green on every runner; check with `gh run list --branch main --limit 3` and `gh run view <id> --log-failed`.

## Location-scale fits can stop at a lower local maximum than the test reference

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
Found on 2026-11-24 while measuring `ls.permutest.location.exact.ML.z.bcglat` on macOS arm64 at `282a6b9`
(`agents/specs/location-scale-fits-stable-across-platforms/measurements.md`). On 8 of the 1210 regular permutations (indices 68, 264, 433, 651,
805, 979, 1110, 1278 in `distinct_label_orders` order) metafopy's `z` differs from the test reference's by more than 1e-4. On permutation 68
metafopy and R both stop at log-likelihood `-12.58248229` (alpha `[-0.778, -0.951]`); the test's reference fit reaches the higher
`-12.57323737` (alpha `[-3.042, 2.526]`). Permutations 264 and 433 likewise.
<!-- doc-counts: accept-end -->

Cause: unconfirmed. The likelihood has at least two local maxima on these permutations; metafopy and R share the lower one, the reference
fit, which starts elsewhere, reaches the higher.

Fix direction: decide which maximum metafopy should report (the higher one is the better fit, but R's is the lower), compare R's behaviour on
the eight permutations, and if metafopy should search more starts, write a spec first. Filed apart from
`agents/specs/location-scale-fits-stable-across-platforms/spec.md`, whose decision D3 checks the exact permutation cases against widened
bounds only.

Note 2026-11-24: permutation 433 alone puts metafopy one over the reference bound of `ls.permutest.location.exact.ML.z.bcglat`, now a
strict-xfail mismatch naming this item; remove its `status = "mismatch"` once this item is fixed.

## Auditor rev 133ca2c reports unaccepted hits under three new rules

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
Found on 2026-11-24 when the auditor hooks' `rev` in `.pre-commit-config.yaml` was moved from `3e232368c42ae4147aa8607ac73570adfc3f786f` to
`133ca2c819d32fbc369ff921c1eba491ab36e222` (commit `d35c7a7`): the pre-push hook `auditme-all` refused the push of `main` with 29 unaccepted hits,
all under rules the earlier rev does not have. The bump was reverted so the CI fix could be pushed (operator choice, 2026-11-24).

- `discarded-tuple-return` (a returned tuple is thrown away): `src/metafopy/plot/gosh.py:280`, `:282`, `:295`, `src/metafopy/plot/infl.py:111`,
  `:266`, `src/metafopy/plot/llplot.py:278`, `src/metafopy/plot/profile.py:83`, `src/metafopy/plot/radial.py:194`, `src/metafopy/plot/rma.py:77`,
  `:90`, `src/metafopy/plot/selmodel.py:230` (each discards `metafopy.plot.plain.plain_plot`'s tuple), `src/tests/correctness/test_derived.py:343`,
  `src/tests/test_glmm_extras.py:141`;
- `assert-never-missing` (an `if/elif` chain on a string ends in a fall-through `else`, not `typing.assert_never`): `src/audit/pycall.py:88`,
  `src/metafopy/bias/fsn.py:96`, `src/metafopy/core/padjust.py:69`, `src/metafopy/effects/measures.py:1999`, `src/metafopy/methods/anova.py:195`,
  `src/metafopy/methods/emm_adjust.py:223`, `:308`, `src/metafopy/methods/emm_contrasts.py:221`, `src/metafopy/models/mv_structs.py:538`,
  `src/metafopy/models/tau2.py:353`, `src/metafopy/plot/forest.py:1238`, `src/metafopy/plot/labbe.py:356`, `src/metafopy/plot/rutils.py:339`;
- `pure-name-has-side-effect` (a query-like name with a side effect): `src/audit/steps.py:98`, `src/metafopy/core/rng.py:267`,
  `src/tests/auditsnippets.py:44`.
<!-- doc-counts: accept-end -->

Cause: the new rules are stricter than the code was written to; each hit is either a real cleanup or a site to accept with a reason.

Fix direction: move the rev to `133ca2c` again on a branch, then fix or accept each hit (`# auditme: accept[<rule>] <reason>`); an
`assert-never-missing` fix that changes which input raises changes behaviour, so check each chain's `else` before replacing it. Push only when
`pre-commit run auditme-all --hook-stage manual --all-files` passes.

