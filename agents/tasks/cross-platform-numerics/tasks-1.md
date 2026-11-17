# Tasks: Cross-platform numerics

**Spec:** `agents/specs/cross-platform-numerics/spec.md`
**Created:** 2026-11-16
**Version:** 1

Row 3 of the run order `agents/tasks/order-ci-platform-failures.md`, from spec version 1.1 (approved on 2026-11-16 in
`agents/reviews/order-ci-platform-failures-specs.md`: D1 *Draft pull request*, D2 *keep* the probe workflow). The spec's Failure groups table names
eleven groups; Tasks 5 to 10 take them in the table's order, sharing a task where groups share a reference.

Rules every task follows:

- **No change to a metafopy result without the operator.** A task whose figures show metafopy below a reference's maximum or off a theoretical value
  (spec, Confirming the cause) stops as `blocked` with the figures from every environment and the two options (fix through a spec, or record a
  correctness mismatch); the other groups' tasks carry on first. `git diff main -- src/metafopy` stays empty unless such a stop was answered.
- **Verdicts** are the spec's four, in its order of preference; every changed tolerance follows the spec's choice "Tolerance margin" (one value for all
  platforms at 10 times the largest measured deviation, refused if larger than the effect the test exists to detect), and every platform-dependent
  case follows its choice "Platform-dependent marks" (assert the observed range; skip by platform only where no range can be stated). Each is written in
  `agents/specs/cross-platform-numerics/measurements.md` with its figures and the date.
- **Outward actions** are limited to D1's answer: push this list's branch, open one draft pull request, read CI with `gh`, and close the pull request
  and delete the remote branch at the end. Never push `main`, never merge the pull request on GitHub, never force-push, never `--no-verify`.
- Probe outputs and CI artifacts go to `$TMPDIR/metafopy-platform-probe/`, outside the repository; only figures copied into `measurements.md` are
  committed.
- The project has no affected-tests tool, so each task names its test files.

---

## Task 1: Check the prerequisites

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Scope (Prerequisite), Order of work step 1, Operator stops

**Technical Goal:** the list starts only where its acceptance can be met: the two harness lists merged, Docker running, `gh` authenticated.

**Subtasks:**

- [x] Check that the headings "A harness test runs Rscript with no skip where R is missing" and "Audit harness tests assume POSIX and fail on Windows"
      are absent from `TODO.md` and present in `DONE.md`, and that `pytest src/tests/test_harness_engine.py src/tests/test_harness_scripts.py` passes
- [x] Check `docker info` exits 0, and `gh auth status` exits 0
- [x] Check the new names do not exist yet: `ls bin/linux-tests.sh bin/linux-tests.Dockerfile bin/platform-probe.py bin/platform-probe.sh
      .github/workflows/platform-probe.yml src/tests/test_platform_probe.py agents/specs/cross-platform-numerics/measurements.md` finds none
- [x] If any check fails, stop as `blocked`, naming the check and what the operator must do (merge the list, start Docker Desktop, `gh auth login`)

**Acceptance Criteria:**

- [x] Every prerequisite holds, recorded in this task's Notes with the date

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes
- Checked on 2026-11-16 (21:25 UTC): both headings are in `DONE.md` and not in `TODO.md`; `pytest src/tests/test_harness_engine.py
  src/tests/test_harness_scripts.py` passes (25 tests, run with the main checkout's venv because this worktree has none); `docker info` and
  `gh auth status` exit 0; none of the new names exists

---

## Task 2: The Linux container

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`bin/linux-tests.sh`), Linux reproduction, Error handling, Settings

**Technical Goal:** `bin/linux-tests.sh` runs the group tests (or the probe) in a pinned Linux container from a copy of the working tree, and the
architecture that reproduces CI's Linux failures is known.

**Subtasks:**

- [x] Write `bin/linux-tests.Dockerfile`: `FROM ubuntu:26.04` pinned to an exact tag, the uv binary copied from `ghcr.io/astral-sh/uv:<version>` pinned to
      an exact release tag (the newest outside the project's resolver quarantine, if any; read `pyproject.toml`'s uv settings)
- [x] Write `bin/linux-tests.sh` as the spec's API and Linux reproduction say: `--arch arm64|amd64` (default `arm64`), `--probe OUT_DIR`, `-- PYTEST_ARGS`
      (default: the group test files of Settings); `docker info` first, exiting 2 with "Docker is not running: start Docker Desktop"; the tree streamed
      in by `tar` from `git ls-files -co --exclude-standard`; named volumes `metafopy-uv-cache` and `metafopy-venv-<arch>`;
      `uv sync --locked --all-extras --python 3.14.8`, then `uv run pytest --mpl <args>` or the probe; never `uv lock`; exit with pytest's or the probe's
      status. Give it `--help`
- [x] Check it by running it: `--help`; a group run on arm64. (The Docker-not-running message is checked by reading the code path, since stopping the
      operator's Docker is not this task's to do)
- [x] Compare the arm64 run's failing tests with the Linux job's `FAILED` lines of run 43883708179 (`gh run view 43883708179 --log-failed`, read-only),
      leaving out the harness tests the two earlier lists fixed. If the lists differ, repeat with `--arch amd64` and compare again
- [x] Create `agents/specs/cross-platform-numerics/measurements.md` (dated, inside `doc-counts` accept markers like the spec's dated records) with the
      image digests (`docker image inspect`), the architecture that reproduces CI's Linux list (or that neither does, with both lists), and the date
- [x] If `uv sync --locked` fails in the container, stop as `blocked` with uv's message (the lock is not edited without instruction)

**Acceptance Criteria:**

- [x] `bin/linux-tests.sh --help` prints its usage; a group run exits with pytest's status; nothing in the checkout is written by a run
- [x] `measurements.md` names the reproducing architecture and the image digests

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `bin/linux-tests.sh -- src/tests/test_nchg.py` runs to completion and its result is recorded in `measurements.md`

**Notes:**

- Depends on Task 1
- Key files: `bin/linux-tests.sh`, `bin/linux-tests.Dockerfile`, `agents/specs/cross-platform-numerics/measurements.md`
- An emulated amd64 run is slow; run it in the background and wait for it
- Done on 2026-11-16: neither architecture reproduces CI's Linux list (figures in `measurements.md`); GNU tar cannot create nested directories in the emulated container, so the image installs `bsdtar` and the script extracts with it
- The pre-push stage reports `ruff-version-drift` at `.pre-commit-config.yaml:3` (the audit's ruff differs from the pre-commit rev): the worktree has no venv of its own, so it uses the main checkout's; this task did not touch that file

---

## Task 3: The probe

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`bin/platform-probe.py`), Tests of the new code, Error handling, Confirming the cause

**Technical Goal:** one command measures every group in the current environment and writes JSON with the environment's identity, so the five
environments can be compared.

**Subtasks:**

- [x] Write `bin/platform-probe.py` with `GroupMeasurement`, `environment()`, `measure_groups()` and `main()` as the spec's API gives them, one measuring
      function per group of the Failure groups table, each recording what that group's confirmation (spec, Confirming the cause) needs and calling the
      same helpers its tests call; and `bin/platform-probe.sh`, the wrapper (`source bin/activate.sh`, then the script), as the project's other `bin/`
      tools are wrapped
- [x] Write `src/tests/test_platform_probe.py` as Tests of the new code says (`measure_groups(['cm-el-tight-se'])`, an unknown group's `KeyError`, `main`
      writing JSON with `environment` and `groups` to `tmp_path`)
- [x] Run the probe on the development Mac and in the container (`bin/linux-tests.sh --arch <the reproducing arch> --probe $TMPDIR/metafopy-platform-probe`),
      outputs outside the repository
- [x] Add to `measurements.md` each environment's `environment()` and the figures per group

**Acceptance Criteria:**

- [x] The probe runs on the development Mac and in the container and writes one JSON each; a group that raises is recorded and the others run
- [x] `measurements.md` holds both environments' figures, dated

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_platform_probe.py` passes

**Notes:**

- Depends on Task 2
- Key files: `bin/platform-probe.py`, `bin/platform-probe.sh`, `src/tests/test_platform_probe.py`
- Done on 2026-11-16: the probe ran on the development Mac and in the arm64 container (figures in `measurements.md`). Running it showed that
  `bin/linux-tests.sh` ran `uv run` on the project's `.python-version` (3.14.7), not 3.14.8; the script now sets `UV_PYTHON`, and `measurements.md`
  carries a correction for Task 2's runs
- `bin/platform-probe.py` uses argparse, as the project has no `cappa` dependency and the spec gives the `--group`/`--out` interface

---

## Task 4: The probe workflow and the draft pull request

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Windows and the CI Mac, Decisions (D1 *Draft pull request*, D2 *keep*)

**Technical Goal:** the probe runs on the three CI platforms through a draft pull request, and its figures complete the five environments.

**Subtasks:**

- [x] Write `.github/workflows/platform-probe.yml`: the `test` job's matrix of `.github/workflows/ci.yml` (`ubuntu-26.04`, `macos-26`,
      `windows-2025-vs2026`, Python 3.14.8, `uv sync --locked --all-extras`), running
      `uv run python bin/platform-probe.py --out probe-${{ runner.os }}.json` and uploading it with `actions/upload-artifact` pinned to its newest release
      tag (`gh api repos/actions/upload-artifact/releases/latest`; an exact tag, never a SHA, per the shared security rule); triggered by
      `workflow_dispatch` and by `pull_request` with the `paths` filter of D2 (`uv.lock`, `src/metafopy/**`, `src/tests/correctness/**`,
      `bin/platform-probe.py`, and the workflow file itself)
- [x] Commit, push this list's branch to `origin` (pre-push hooks run; never `--no-verify`), and open a draft pull request against `main` with
      `gh pr create --draft`, its body saying it exists to run CI for this task list and will be closed unmerged. Record its number in this task's Notes
- [x] Wait for the CI and probe runs of the pull request's head commit (poll `gh run list --branch <branch>` every five minutes, for at most two hours;
      past that, stop as `blocked` naming the runs). Download the probe artifacts with `gh run download <id>` into `$TMPDIR/metafopy-platform-probe/`;
      a missing platform's artifact stops the task as `blocked`, naming the run id and job (spec, Error handling)
- [x] Add the three CI environments and their figures to `measurements.md`, with the run ids and the date

**Acceptance Criteria:**

- [x] The probe workflow ran on all three platforms; `measurements.md` holds figures from five environments
- [x] Exactly one draft pull request is open for this branch

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The push passed the pre-push hooks

**Notes:**

- Depends on Task 3
- Key files: `.github/workflows/platform-probe.yml`
- Draft pull request: number 74 (https://github.com/bnomis/metafopy/pull/74); probe run 43972461382, CI run 43972460854
- CI runs about 40 minutes on Windows (2026-11-16); this task's CI results also show whether the two harness lists' skips hold on Windows

---

## Task 5: The location-scale reference fits

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Failure groups (`ls-bcg-reference`, `ls-identity-alpha`, `ls-identity-maximum`), Confirming the cause, Verdicts, Tests of the new code

**Technical Goal:** the three groups that compare against `formulas.ls_reference_fit` get a confirmed cause and a verdict, and their tests pass in the
five environments.

**Subtasks:**

- [x] Confirm each group's cause from the five environments' figures as the spec's Confirming the cause says (every start's message, `-fun`, the
      analytic score, metafopy's `alpha` and log-likelihood). If the reference finds a higher log-likelihood than metafopy's anywhere, stop as `blocked`
      (operator stop; see the header)
- [x] For a confirmed reference defect, make `ls_reference_fit` (and `ls_starts`, in `src/tests/correctness/formulas.py`) robust as verdict 2 says: more
      starts, keep a result whose score passes a stated check whatever the optimiser's message, a final polish (Newton on the analytic score; projected
      for the identity link), a final small-score check; never call metafopy
- [x] Write the tests of Tests of the new code for it (a start forced to "precision loss" is still accepted when its score passes; a worse stationary
      point is not chosen over a better one), each run once against the old code and seen to fail
- [x] Where verdict 2 cannot hold a group, use verdict 3 or 4 under the header's rules
- [x] Record each group's figures and verdict in `measurements.md`; run the group's tests locally and in the container

**Acceptance Criteria:**

- [x] The groups' tests pass on the development Mac and in the container; `measurements.md` holds each group's verdict
- [x] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/test_rma_ls.py src/tests/correctness/test_formulas.py`
      passes (leave out a file that does not exist; name the file holding the new reference-fit tests)
- [x] `bin/linux-tests.sh -- src/tests/correctness/test_derived.py src/tests/test_rma_ls.py -k "ls"` passes

**Notes:**

- Depends on Task 4
- Done on 2026-11-17: reference defect confirmed in all three groups and fixed by verdict 2 (figures in `measurements.md`); no verdict 3 or 4 was needed.
  An early version that ran all five starts for every permutation refit made `ls.permutest.location.exact.ML.z.bcglat` fail in the arm64 container (11
  permutations without a reference fit) and slow; the log-link search now stops once two kept starts agree, and the case passes on the Mac
- Final runs of `test_derived.py`, `test_rma_ls.py` and `test_formulas_ls.py` with `-k "ls or identity or bcg"`: Mac 480 passed, 1 xfailed; arm64
  container 478 passed, 2 failed, both outside this task's groups and not caused by it: `glmm.cor.um.rs.ml.bcg` (Task 7's `glmm-um-rs-score`) and
  `test_identity_fit_bd_is_a_stationary_point` (asserts metafopy's own score below 1e-6, got 4.0e-06; it failed in the arm64 container before this task,
  in Task 2's run, and is not in CI's Linux list, so the group table does not name it). The last one is why the Definition of Done line for the
  container, whose `-k "ls"` also selects that file's tests, is ticked for the three groups' tests only
- Key files: `src/tests/correctness/formulas.py` (`ls_reference_fit`, `ls_starts`, `ls_score`), `src/tests/correctness/test_derived.py` (`LS_RELATIVE`,
  `ls_agree`), `src/tests/test_rma_ls.py`

---

## Task 6: The decreasing selection model's reference maximum

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Failure groups (`selmodel-decreasing`), Confirming the cause, Verdicts

**Technical Goal:** `selmodel.decreasing.bangertdrowns2004` gets a confirmed cause and a verdict, and passes in the five environments.

**Subtasks:**

- [x] Confirm the cause as the spec says (the log-likelihood at each platform's reference end point, at metafopy's point, and along the ray from
      metafopy's point to the runaway point). If any end point beats metafopy's by more than `SELMODEL_LL_TOLERANCE`, stop as `blocked` (operator stop)
- [x] For a confirmed reference defect, make `selmodel_maximum` (`src/tests/correctness/test_derived.py`) robust as verdict 2 says (several starts,
      bounds on `beta` wide enough not to bind at the maximum, a final small-score check), with a test that fails against the old code
- [x] Record the figures and verdict in `measurements.md`; run the case locally and in the container

**Acceptance Criteria:**

- [x] The case passes on the development Mac and in the container; `measurements.md` holds its verdict
- [x] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_derived.py -k selmodel` passes
- [x] `bin/linux-tests.sh -- src/tests/correctness/test_derived.py -k selmodel` passes

**Notes:**

- Depends on Task 4

---

## Task 7: The GLMM UM.RS score and the CM.EL tight-optimum check

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Failure groups (`glmm-um-rs-score`, `cm-el-tight-se`), Confirming the cause, Verdicts

**Technical Goal:** the two GLMM groups get a confirmed cause and a verdict, and pass in the five environments.

**Subtasks:**

- [x] `glmm-um-rs-score`: measure the reference optimum's score by each step pair and its spread over steps 2e-5 to 4e-4, beside the analytic score where
      `glmm_formulas` has one. If the score is within the noise floor on each platform, set `GRADIENT_TOLERANCE` (or the case's own) under verdict 3, or
      make `maximise` polish to a smaller score under verdict 2; otherwise stop as `blocked`
- [x] `cm-el-tight-se`: measure the Nelder-Mead end point, its `neg_loglik`, the score and the standard error per platform. If the standard error's
      spread follows the end point's, set the `rel` of `test_ml_se_matches_spec_measurement_at_tight_optimum` under verdict 3, or tighten the search
      under verdict 2; otherwise stop as `blocked`
- [x] Record both groups' figures and verdicts in `measurements.md`; run their tests locally and in the container

**Acceptance Criteria:**

- [x] Both groups' tests pass on the development Mac and in the container; `measurements.md` holds each verdict
- [x] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_nchg.py src/tests/correctness/test_derived.py -k "glmm or nchg"` passes
- [x] `bin/linux-tests.sh -- src/tests/test_nchg.py src/tests/correctness/test_derived.py -k "glmm or nchg"` passes

**Notes:**

- Depends on Task 4
- Done on 2026-11-17: both groups verdict 3 (figures in `measurements.md`). In the arm64 container the Definition of Done run has one failure, `glmm.fit.or.um.rs.fe.nielweise2007.beta`, an arm64-only difference already seen in Task 2's run, not in CI's Linux list nor the group table
- Key files: `src/tests/correctness/glmm_formulas.py` (`maximise`, `GRADIENT_TOLERANCE`), `src/tests/test_nchg.py`

---

## Task 8: The rma_mv REML score

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Failure groups (`mv-reml-score`), Confirming the cause, Verdicts, Operator stops

**Technical Goal:** `property.mv.reml-score.kon` gets a confirmed cause and a verdict.

**Subtasks:**

- [x] Measure metafopy's optimiser message and gradient in its own parameterisation, the REML score in `sigma2`, and its scale `tr(P Z Z')`, per
      platform
- [x] If the optimiser reports convergence and the score is a rounding-size fraction of its scale on every platform, set the case's `tolerance` relative
      to that scale under verdict 3, with the comment the header asks for. Otherwise metafopy is off a theoretical value: stop as `blocked` (operator
      stop) with the figures
- [x] Record the figures and verdict in `measurements.md`; run the case locally and in the container

**Acceptance Criteria:**

- [x] The case passes on the development Mac and in the container, or the task is `blocked` with the figures
- [x] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_properties.py -k mv` passes
- [x] `bin/linux-tests.sh -- src/tests/correctness/test_properties.py -k mv` passes

**Notes:**

- Depends on Task 4
- Stopped on 2026-11-17: the probe now records the optimiser message and gradient, and the three local environments give a gradient of 1e-10 and a score
  of 1e-11 of the scale, but CI Linux's score (1.1e-06 of the scale) is not reproduced locally and its gradient is unrecorded. Needs one CI probe run of this
  commit (Task 11's round) before the verdict; see `measurements.md`
- Decided 2026-11-17 by Claude, not yet reviewed (queue decision): push this list's branch now to update draft pull request #74 and wait for its
  CI and probe runs, as Task 4 waits, then judge this group on CI Linux's recorded gradient and score. The answer to D1 (*Draft pull request*,
  2026-11-16) allows this list to push its branch and read CI; pushing before Task 11 is within it. Never push `main`
- Stopped on 2026-11-17 (`blocked`, operator): the CI probe run 44002318304 of commit `f57ab45` gives a gradient of 6.0e-07 (Linux), 1.3e-10 (Mac) and
  1.8e-05 (Windows), all under the 1e-4 slack, with "precision loss" and `success` false on every platform. The score is 1.3e-07 and 3.8e-08 of the scale on
  Linux and up to 1.1e-06 on Windows, against 1e-11 on the Mac, so the condition for verdict 3 (rounding-size score) is not met. The options are in
  `measurements.md` ("The CI probe run of commit `f57ab45`"). The CI test run of the same head also fails this case on Linux (1.3e-05 against 1e-6)
- Answered by the operator on 2026-11-17 (03:40 UTC): *range now, file the fix*. (1) The test asserts the measured range: the REML score of `sigma2`
  within 2e-06 of its scale `tr(P Z Z')` (the largest measured, 1.1e-06 on CI Windows, rounded up), as verdict 4, with a comment naming probe run
  44002318304, the five environments and the date, and an entry in `docs/differences-notes.md` ("Optimisers and flat likelihoods"). (2) File a new
  `TODO.md` item, at the top, in the shape of the others, dated 2026-11-17: "The rma_mv REML fit stops short of the optimum on Linux and Windows",
  with the table of `measurements.md`, the cause (BFGS ends on "precision loss" and `mv_opt.maximise` accepts the stop under `SLACK_GRADIENT` 1e-4),
  and the fix direction: tighten the stop in `src/metafopy` through a spec (for example a Newton polish on the analytic score), then narrow this
  test's range. `src/metafopy` stays unchanged in this list
- Key files: `src/tests/correctness/test_properties.py` (`mv_score_is_zero`), `src/tests/correctness/cases/*.toml`

---

## Task 9: The stepcon interval and the halfnorm print

**Status:** in-progress
**Priority:** medium
**Complexity:** small
**Spec Reference:** Failure groups (`stepcon-rau-mods`, `selmodel-print-halfnorm`), Confirming the cause, Verdicts

**Technical Goal:** the two R-fixture comparisons of quantities near zero or at a bound get a confirmed cause and a verdict, and pass in the five
environments.

**Subtasks:**

- [ ] Measure each compared quantity and its standard error on each platform. If the spread is a small fraction of the standard error, set the
      comparison's tolerance under verdict 3 (`LOOSE`/`ATOL` for the case in `src/tests/test_stepcon.py`; `NOISY_HESSIAN`/`NOISY_ATOL` for the case in
      `src/tests/test_selmodel.py`), for that case only; otherwise stop as `blocked`
- [ ] Record both groups' figures and verdicts in `measurements.md`; run their tests locally and in the container

**Acceptance Criteria:**

- [ ] Both tests pass on the development Mac and in the container; no other case's tolerance changes
- [ ] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_stepcon.py src/tests/test_selmodel.py` passes
- [ ] `bin/linux-tests.sh -- src/tests/test_stepcon.py src/tests/test_selmodel.py` passes

**Notes:**

- Depends on Task 4
- Stopped on 2026-11-17 (`blocked`): `stepcon-rau-mods` is done (`LOOSE_ATOL` 2.5e-05, verdict 3). `selmodel-print-halfnorm` needs the operator: its
  `ci.lb` deviates by up to 0.0075 (CI Mac) and `se` spreads 5 per cent of itself, and the margin rule's 0.075 is larger than the effect the test detects.
  The three options are in `measurements.md` ("Verdicts: the stepcon interval and the halfnorm print"). The container run of both tests is also open
- Decided 2026-11-17 by Claude, not yet reviewed (queue decision): option (a), as verdict 4, not verdict 3. Verdict 3 is refused (the margin's 0.075
  exceeds the effect the test detects), so the spec sends the group to verdict 2 or 4; `delta` sits at its bound on every environment, which is verdict
  4's case, and the spec's choice "Platform-dependent marks" is to assert the observed range. So `NOISY_ATOL` 0.01 for this case's printed `ci.lb` only
  (the measured deviations, up to 0.0075, with the five environments and the date in its comment), every other printed field still compared as now,
  and an entry in `docs/differences-notes.md` ("Optimisers and flat likelihoods") naming the measurement. Then run both tests in the container

---

## Task 10: The ridge stops of the vif simulation and the permutation boundary fits

**Status:** done
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Failure groups (`ls-vif-sim`, `permutest-ls-boundary`), Confirming the cause, Verdicts, the choice "Platform-dependent marks"

**Technical Goal:** the two tests that count optimiser stops along a flat ridge assert the range every platform gives, and pass in the five
environments.

**Subtasks:**

- [x] Measure which draws or permutations are NaN or boundary fits, and their `alpha` at the stop, on each platform. If the only difference is the count
      of boundary stops along a ridge (the stop accepted as an optimiser difference on 2026-11-04), mark the case platform-dependent under verdict 4: the
      test asserts the measured range (for example 0 or 1 NaN draws; `pval_alpha` whose denominator lies in the measured range) and checks the rest
      exactly; replace the `sys.platform == 'win32'` tolerance in `test_audit4_numerics.py`. Otherwise stop as `blocked`
- [x] Add an entry to `docs/differences-notes.md` ("Optimisers and flat likelihoods") naming the measurement, as verdict 4 requires
- [x] Record both groups' figures and verdicts in `measurements.md`; run their tests locally and in the container

**Acceptance Criteria:**

- [x] Both tests pass on the development Mac and in the container, asserting the measured ranges
- [x] `git diff main -- src/metafopy` is empty

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_ls_vif.py src/tests/test_audit4_numerics.py` passes
- [x] `bin/linux-tests.sh -- src/tests/test_ls_vif.py src/tests/test_audit4_numerics.py` passes

**Notes:**

- Depends on Task 4

---

## Task 11: The CI round

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Order of work step 5, Acceptance, Decisions (D1)

**Technical Goal:** CI's three test jobs and `lint` pass on the draft pull request's head, which is what the merge will be.

**Subtasks:**

- [ ] Push the branch (pre-push hooks run) and wait for the pull request's CI run of the new head, as Task 4 waits (at most two hours, then `blocked`)
- [ ] If a test job fails, read its `FAILED` lines (`gh run view <id> --log-failed`), add the new figures to `measurements.md`, return to the task of the
      failing group (reopen it as `in-progress` with a Notes line naming the run), and repeat this task after it. After three rounds with a job still
      failing, stop as `blocked` with the failing tests of each round
- [ ] When all three test jobs and `lint` pass, record the run id and the date in `measurements.md` and in this task's Notes

**Acceptance Criteria:**

- [ ] The pull request's latest CI run has every job passing, recorded with its id

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `gh run view <id> --json jobs` shows every job `success`

**Notes:**

- Depends on Tasks 5 to 10
- A pull request's CI tests the merge of the branch with `main`, so a pass here is the pass the merge will have, if `main` does not move meanwhile

---

## Task 12: The full test run, the records and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Order of work step 6, Acceptance

**Technical Goal:** the branch is checked once locally, the records are complete, the draft pull request is closed, the spec is implemented and the item
is closed.

**Subtasks:**

- [ ] Check `measurements.md` holds every group's figures and verdict, and that every changed tolerance and platform-dependent mark has a comment with
      the measured spread, the environments and the date
- [ ] Close the draft pull request without merging (`gh pr close <number> --delete-branch`), with a comment naming this list; the branch merges into
      `main` locally through the run order
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md`
- [ ] Move the item "Numerical tests whose results or tolerances hold only on the development machine" from `TODO.md` to `DONE.md`: append its heading
      and body unchanged at the end of `DONE.md`, then a line `Done: 2026-MM-DD, agents/specs/cross-platform-numerics/spec.md,
      agents/tasks/cross-platform-numerics/tasks-1.md; CI run <id> passed on all three platforms` with the date of the commit. (The spec's Order of work
      moves it after the merge; the pull request's CI already tests the merge with `main`, so it moves here)
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/cross-platform-numerics/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/cross-platform-numerics/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The spec is `implemented`; the item is in `DONE.md` with its body unchanged and no longer in `TODO.md`; no pull request of this branch is open
- [ ] `git diff main -- src/metafopy` is empty, unless an operator stop confirmed a change

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/cross-platform-numerics/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 11
- No version bump: no library output changes, unless an operator stop confirmed one; then bump against `main` here
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
