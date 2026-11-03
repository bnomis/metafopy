# Tasks: R fused arithmetic

**Spec:** `agents/specs/r-fused-arithmetic/spec.md`
**Created:** 2026-11-03
**Version:** 1

The review of 2026-11-03 (`agents/specs/r-fused-arithmetic/review.md`)
approved every part of the spec. The project has no affected-tests tool, so
each task names its test files; the last task runs every test file that
imports a changed module or calls `gosh`.

---

## Task 1: Fused forms in dbinom, following the platform

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Implementation - The fused forms

**Technical Goal:** `dbinom` reproduces R built with or without fused multiply-add, choosing by platform, so that random `gosh` subsets match the R
the user runs on the same kind of machine.

**Subtasks:**

- [x] Add `platform_fuses()` (true when `platform.machine().lower()` is `arm64` or `aarch64`) and `FUSED = platform_fuses()` to
      `src/metafopy/core/dbinom.py`
- [x] Give `bd0`, `pow1p` and `dbinom_raw` a `fused: bool` argument and use the fused forms of the spec's table when it is true; give `dbinom` a
      `fused: bool | None = None` argument, `None` meaning `FUSED`, passed down to every call
- [x] Rewrite the module docstring: the last bit depends on whether R's build fuses `a*b + c` (arm64 clang does, x86_64 R does not), as traced
      on 2026-11-03
- [x] In `src/tests/test_diagnostics.py`, make `test_dbinom_bits_match_r` call `dbinom(..., fused=False)` and drop its `refutil.linux_only`
      mark (the fixture is R 4.5.1 on Linux, unfused); add `refutil.assert_bits(got, expected, exact)` to `src/tests/refutil.py` (last bit
      when `exact`, else `rtol=1e-12`) and compare with `exact=sys.platform.startswith('linux')` (decision of 2026-11-03, see Notes)
- [x] Add tests: `platform_fuses` with `platform.machine` patched to `arm64`, `aarch64` and `x86_64`; `dbinom(c(4, 10, 12), 13, 0.5)`,
      `dbinom(0, 2:4, 0.3)` and `dbinom(2:4, 2:4, 0.7)` with `fused=True` equal R 4.6.2's bits as the spec's Motivation lists them
      (`0x1.657fffffffffdp-4`, `0x1.1dffffffffffep-5`, `0x1.9fffffffffff4p-10`; the `pow1p` values from R, written as hex literals),
      compared through `refutil.assert_bits` with `exact=sys.platform == 'darwin'` (written already; only the comparison changes)
- [x] Remove the `# auditme: accept[spec-api-signature-drift]` line above `def dbinom` in the spec's API block, now that the code has `fused`

**Acceptance Criteria:**

- [x] `dbinom(..., fused=False)` gives the bits of `audit3.json`'s `gosh.dbinom_k` on Linux, and agrees with them to `rtol=1e-12` elsewhere
- [x] `dbinom(..., fused=True)` gives R 4.6.2's bits for the values above on macOS, and agrees with them to `rtol=1e-12` elsewhere
- [x] After `metafopy.set_seed(1234)`, `gosh(rma(yi, vi, data=dat), subsets=20)` on bcg (`measure='RR'`) with `FUSED` true gives R's first
      `k` values `3 5 5 5 5 5 5 6 6 6` and first estimate `-1.1366851`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_diagnostics.py` passes

**Notes:**

- Key files: `src/metafopy/core/dbinom.py`, `src/tests/test_diagnostics.py`, `agents/specs/r-fused-arithmetic/spec.md`
- `math.fma` is in the standard library from Python 3.13; the project needs 3.14
- R's `pow1p` values: `Rscript --vanilla -e 'cat(sprintf("%a", c(dbinom(0, 2:4, 0.3), dbinom(2:4, 2:4, 0.7))))'`; on 2026-11-03 they were
  `0x1.f5c28f5c28f5dp-2 0x1.5f3b645a1cac1p-2 0x1.ebb98c7e2823ep-3 0x1.f5c28f5c28f5bp-2 0x1.5f3b645a1cabfp-2 0x1.ebb98c7e2823ep-3`

---

**Unblocked 2026-11-03 (operator decision, recorded in the spec's Decisions):** with `fused=False` on arm64 macOS, `dbinom` misses 7 of
the 844 values of `audit3.json` (k = 13, 28, 210, 400). Correctly rounded `log`, `exp` and `log1p` miss the same 7, and moving one `log1p`,
`log` or `exp` result by 1 ulp reproduces each: glibc's last bits, not the fused arithmetic. Each fixture is compared to the last bit on the
C library that made it and to `rtol=1e-12` elsewhere; the seeded `gosh` tests run only on their fixture's platform (Task 2).

---

## Task 2: A fused reference from R, and seeded gosh tests of both variants

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation - Fixtures and tests; Documentation

**Technical Goal:** both arithmetics are tested against values R made: to the last bit on the C library that made each fixture, and to a
relative tolerance elsewhere; the Linux-only skip gets its true reason.

**Subtasks:**

- [x] Write `src/tests/r/generate_dbinom_fused.R`: `dbinom(0:k, k, p)` as `%a` for k in 2, 5, 8, 13, 16, 24, 28, 37, 90, 210, 400 and p in
      0.5, 0.3, 0.7; one seeded `gosh` reference (`set.seed(99)`, `gosh(r13, subsets = 100)`) in the shape `REF['gosh_random_global']` has in
      `src/tests/r/generate_diagnostics.R`; and `R.version.string`, `R.version$platform` and the fusion probe
      (`sprintf("%a", dbinom(10, 13, 0.5))`)
- [x] Run it with the pinned R (`LC_ALL=C.UTF-8 Rscript src/tests/r/generate_dbinom_fused.R`) to write
      `src/tests/reference/phase12/dbinom_fused.json`; check the probe is `0x1.1dffffffffffep-5`, or stop the task as `blocked` naming the R
      found
- [x] Add a test with `fused=True` over every k and p of the new fixture, through `refutil.assert_bits` with `exact=sys.platform == 'darwin'`
- [x] Make `test_gosh_seeded` (`src/tests/test_audit3_misc.py`) and `test_gosh_random_subsets_global_seed` (`src/tests/test_diagnostics.py`)
      patch `metafopy.core.dbinom.FUSED` to `False` with `monkeypatch` and keep `refutil.linux_only`, and add a test that patches it to
      `True` against the new `gosh` reference, marked by a new `refutil.macos_only`
- [x] Give `refutil.linux_only` the reason "glibc's log, log1p and exp set the last bit of R's dbinom()" and correct its comment; give
      `refutil.macos_only` the reason "Apple's libm sets the last bit of R's dbinom()"; leave either mark only on seeded `gosh` tests
- [x] Update the `seed` entry of `gosh_rma_uni`'s docstring (`src/metafopy/methods/gosh.py`) and `set_seed`'s docstring
      (`src/metafopy/core/rng.py`): random subsets match R built for the same kind of machine (fused on arm64, not on x86_64)

**Acceptance Criteria:**

- [x] `grep -rn 'linux_only\|macos_only' src/tests` finds only `refutil.py` and seeded `gosh` tests
- [x] On this machine every `dbinom` fixture test and the fused seeded `gosh` test run, not skip

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_diagnostics.py src/tests/test_audit3_misc.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/tests/r/generate_dbinom_fused.R`, `src/tests/reference/phase12/dbinom_fused.json`, `src/tests/refutil.py`,
  `src/tests/test_diagnostics.py`, `src/tests/test_audit3_misc.py`, `src/metafopy/methods/gosh.py`, `src/metafopy/core/rng.py`
- `refutil.load_phase12(name)` reads `src/tests/reference/phase12/<name>.json`

---

## Task 3: Pin R's platform and arithmetic

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation - The pin; Error handling; Settings; Documentation

**Technical Goal:** `check-r` refuses an R build other than the pinned one, and an R and a Python that disagree on fusing, as it refuses a
different package version.

**Subtasks:**

- [ ] Add `r_identity()` to `src/audit/r/common.R`, returning `R` (version), `R_platform` (`R.version$platform`) and `R_fma` (`"fused"` when
      `sprintf("%a", dbinom(10, 13, 0.5))` is `"0x1.1dffffffffffep-5"`, `"unfused"` when it is `"0x1.1ep-5"`, else `"unknown"`)
- [ ] Use it in `src/audit/r/versions.R` and `src/audit/r/setup.R` in place of `vers$R`; make `setup.R` drop `R`, `R_platform` and `R_fma`
      from the packages to install
- [ ] Add `"R_platform": "aarch64-apple-darwin27.0.0"` and `"R_fma": "fused"` to `src/audit/data/r-versions.json`, after checking that
      `Rscript src/audit/r/versions.R <out.json>` writes those values on this machine
- [ ] In `check_r` (`src/audit/steps.py`), refuse when the pin's `R_fma` disagrees with `metafopy.core.dbinom.FUSED`, naming both and the
      three ways on; with `--allow-r-drift` the difference becomes a note, as for drift
- [ ] Tests in `src/tests/test_harness_compare.py`: a different `R_platform` or `R_fma` is drift named by `check_r`; `R_fma` against a
      patched `FUSED` raises, and becomes a note with the flag
- [ ] Name the two entries in `src/audit/README.md` where it describes the pin

**Acceptance Criteria:**

- [ ] `bin/metafor-audit.sh --steps check-r` passes on this machine
- [ ] With `R_fma` changed to `"unfused"` in a copy of the pin, `check_r` raises naming `R_fma fused -> unfused` (or the reverse) and the
      three ways on

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 1 (`FUSED`)
- Key files: `src/audit/r/common.R`, `src/audit/r/versions.R`, `src/audit/r/setup.R`, `src/audit/data/r-versions.json`, `src/audit/steps.py`,
  `src/audit/README.md`, `src/tests/test_harness_compare.py`
- `library_drift` already compares every key of the pin, so the two entries need no change there
- Underscores keep the entries apart from package names, which cannot contain one

---

## Task 4: Rerun the gosh cases, settle the TODO item and close the spec

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation - Rerun and the rest of the item

**Technical Goal:** the audit's `gosh` cases are rerun on the pinned R with the new arithmetic, the `TODO.md` item records what remains, and
the spec is closed.

**Subtasks:**

- [ ] Run `bin/metafor-audit.sh --only gosh` in the background; record the run directory and its causes in this task's Notes
- [ ] Compare its causes with those of the `TODO.md` item "gosh gives different subsets and results from R on every fit"
- [ ] If no cause of the item remains: move the item to the end of `DONE.md`, its heading and body unchanged, then
      `Done: <date> - agents/specs/r-fused-arithmetic/spec.md, agents/tasks/r-fused-arithmetic/tasks-1.md.`, and remove it from `TODO.md`
- [ ] Otherwise: add a dated note to the item naming the run, the causes that went and the causes that remain, and change its `Spec:` line's
      text to say the spec is implemented; the item stays open. A cause that is new and belongs to no item is listed in the note, not fixed
- [ ] Run `python3 ~/.claude/skills/todo-review/verify_bodies.py <baseline> TODO.md DONE.md`, the baseline being `TODO.md` at the commit that
      added this task list (`git show $(git log --diff-filter=A --format=%H -- agents/tasks/r-fused-arithmetic/tasks-1.md):TODO.md`); a note
      added to the item is the only change allowed
- [ ] Set the spec's `**Status:**` to `implemented`, and its entry in `agents/specs/index.md` to say so

**Acceptance Criteria:**

- [ ] The run reports no `$incl` difference for a `gosh` case
- [ ] The `TODO.md` item is in `DONE.md` with its `Done:` line, or carries a dated note listing what remains

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_diagnostics.py src/tests/test_audit3_misc.py src/tests/test_audit8_args.py src/tests/test_audit8_crashes.py
      src/tests/test_audit8_fields.py src/tests/test_fits.py src/tests/test_plots.py src/tests/test_plots_image.py src/tests/test_profile_parallel.py
      src/tests/test_rma_ls.py src/tests/test_harness_compare.py src/tests/test_harness_engine.py` passes (every test file that imports
      `metafopy.core.dbinom` or `audit.steps`, or calls `gosh`)

**Notes:**

- Depends on Tasks 1 to 3
- The run takes long; start it in the background and wait for it

---

## Task 5: Decide how the unfused variant is tested off Linux

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Fixtures and tests (needs an operator decision; the spec assumes the unfused fixture matches everywhere)

**Technical Goal:** settle what runs `fused=False` against the Linux fixture, given that Apple's libm differs from glibc's in `log`, `exp` and
`log1p` (found in Task 1), so Tasks 1 and 2 can drop `refutil.linux_only` honestly.

**Subtasks:**

- [x] Operator chooses: keep `linux_only` on the unfused fixture test with the true reason (the C library's `log`/`exp`); compare unfused
      results to the fixture with a tolerance of a few ulp off Linux; or cover the unfused arithmetic only through the Python-side fma-free
      reference
- [x] Amend the spec's Decisions, Fixtures and tests, and Task 1 and Task 2 acceptance criteria to match

**Decision (operator, 2026-11-03):** exact where made - each fixture to the last bit on the C library that made it (glibc on Linux, Apple's
libm on macOS), `rtol=1e-12` elsewhere; seeded `gosh` tests only on their fixture's platform. Recorded in the spec's Decisions and in
Tasks 1 and 2.

**Notes:**

- Blocks Tasks 1 and 2
