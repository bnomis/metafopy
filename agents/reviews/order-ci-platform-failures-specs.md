# Review: specs of the CI platform failures order

**Date:** 2026-11-16
**Order:** `agents/tasks/order-ci-platform-failures.md` (written after this review is answered)
**Review page:** https://claude.ai/artifact/NPXPMUthes8x1BDiDBPP5h

## What is reviewed and how

The order covers the first three `TODO.md` items, filed on 2026-11-16 because CI on `main` has failed on every completed run since 2026-11-01:
"A harness test runs Rscript with no skip where R is missing", "Audit harness tests assume POSIX and fail on Windows" and "Numerical tests whose
results or tolerances hold only on the development machine". The first two become small task lists with no spec; the third became the spec
`agents/specs/cross-platform-numerics/spec.md`. Approve or flag the plan and the spec, and answer each decision. The task lists are written only after
every item and decision has an answer.

## Automatic checks

- The spec has the specdev sections, a Decisions section, and carries its `TODO.md` item's dated findings unchanged (quoted under Motivation): pass.
- CI runs 43832306599 (`ff7eb2b`) and 43883708179 (`dae8748`) fail the same tests (Linux 29, macOS 7, Windows 33), so the failures are per platform, not
  run-to-run noise (checked 2026-11-16): pass.
- Docker is installed (`/Users/simonb/.docker/bin/docker`) but its daemon is not running (checked 2026-11-16 18:24 UTC): needs attention, see
  Prerequisites.
- The spec is about 340 lines, over the specdev guide of about 200; it has one table row per failure group: note.

## The items

### 1. The plan: two small task lists, no spec

- `rscript-skip-guard`: skip `test_r_takes_the_cluster_port_it_is_given` (`src/tests/test_harness_engine.py`) when `Rscript` is not on the path.
- `audit-windows-tests`, after the first (both edit `src/tests/test_harness_engine.py`): skip the eight harness tests on Windows with their reason, and
  say in `src/audit/README.md` that the audit runs on macOS and Linux. I chose skipping over porting the harness: it uses `preexec_fn`, resource limits,
  resident-memory reads and `SIGALRM`, none of which Windows has, and the audit is run on the development Mac only. Reversible by porting later.
- **Recommendation:** approve.

### 2. The spec `agents/specs/cross-platform-numerics/spec.md`

- **What it changes:** the remaining failures, in eleven groups (location-scale `bcg` reference fits, identity-link `alpha`, selmodel decreasing, GLMM
  UM.RS score, CM.EL tight-optimum standard error, `rma_mv` REML score, `stepcon`, `ls_vif` sim, `permutest` boundary fits, and two macOS-only tests).
  Each group's cause is confirmed from figures on the development Mac, a Linux container and the three CI platforms (`measurements.md`), then it gets one
  verdict: fix metafopy (operator stop), make the test's reference fit robust, set a tolerance from the measured spread, or mark the case
  platform-dependent. New: `bin/linux-tests.sh` with its Dockerfile, `bin/platform-probe.py`, `.github/workflows/platform-probe.yml`.
- **Why:** in every disagreement read so far metafopy prints the same value on every platform and the test's reference moves (on Windows the
  location-scale reference ends at a lower log-likelihood, -9.4736 against metafopy's -7.5393).
- **Choices made in the spec, reversible by amending it:** one tolerance for all platforms at 10 times the largest measured deviation (the two Macs share
  `sys.platform`, so per-platform values cannot tell them apart); a platform-dependent case asserts the observed range rather than being skipped; the
  container runs natively on arm64 first and falls back to emulated amd64 if it does not reproduce CI's Linux failures.
- **Stops for you:** a group whose figures show metafopy below a reference's maximum or off a theoretical value (likeliest: the `rma_mv` REML score,
  `bcg_year_identity`, selmodel decreasing); unmet prerequisites.
- **Recommendation:** approve.

## Decisions

### D1. CI runs before the merge

- **What is decided:** whether the cross-platform list may push its worktree branch and open a draft pull request so CI and the probe run on all three
  platforms before the merge to `main`.
- **Background:** CI triggers only on a push to `main` and on pull requests (`.github/workflows/ci.yml`, 2026-11-16); Windows can be observed only in CI,
  and the CI Mac fails tests the development Mac passes. The repository is public. Pushing is yours to allow.
- **What it affects:** whether acceptance is proved before the merge; how many merges to `main` the work takes; outward actions by an unattended queue.
- **Options:**
  - *Draft pull request* (recommended): the list pushes its branch, opens a draft PR, reads CI with `gh` and loops until green (about 40 minutes per
    Windows run); after the local merge the PR is closed and the remote branch deleted.
  - *After the merge only*: the list merges on figures from the Mac and the container; a group still failing on Windows or the CI Mac becomes another
    round and another merge; `main` stays red meanwhile.
  - *Operator pushes*: the list stops at each CI round for you to push and open the PR.
- **Recommendation:** draft pull request - the narrowest outward action that shows Windows and the CI Mac before the merge.

### D2. The probe workflow after the work

- **What is decided:** whether `.github/workflows/platform-probe.yml` stays once CI is green.
- **Background:** it is needed during the work; afterwards it re-measures the groups when wheels or runner images change.
- **What it affects:** CI minutes on pull requests (Dependabot opens some); a later cross-platform failure starts with figures.
- **Options:**
  - *Keep* (recommended), triggered by hand and by pull requests touching `uv.lock`, `src/metafopy/**`, `src/tests/correctness/**` or the probe.
  - *Delete* it in the list's last task.
- **Recommendation:** keep - a dependency update is when the spread can move, and the filter spares other pull requests.

## Prerequisites

- **Docker running** before the order starts: the cross-platform list reproduces CI's Linux failures in a container, because this Mac's numpy and scipy
  use Apple's Accelerate while CI's Linux and Windows wheels use OpenBLAS, and no Linux machine is available otherwise. Start Docker Desktop.
- **`gh` authenticated** (it is, as of 2026-11-16): the lists read CI logs, runs and the probe's artifacts.

## Answers

Answered on the review page on 2026-11-16 (19:25 UTC), read back from its `answers` store; no comments:

| Item or decision | Answer |
|---|---|
| 1. The plan: two small task lists, no spec | approve |
| 2. The spec `agents/specs/cross-platform-numerics/spec.md` | approve |
| D1. CI runs before the merge | draft pull request |
| D2. The probe workflow after the work | keep |

Applied: the spec is version 1.1, approved, with both answers recorded under its Decisions.
