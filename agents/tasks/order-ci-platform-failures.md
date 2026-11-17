# Order: CI platform failures

Written on 2026-11-16 by `/todo-run-order 1-3`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-ci-platform-failures.md
```

It covers these `TODO.md` items, in this order:

- "A harness test runs Rscript with no skip where R is missing" (row 1);
- "Audit harness tests assume POSIX and fail on Windows" (row 2);
- "Numerical tests whose results or tolerances hold only on the development machine" (row 3), through
  `agents/specs/cross-platform-numerics/spec.md`.

The plan and the spec were reviewed in `agents/reviews/order-ci-platform-failures-specs.md` and answered on 2026-11-16 (every recommendation
accepted): the spec is at version 1.1, row 3 runs CI through a draft pull request, and the probe workflow is kept.

## Prerequisites

- **Docker running** (Docker Desktop, `/Users/simonb/.docker/bin/docker`): row 3 reproduces CI's Linux failures in a container, because this Mac's
  numpy and scipy use Apple's Accelerate while CI's Linux and Windows wheels use OpenBLAS. Row 3's Task 1 stops if `docker info` fails.
- **`gh` authenticated**: row 3 reads CI logs and artifacts, pushes its branch and opens one draft pull request (decision D1).

## Stops for you

Row 3 stops in Tasks 5 to 10 only if a failure group's figures show metafopy below a reference's maximum or off a theoretical value, since metafopy's
output changes only after you confirm it (`claude/rules/metafopy.md`); the other groups carry on first. Its Task 1 stops if a prerequisite above is
missing, so starting Docker before the order lets it run straight through. Rows 1 and 2 run straight through.

## Settings

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | https://claude.ai/artifact/7H489K9ANLy1wUF4pdE6V4 |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `rscript-skip-guard/tasks-1.md` | - | - | merged e7f4567 |
| 2 | `audit-windows-tests/tasks-1.md` | 1 | - | merged 77b7db7 |
| 3 | `cross-platform-numerics/tasks-1.md` | 2 | a group whose figures show metafopy wrong, only if one does (Tasks 5-10) | merged 0abbe0c |

Checkpoint after 3: running
