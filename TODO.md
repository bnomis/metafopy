# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Extend the correctness tests beyond the core models

Found on 2026-11-12 while reviewing `TODO.md`: the correctness-tests spec (`agents/specs/correctness-tests/spec.md`, introduction) covers `escalc`,
`rma_uni`, `rma_mh` and `rma_peto`, and says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and
`robust`." No such spec exists yet, so those functions are checked only against R metafor, not against their defining references.

Fix direction: one spec per area with `/specdev`, each listing its defining references, the sources to fetch or ask the user for, and its coverage items in
`src/tests/correctness/test_coverage.py`; do the harness item above first, since several of these areas report back-transformed or interval values.

Spec: `agents/specs/correctness-mv-robust/spec.md`; tasks: `agents/tasks/correctness-mv-robust/tasks-1.md`. Spec:
`agents/specs/correctness-glmm/spec.md`; tasks: `agents/tasks/correctness-glmm/tasks-1.md`. Spec: `agents/specs/correctness-location-scale/spec.md`;
tasks: `agents/tasks/correctness-location-scale/tasks-1.md`. Spec: `agents/specs/correctness-publication-bias/spec.md`; tasks:
`agents/tasks/correctness-publication-bias/tasks-1.md`.
