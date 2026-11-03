# Review: R fused arithmetic spec

**Review page:** https://claude.ai/artifact/7MaNVhFq7778THBJjjHC26

## What is being reviewed and how

The spec `agents/specs/r-fused-arithmetic/spec.md` (proposed 2026-11-03)
makes metafopy's port of R's `dbinom()` reproduce R built with fused
multiply-add (arm64) or without it (x86_64), following the platform it runs
on, so that random `gosh` subsets match R; it tests both variants on every
platform and adds R's platform and arithmetic to the audit's R oracle pin.
It comes from the `TODO.md` item "gosh gives different subsets and results
from R on every fit". The two design choices were answered on 2026-11-03
before the draft and are recorded in the spec, so there is no open
decision. Approve or flag each part below; implementation waits until every
part has an answer.

## Automatic checks

- Every template section present (Module location, API, Error handling,
  Implementation, Settings, External dependencies): pass.
- API has type-annotated signatures: pass.
- Every existing module path named exists; the new fixture, its generator
  and the spec's own files are marked new: pass.
- Size: under the ~200-line guide (`wc -l` on the spec): pass.
- The fused forms checked against R 4.6.2 on 2026-11-03: `dbinom(c(4, 10,
  12), 13, 0.5)`, `dbinom(0, 2:4, 0.3)`, `dbinom(2:4, 2:4, 0.7)` and a seeded
  `gosh(res, subsets = 20)`: pass.

## The items

1. **Decisions recorded** - follow the platform, test both variants; pin
   R's platform and arithmetic, refused unless `--allow-r-drift`.
2. **The cause** - R on arm64 fuses `a*b + c`, the port rounds twice; one
   bit of a probability differs, R's unstable heapsort swaps equal weights,
   and every random subset after it differs.
3. **The fused forms** - `math.fma` in `bd0`'s log branches, `dbinom_raw`'s
   `lc - 0.5*lf` and `pow1p`'s short forms; `dbinom(fused=)` with the
   default `FUSED = platform_fuses()`.
4. **Tests of both variants** - the Linux reference with `fused=False`, a
   new reference from the pinned R with `fused=True`, seeded `gosh` tests
   patching `FUSED`; the Linux-only skip removed.
5. **The pin** - `R_platform` and `R_fma` in `r-versions.json`; `check-r`
   refuses a different build, or R and Python disagreeing on fusing.
6. **Documentation** - `gosh` and `set_seed` docstrings, the audit README.
7. **Rerun and the rest of the item** - `bin/metafor-audit.sh --only gosh`;
   remaining causes get a dated note on the item.
8. **Out of scope** - normal draws (SciPy's `ndtri`), `ptukey`, `dnorm`:
   last-bit differences that nothing sorts on.

## Decisions

None open. The answers below were given on 2026-11-03 in the question
dialog, before the draft:

- Arithmetic: follow the platform (fused on arm64, unfused elsewhere), with
  both variants tested.
- The pin: records R's platform and arithmetic; `check-r` refuses a
  mismatch.

## Answers

Answered on 2026-11-03 on the review page:

- Items 1 to 8: approved, no comments.
