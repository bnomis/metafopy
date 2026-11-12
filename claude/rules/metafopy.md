# metafopy-specific rules

- The publication that defines a method is the first oracle; R metafor is the second. Port numerics line for line from metafor and keep the committed
  JSON parity fixtures. Where the two disagree, a correctness case in `src/tests/correctness/` records it, and metafopy's output changes only after
  the user confirms it.
- Randomised procedures use the R-exact RNG in `metafopy.core.rng`; never `numpy.random` for ported procedures.
- Result classes are frozen dataclasses with a metafor-style `__str__`.
