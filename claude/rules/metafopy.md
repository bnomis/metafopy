# metafopy-specific rules

- R metafor is the oracle: port numerics line for line and compare against committed JSON fixtures.
- Randomised procedures use the R-exact RNG in `metafopy.core.rng`; never `numpy.random` for ported procedures.
- Result classes are frozen dataclasses with a metafor-style `__str__`.
