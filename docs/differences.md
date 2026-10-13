# Differences from metafor

- Names: `.` becomes `_`; indices are 0-based.
- Optimisers: metafor's `optim` BFGS and Nelder-Mead are ported exactly; other optimisers map to SciPy and agree to optimiser tolerance (about 1e-4 relative).
- Not ported: `reporter()`, `emmprep`, `matreg`, `optbeta`/`vccon`/`cvvc` for `rma_mv`, the saddlepoint and HTA/HTB/EDF/EDT tests in `robust`, `rma_glmm` measures `PR`, `RR`, `RD`, `PLN`.
