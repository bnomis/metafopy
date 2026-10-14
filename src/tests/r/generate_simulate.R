# Reference values for simulate() on rma.uni, rma.ls and rma.mv fits.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_simulate.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$yi[3] <- NA
m <- function(x) unname(as.matrix(x))
out <- list()
uni <- rma(yi, vi, mods = ~ablat, data = dat)
out$uni <- m(simulate(uni, nsim = 3, seed = 1234))
out$uni_rows <- rownames(simulate(uni, nsim = 1, seed = 1))
out$uni_olim <- m(simulate(uni, nsim = 2, seed = 42, olim = c(-1.5, 0)))
out$uni_withvb <- m(simulate(uni, nsim = 2, seed = 7, withvb = TRUE))
set.seed(99)
out$uni_global <- m(simulate(uni, nsim = 2))
out$uni_global_next <- rnorm(2)
ls <- rma(yi, vi, mods = ~ablat, scale = ~year, data = dat)
out$ls <- m(simulate(ls, nsim = 2, seed = 5))
mv <- rma.mv(yi, vi, random = ~ 1 | district/school, data = dat.konstantopoulos2011)
out$mv <- m(simulate(mv, nsim = 2, seed = 11))
write_json(out, "src/tests/reference/phase7/simulate.json", digits = NA, pretty = TRUE)
