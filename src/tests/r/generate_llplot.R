# Reference values for llplot(): the likelihoods it returns for the bcg data, as sums over each study (trapezoid rule) and values at fixed
# positions, for measure = "GEN" and measure = "OR" (which needs the BiasedUrn package), scaled and not.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_llplot.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
dat <- escalc(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
num <- function(x) unname(as.vector(x))
summ <- function(lls) list(dim = dim(lls), nna = num(rowSums(is.na(lls))), sums = num(rowSums(lls, na.rm = TRUE)), at = num(lls[, c(1, 100, 250, 400, 500)]), max = max(lls, na.rm = TRUE))

out <- list()
out$gen <- summ(llplot(measure = "GEN", yi = yi, vi = vi, data = dat, xvals = 500))
out$gen_unscaled <- summ(llplot(measure = "GEN", yi = yi, vi = vi, data = dat, xvals = 500, scale = FALSE, level = 95, xlim = c(-3, 2)))
out$or <- summ(llplot(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, xvals = 500, level = 99))
dd <- dat.bcg
dd$tpos[2] <- 0
dd$cpos[2] <- 0
dd$tpos[5] <- 0
out$or_zero <- summ(suppressWarnings(llplot(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dd, xvals = 500, level = 99, subset = c(1:8, 10:13))))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase2/llplot.json")
