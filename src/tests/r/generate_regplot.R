# Reference values for regplot(): the data frame it returns (coordinates, symbol sizes, labels, fitted values) and the
# predictions behind the regression line and the bands, for the bcg data with absolute latitude as the moderator.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_regplot.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ ablat, data = dat)
dat$cat <- rep(c("a", "b"), length.out = nrow(dat))
res2 <- rma(yi, vi, mods = ~ ablat + year, data = dat)

num <- function(x) unname(as.vector(x))
sav2list <- function(s) list(xi = num(s$xi), yi = num(s$yi), psize = num(s$psize), label = as.logical(s$label), pred = num(s$pred), order = num(s$order) - 1L)

out <- list()
out$default <- sav2list(regplot(res))
out$pi_label <- sav2list(regplot(res, pi = TRUE, label = "piout", plim = c(NA, 2, 0.5)))
out$ciout <- sav2list(regplot(res, label = "ciout", psize = "seinv", plim = c(0.5, 3)))
out$transf <- sav2list(regplot(res, transf = exp, psize = 1.5, label = c(1, 4)))
out$olim <- sav2list(regplot(res, olim = c(-1.2, 0), xvals = seq(10, 60, length.out = 6)))
out$mod2 <- sav2list(regplot(res2, mod = "year", label = "all"))

xs <- seq(min(dat$ablat) - 0.04 * diff(range(dat$ablat)), max(dat$ablat) + 0.04 * diff(range(dat$ablat)), length.out = 1000)
p <- predict(res, newmods = xs)
out$band <- list(sum_pred = sum(p$pred), sum_ci_lb = sum(p$ci.lb), sum_ci_ub = sum(p$ci.ub), sum_pi_lb = sum(p$pi.lb), sum_pi_ub = sum(p$pi.ub),
                 first = list(pred = p$pred[1], ci_lb = p$ci.lb[1], pi_ub = p$pi.ub[1]))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase2/regplot.json")
