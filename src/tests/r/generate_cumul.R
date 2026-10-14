# Reference values for forest() and plot() of cumul() results: the values returned by forest.cumul.rma() and the
# data frame returned by plot.cumul.rma(), for the bcg data ordered by year, with and without study labels.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_cumul.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$label <- paste(dat$author, dat$year)
res <- rma(yi, vi, data = dat, slab = label)
res0 <- rma(yi, vi, data = dat)
resfe <- rma(yi, vi, data = dat, method = "FE")

num <- function(x) unname(as.vector(x))
fres <- function(r) list(xlim = num(r$xlim), alim = num(r$alim), at = num(r$at), ylim = num(r$ylim), rows = num(r$rows), cex = r$cex, textpos = num(r$textpos))
pres <- function(d) list(estim = num(d$estim), yval = num(d$yval), col = as.character(d$col))
cum <- cumul(res, order = dat$year)
cum0 <- cumul(res0, order = dat$year)
cumfe <- cumul(resfe, order = dat$year)

out <- list()
out$forest <- fres(forest(cum))
out$forest_slab_null <- fres(forest(cum0))
out$forest_transf <- fres(forest(cum, atransf = exp, at = log(c(0.25, 0.5, 1)), xlim = c(-6, 3), digits = c(2L, 3L)))
out$forest_exp <- fres(forest(cumul(res, order = dat$year, transf = exp), refline = 1, steps = 3))
out$plot_tau2 <- pres(plot(cum))
out$plot_I2 <- pres(plot(cum, yaxis = "I2", transf = exp))
out$plot_H2 <- pres(plot(cum, yaxis = "H2", atransf = exp, at = log(c(0.4, 0.6, 0.8))))
out$plot_fe <- pres(plot(cumfe))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase2/cumul_plots.json")
