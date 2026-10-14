# Reference values for plot() of a location-scale permutest() result (alpha= and QS=): the histograms drawn, the lines and the plot
# region, recorded on a null device as in generate_permutest_vif_plots.R, together with the permutation distributions they are drawn from.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_permutest_ls_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ ablat, scale = ~ year, data = dat)

num <- function(x) unname(as.vector(x))
vsum <- function(v) { v <- num(v); if (length(v) <= 20L) v else list(n = length(v), sum = sum(v), first = v[1:3], last = v[(length(v) - 2L):length(v)]) }
.log <- new.env()
.log$ev <- list()
add <- function(type, ...) .log$ev[[length(.log$ev) + 1L]] <- c(list(type = type), list(...))

ct <- metafor:::.coltail
e_ct <- new.env(parent = environment(ct))
assign("lines", function(x, ...) {
  add("tail", breaks = num(x$breaks), density = num(x$density), counts = num(x$counts))
  graphics::lines(x, ...)
}, envir = e_ct)
environment(ct) <- e_ct

instrument <- function(f) {
  e <- new.env(parent = environment(f))
  assign(".coltail", ct, envir = e)
  assign("hist", function(x, breaks, plot = TRUE, ylim = NULL, main = NULL, xlab = NULL, freq = NULL, ...) {
    h <- graphics::hist(x, breaks = breaks, plot = plot, ylim = ylim, main = main, xlab = xlab, freq = freq, ...)
    if (plot) add("hist", breaks = num(h$breaks), density = num(h$density), counts = num(h$counts), ylim = num(ylim), main = main, xlab = xlab)
    invisible(h)
  }, envir = e)
  assign("abline", function(...) { a <- list(...); add("abline", v = num(a$v), h = num(a$h), lwd = num(a$lwd), lty = as.character(a$lty)); graphics::abline(...) }, envir = e)
  assign("lines", function(x, y = NULL, ...) {
    a <- list(...)
    xy <- grDevices::xy.coords(x, y)
    add("lines", x = vsum(xy$x), y = vsum(xy$y), lwd = num(a$lwd), col = as.character(a$col))
    graphics::lines(x, y, ...)
  }, envir = e)
  assign("segments", function(x0, y0, x1, y1, ...) { a <- list(...); add("segments", x0 = num(x0), x1 = num(x1), lwd = num(a$lwd)); graphics::segments(x0, y0, x1, y1, ...) }, envir = e)
  assign("arrows", function(x0, y0, x1, y1, ...) { add("arrows", x0 = num(x0), y0 = num(y0), x1 = num(x1), y1 = num(y1)); graphics::arrows(x0, y0, x1, y1, ...) }, envir = e)
  environment(f) <- e
  f
}
run <- function(f, ...) {
  .log$ev <- list()
  f(...)
  list(events = .log$ev, usr = num(par("usr")))
}

set.seed(1234)
pt <- suppressWarnings(permutest(res, iter = 200, progbar = FALSE))
pplot <- instrument(metafor:::plot.permutest.rma.uni)
mains <- function(r) unlist(lapply(r$events, function(ev) if (ev$type == "hist") ev$main else NULL))

out <- list()
out$permutest <- list(zval_perm = lapply(pt$zval.perm, num), beta_perm = lapply(pt$beta.perm, num), QM_perm = num(pt$QM.perm),
                      zval_alpha_perm = lapply(pt$zval.alpha.perm, num), alpha_perm = lapply(pt$alpha.perm, num), QS_perm = num(pt$QS.perm),
                      zval = num(pt$zval), beta = num(pt$beta[, 1]), QM = pt$QM, zval_alpha = num(pt$zval.alpha), alpha = num(pt$alpha[, 1]),
                      QS = pt$QS, coef_names = rownames(pt$beta), alpha_names = rownames(pt$alpha))
# with one scale coefficient the chi-square(1) reference density is infinite at 0, which R cannot plot, so it is left out
out$p_qs <- run(pplot, pt, beta = NA, alpha = NA, QS = TRUE, lwd = c(2, 0, 1, 4))
out$p_alpha <- run(pplot, pt, beta = NA, alpha = 2, lwd = c(2, 1, 1, 4))
out$p_alpha_coef <- run(pplot, pt, beta = NA, alpha = 2, stat = "coef")
out$mains_all <- mains(run(pplot, pt, QM = TRUE, QS = TRUE))
out$mains_default <- mains(run(pplot, pt))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase7/permutest_ls_plots.json")
