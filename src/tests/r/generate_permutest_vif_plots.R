# Reference values for plot() of permutest() and vif(sim = TRUE) results: the histograms drawn (breaks and densities of the main histogram
# and of every shaded tail), the lines and the plot region, recorded on a null device, together with the permutation distributions and
# the simulated VIFs they are drawn from (the Python tests plot those same distributions).
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_permutest_vif_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ ablat + year, data = dat)

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
pt <- permutest(res, iter = 500, progbar = FALSE)
pplot <- instrument(metafor:::plot.permutest.rma.uni)

out <- list()
out$permutest <- list(zval_perm = lapply(pt$zval.perm, num), beta_perm = lapply(pt$beta.perm, num), QM_perm = num(pt$QM.perm),
                      zval = num(pt$zval), beta = num(pt$beta[, 1]), QM = pt$QM, ddf = if (is.na(pt$ddf)) NULL else pt$ddf, coef_names = rownames(pt$beta))
out$p_default <- run(pplot, pt, beta = 2)
out$p_all_curves <- run(pplot, pt, beta = 3, lwd = c(2, 1, 1, 4), breaks = 10)
out$p_greater <- run(pplot, pt, beta = 2, alternative = "greater", lwd = c(2, 1, 1, 4))
out$p_less_trim <- run(pplot, pt, beta = 2, alternative = "less", trim = 0.1, adjust = 2, lwd = c(1, 2, 0, 2))
out$p_px2 <- run(pplot, pt, beta = 2, p2defn = "px2", lwd = c(2, 1, 1, 0))
out$p_px2_other <- run(pplot, pt, beta = 3, p2defn = "px2")
out$p_qm <- run(pplot, pt, beta = NA, QM = TRUE, lwd = c(2, 1, 1, 4))
out$p_coef <- run(pplot, pt, beta = 2, stat = "coef", lwd = c(2, 1, 1, 0))
out$p_freq <- run(pplot, pt, beta = 2, freq = TRUE)

set.seed(4321)
v <- vif(res, btt = 2, sim = TRUE, reps = 300, progbar = FALSE)
out$vif <- list(sim = lapply(seq_len(ncol(v$sim)), function(j) num(v$sim[, j])), vifs = num(v$vifs), names = names(v$vifs), m = v$vif[[1]]$m)
vplot <- instrument(metafor:::plot.vif.rma)
out$v_default <- run(vplot, v)
out$v_density <- run(vplot, v, lwd = c(2, 2), breaks = 8, trim = 0.1, adjust = 1.5)
out$v_freq <- run(vplot, v, freq = TRUE)
out$v_none <- run(vplot, v, lwd = c(0, 0))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase2/permutest_vif_plots.json")
