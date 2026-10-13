# Reference values for the plot functions: forest(), addpoly(), funnel(), baujat(), radial(), qqnorm(),
# labbe(), plot.gosh(), plot.infl() and plot.profile().
#
# Each plot function is run on a null device with the graphics primitives it calls (polygon, segments,
# rect, text, points, abline, axis, mtext, lines, ...) replaced by wrappers that record their arguments
# before drawing; the recorded geometry is dumped together with the values the function returns
# (xlim, alim, at, rows, ... for forest(); the plotted coordinates for the others) and the usr/axis
# limits of the plot region. Long coordinate vectors (polygons, smooth curves) are stored as a summary
# (length, sum and first/last values).
#
# Usage (from the repository root):  Rscript tests/r/generate_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
pdf(NULL, width = 7, height = 7)

### recording machinery ---------------------------------------------------------------------------------

.rec <- new.env()
.rec$log <- list()
rec_reset <- function() .rec$log <- list()
rec_add <- function(type, ...) .rec$log[[length(.rec$log) + 1L]] <- c(list(type = type), list(...))

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
vsum <- function(v) {
  v <- num(v)
  if (is.null(v) || length(v) <= 40L) return(v)
  list(n = length(v), sum = sum(v, na.rm = TRUE), head = v[1:3], tail = v[(length(v) - 2L):length(v)])
}
first <- function(v) if (is.null(v)) NULL else num(v)[1]
chr <- function(v) if (is.null(v)) NULL else as.character(v)

wrappers <- list(
  polygon = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("polygon", x = vsum(x), y = vsum(y), col = first(a$col), border = first(a$border))
    graphics::polygon(x, y, ...)
  },
  segments = function(x0, y0, x1, y1 = x0, ...) {
    a <- list(...)
    rec_add("segments", x0 = num(x0), y0 = num(y0), x1 = num(x1), y1 = num(y1), lty = chr(a$lty), col = first(a$col))
    graphics::segments(x0, y0, x1, y1, ...)
  },
  rect = function(xleft, ybottom, xright, ytop, ...) {
    a <- list(...)
    rec_add("rect", xleft = vsum(xleft), ybottom = vsum(ybottom), xright = vsum(xright), ytop = vsum(ytop), col = first(a$col))
    graphics::rect(xleft, ybottom, xright, ytop, ...)
  },
  text = function(x, y = NULL, labels = NULL, ...) {
    a <- list(...)
    rec_add("text", x = num(x), y = num(y), labels = if (is.expression(labels)) paste(deparse(labels[[1]]), collapse = "") else chr(labels),
            pos = num(a$pos), font = first(a$font), cex = first(a$cex), col = chr(a$col))
    graphics::text(x, y, labels, ...)
  },
  points = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("points", x = num(x), y = num(y), pch = num(a$pch), cex = num(a$cex), col = chr(a$col), bg = chr(a$bg))
    graphics::points(x, y, ...)
  },
  abline = function(a = NULL, b = NULL, h = NULL, v = NULL, ...) {
    d <- list(...)
    rec_add("abline", a = num(a), b = num(b), h = num(h), v = num(v), lty = chr(d$lty), col = chr(d$col))
    graphics::abline(a, b, h, v, ...)
  },
  lines = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("lines", x = vsum(x), y = vsum(y), lty = chr(a$lty), col = chr(a$col))
    graphics::lines(x, y, ...)
  },
  axis = function(side, at = NULL, labels = TRUE, ...) {
    rec_add("axis", side = side, at = num(at), labels = if (is.logical(labels)) NULL else chr(labels))
    graphics::axis(side, at, labels, ...)
  },
  mtext = function(text, side = 3, line = 0, ...) {
    a <- list(...)
    rec_add("mtext", text = if (is.expression(text)) paste(deparse(text[[1]]), collapse = "") else chr(text), side = side, line = line, at = first(a$at), adj = first(a$adj))
    graphics::mtext(text, side, line, ...)
  },
  grid = function(...) {
    a <- list(...)
    rec_add("grid", col = chr(a$col), x = num(axTicks(1)), y = num(axTicks(2)))
    graphics::grid(...)
  },
  plot = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("plot", xlim = num(a$xlim), ylim = num(a$ylim), xlab = if (is.null(a$xlab)) NULL else if (is.expression(a$xlab)) paste(deparse(a$xlab[[1]]), collapse = "") else chr(a$xlab),
            ylab = if (is.null(a$ylab)) NULL else if (is.expression(a$ylab)) paste(deparse(a$ylab[[1]]), collapse = "") else chr(a$ylab))
    graphics::plot(x, y, ...)
  }
)

instrument <- function(f) {
  e <- new.env(parent = environment(f))
  for (nm in names(wrappers)) assign(nm, wrappers[[nm]], envir = e)
  environment(f) <- e
  f
}

# collect what was recorded since rec_reset(), by primitive
collect <- function() {
  log <- .rec$log
  types <- unique(vapply(log, function(l) l$type, ""))
  out <- lapply(types, function(t) lapply(Filter(function(l) l$type == t, log), function(l) { l$type <- NULL; l }))
  names(out) <- types
  out$usr <- num(par("usr"))
  out$axticks <- list(x = num(axTicks(1)), y = num(axTicks(2)),
                      xl = format(axTicks(1), trim = TRUE), yl = format(axTicks(2), trim = TRUE))
  out
}

fforest.rma <- instrument(metafor:::forest.rma)
fforest.default <- instrument(metafor:::forest.default)
faddpoly.default <- instrument(metafor:::addpoly.default)
faddpoly.rma <- instrument(metafor:::addpoly.rma)
environment(faddpoly.rma)$addpoly <- function(...) faddpoly.default(...)
ffunnel.rma <- instrument(metafor:::funnel.rma)
ffunnel.default <- instrument(metafor:::funnel.default)
fbaujat.rma <- instrument(metafor:::baujat.rma)
fradial.rma <- instrument(metafor:::radial.rma)
fqqnorm.rma.uni <- instrument(metafor:::qqnorm.rma.uni)
flabbe.rma <- instrument(metafor:::labbe.rma)
fplot.infl.rma.uni <- instrument(metafor:::plot.infl.rma.uni)
fplot.profile.rma <- instrument(metafor:::plot.profile.rma)

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
dat$label <- paste(dat$author, dat$year)
out <- list()

forest_out <- function(r) {
  l <- lapply(r, function(v) if (is.numeric(v) || is.character(v)) num(v) else v)
  l
}
run <- function(expr) {
  rec_reset()
  plot.new()
  ret <- force(expr)
  list(ret = if (is.null(ret)) NULL else lapply(ret, num), rec = collect())
}

### forest -----------------------------------------------------------------------------------------------

res <- rma(yi, vi, data = dat, slab = label)
res_noslab <- rma(yi, vi, data = dat)
res_mods <- rma(yi, vi, mods = ~ ablat, data = dat, slab = label)
res_ee <- rma(yi, vi, data = dat, method = "EE", slab = label)

out$forest_re <- run(fforest.rma(res))
out$forest_noslab <- run(fforest.rma(res_noslab))
out$forest_exp <- run(fforest.rma(res, atransf = exp, at = log(c(0.05, 0.25, 1, 4)), xlim = c(-16, 6), ilab = cbind(dat$tpos, dat$tneg, dat$cpos, dat$cneg),
                                  ilab.lab = c("TB+", "TB-", "CTRL+", "CTRL-"), ilab.xpos = c(-9.5, -8, -6, -4.5), cex = 0.75, header = c("Author(s) and Year", "RR [95% CI]"),
                                  digits = 3L, efac = c(1, 2)))
out$forest_transf <- run(fforest.rma(res, transf = exp, alim = c(0, 5), refline = 1, order = "obs", showweights = TRUE, psize = 1, pch = 18, shade = "zebra2"))
out$forest_mods <- run(fforest.rma(res_mods, order = "fit", addpred = TRUE, shade = TRUE, plim = c(1, 2), colout = "blue"))
out$forest_mods_resid <- run(fforest.rma(res_mods, order = "resid", level = 90, annotate = FALSE, xlab = "Log RR", header = FALSE, ylim = c(-1, 16)))
out$forest_pi_polygon <- run(fforest.rma(res, predstyle = "polygon", showweights = TRUE, mlab = "RE Model", col = c("blue", "red")))
out$forest_pi_bar <- run(fforest.rma(res, predstyle = "bar", level = 90, atransf = exp, at = log(c(0.1, 1, 10)), xlim = c(-8, 6)))
out$forest_pi_line <- run(fforest.rma(res, addpred = TRUE, steps = 7, alim = c(-3, 3), rows = 15:3, ylim = c(-1, 19)))
out$forest_ee <- run(fforest.rma(res_ee, addfit = FALSE, xlab = c("Left", "Mid", "Right"), refline = NULL, top = 4))
out$forest_arrows <- run(fforest.rma(res, alim = c(-1, 1), xlim = c(-6, 4), olim = c(-1.2, 1.2), cex = 1.2, width = 7, annosym = c(" (", " to ", ")")))
out$forest_dist <- run(fforest.rma(res, order = "prec", rowadj = c(0.1, -0.1, 0), textpos = c(-7, 5), xlim = c(-8, 6), lty = c("dashed", "dashed", "solid")))

# subgroups
dat$alloc <- dat.bcg$alloc
res_r <- rma(yi, vi, data = dat, subset = alloc == "random", slab = label)
res_a <- rma(yi, vi, data = dat, subset = alloc == "alternate", slab = label)
res_s <- rma(yi, vi, data = dat, subset = alloc == "systematic", slab = label)
rows_all <- c(3:8, 12:14, 18:21)
rec_reset(); plot.new()
fit_all <- rma(yi, vi, data = dat, slab = label, subset = order(dat$alloc))
rr <- fforest.rma(fit_all, rows = rows_all, ylim = c(-1, 25), xlim = c(-8, 5), addfit = FALSE, cex = 0.8)
faddpoly.rma(res_a, row = 16.5, mlab = "RE Model for Subgroup A", cex = 0.8)
faddpoly.rma(res_r, row = 9.5, mlab = "RE Model for Subgroup B", cex = 0.8, col = "red", border = "red")
faddpoly.rma(res_s, row = 1.5, mlab = "RE Model for Subgroup C", cex = 0.8, annotate = FALSE)
out$forest_subgroups <- list(ret = lapply(rr, num), rec = collect())

# plain yi/vi form
yi <- dat$yi; vi <- dat$vi; slab <- dat$label
out$forest_default <- run(fforest.default(yi, vi, slab = slab))
out$forest_default_opts <- run(fforest.default(yi, sei = sqrt(vi), slab = slab, order = "prec", showweights = TRUE, transf = exp, alim = c(0, 4), refline = 1, ilab = dat$year, ilab.lab = "Year", ilab.xpos = -3,
                                               ilab.pos = 2, col = rep(c("black", "red"), length.out = 13), shade = "zebra", psize = 1:13 / 5, pch = 19, header = "Trial", xlab = c("favours A", "favours B"), cex = 0.9))
out$forest_default_ci <- run(fforest.default(yi, ci.lb = yi - 0.3, ci.ub = yi + 0.5, level = 90, slab = NA, efac = c(1, 1.5), digits = c(2L, 1L, 3L), atransf = function(x) x / 2))
yi_na <- yi; yi_na[c(3, 7)] <- NA
out$forest_default_na <- run(fforest.default(yi_na, vi, slab = slab, subset = -1, rows = c(13:2) + 0, ylim = c(0, 17)))
out$forest_default_sub <- run(fforest.default(yi, vi, slab = slab, order = dat$year, subset = dat$year > 1950, shade = dat$year > 1960 & dat$year < 1970 | dat$year == 1948 | TRUE))

# addpoly, numeric form
rec_reset(); plot.new()
rr <- fforest.default(yi, vi, slab = slab, ylim = c(-4, 17), xlim = c(-8, 6))
faddpoly.default(c(-0.5, 0.2), sei = c(0.1, 0.2), rows = c(-1.5, -2.5), mlab = c("A", "B"), cex = 0.9)
faddpoly.default(0.3, ci.lb = -0.2, ci.ub = 0.9, pi.lb = -0.8, pi.ub = 1.4, rows = -3.2, predstyle = "polygon", mlab = "C", annotate = FALSE)
faddpoly.default(-0.3, vi = 0.05, rows = -1, transf = exp, efac = 2, constarea = TRUE)
out$addpoly_default <- list(ret = lapply(rr, num), rec = collect())
rec_reset(); plot.new()
rr <- fforest.rma(res, ylim = c(-5, 17), addfit = FALSE)
faddpoly.rma(res, row = -1.5, addpred = TRUE, mlab = "RE model", atransf = exp)
faddpoly.rma(res, row = -3, predstyle = "bar", level = 90, annotate = TRUE, digits = 3L, col = "gray")
out$addpoly_rma <- list(ret = lapply(rr, num), rec = collect())

### funnel -----------------------------------------------------------------------------------------------

res_ni <- rma(yi, vi, data = dat, slab = label, ni = n)
res_mods_ni <- rma(yi, vi, mods = ~ ablat, data = dat, slab = label, ni = n)
res_tf <- trimfill(res_ni)
for (ya in c("sei", "vi", "seinv", "vinv", "ni", "ninv", "sqrtni", "sqrtninv", "lni", "wi")) {
  out[[paste0("funnel_", ya)]] <- local({ yy <- ya; run(ffunnel.rma(res_ni, yaxis = yy)) })
}
out$funnel_levels <- run(ffunnel.rma(res_ni, level = c(90, 95, 99), shade = c("white", "gray55", "gray75"), refline = 0, back = "gray90", hlines = "white",
                                     steps = 7, atransf = exp, at = log(c(0.25, 0.5, 1, 2)), digits = c(2L, 2L), label = "out", xlim = c(-2.5, 2), ylim = c(0, 1.2)))
out$funnel_levels_ramp <- run(ffunnel.rma(res_ni, level = c(90, 95, 99), yaxis = "seinv", label = 3, offset = 0.2, lty = c(2, 4), pch = 15, col = "blue", bg = "red", addtau2 = TRUE))
out$funnel_vi_refline2 <- run(ffunnel.rma(res_ni, yaxis = "vi", refline = -0.5, level = 90, xlab = "Effect", ylab = "V", digits = 3L, label = TRUE))
out$funnel_mods <- run(ffunnel.rma(res_mods_ni, yaxis = "seinv"))
out$funnel_mods_stud <- run(ffunnel.rma(res_mods_ni, type = "rstudent", label = 2L, steps = 4))
out$funnel_ee <- run(ffunnel.rma(res_ee, yaxis = "vinv", level = c(95, 99)))
out$funnel_tf <- run(ffunnel.rma(res_tf))
out$funnel_tf_opts <- run(ffunnel.rma(res_tf, yaxis = "seinv", pch.fill = 17, col = c("black", "red"), bg = c("gray", "yellow"), level = c(90, 95, 99), label = "all"))
out$funnel_tf_ni <- run(ffunnel.rma(res_tf, yaxis = "ni"))
out$funnel_pchvec <- run(ffunnel.rma(res_ni, pch = rep(c(1, 19), length.out = 13), col = rep(c("black", "blue"), length.out = 13), refline = 0.1))
out$funnel_default <- run(ffunnel.default(yi, vi))
out$funnel_default_opts <- run(ffunnel.default(yi, sei = sqrt(vi), yaxis = "seinv", level = c(90, 99), refline = -0.7, slab = slab, label = 4, subset = -1))
out$funnel_default_ni <- run(ffunnel.default(yi, ni = dat$n, yaxis = "sqrtni", xlab = "A"))
out$funnel_default_wi <- run(ffunnel.default(yi, vi, yaxis = "wi", ylim = c(1, 20)))
out$funnel_default_na <- run(ffunnel.default(yi_na, vi))

### baujat -----------------------------------------------------------------------------------------------

out$baujat_re <- run(fbaujat.rma(res))
out$baujat_ee <- run(fbaujat.rma(res_ee, symbol = "slab", cex = 0.6, grid = "red", xlab = "X", ylab = "Y"))
out$baujat_mods <- run(fbaujat.rma(res_mods, symbol = 16, grid = FALSE, xlim = c(0, 12), ylim = c(0, 3)))
out$baujat_symvec <- run(fbaujat.rma(res_mods, symbol = rep(c(1, 2, 19), length.out = 13), cex = 1.2))

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "tests/reference/phase2/plots.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote tests/reference/phase2/plots.json\n")
