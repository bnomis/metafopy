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
# Usage (from the repository root):  Rscript src/tests/r/generate_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
pdf(NULL, width = 7, height = 7)

### recording machinery ---------------------------------------------------------------------------------

.rec <- new.env()
.rec$log <- list()
# note: forest() sets par(family, font) and does not reset them, so reset them here to keep the cases independent
rec_reset <- function() {
  .rec$log <- list()
  par(family = "", font = 1)
}
rec_add <- function(type, ...) .rec$log[[length(.rec$log) + 1L]] <- c(list(type = type), list(...))

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
vsum <- function(v) {
  v <- num(v)
  if (is.null(v) || length(v) <= 40L) return(v)
  list(n = length(v), sum = sum(v, na.rm = TRUE), head = v[1:3], tail = v[(length(v) - 2L):length(v)])
}
first <- function(v) if (is.null(v)) NULL else num(v)[1]
chr <- function(v) if (is.null(v)) NULL else as.character(v)
# a long character vector (the colours of many rectangles) as a summary
csum <- function(v) {
  v <- chr(v)
  if (is.null(v) || length(v) <= 40L) return(v)
  list(n = length(v), head = v[1:3], tail = v[(length(v) - 2L):length(v)], uniq = length(unique(v)))
}
# the font family and font set via par(), recorded only when they differ from the defaults
fontinfo <- function() {
  l <- list()
  if (nzchar(par("family"))) l$family <- par("family")
  if (par("font") != 1) l$fontpar <- par("font")
  l
}

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
    do.call(rec_add, c(list("rect", xleft = vsum(xleft), ybottom = vsum(ybottom), xright = vsum(xright), ytop = vsum(ytop), col = first(a$col)),
                       if (length(a$col) > 1L) list(cols = csum(a$col))))
    graphics::rect(xleft, ybottom, xright, ytop, ...)
  },
  text = function(x, y = NULL, labels = NULL, ...) {
    a <- list(...)
    do.call(rec_add, c(list("text", x = num(x), y = num(y), labels = if (is.expression(labels)) paste(deparse(labels[[1]]), collapse = "") else chr(labels),
                            pos = num(a$pos), font = first(a$font), cex = first(a$cex), col = chr(a$col)), fontinfo()))
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
    xy <- grDevices::xy.coords(x, y)
    rec_add("lines", x = vsum(xy$x), y = vsum(xy$y), lty = chr(a$lty), col = first(a$col), lwd = first(a$lwd))
    graphics::lines(x, y, ...)
  },
  axis = function(side, at = NULL, labels = TRUE, ...) {
    do.call(rec_add, c(list("axis", side = side, at = num(at), labels = if (is.logical(labels)) NULL else chr(labels)), fontinfo()))
    graphics::axis(side, at, labels, ...)
  },
  mtext = function(text, side = 3, line = 0, ...) {
    a <- list(...)
    do.call(rec_add, c(list("mtext", text = if (is.expression(text)) paste(deparse(text[[1]]), collapse = "") else chr(text), side = side, line = line, at = first(a$at), adj = first(a$adj)),
                       if (!is.null(a$font)) list(font = first(a$font)), fontinfo()))
    graphics::mtext(text, side, line, ...)
  },
  pairs = function(x, ...) {
    a <- list(...)
    rec_add("pairs", nr = nrow(x), nc = ncol(x), colsum = num(colSums(x, na.rm = TRUE)), nna = num(colSums(is.na(x))), colhead = as.vector(unlist(x[1, ])), pch = first(a$pch), cex = first(a$cex),
            cols = if (is.character(a$col)) unique(a$col) else NULL, ncol = length(a$col), labels = if (is.expression(a$labels)) vapply(a$labels, function(l) paste(deparse(l), collapse = ""), "") else chr(a$labels))
    graphics::pairs(x, ...)
  },
  grid = function(...) {
    a <- list(...)
    rec_add("grid", col = chr(a$col), x = num(axTicks(1)), y = num(axTicks(2)))
    graphics::grid(...)
  },
  plot = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("plot", xlim = num(a$xlim), ylim = num(a$ylim), xlab = if (is.null(a$xlab)) NULL else if (is.expression(a$xlab)) paste(deparse(a$xlab[[1]]), collapse = "") else chr(a$xlab),
            ylab = if (is.null(a$ylab)) NULL else if (is.expression(a$ylab)) paste(deparse(a$ylab[[1]]), collapse = "") else chr(a$ylab),
            xlab_expr = is.expression(a$xlab), ylab_expr = is.expression(a$ylab),
            xdat = if (is.numeric(x)) vsum(x) else NULL, ydat = if (is.numeric(y)) vsum(y) else NULL,
            pch = first(a$pch), cex = first(a$cex), cols = if (is.character(a$col)) unique(a$col) else NULL, ncol = length(a$col))
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
fplot.gosh.rma <- instrument(metafor:::plot.gosh.rma)
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
  list(ret = if (is.null(ret) || inherits(ret, "density")) NULL else lapply(ret, num), rec = collect())
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

# prediction interval styles 'shade' and 'dist', and fonts
res_knha <- rma(yi, vi, data = dat, slab = label, test = "knha")
out$forest_pi_shade <- run(fforest.rma(res, predstyle = "shade"))
out$forest_pi_shade_exp <- run(fforest.rma(res, predstyle = "shade", transf = exp, alim = c(0, 5), refline = 1, col = c("black", "darkblue", "gray90"), showweights = TRUE))
out$forest_pi_shade_knha <- run(fforest.rma(res_knha, predstyle = "shade", level = 90, atransf = exp, at = log(c(0.1, 0.5, 1, 2)), xlim = c(-6, 4)))
out$forest_pi_dist <- run(fforest.rma(res, predstyle = "dist"))
out$forest_pi_dist_exp <- run(fforest.rma(res, predstyle = "dist", transf = exp, alim = c(0, 5), refline = 1, efac = c(1, 2)))
out$forest_pi_dist_tail <- run(fforest.rma(res, predstyle = "dist", predtail = "<-0.5", level = 90, mlab = "RE", col = c("black", "gray40", "lightblue", "pink"),
                                            border = c("black", "blue"), olim = c(-2.5, 1.5), alim = c(-2, 1), digits = 1L))
out$forest_pi_dist_knha <- run(fforest.rma(res_knha, predstyle = "dist", predtail = 0.2))
out$forest_pi_dist_predlim <- run(fforest.rma(res, predstyle = "dist", transf = exp, predlim = c(0.05, 3), alim = c(0, 3), refline = 1))
out$forest_pi_dist_ilogit <- run(fforest.rma(res, predstyle = "dist", transf = transf.ilogit, alim = c(0, 1), refline = 0.5))
pdxs <- seq(-3, 2, length.out = 301)
out$forest_pi_dist_preddist <- run(fforest.rma(res, predstyle = "dist", preddist = list(pdxs, dnorm(pdxs, -0.7, 0.6), level = 90)))
out$forest_pi_shade_preddist <- run(fforest.rma(res, predstyle = "shade", preddist = list(pdxs, dnorm(pdxs, -0.7, 0.6), pi.lb = -1.8, pi.ub = 0.4), predlim = c(-2.5, 1)))
out$forest_fonts <- run(fforest.rma(res, fonts = c("serif", "mono"), ilab = dat$year, ilab.lab = "Year", ilab.xpos = -7, xlim = c(-10, 6)))
out$forest_fonts_named <- run(fforest.rma(res, fonts = c(mono = 2, serif = 3, sans = 4), predstyle = "polygon", ilab = dat$year, ilab.lab = "Year", ilab.xpos = -7, xlim = c(-10, 6), xlabfont = 3))
out$forest_fonts_one <- run(fforest.rma(res, fonts = "serif", predstyle = "dist", showweights = TRUE))

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
out$forest_default_fonts <- run(fforest.default(yi, vi, slab = slab, fonts = c(mono = 1, serif = 3), ilab = dat$year, ilab.xpos = -7, xlim = c(-10, 6)))
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

# addpoly with the prediction styles 'shade' and 'dist', and with fonts
rec_reset(); plot.new()
rr <- fforest.rma(res, ylim = c(-8, 17), addfit = FALSE, fonts = c("serif", "mono"))
faddpoly.rma(res, row = -1.5, predstyle = "shade")
faddpoly.rma(res, row = -4, predstyle = "dist", level = 90, predtail = ">-0.3", atransf = exp)
faddpoly.rma(res, row = -6.5, predstyle = "polygon", fonts = c("sans", "mono"))
out$addpoly_rma_pred <- list(ret = lapply(rr, num), rec = collect())
rec_reset(); plot.new()
rr <- fforest.rma(res, ylim = c(-8, 17), addfit = FALSE, transf = exp, alim = c(0, 5))
faddpoly.rma(res, row = -1.5, predstyle = "dist", transf = exp, efac = c(1, 1.5))
faddpoly.rma(res_knha, row = -4.5, predstyle = "shade", transf = exp)
out$addpoly_rma_pred_exp <- list(ret = lapply(rr, num), rec = collect())
rec_reset(); plot.new()
rr <- fforest.default(yi, vi, slab = slab, ylim = c(-8, 17), xlim = c(-8, 6))
faddpoly.default(0.3, ci.lb = -0.2, ci.ub = 0.9, pi.lb = structure(-0.8, level = 0.1, dist = "t", ddf = 6, se = 0.55), pi.ub = 1.4, rows = -2, predstyle = "dist", mlab = "C")
faddpoly.default(-0.3, ci.lb = -0.7, ci.ub = 0.1, pi.lb = structure(-1.2, se = 0.5), pi.ub = 0.6, rows = -5, predstyle = "shade", annotate = FALSE, fonts = c(mono = 2, serif = 3))
faddpoly.default(-0.3, ci.lb = -0.7, ci.ub = 0.1, pi.lb = -1.2, pi.ub = 0.6, rows = -7, predstyle = "dist", preddist = list(pdxs, dnorm(pdxs, -0.3, 0.5)), predlim = c(-2, 1))
out$addpoly_default_pred <- list(ret = lapply(rr, num), rec = collect())

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

### radial -----------------------------------------------------------------------------------------------

res_ee_nl <- res_ee
out$radial_re <- run(fradial.rma(res))
out$radial_ee_center <- run(fradial.rma(res_ee, center = TRUE, steps = 5, digits = 3L))
out$radial_opts <- run(fradial.rma(res, xlim = c(0, 12), zlim = c(-6, 6), atz = c(-4, 0, 4), aty = c(-1.5, -1, 0, 0.5), transf = exp, level = 90, cex = 1.2, pch = 19, arc.res = c(60, 30)))
out$radial_labels <- run(fradial.rma(res, xlab = "Precision", zlab = "Z", back = "gray90", col = "blue", bg = "red", cex.axis = 0.8, cex.lab = 1.1))

### qqnorm -----------------------------------------------------------------------------------------------

out$qqnorm_re <- run(fqqnorm.rma.uni(res, seed = 1234))
out$qqnorm_stud <- run(fqqnorm.rma.uni(res_mods, type = "rstudent", seed = 42, label = TRUE, grid = TRUE, level = 90, reps = 300, pos = 24))
out$qqnorm_bonf <- run(fqqnorm.rma.uni(res_ee, seed = 7, bonferroni = TRUE, smooth = FALSE, label = "out", envelope = "gray85", lty = c("dashed", "solid"), pch = 19, col = "blue"))
out$qqnorm_label <- run(fqqnorm.rma.uni(res, seed = 99, label = 3, offset = 0.5, pos = 13, bass = 2, reps = 200))
out$qqnorm_noenv <- run(fqqnorm.rma.uni(res_mods, envelope = FALSE, label = 2, pos = 4))

### labbe ------------------------------------------------------------------------------------------------

raw <- dat.bcg
raw$label <- paste(raw$author, raw$year)
raw$x1i <- raw$tpos; raw$t1i <- (raw$tpos + raw$tneg) / 100; raw$x2i <- raw$cpos; raw$t2i <- (raw$cpos + raw$cneg) / 100
res_rr <- rma(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, measure = "RR", slab = label)
res_or <- rma(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, measure = "OR", slab = label)
res_rd <- rma(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, measure = "RD", slab = label)
res_as <- rma(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, measure = "AS", slab = label)
res_irr <- rma(x1i = x1i, t1i = t1i, x2i = x2i, t2i = t2i, data = raw, measure = "IRR", slab = label)
res_ird <- rma(x1i = x1i, t1i = t1i, x2i = x2i, t2i = t2i, data = raw, measure = "IRD", slab = label)
res_irsd <- rma(x1i = x1i, t1i = t1i, x2i = x2i, t2i = t2i, data = raw, measure = "IRSD", slab = label)
out$labbe_rr <- run(flabbe.rma(res_rr))
out$labbe_or_ci <- run(flabbe.rma(res_or, ci = TRUE, pi = TRUE, grid = TRUE, transf = exp, flip = TRUE, pch = 19, col = "blue", lty = c("dotted", "solid")))
out$labbe_rd <- run(flabbe.rma(res_rd, plim = c(NA, 4, 1), ci = "gray80", lim = c(0, 0.2), xlab = "A", ylab = "B", add = 0.25))
out$labbe_as <- run(flabbe.rma(res_as, psize = 1:13 / 4, pi = TRUE, to = "all", legend = TRUE))
out$labbe_irr <- run(flabbe.rma(res_irr, xlim = c(-8, -2), ylim = c(-8, -2), grid = "red"))
out$labbe_ird <- run(flabbe.rma(res_ird, ci = TRUE, bg = "yellow", plim = c(1, NA, 2)))
out$labbe_irsd <- run(flabbe.rma(res_irsd, flip = TRUE, llim = c(-0.05, 0.1), transf = function(x) x * 100))

### plot.profile -----------------------------------------------------------------------------------------

pf <- profile(res, plot = FALSE, steps = 15, xlim = c(0.01, 0.9))
out$profile_re <- run(fplot.profile.rma(pf, cline = TRUE))
pf2 <- profile(res, plot = FALSE, steps = 9, xlim = c(0.4, 0.9), sqrt = TRUE)
out$profile_sqrt <- run(fplot.profile.rma(pf2, refline = FALSE, pch = 17, ylim = c(-14, -12), cline = 0.1))
pf3 <- profile(res_mods, plot = FALSE, steps = 8, xlim = c(0.01, 0.5), exp = TRUE)
out$profile_exp <- run(fplot.profile.rma(pf3, xlab = "Tau squared", ylab = "L", main = "Title", xlim = c(0, 0.6), cline = TRUE))

### plot.infl --------------------------------------------------------------------------------------------

inf_re <- influence(res)
out$infl_all <- run(fplot.infl.rma.uni(inf_re))
out$infl_sub <- run(fplot.infl.rma.uni(inf_re, plotinf = c(1, 3, 4, 7), logcov = FALSE, las = 2, bg = "blue", bg.infl = "green", slab.style = 2))
inf_mods <- influence(res_mods)
out$infl_dfbs <- run(fplot.infl.rma.uni(inf_mods, plotinf = FALSE, plotdfbs = TRUE))
out$infl_dfbs_only2 <- run(fplot.infl.rma.uni(inf_mods, plotinf = c(2, 5), plotdfbs = 2, pch = 19))
dat_na <- dat; dat_na$yi[3] <- NA; dat_na$ablat[7] <- NA
res_na <- rma(yi, vi, mods = ~ ablat, data = dat_na, slab = label)
inf_na <- influence(res_na)
out$infl_na_omit <- run(fplot.infl.rma.uni(inf_na, plotinf = c(1, 2, 6)))
options(na.action = "na.exclude")
inf_na_ex <- influence(res_na)
out$infl_na_exclude <- run(fplot.infl.rma.uni(inf_na_ex, plotinf = c(1, 2, 6, 8)))
options(na.action = "na.omit")

### plot.gosh --------------------------------------------------------------------------------------------

dat8 <- dat[1:8, ]
res8 <- rma(yi, vi, data = dat8, slab = label)
res8m <- rma(yi, vi, mods = ~ ablat, data = dat8, slab = label)
g8 <- gosh(res8, progbar = FALSE)
g8m <- gosh(res8m, progbar = FALSE)
out$gosh_info <- list(k = nrow(g8$res), km = nrow(g8m$res), cols = colnames(g8m$res), est = num(g8$res[, "estimate"]), I2 = num(g8$res[, "I2"]))
out$gosh_default <- run(fplot.gosh.rma(g8))
out$gosh_opts <- run(fplot.gosh.rma(g8, het = "tau2", out = 3, hh = c(0.2, 0.4), breaks = c(6, 10), adjust = c(1, 2), lwd = c(1, 0), alpha = c(0.3, 0.5, 0.8), cex = 1.5, pch = 1,
                                    border = c("gray", "black"), labels = c("Est", "T2")))
out$gosh_nox <- run(fplot.gosh.rma(g8, het = "QE", xhist = FALSE, col = "blue", xlim = c(-1.5, 0.5), trim = 0.05))
out$gosh_noy <- run(fplot.gosh.rma(g8, het = "H2", yhist = FALSE, breaks = "Scott"))
out$gosh_nohist <- run(fplot.gosh.rma(g8, xhist = FALSE, yhist = FALSE, ylim = c(0, 100)))
out$gosh_fd <- run(fplot.gosh.rma(g8, het = "tau", breaks = list("FD", 12), out = 2, col = c("red", "green")))
out$gosh_pairs <- run(fplot.gosh.rma(g8m))
out$gosh_pairs_opts <- run(fplot.gosh.rma(g8m, het = "H2", out = 2, alpha = 0.4, breaks = 7, adjust = 2, lwd = 1, trim = list(0.02, c(0.1, 0.05), 0), border = "red", hh = 0.5, labels = c("H", "a", "b")))

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "src/tests/reference/phase2/plots.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase2/plots.json\n")
