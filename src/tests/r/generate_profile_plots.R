# Reference geometry for plot.profile.rma() applied to the profiles of models with several components (rma.mv, rma.ls, selmodel), and for
# a single profile of a location-scale model. metafor draws one panel per component of the list that profile() returns; each panel is the
# plot of that component's own profile, which is recorded here component by component (plot, abline, points and lines calls plus the
# usr/axis limits), as generate_plots.R does for rma.uni profiles.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_profile_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

pdf(NULL, width = 7, height = 7)
.rec <- new.env()
.rec$log <- list()
rec_add <- function(type, ...) .rec$log[[length(.rec$log) + 1L]] <- c(list(type = type), list(...))
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
chr <- function(v) if (is.null(v)) NULL else as.character(v)
first <- function(v) if (is.null(v)) NULL else num(v)[1]

wrappers <- list(
  abline = function(a = NULL, b = NULL, h = NULL, v = NULL, ...) {
    d <- list(...)
    rec_add("abline", a = num(a), b = num(b), h = num(h), v = num(v), lty = chr(d$lty), col = chr(d$col))
    graphics::abline(a, b, h, v, ...)
  },
  points = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("points", x = num(x), y = num(y), pch = num(a$pch), cex = num(a$cex), col = chr(a$col), bg = chr(a$bg))
    graphics::points(x, y, ...)
  },
  plot = function(x, y = NULL, ...) {
    a <- list(...)
    rec_add("plot", xlim = num(a$xlim), ylim = num(a$ylim), xlab = if (is.null(a$xlab)) NULL else if (is.expression(a$xlab)) paste(deparse(a$xlab[[1]]), collapse = "") else chr(a$xlab),
            ylab = if (is.null(a$ylab)) NULL else if (is.expression(a$ylab)) paste(deparse(a$ylab[[1]]), collapse = "") else chr(a$ylab),
            main = if (is.null(a$main)) NULL else if (is.expression(a$main)) paste(deparse(a$main[[1]]), collapse = "") else chr(a$main),
            xlab_expr = is.expression(a$xlab), ylab_expr = is.expression(a$ylab),
            xdat = num(x), ydat = num(y), pch = first(a$pch), cex = first(a$cex))
    graphics::plot(x, y, ...)
  }
)
instrument <- function(f) {
  e <- new.env(parent = environment(f))
  for (nm in names(wrappers)) assign(nm, wrappers[[nm]], envir = e)
  environment(f) <- e
  f
}
fplot.profile.rma <- instrument(metafor:::plot.profile.rma)

collect <- function() {
  log <- .rec$log
  types <- unique(vapply(log, function(l) l$type, ""))
  out <- lapply(types, function(t) lapply(Filter(function(l) l$type == t, log), function(l) { l$type <- NULL; l }))
  names(out) <- types
  out$usr <- num(par("usr"))
  out$axticks <- list(x = num(axTicks(1)), y = num(axTicks(2)), xl = format(axTicks(1), trim = TRUE), yl = format(axTicks(2), trim = TRUE))
  out
}
run <- function(expr) {
  .rec$log <- list()
  plot.new()
  force(expr)
  collect()
}
parts <- function(pf) if (pf$comps > 1L) lapply(seq_len(pf$comps), function(j) pf[[j]]) else list(pf)
panels <- function(pf, ...) lapply(parts(pf), function(p) run(fplot.profile.rma(p, ...)))
table <- function(pf) lapply(parts(pf), function(p) list(grid = num(p[[1]]), ll = num(p$ll), ylim = num(p$ylim), vc = p$vc, maxll = p$maxll))

out <- list()

kon <- dat.konstantopoulos2011
kres <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon)
pf_mv <- profile(kres, steps = 5, plot = FALSE, progbar = FALSE)
out$mv <- list(table = table(pf_mv), panels = panels(pf_mv), panels_opts = panels(pf_mv, cline = TRUE, pch = 17, ylim = c(-70, -10)))

bd <- dat.bangertdrowns2004
bd$ni100 <- bd$ni / 100
rls <- rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd)
pf_ls <- profile(rls, steps = 6, plot = FALSE, progbar = FALSE)
out$ls <- list(table = table(pf_ls), panels = panels(pf_ls))
pf_ls1 <- profile(rls, alpha = 2, steps = 6, plot = FALSE, progbar = FALSE)
out$ls_single <- list(table = table(pf_ls1), panels = panels(pf_ls1, cline = TRUE, refline = FALSE))
rls0 <- rma(yi, vi, scale = ~ 1, data = bd)
pf_ls0 <- profile(rls0, steps = 5, plot = FALSE, progbar = FALSE)
out$ls_intercept <- list(table = table(pf_ls0), panels = panels(pf_ls0))

hack <- dat.hackshaw1998
sel <- selmodel(rma(yi, vi, data = hack, method = "ML"), type = "stepfun", steps = 0.025)
pf_sel <- profile(sel, steps = 5, plot = FALSE, progbar = FALSE)
out$sel <- list(table = table(pf_sel), panels = panels(pf_sel))

write_json(out, "src/tests/reference/phase11/profile_plots.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
