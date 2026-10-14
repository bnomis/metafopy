# Reference values for the functions metafor writes once for all 'rma' objects, run on fits of every class:
# forest(), funnel(), radial(), baujat(), qqnorm(), labbe(), addpoly() of predict() results and plot() (geometry recorded
# with the machinery of generate_plots.R), ranktest(), predict(), tes() and gosh() for rma.mh / rma.peto / rma.mv /
# rma.glmm / location-scale / selection models, and the error messages metafor gives for the classes a function does
# not support.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_fits.R

src <- readLines("src/tests/r/generate_plots.R")
eval(parse(text = src[seq_len(grep("^fforest.rma <- ", src)[1] - 1L)]))

wrappers$title <- function(main = NULL, ...) {
  rec_add("title", main = chr(main))
  graphics::title(main, ...)
}

fforest.rma <- instrument(metafor:::forest.rma)
ffunnel.rma <- instrument(metafor:::funnel.rma)
fradial.rma <- instrument(metafor:::radial.rma)
fbaujat.rma <- instrument(metafor:::baujat.rma)
fqqnorm.rma.mh <- instrument(metafor:::qqnorm.rma.mh)
fqqnorm.rma.peto <- instrument(metafor:::qqnorm.rma.peto)
flabbe.rma <- instrument(metafor:::labbe.rma)
faddpoly.default <- instrument(metafor:::addpoly.default)
faddpoly.predict.rma <- instrument(metafor:::addpoly.predict.rma)
environment(faddpoly.predict.rma)$addpoly <- function(...) faddpoly.default(...)
fplot.rma.uni <- instrument(metafor:::plot.rma.uni)
fplot.rma.mh <- instrument(metafor:::plot.rma.mh)
fplot.rma.peto <- instrument(metafor:::plot.rma.peto)

run <- function(expr) {
  rec_reset()
  plot.new()
  ret <- force(expr)
  list(ret = if (is.null(ret)) NULL else lapply(ret, num), rec = collect())
}
# for plot(): only the primitives of the panels plot.rma.*() draws itself (titles, the residual panels)
run_plot <- function(expr) {
  rec_reset()
  force(expr)
  par(mfrow = c(1, 1))
  log <- .rec$log
  types <- unique(vapply(log, function(l) l$type, ""))
  out <- lapply(types, function(t) lapply(Filter(function(l) l$type == t, log), function(l) { l$type <- NULL; l }))
  names(out) <- types
  out
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$label <- paste(dat.bcg$author, dat.bcg$year)
bcg <- dat.bcg
bcg$label <- paste(bcg$author, bcg$year)
bcg_na <- bcg
bcg_na$tpos[3] <- NA

fits <- list(
  mh = rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label),
  mh0 = rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg),
  mh_rr = rma.mh(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label),
  mh_na = suppressWarnings(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg_na, slab = label)),
  peto = rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label),
  mv = rma.mv(yi, vi, random = ~ 1 | trial, data = dat, slab = label),
  mvm = rma.mv(yi, vi, mods = ~ ablat, random = ~ 1 | trial, data = dat, slab = label),
  ls = rma(yi, vi, scale = ~ 1, data = dat, slab = label),
  sel = selmodel(rma(yi, vi, data = dat, slab = label), type = "logistic"),
  glmm = rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label),
  uni = rma(yi, vi, data = dat, slab = label),
  unim = rma(yi, vi, mods = ~ ablat, data = dat, slab = label)
)

out <- list()

### forest -------------------------------------------------------------------------------------------------

for (nm in c("mh", "mh0", "mh_na", "peto", "mv", "mvm", "sel", "glmm")) out[[paste0("forest_", nm)]] <- run(fforest.rma(fits[[nm]]))
out$forest_mh_opts <- run(fforest.rma(fits$mh, atransf = exp, at = log(c(0.05, 0.25, 1, 4)), xlim = c(-16, 6), showweights = TRUE, order = "obs"))

### funnel -------------------------------------------------------------------------------------------------

for (nm in c("mh", "peto", "mv", "mvm", "ls", "sel", "glmm")) out[[paste0("funnel_", nm)]] <- run(ffunnel.rma(fits[[nm]]))
out$funnel_mh_wi <- run(ffunnel.rma(fits$mh, yaxis = "wi"))
out$funnel_mh_ni <- run(ffunnel.rma(fits$mh, yaxis = "ni", level = c(90, 95, 99), shade = c("white", "gray55", "gray75")))

### radial, baujat, qqnorm, labbe ---------------------------------------------------------------------------

for (nm in c("mh", "peto", "glmm")) out[[paste0("radial_", nm)]] <- run(fradial.rma(fits[[nm]]))
for (nm in c("mh", "mh_na", "peto")) out[[paste0("baujat_", nm)]] <- run(fbaujat.rma(fits[[nm]]))
out$baujat_mh_slab <- run(fbaujat.rma(fits$mh, symbol = "slab"))
out$qqnorm_mh <- run(fqqnorm.rma.mh(fits$mh, label = TRUE))
out$qqnorm_peto <- run(fqqnorm.rma.peto(fits$peto, type = "rstudent", grid = TRUE))
out$labbe_mh <- run(flabbe.rma(fits$mh))
out$labbe_mh_rr <- run(flabbe.rma(fits$mh_rr, ci = TRUE, transf = exp))
out$labbe_glmm <- run(flabbe.rma(fits$glmm, pi = TRUE, ci = TRUE))

### addpoly(predict()) --------------------------------------------------------------------------------------

addp <- function(fit, ...) {
  forest(fits$uni, ylim = c(-5, 16))
  faddpoly.predict.rma(predict(fit), ...)
}
out$addpoly_pred_mh <- run(addp(fits$mh, rows = -3, mlab = "MH"))
out$addpoly_pred_uni <- run(addp(fits$uni, addpred = TRUE))
out$addpoly_pred_mods <- run(addp(fits$unim))

### plot() ----------------------------------------------------------------------------------------------------

out$plot_uni <- run_plot(fplot.rma.uni(fits$uni))
out$plot_unim <- run_plot(fplot.rma.uni(fits$unim))
out$plot_mh <- run_plot(fplot.rma.mh(fits$mh))
out$plot_peto <- run_plot(fplot.rma.peto(fits$peto))

### numbers ---------------------------------------------------------------------------------------------------

rank <- list()
for (nm in c("mh", "peto", "mv", "ls", "sel", "glmm")) {
  r <- ranktest(fits[[nm]])
  rank[[nm]] <- list(tau = num(r$tau), pval = num(r$pval))
}
out$ranktest <- rank

pr <- function(p) list(pred = num(p$pred), se = num(p$se), ci.lb = num(p$ci.lb), ci.ub = num(p$ci.ub), pi.lb = num(p$pi.lb), pi.ub = num(p$pi.ub))
out$predict <- list(mh = pr(predict(fits$mh)), mh_exp = pr(predict(fits$mh, transf = exp, level = 90)), peto = pr(predict(fits$peto)),
                    mh_rr = pr(predict(fits$mh_rr, transf = exp)))

te <- function(t) list(k = t$k, O = t$O, E = num(t$E), OEratio = num(t$OEratio), X2 = num(t$X2), pval = num(t$pval), power = num(t$power), theta = num(t$theta))
out$tes <- list(mh = te(tes(fits$mh)), peto = te(tes(fits$peto)))

gs <- function(g) list(nrow = nrow(g$res), cols = colnames(g$res), est = num(g$res$estimate), QE = num(g$res$QE), I2 = num(g$res$I2), H2 = num(g$res$H2), k = num(g$res$k),
                       fit = num(g$fit))
mh8 <- rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg[1:8, ])
peto8 <- rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg[1:8, ])
out$gosh <- list(mh = gs(gosh(mh8, progbar = FALSE)), peto = gs(gosh(peto8, progbar = FALSE)))

### errors ----------------------------------------------------------------------------------------------------

msg <- function(expr) tryCatch({ suppressWarnings(force(expr)); "OK" }, error = function(e) conditionMessage(e))
rob <- robust(fits$uni, cluster = 1:13)
errs <- list(
  forest_ls = msg(forest(fits$ls)),
  forest_glmm_weights = msg(forest(fits$glmm, showweights = TRUE)),
  forest_sel_weights = msg(forest(fits$sel, showweights = TRUE)),
  predict_mh_newmods = msg(predict(fits$mh, newmods = 1)),
  tes_mv = msg(tes(fits$mv)), tes_ls = msg(tes(fits$ls)), tes_sel = msg(tes(fits$sel)), tes_glmm = msg(tes(fits$glmm)), tes_robust = msg(tes(rob)),
  gosh_mv = msg(gosh(fits$mv)), gosh_ls = msg(gosh(fits$ls)), gosh_sel = msg(gosh(fits$sel)), gosh_glmm = msg(gosh(fits$glmm)), gosh_robust = msg(gosh(rob)),
  radial_mv = msg(radial(fits$mv)), radial_ls = msg(radial(fits$ls)), radial_sel = msg(radial(fits$sel)), radial_robust = msg(radial(rob)),
  baujat_mv = msg(baujat(fits$mv)), baujat_ls = msg(baujat(fits$ls)), baujat_sel = msg(baujat(fits$sel)), baujat_glmm = msg(baujat(fits$glmm)),
  qqnorm_mv = msg(qqnorm(fits$mv)), qqnorm_sel = msg(qqnorm(fits$sel)), qqnorm_glmm = msg(qqnorm(fits$glmm)),
  labbe_mv = msg(labbe(fits$mv)), labbe_ls = msg(labbe(fits$ls)), labbe_sel = msg(labbe(fits$sel)), labbe_peto = msg(labbe(fits$peto)),
  plot_mv = msg(plot(fits$mv)), plot_ls = msg(plot(fits$ls)), plot_glmm = msg(plot(fits$glmm)), plot_robust = msg(plot(rob)),
  regtest_mv = msg(regtest(fits$mv)), regtest_mh = msg(regtest(fits$mh)), regtest_glmm = msg(regtest(fits$glmm)), regtest_sel = msg(regtest(fits$sel)),
  fsn_mv = msg(fsn(fits$mv)), fsn_glmm = msg(fsn(fits$glmm)), fsn_ls = msg(fsn(fits$ls)),
  trimfill_mv = msg(trimfill(fits$mv)), trimfill_mh = msg(trimfill(fits$mh)), trimfill_glmm = msg(trimfill(fits$glmm)), trimfill_ls = msg(trimfill(fits$ls)),
  hc_mv = msg(hc(fits$mv)), hc_glmm = msg(hc(fits$glmm)), hc_sel = msg(hc(fits$sel)), hc_ls = msg(hc(fits$ls))
)
out$errors <- errs

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
dir.create("src/tests/reference/phase7", showWarnings = FALSE)
write_json(out, "src/tests/reference/phase7/fits.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase7/fits.json\n")
