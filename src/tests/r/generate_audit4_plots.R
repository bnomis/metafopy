# Reference values for the gaps closed after the fourth audit in the plot area: vif() of rma.mv and selmodel fits with moderators (and the
# intercept= and fixed= arguments for all classes that take them), regplot() of rma.mv, location-scale and selmodel fits (with the grep= and
# fixed= arguments), forest(tabfig=) and the layout of forest() for selmodel fits.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit4_plots.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
quiet <- function(expr) suppressWarnings(suppressMessages(expr))

out <- list()
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
kon <- dat.konstantopoulos2011
kon$z <- sin(1:56)

### vif ----------------------------------------------------------------------------------------------------------------------------------------
vifl <- function(v) {
  list(vif = num(v$vifs), sif = num(sapply(v$vif, function(x) x$sif)), m = num(sapply(v$vif, function(x) x$m)),
       coefs = as.character(sapply(v$vif, function(x) x$coefs)), names = names(v$vifs), print = printed(v),
       sim = if (is.null(v$sim)) NULL else lapply(seq_len(nrow(v$sim)), function(i) num(v$sim[i, ])), sim_type = attr(v$sim, "type"),
       prop = num(v$prop))
}
vifs <- function(expr) {
  r <- tryCatch(expr, error = function(e) e)
  if (inherits(r, "error")) list(error = conditionMessage(r)) else vifl(r)
}

mv <- rma.mv(yi, vi, mods = ~ year + z, random = ~ 1 | district/school, data = kon)
mv_t <- rma.mv(yi, vi, mods = ~ year + z, random = ~ 1 | district/school, data = kon, test = "t")
res_uni <- rma(yi, vi, mods = ~ ablat + year, data = dat)
res_ls <- rma(yi, vi, mods = ~ ablat + year, scale = ~ ablat, data = dat)
sel <- selmodel(res_uni, type = "halfnorm")
sel_step <- selmodel(rma(yi, vi, mods = ~ ablat + year, data = dat), type = "stepfun", steps = 0.025)

out$vif <- list(
  mv_default = vifs(vif(mv)),
  mv_table = vifs(vif(mv, table = TRUE)),
  mv_table_t = vifs(vif(mv_t, table = TRUE)),
  mv_intercept = vifs(vif(mv, intercept = TRUE)),
  mv_intercept_table = vifs(vif(mv, intercept = TRUE, table = TRUE)),
  mv_btt = vifs(vif(mv, btt = 2:3)),
  mv_btt_name = vifs(vif(mv, btt = "z")),
  mv_btt_grep = vifs(vif(mv, btt = "y.")),
  mv_btt_fixed = vifs(vif(mv, btt = "y.", fixed = TRUE)),
  mv_btt_list = vifs(vif(mv, btt = list(2, 3, 2:3))),
  mv_reestimate = vifs(vif(mv, reestimate = TRUE)),
  mv_reestimate_btt = vifs(vif(mv, btt = 2:3, reestimate = TRUE, intercept = TRUE)),
  mv_sim = vifs(vif(mv, sim = 4, seed = 1, progbar = FALSE)),
  mv_sim_reestimate = vifs(vif(mv, sim = 3, seed = 2, reestimate = TRUE, progbar = FALSE)),
  sel_default = vifs(vif(sel)),
  sel_table = vifs(vif(sel, table = TRUE)),
  sel_intercept = vifs(vif(sel, intercept = TRUE)),
  sel_btt = vifs(vif(sel, btt = 2:3)),
  sel_btt_name = vifs(vif(sel_step, btt = "ablat")),
  sel_sim = vifs(vif(sel, sim = 3)),
  sel_reestimate = vifs(vif(sel, reestimate = TRUE)),
  uni_intercept = vifs(vif(res_uni, intercept = TRUE)),
  uni_intercept_table = vifs(vif(res_uni, intercept = TRUE, table = TRUE)),
  uni_intercept_reestimate = vifs(vif(res_uni, intercept = TRUE, reestimate = TRUE)),
  uni_fixed = vifs(vif(res_uni, btt = "ab", fixed = TRUE)),
  uni_fixed_none = vifs(vif(res_uni, btt = ".b", fixed = TRUE)),
  uni_sim_intercept = vifs(vif(res_uni, intercept = TRUE, sim = 4, seed = 3, progbar = FALSE)),
  ls_intercept = {
    v <- vif(res_ls, intercept = TRUE)
    list(beta = vifl(v$beta), alpha = vifl(v$alpha))
  },
  ls_fixed = {
    v <- vif(res_ls, btt = "ab", att = "ab", fixed = TRUE)
    list(beta = vifl(v$beta), alpha = vifl(v$alpha))
  }
)

### regplot ------------------------------------------------------------------------------------------------------------------------------------
pdf(NULL, width = 7, height = 7)
sav2list <- function(s) list(xi = num(s$xi), yi = num(s$yi), psize = num(s$psize), label = as.logical(s$label), pred = num(s$pred),
                             order = num(s$order) - 1L, ids = num(s$ids) - 1L, pch = num(s$pch), col = as.character(s$col), bg = as.character(s$bg),
                             slab = as.character(s$slab), offset = num(attr(s, "offset")), labsize = attr(s, "labsize"))
band <- function(m, mod.pos = 2) {
  X <- m$X
  xi <- X[, mod.pos]
  r <- diff(range(xi))
  xs <- seq(min(xi) - 0.04 * r, max(xi) + 0.04 * r, length.out = 1000)
  Xnew <- rbind(colMeans(X))[rep(1, 1000), , drop = FALSE]
  Xnew[, mod.pos] <- xs
  p <- predict(m, newmods = Xnew[, -1, drop = FALSE])
  list(sum_pred = sum(p$pred), sum_ci_lb = sum(p$ci.lb), sum_ci_ub = sum(p$ci.ub),
       sum_pi_lb = if (is.null(p$pi.lb)) NULL else sum(p$pi.lb), sum_pi_ub = if (is.null(p$pi.ub)) NULL else sum(p$pi.ub))
}
res_one <- rma(yi, vi, mods = ~ ablat, data = dat)
mv_one <- rma.mv(yi, vi, mods = ~ ablat, random = ~ 1 | trial, data = dat)
ls_one <- rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = dat)
sel_one <- selmodel(res_one, type = "halfnorm")
mv_two <- rma.mv(yi, vi, mods = ~ ablat + year, random = ~ 1 | trial, data = dat)
ls_two <- rma(yi, vi, mods = ~ ablat + year, scale = ~ year, data = dat)
sel_two <- selmodel(res_uni, type = "halfnorm")

out$regplot <- list(
  mv_default = c(sav2list(regplot(mv_one)), list(band = band(mv_one))),
  mv_pi_label = c(sav2list(regplot(mv_one, pi = TRUE, label = "piout")), list(band = band(mv_one))),
  mv_mod = c(sav2list(regplot(mv_two, mod = "year", label = "ciout", psize = "seinv")), list(band = band(mv_two, 3))),
  ls_default = c(sav2list(regplot(ls_one)), list(band = band(ls_one))),
  ls_label = c(sav2list(suppressWarnings(regplot(ls_one, pi = TRUE, label = "piout"))), list(band = band(ls_one))),
  ls_mod = c(sav2list(regplot(ls_two, mod = "year", label = "all", transf = exp)), list(band = band(ls_two, 3))),
  sel_default = c(sav2list(regplot(sel_one)), list(band = band(sel_one))),
  sel_pi_label = c(sav2list(regplot(sel_one, pi = TRUE, label = "piout")), list(band = band(sel_one))),
  sel_mod = c(sav2list(regplot(sel_two, mod = "year", label = c(1, 5, 9), xvals = seq(1950, 1990, length.out = 5))), list(band = band(sel_two, 3))),
  grep = sav2list(regplot(res_uni, mod = "ea", grep = TRUE)),
  grep_fixed = sav2list(regplot(res_uni, mod = "ea", grep = TRUE, fixed = TRUE)),
  grep_regex = sav2list(regplot(res_uni, mod = "^a.", grep = TRUE)),
  grep_regex_fixed = tryCatch(regplot(res_uni, mod = "^a.", grep = TRUE, fixed = TRUE), error = function(e) list(error = conditionMessage(e))),
  grep_ambiguous = tryCatch(regplot(res_uni, mod = "a.", grep = TRUE), error = function(e) list(error = conditionMessage(e)))
)
out$regplot$grep_regex_fixed <- if (inherits(out$regplot$grep_regex_fixed, "regplot")) NULL else out$regplot$grep_regex_fixed
out$regplot$grep_ambiguous <- if (inherits(out$regplot$grep_ambiguous, "regplot")) NULL else out$regplot$grep_ambiguous

### forest: tabfig and the layout for selmodel fits ----------------------------------------------------------------------------------------------
res_plain <- rma(yi, vi, data = dat)
texts <- character()
invisible(suppressMessages(trace("text.default", tracer = quote(if (is.character(labels)) texts <<- c(texts, labels)), print = FALSE, where = asNamespace("graphics"))))
ftext <- function(...) {
  texts <<- character()
  s <- do.call(forest, list(...))
  list(xlim = num(s$xlim), alim = num(s$alim), textpos = num(s$textpos), areas = num(s$areas), annotations = grep("\\[", texts, value = TRUE))
}
out$forest <- list(
  default = ftext(res_plain),
  tabfig1 = ftext(res_plain, tabfig = 1),
  tabfig2 = ftext(res_plain, tabfig = 2),
  tabfig3 = ftext(res_plain, tabfig = 3),
  tabfig1_prediction = ftext(res_plain, tabfig = 1, addpred = TRUE),
  tabfig_over_annosym = ftext(res_plain, tabfig = 3, annosym = c(" (", " / ", ")")),
  tabfig1_default = {
    texts <- character()
    s <- forest(dat$yi, dat$vi, tabfig = 1)
    list(xlim = num(s$xlim), annotations = grep("\\[", texts, value = TRUE))
  },
  tabfig1_cumul = ftext(cumul(res_plain, order = order(dat$year)), tabfig = 1),
  selmodel = ftext(sel_one),
  selmodel_slab = ftext(sel_one, slab = paste0("Trial ", dat$trial)),
  selmodel_mods = ftext(sel_two),
  selmodel_addpred = ftext(selmodel(res_plain, type = "halfnorm"), addpred = TRUE)
)
invisible(suppressMessages(untrace("text.default", where = asNamespace("graphics"))))

write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase12/audit4_plots.json")
