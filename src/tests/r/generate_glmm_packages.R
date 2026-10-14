# Reference values for rma.glmm() with control = list(package = "GLMMadaptive") and package = "glmmTMB" (the default is lme4,
# see generate_glmm.R), on the bcg, collins1985a, nielweise2007 and hart1999 data.
# GLMMadaptive integrates the random effects with adaptive Gauss-Hermite quadrature (nAGQ points per dimension) and stops its
# EM and quasi-Newton iterations at relative tolerances of 1e-5..1e-8; glmmTMB uses the Laplace approximation (nAGQ is ignored)
# with exact automatic derivatives. Fits that fail in R (for example GLMMadaptive's "large coefficient" error) are stored under "error".
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_glmm_packages.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
  library(GLMMadaptive)
  library(glmmTMB)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

core <- function(expr) {
  warns <- character()
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(withCallingHandlers(suppressMessages(expr), warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
                  error = function(e) conditionMessage(e))
  if (is.character(res)) return(list(error = res, warnings = I(warns)))
  elapsed <- proc.time()[["elapsed"]] - t0
  list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval), ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub),
    vb = num(res$vb), coef_names = rownames(res$beta),
    tau2 = res$tau2, se_tau2 = res$se.tau2, sigma2 = res$sigma2, rho = res$rho, ci_lb_tau2 = res$ci.lb.tau2, ci_ub_tau2 = res$ci.ub.tau2,
    I2 = res$I2, H2 = res$H2, vt = res$vt,
    QE_Wld = res$QE.Wld, QEp_Wld = res$QEp.Wld, QE_LRT = res$QE.LRT, QEp_LRT = res$QEp.LRT, QE_df = res$QE.df,
    QM = res$QM, QMp = res$QMp, QMdf = num(res$QMdf),
    k = res$k, k_f = res$k.f, k_yi = res$k.yi, k_eff = res$k.eff, k_all = res$k.all, p = res$p, p_eff = res$p.eff, parms = res$parms,
    int_only = res$int.only, method = res$method, model = res$model,
    fit_ml = num(res$fit.stats$ML), yi = num(res$yi), vi = num(res$vi), X = num(res$X), slab = res$slab,
    warnings = I(warns), elapsed = elapsed,
    print = printed(res), print_showfit = printed(res, showfit = TRUE)
  )
}

bcg <- dat.bcg
col <- dat.collins1985a
nw <- dat.nielweise2007
hart <- dat.hart1999
bna <- bcg
bna$ablat[3] <- NA
b2 <- function(measure = "OR", ...) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = measure, ...)
c2 <- function(measure = "OR", ...) rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = measure, ...)
n2 <- function(measure = "OR", ...) rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = measure, ...)
irr <- function(...) rma.glmm(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = hart, measure = "IRR", ...)
p1 <- function(measure, ...) rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = measure, ...)
bcg$ti <- bcg$tpos + bcg$tneg
rate <- function(...) rma.glmm(xi = tpos, ti = ti, data = bcg, measure = "IRLN", ...)

# every case takes the control list holding the package
cases <- list(
  fs_ml = function(ctl, ...) b2(model = "UM.FS", control = ctl, ...),
  fs_ml_mod = function(ctl) b2(model = "UM.FS", mods = ~ ablat, control = ctl),
  fs_ml_coding1 = function(ctl) b2(model = "UM.FS", coding = 1, control = ctl),
  fs_ml_t = function(ctl) b2(model = "UM.FS", test = "t", level = 90, control = ctl),
  fs_ml_mod_na = function(ctl) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna, measure = "OR", model = "UM.FS", mods = ~ ablat, control = ctl),
  fs_collins = function(ctl) c2(model = "UM.FS", control = ctl),
  fs_nw = function(ctl) n2(model = "UM.FS", control = ctl),
  rs_ml = function(ctl) b2(model = "UM.RS", control = ctl),
  rs_ml_cor = function(ctl) b2(model = "UM.RS", cor = TRUE, control = ctl),
  rs_fe = function(ctl) b2(model = "UM.RS", method = "FE", control = ctl),
  rs_ml_mod = function(ctl) b2(model = "UM.RS", mods = ~ ablat, control = ctl),
  rs_collins = function(ctl) c2(model = "UM.RS", control = ctl),
  al_ml = function(ctl) b2(model = "CM.AL", control = ctl),
  al_ml_mod = function(ctl) b2(model = "CM.AL", mods = ~ ablat, control = ctl),
  al_collins = function(ctl) c2(model = "CM.AL", control = ctl),
  al_nw = function(ctl) n2(model = "CM.AL", control = ctl),
  irr_fs = function(ctl) irr(model = "UM.FS", control = ctl),
  irr_rs = function(ctl) irr(model = "UM.RS", control = ctl),
  irr_al = function(ctl) irr(model = "CM.AL", control = ctl),
  irr_el = function(ctl) irr(model = "CM.EL", control = ctl),
  irr_el_mod = function(ctl) irr(model = "CM.EL", mods = ~ year, control = ctl),
  plo = function(ctl) p1("PLO", control = ctl),
  plo_mod = function(ctl) p1("PLO", mods = ~ ablat, control = ctl),
  irln = function(ctl) rate(control = ctl),
  irln_mod = function(ctl) rate(mods = ~ ablat, control = ctl)
)

out <- list()
for (pk in c("GLMMadaptive", "glmmTMB")) {
  res <- list()
  for (nm in names(cases)) {
    cat(pk, nm, "\n")
    res[[nm]] <- core(cases[[nm]](list(package = pk)))
  }
  # the number of quadrature points only matters for GLMMadaptive (nAGQ is ignored by glmmTMB and fixed at 1 for lme4's UM.RS)
  if (pk == "GLMMadaptive") {
    res$fs_nagq3 <- core(b2(model = "UM.FS", nAGQ = 3, control = list(package = pk)))
    res$fs_nagq15 <- core(b2(model = "UM.FS", nAGQ = 15, control = list(package = pk)))
    res$rs_nagq3 <- core(b2(model = "UM.RS", nAGQ = 3, control = list(package = pk)))
    res$al_nagq1 <- core(b2(model = "CM.AL", nAGQ = 1, control = list(package = pk)))
    res$plo_nagq1 <- core(p1("PLO", nAGQ = 1, control = list(package = pk)))
  }
  # the experimental measures
  res$rr_fs_ml <- core(b2("RR", model = "UM.FS", control = list(package = pk)))
  res$rd_fs_ml <- core(b2("RD", model = "UM.FS", control = list(package = pk)))
  res$rr_rs_ml <- core(b2("RR", model = "UM.RS", control = list(package = pk)))
  res$pr_ml <- core(p1("PR", control = list(package = pk)))
  res$pln_ml <- core(p1("PLN", control = list(package = pk)))
  out[[pk]] <- res
}

# a partial name of the package is accepted, an unknown name is not
err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(bad_package = err(b2(model = "UM.FS", control = list(package = "nlme"))))
out$partial <- core(b2(model = "UM.FS", control = list(package = "glmmT")))

out$meta <- list(metafor = as.character(packageVersion("metafor")), GLMMadaptive = as.character(packageVersion("GLMMadaptive")),
                 glmmTMB = as.character(packageVersion("glmmTMB")), R = R.version.string)
dir.create("src/tests/reference/phase8", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase8/glmm_packages.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase8/glmm_packages.json\n")
