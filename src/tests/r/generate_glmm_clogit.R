# Reference values for rma.glmm(model = "CM.EL", measure = "OR") fitted with control = list(optimizer = "clogit") (survival::clogit)
# and control = list(optimizer = "clogistic") (Epi::clogistic), with the clogitCtrl and clogisticCtrl lists, on small datasets
# (clogit() and clogistic() work on individual patient data, so the large trials of bcg are left out).
#
# Usage (from the repository root):  R_LIBS=... LC_ALL=C.UTF-8 Rscript src/tests/r/generate_glmm_clogit.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

core <- function(expr) {
  message("running ", deparse(substitute(expr))[1])
  warns <- character()
  res <- tryCatch(withCallingHandlers(expr, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
                  error = function(e) conditionMessage(e))
  if (is.character(res)) return(list(error = res, warnings = I(warns)))
  list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval), ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub),
    vb = num(res$vb), coef_names = rownames(res$beta),
    tau2 = res$tau2, I2 = res$I2, H2 = res$H2, vt = res$vt,
    QE_Wld = res$QE.Wld, QEp_Wld = res$QEp.Wld, QE_LRT = res$QE.LRT, QEp_LRT = res$QEp.LRT, QE_df = res$QE.df,
    QM = res$QM, QMp = res$QMp, QMdf = num(res$QMdf),
    k = res$k, k_f = res$k.f, k_yi = res$k.yi, k_eff = res$k.eff, p = res$p, p_eff = res$p.eff, parms = res$parms,
    int_only = res$int.only, method = res$method, model = res$model,
    fit_ml = num(res$fit.stats$ML),
    warnings = I(warns),
    print = printed(res), print_showfit = printed(res, showfit = TRUE)
  )
}

bcg <- dat.bcg
bsub <- subset(bcg, trial %in% c(1, 2, 3, 7, 10))
bna <- bsub
bna$ablat[2] <- NA
col <- dat.collins1985a
nw <- dat.nielweise2007
lee <- dat.lee2004

b2 <- function(data, model = "CM.EL", ...) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = data, measure = "OR", model = model, ...)
c2 <- function(...) rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", model = "CM.EL", ...)
n2 <- function(data, ...) rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = data, measure = "OR", model = "CM.EL", ...)

out <- list()
for (opt in c("clogit", "clogistic")) {
  ctl <- function(...) list(optimizer = opt, ...)
  o <- list()
  o$bcg_fe <- core(b2(bsub, method = "FE", control = ctl()))
  o$bcg_ee <- core(b2(bsub, method = "EE", control = ctl()))
  o$bcg_mod <- core(b2(bsub, mods = ~ ablat, method = "FE", control = ctl()))
  o$bcg_mod2 <- core(b2(bsub, mods = ~ ablat + year, method = "FE", control = ctl()))
  o$bcg_mod_na <- core(b2(bna, mods = ~ ablat, method = "FE", control = ctl()))
  o$bcg_t <- core(b2(bsub, mods = ~ ablat, method = "FE", test = "t", level = 90, control = ctl()))
  o$bcg_noscale <- core(b2(bsub, mods = ~ ablat, method = "FE", control = ctl(scaleX = FALSE)))
  o$nw_fe <- core(n2(nw, method = "FE", control = ctl()))
  o$nw_mod <- core(n2(nw, mods = ~ year, method = "FE", control = ctl()))
  o$lee_fe <- core(n2(lee, method = "FE", control = ctl()))
  o$lee_mod <- core(n2(lee, mods = ~ year, method = "FE", control = ctl()))
  o$col_fe <- core(c2(method = "FE", control = ctl()))
  o$col_mod <- core(c2(mods = ~ year, method = "FE", control = ctl()))
  out[[opt]] <- o
}

### control lists passed on to clogit() and clogistic()
o <- list()
o$clogit_eps <- core(b2(bsub, mods = ~ ablat, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(control = survival::coxph.control(eps = 1e-5)))))
o$clogit_init <- core(b2(bsub, mods = ~ ablat, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(init = c(-0.5, 0.1)))))
o$clogit_iter <- core(n2(nw, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(control = survival::coxph.control(iter.max = 3)))))
o$clogit_exact <- core(b2(bsub, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(method = "exact"))))
for (m in c("efron", "breslow", "approximate")) {
  o[[paste0("ties_", m)]] <- core(b2(bsub, mods = ~ ablat, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(method = m))))
  o[[paste0("ties_", m, "_nw")]] <- core(n2(nw, method = "FE", control = list(optimizer = "clogit", clogitCtrl = list(method = m))))
}
o$clogistic_eps <- core(b2(bsub, mods = ~ ablat, method = "FE", control = list(optimizer = "clogistic", clogisticCtrl = list(eps = 1e-3))))
o$clogistic_init <- core(b2(bsub, mods = ~ ablat, method = "FE", control = list(optimizer = "clogistic", clogisticCtrl = list(init = c(-0.5, 0.1)))))
o$clogistic_iter <- core(n2(nw, method = "FE", control = list(optimizer = "clogistic", clogisticCtrl = list(iter.max = 3))))
out$ctrl <- o

### arguments and errors
err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(
  clogit_ml = err(b2(bsub, control = list(optimizer = "clogit"))),
  clogistic_ml = err(b2(bsub, control = list(optimizer = "clogistic"))),
  clogit_ml_um = err(b2(bsub, model = "UM.FS", control = list(optimizer = "clogit"))),
  clogit_k1 = err(b2(bsub[1, ], method = "FE", control = list(optimizer = "clogit"))),
  clogistic_k1 = err(b2(bsub[1, ], method = "FE", control = list(optimizer = "clogistic"))),
  bad_optimizer = err(b2(bsub, method = "FE", control = list(optimizer = "bogus")))
)

### other models ignore the optimizer
out$ignored <- list(
  al = core(b2(bsub, model = "CM.AL", method = "FE", control = list(optimizer = "clogit"))),
  um = core(b2(bsub, model = "UM.FS", method = "FE", control = list(optimizer = "clogistic")))
)

out$meta <- list(metafor = as.character(packageVersion("metafor")), survival = as.character(packageVersion("survival")),
                 Epi = as.character(packageVersion("Epi")), R = R.version.string)
dir.create("src/tests/reference/phase9", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase9/glmm_clogit.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase9/glmm_clogit.json\n")
