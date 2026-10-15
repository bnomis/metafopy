# Reference values for the fourth audit's predict() gaps: predtype/pi.type and newvi of predict() for every fit class, the refusal of newmods by
# models without moderators, na.action (na.omit, na.exclude, na.pass) of fitted(), predict() and hatvalues(), and the location-scale fits whose
# scale model has a missing value (residuals(), weights(), rstandard(), fitted(), predict() under na.pass).
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit4_predict.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(clubSandwich)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dna <- dat
dna$yi[3] <- NA
dna$ablat[5] <- NA
bna <- dat.bcg
bna$tpos[3] <- NA
bna$cneg[5] <- NA
kon <- dat.konstantopoulos2011

# the fields of a predict() result
pr <- function(p) {
  list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), pi_lb = num(p$pi.lb), pi_ub = num(p$pi.ub),
       slab = as.character(p$slab), pi_se = num(p$pi.se), pi_dist = p$pi.dist, pi_ddf = num(p$pi.ddf), prob = num(p$prob), ddf = num(p$ddf))
}

out <- list()

### predtype and newvi -------------------------------------------------------------------------------------------------------------------------
types <- list(
  list(), list(predtype = "simple"), list(predtype = "riley"), list(predtype = "t"), list(pi.type = "riley"), list(predtype = "ri"),
  list(predtype = "riley", newvi = "vec"), list(newvi = 0.05), list(predtype = "t", newvi = "vec", level = 90),
  list(predtype = "riley", adjust = TRUE), list(predtype = "simple", prob = "<0"), list(predtype = "riley", prob = ">-0.5", transf = exp),
  list(hetvar = 0.2, predtype = "riley", newvi = 0.1)
)
nm <- c(10, 30, 50)
nv <- function(n) seq(0.01, by = 0.005, length.out = n)
runtypes <- function(res, newmods = NULL, extra = list()) {
  knew <- if (!is.null(newmods)) NROW(newmods) else if (res$int.only) 1 else res$k.f
  lapply(types, function(a) {
    if (identical(a$newvi, "vec")) a$newvi <- nv(knew)
    args <- c(list(res), if (is.null(newmods)) list() else list(newmods = newmods), extra, a)
    p <- quiet(do.call(predict, args))
    c(list(args = names(a), vals = lapply(a, function(v) if (is.function(v)) "exp" else v)), pr(p))
  })
}
uni_reml <- rma(yi, vi, mods = ~ablat, data = dat)
uni_knha <- rma(yi, vi, mods = ~ablat, data = dat, test = "knha")
uni_ee <- rma(yi, vi, mods = ~ablat, data = dat, method = "EE")
uni_int <- rma(yi, vi, data = dat)
uni_fix_tau2 <- rma(yi, vi, mods = ~ablat, data = dat, tau2 = 0.1)
mv_mods <- rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dat)
mv_t <- rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dat, test = "t")
mv_int <- rma.mv(yi, vi, random = ~1 | trial, data = dat)
mv_kon <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~1 | district / school, data = kon, test = "t")
rob <- robust(uni_reml, cluster = dat$trial)
rob_cs <- robust(uni_reml, cluster = dat$trial, clubSandwich = TRUE)
rob_mv <- robust(mv_mods, cluster = dat$trial)
sel <- quiet(selmodel(uni_reml, type = "halfnorm"))
sel_int <- quiet(selmodel(uni_int, type = "halfnorm"))
mh <- rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
peto <- rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
out$types <- list(
  uni_reml = runtypes(uni_reml, nm), uni_reml_fit = runtypes(uni_reml), uni_knha = runtypes(uni_knha, nm), uni_ee = runtypes(uni_ee, nm),
  uni_int = runtypes(uni_int), uni_fix_tau2 = runtypes(uni_fix_tau2, nm),
  mv_mods = runtypes(mv_mods, nm), mv_t = runtypes(mv_t, nm), mv_int = runtypes(mv_int), mv_kon = runtypes(mv_kon, c(-5, 0, 5)),
  rob = runtypes(rob, nm), rob_cs = runtypes(rob_cs, nm), rob_mv = runtypes(rob_mv, nm),
  sel = runtypes(sel, nm), sel_int = runtypes(sel_int),
  mh = runtypes(mh), peto = runtypes(peto)
)
out$parms <- list(uni = uni_reml$parms, mv = mv_mods$parms, mv_kon = mv_kon$parms, sel = sel$parms, rob = rob$parms)

# parms of a model with an unknown number of variance components: berkey1998 with an unstructured tau2 matrix
ber <- dat.berkey1998
V <- vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = ber)
ber_un <- rma.mv(yi, V, mods = ~ outcome - 1, random = ~ outcome | trial, struct = "UN", data = ber)
ber_diag <- rma.mv(yi, V, mods = ~ outcome - 1, random = ~ outcome | trial, struct = "DIAG", data = ber)
berpr <- function(res, ...) {
  lapply(list(list(), list(predtype = "riley"), list(predtype = "simple"), list(predtype = "t", newvi = 0.02)), function(a) {
    p <- quiet(do.call(predict, c(list(res, newmods = diag(2), ...), a)))
    c(list(args = names(a)), pr(p))
  })
}
out$ber <- list(
  V = mat(V), yi = num(ber$yi), outcome = as.character(ber$outcome), trial = num(ber$trial),
  un = c(list(parms = ber_un$parms), list(runs = berpr(ber_un, tau2.levels = c(1, 2)))),
  diag = c(list(parms = ber_diag$parms), list(runs = berpr(ber_diag, tau2.levels = c(1, 2))))
)

### newmods for models without moderators -----------------------------------------------------------------------------------------------------
out$int_only_errors <- list(
  mv = err(predict(mv_int, newmods = 30)), mh = err(predict(mh, newmods = 30)), peto = err(predict(peto, newmods = 30)),
  sel = err(predict(sel_int, newmods = 30)), rob = err(predict(robust(uni_int, cluster = dat$trial), newmods = 30)),
  uni = err(predict(uni_int, newmods = 30)), mv_rob = err(predict(robust(mv_int, cluster = dat$trial), newmods = 30)),
  mv_ok = num(predict(mv_mods, newmods = 30)$pred)
)

### na.action ----------------------------------------------------------------------------------------------------------------------------------
acts <- c("na.omit", "na.exclude", "na.pass")
newm <- c(10, NA, 30)
na_fits <- function(mk) {
  res <- list()
  for (act in acts) {
    options(na.action = act)
    r <- quiet(mk())
    res[[act]] <- list(fitted = num(fitted(r)))
  }
  options(na.action = "na.omit")
  res
}
na_all <- function(mk, hat = TRUE, newmods = NULL, extra = list()) {
  res <- list()
  options(na.action = "na.omit")
  r <- quiet(mk())
  for (act in acts) {
    options(na.action = act)
    h <- if (hat) try(num(hatvalues(r)), silent = TRUE) else NULL
    if (inherits(h, "try-error")) h <- NULL
    hm <- if (hat) try(mat(hatvalues(r, type = "matrix")), silent = TRUE) else NULL
    if (inherits(hm, "try-error")) hm <- NULL
    f <- fitted(r)
    safe <- function(expr) tryCatch(pr(quiet(expr)), error = function(e) list(error = conditionMessage(e)))
    entry <- list(fitted = if (is.list(f)) lapply(f, num) else num(f), hat = h, hat_matrix = hm, predict = safe(do.call(predict, c(list(r), extra))))
    if (!is.null(newmods)) entry$predict_newmods <- safe(do.call(predict, c(list(r, newmods = newmods), extra)))
    entry$predict_vcov <- tryCatch(mat(quiet(predict(r, vcov = TRUE))$vcov), error = function(e) NULL)
    res[[act]] <- entry
  }
  options(na.action = "na.omit")
  res
}
out$na <- list(
  uni_mods = na_all(function() rma(yi, vi, mods = ~ablat, data = dna), newmods = newm),
  uni_int = na_all(function() rma(yi, vi, data = dna)),
  uni_ee = na_all(function() rma(yi, vi, mods = ~ablat, data = dna, method = "EE")),
  mv_mods = na_all(function() rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dna), newmods = newm),
  mv_int = na_all(function() rma.mv(yi, vi, random = ~1 | trial, data = dna)),
  rob = na_all(function() robust(rma(yi, vi, mods = ~ablat, data = dna), cluster = dna$trial), newmods = newm),
  rob_cs = na_all(function() robust(rma(yi, vi, mods = ~ablat, data = dna), cluster = dna$trial, clubSandwich = TRUE), newmods = newm),
  mh = na_all(function() rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna), hat = FALSE),
  peto = na_all(function() rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna), hat = FALSE),
  sel = na_all(function() selmodel(rma(yi, vi, mods = ~ablat, data = dna), type = "halfnorm"), hat = FALSE),
  ls_loc = na_all(function() rma(yi, vi, mods = ~ablat, scale = ~1, data = dna), newmods = newm),
  ls_scale = na_all(function() rma(yi, vi, scale = ~ablat, data = dna))
)
### location-scale fits with a missing value in the scale model ------------------------------------------------------------------------------
ls_na <- function(mk) {
  res <- list()
  options(na.action = "na.omit")
  r <- quiet(mk())
  for (act in acts) {
    options(na.action = act)
    res[[act]] <- list(
      residuals = num(residuals(r)), weights = num(weights(r)), rstandard = num(rstandard(r)$z), rstandard_resid = num(rstandard(r)$resid),
      rstandard_slab = as.character(rstandard(r)$slab), blup = num(blup(r)$pred), pearson = num(residuals(r, type = "pearson")),
      cholesky = num(residuals(r, type = "cholesky")), rstandard_cond = num(rstandard(r, type = "conditional")$z)
    )
  }
  options(na.action = "na.omit")
  res
}
out$ls_na <- list(
  scale_only = ls_na(function() rma(yi, vi, scale = ~ablat, data = dna)),
  both = ls_na(function() rma(yi, vi, mods = ~ablat, scale = ~ablat, data = dna)),
  loc_only = ls_na(function() rma(yi, vi, mods = ~ablat, scale = ~1, data = dna))
)

write_json(out, "src/tests/reference/phase12/audit4_predict.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
