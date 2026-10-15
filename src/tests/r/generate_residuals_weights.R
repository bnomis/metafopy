# Reference values for residuals(type = ...) and weights(type = ...) of fits of every class (rma.uni with and without moderators and
# with several estimators, subset and missing values, rma.mv, rma.mh, rma.peto, rma.glmm, location-scale and selection models) under each
# na.action, including the error messages R gives where a type is not available.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_residuals_weights.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) unname(as.vector(x))
nas <- c("na.omit", "na.exclude", "na.pass", "na.fail")

probe <- function(fun) {
  out <- list()
  for (na in nas) {
    options(na.action = na)
    r <- try(suppressWarnings(fun()), silent = TRUE)
    out[[na]] <- if (inherits(r, "try-error")) list(error = conditionMessage(attr(r, "condition")))
                 else list(value = if (is.matrix(r)) unname(r) else num(r))
  }
  options(na.action = "na.omit")
  out
}

types_resid <- c("response", "rstandard", "rstudent", "pearson", "cholesky")

collect <- function(o, wtypes = c("diagonal", "matrix")) {
  out <- list(residuals = list(), weights = list())
  for (ty in types_resid) out$residuals[[ty]] <- probe(function() residuals(o, type = ty))
  for (ty in wtypes) out$weights[[ty]] <- probe(function() weights(o, type = ty))
  out
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$yi[3] <- NA
dat$ablat[7] <- NA
datv <- dat
datv$vi[5] <- NA
dat$n <- seq_len(nrow(dat))

raw <- dat.bcg
raw$tpos[3] <- NA

out <- list()
out$uni_mods <- collect(rma(yi, vi, mods = ~ablat, data = dat))
out$uni_nomods_dl <- collect(rma(yi, vi, data = dat, method = "DL"))
out$uni_fe_mods <- collect(rma(yi, vi, mods = ~ablat + year, data = dat, method = "FE"))
out$uni_ml_subset <- collect(rma(yi, vi, mods = ~ablat, data = dat, method = "ML", subset = 1:10))
out$uni_vi_na <- collect(rma(yi, vi, data = datv))
out$uni_weights <- collect(rma(yi, vi, mods = ~ablat, data = dat, weights = n))
out$uni_unweighted <- collect(rma(yi, vi, mods = ~ablat, data = dat, weighted = FALSE))
out$uni_knha <- collect(rma(yi, vi, mods = ~ablat, data = dat, test = "knha"))
out$uni_robust <- collect(robust(rma(yi, vi, data = dat), cluster = dat$trial))
out$uni_complete <- collect(rma(yi, vi, data = escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)))
out$ls <- collect(rma(yi, vi, mods = ~ablat, scale = ~year, data = dat))
out$sel <- collect(selmodel(rma(yi, vi, data = dat), type = "beta"))
out$mh <- collect(rma.mh(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, subset = 1:11))
out$peto <- collect(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw))
out$mh_complete <- collect(rma.mh(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg))
out$glmm <- collect(rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = raw, mods = ~ablat))
out$mv_konst <- collect(rma.mv(yi, vi, random = ~1 | district/school, data = dat.konstantopoulos2011), c("diagonal", "matrix", "rowsum"))
out$mv_na_mods <- collect(rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dat), c("diagonal", "matrix", "rowsum"))
out$mv_na_int <- collect(rma.mv(yi, vi, random = ~1 | trial, data = dat), c("diagonal", "matrix", "rowsum"))

# the type is matched partially, as match.arg does
o <- rma(yi, vi, mods = ~ablat, data = dat)
out$partial <- list(resp = num(residuals(o, type = "resp")), pear = num(residuals(o, type = "pea")), rstan = num(residuals(o, type = "rstan")))
out$partial_error <- tryCatch({ residuals(o, type = "r"); NULL }, error = function(e) conditionMessage(e))

write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), "src/tests/reference/phase9/residuals_weights.json")
