# Reference values for blup(), ranef() and vif() on rma.uni models:
# BLUPs for RE (REML, DL, ML), mixed-effects, EE and Knapp-Hartung fits,
# level=90, transf=exp, user weights (an error in metafor) and fits with
# missing values under na.omit / na.exclude; (G)VIFs from the examples on
# metafor's vif help page, incl. btt groups, table=TRUE and reestimate=TRUE.
# Printed output is captured as well.
#
# Usage (from the repository root):  Rscript tests/r/generate_blup_vif.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
lst <- function(r) list(pred = num(r$pred), se = num(r$se), pi_lb = num(r$pi.lb), pi_ub = num(r$pi.ub), slab = as.character(r$slab))
both <- function(res, ...) list(blup = lst(blup(res, ...)), ranef = lst(ranef(res, ...)))
printed <- function(x) paste(capture.output(print(x)), collapse = "\n")

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
out <- list()

### BLUPs ----------------------------------------------------------------------

res <- rma(yi, vi, data = dat)
out$blup_REML     <- both(res)
out$blup_DL       <- both(rma(yi, vi, data = dat, method = "DL"))
out$blup_ML       <- both(rma(yi, vi, data = dat, method = "ML"))
out$blup_mods     <- both(rma(yi, vi, mods = ~ ablat + year, data = dat))
out$blup_EE       <- both(rma(yi, vi, data = dat, method = "EE"))
out$blup_knha     <- both(rma(yi, vi, mods = ~ ablat, data = dat, test = "knha"))
out$blup_level90  <- both(res, level = 90)
out$blup_level90_fit <- both(rma(yi, vi, data = dat, level = 90))
out$blup_exp      <- both(res, transf = exp)
out$blup_slab     <- both(rma(yi, vi, data = dat, slab = paste(author, year)))
out$weights_error <- list(blup = inherits(try(blup(rma(yi, vi, weights = n, data = dat)), silent = TRUE), "try-error"),
                          ranef = inherits(try(ranef(rma(yi, vi, weights = n, data = dat)), silent = TRUE), "try-error"))

### missing values: yi for study 3 and ablat for study 7 set to NA
dna <- dat
dna$yi[3] <- NA
dna$ablat[7] <- NA
res_na <- suppressWarnings(rma(yi, vi, mods = ~ ablat, data = dna))
out$blup_na_omit <- both(res_na)
options(na.action = "na.exclude")
out$blup_na_exclude <- both(res_na)
options(na.action = "na.omit")

out$print_blup_exp  <- printed(blup(res, transf = exp))
out$print_ranef     <- printed(ranef(res))
out$print_blup_mods <- printed(blup(rma(yi, vi, mods = ~ ablat + year, data = dat)))

### VIFs -------------------------------------------------------------------------

vifl <- function(v) {
  r <- list(vif = num(v$vifs), sif = num(sapply(v$vif, function(x) x$sif)), m = num(sapply(v$vif, function(x) x$m)),
            coefs = as.character(sapply(v$vif, function(x) x$coefs)), names = names(v$vifs), print = printed(v))
  if (!is.null(v$table)) r$table <- lapply(as.list(v$table), num)
  r
}

bd <- dat.bangertdrowns2004
res_bd <- rma(yi, vi, mods = ~ length + wic + feedback + info + pers + imag + meta, data = bd)
out$vif_bd         <- vifl(vif(res_bd))
out$vif_bd_reest   <- vifl(vif(res_bd, reestimate = TRUE))
u <- scale
res_bdu <- rma(yi, vi, mods = ~ u(length) + u(wic) + u(feedback) + u(info) + u(pers) + u(imag) + u(meta), data = bd)
out$vif_bdu_reest  <- vifl(vif(res_bdu, reestimate = TRUE))
out$vif_bdu_table  <- vifl(vif(res_bdu, reestimate = TRUE, table = TRUE))

mc <- escalc(measure = "ZCOR", ri = ri, ni = ni, data = dat.mcdaniel1994)
res_mc <- rma(yi, vi, mods = ~ factor(type) + factor(struct), data = mc)
out$vif_mc <- vifl(vif(res_mc))

res_bcg <- rma(yi, vi, mods = ~ ablat + alloc + year, data = dat)
out$vif_bcg_table   <- vifl(vif(res_bcg, table = TRUE))
out$vif_bcg_btt     <- vifl(vif(res_bcg, btt = 3:4))
out$vif_bcg_btt1    <- vifl(vif(res_bcg, btt = 2))
out$vif_bcg_str     <- vifl(vif(res_bcg, btt = "alloc"))
out$vif_bcg_list    <- vifl(vif(res_bcg, btt = list(2, 3:4, 5)))
out$vif_bcg_list_reest <- vifl(vif(res_bcg, btt = list(2, 3:4, 5), reestimate = TRUE))
out$vif_bcg_reest   <- vifl(vif(res_bcg, reestimate = TRUE))
out$vif_bcg_knha_table <- vifl(vif(rma(yi, vi, mods = ~ ablat + alloc + year, data = dat, test = "knha"), table = TRUE))
out$vif_bcg_ee_reest <- vifl(vif(rma(yi, vi, mods = ~ ablat + alloc + year, data = dat, method = "EE"), reestimate = TRUE))

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "tests/reference/extras/blup_vif.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote tests/reference/extras/blup_vif.json\n")
