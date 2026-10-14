# Reference output of verbose= in rma(), rma.mh(), rma.peto(), rma.glmm(), confint(), trimfill() and fsn().
# cat() output goes to stdout and message() output to stderr; both are captured separately.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_verbose.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
grab <- function(expr) {
  msg <- character(0)
  out <- withCallingHandlers(
    capture.output(invisible(suppressWarnings(expr))),
    message = function(m) {
      msg <<- c(msg, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  list(stdout = paste0(paste(out, collapse = "\n"), "\n"), stderr = paste(msg, collapse = ""))
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, data = dat)
out <- list()
out$uni_reml_1 <- grab(rma(yi, vi, data = dat, verbose = TRUE))
out$uni_reml_2 <- grab(rma(yi, vi, mods = ~ablat, data = dat, verbose = 2))
out$uni_pm <- grab(rma(yi, vi, data = dat, method = "PM", verbose = TRUE))
out$uni_dl <- grab(rma(yi, vi, data = dat, method = "DL", verbose = 2))
out$mh <- grab(rma.mh(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, verbose = TRUE))
out$peto <- grab(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, verbose = TRUE))
out$glmm <- grab(rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, model = "UM.FS", verbose = 1))
out$confint_qp <- grab(confint(res, verbose = TRUE))
out$confint_pl <- grab(confint(res, type = "PL", verbose = TRUE))
dat2 <- dat.hackshaw1998
out$trimfill <- grab(trimfill(rma(yi, vi, data = dat2, method = "FE"), verbose = TRUE))
out$fsn <- grab(fsn(res, verbose = TRUE))
write_json(out, "src/tests/reference/phase7/verbose.json", auto_unbox = TRUE, digits = NA, pretty = TRUE)
