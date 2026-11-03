# Reference values for the last bit of R's dbinom() and of the random subsets gosh() draws with it, from the R this machine runs.
# R built with fused multiply-add (arm64 clang) and R built without it give different last bits; the fixture records which one made it.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_dbinom_fused.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
r13 <- rma(yi, vi, data = dat)

out <- list()

### dbinom --------------------------------------------------------------------------
out$dbinom <- list()
for (k in c(2, 5, 8, 13, 16, 24, 28, 37, 90, 210, 400)) {
  for (p in c(0.5, 0.3, 0.7)) {
    out$dbinom[[length(out$dbinom) + 1]] <- list(k = k, p = p, x = 0:k, hex = sprintf("%a", dbinom(0:k, k, p)))
  }
}

### gosh ----------------------------------------------------------------------------
gsh <- function(x, ...) {
  r <- suppressMessages(gosh(x, progbar = FALSE, ...))
  list(res = lapply(r$res, num), incl = lapply(seq_len(nrow(r$incl)), function(i) as.integer(r$incl[i, ])), fit = num(r$fit), k = r$k,
       print = printed(r))
}
set.seed(99)
out$gosh_random_global <- gsh(r13, subsets = 100)

### Which R made this ---------------------------------------------------------------
out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string, platform = R.version$platform,
                 probe = sprintf("%a", dbinom(10, 13, 0.5)))
write_json(out, "src/tests/reference/phase12/dbinom_fused.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase12/dbinom_fused.json\n")
