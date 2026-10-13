suppressMessages({library(metafor); library(jsonlite)})
src <- deparse(metafor:::.setlab)
ml <- src[grepl("measure", src) & grepl('"', src)]
toks <- unique(unlist(regmatches(ml, gregexpr('"[A-Za-z0-9]+"', ml))))
toks <- gsub('"', "", toks)
measures <- toks[grepl("^[A-Z0-9]+$", toks) & nchar(toks) >= 2]
fnames <- c(grep("^transf\\.", ls("package:metafor"), value = TRUE), "exp", "tanh", "plogis", "pnorm", "sqrt")
fnames <- fnames[sapply(fnames, function(f) is.function(get(f)))]
lab <- function(m, tc, ac, g, s) {
  r <- tryCatch(metafor:::.setlab(m, tc, ac, gentype = g, short = s), error = function(e) NA)
  if (is.expression(r)) paste0("EXPR:", paste(deparse(r[[1]]), collapse = "")) else as.character(r)
}
other <- "function (x) x"
out <- list()
for (m in measures) {
  base <- list()
  for (g in 1:3) for (s in c(TRUE, FALSE)) {
    key <- paste0(g, ifelse(s, "s", "l"))
    base[[key]] <- list(none = lab(m, "FALSE", "FALSE", g, s), other_t = lab(m, other, "FALSE", g, s), other_a = lab(m, "FALSE", other, g, s))
    ex <- list()
    for (f in fnames) {
      d <- paste(deparse(get(f)), collapse = "\n")
      lt <- lab(m, d, "FALSE", g, s); la <- lab(m, "FALSE", d, g, s)
      if (!identical(lt, base[[key]]$other_t)) ex[[paste0("t:", f)]] <- lt
      if (!identical(la, base[[key]]$other_a)) ex[[paste0("a:", f)]] <- la
    }
    base[[key]]$ex <- ex
  }
  out[[m]] <- base
}
write(toJSON(out, auto_unbox = TRUE, pretty = FALSE), "setlab.json")
cat(length(measures), "measures;", length(fnames), "funcs\n")
