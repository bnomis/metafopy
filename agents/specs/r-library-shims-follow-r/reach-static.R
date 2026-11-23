# Static count of the calls to the R functions behind the shims of agents/specs/r-library-shims-follow-r/spec.md, in every script the
# scripts step runs (the \examples of each Rd page through tools::Rd2ex, and each tests/testthat/test_*.R file), one row per call site.
#
#   LC_ALL=C.UTF-8 R_LIBS=.audit/Rlib Rscript agents/specs/r-library-shims-follow-r/reach-static.R .audit/source/metafor <out.json>
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; out <- args[2]
targets <- c("as.data.frame","as.numeric","cbind","class","inherits","is.na","length","list","matrix","seq","tail","+","-","*","/","^")
tmp <- tempfile(); dir.create(tmp)
files <- list()
for (rd in sort(list.files(file.path(src, "man"), pattern = "\\.Rd$"))) {
  dest <- file.path(tmp, sub("\\.Rd$", ".R", rd))
  suppressWarnings(tools::Rd2ex(file.path(src, "man", rd), dest))
  if (file.exists(dest)) files[[paste0("rd/", sub("\\.Rd$", "", rd))]] <- dest
}
for (f in sort(list.files(file.path(src, "tests", "testthat"), pattern = "^test_.*\\.[rR]$")))
  files[[paste0("test/", sub("^test_", "", sub("\\.[rR]$", "", f)))]] <- file.path(src, "tests", "testthat", f)
rows <- list()
walk <- function(e, id) {
  if (is.call(e)) {
    fn <- e[[1]]
    if (is.name(fn) && as.character(fn) %in% targets) {
      nm <- names(e); if (is.null(nm)) nm <- rep("", length(e))
      rows[[length(rows) + 1L]] <<- list(script = id, fn = as.character(fn), nargs = length(e) - 1L,
        argnames = paste(nm[-1], collapse = ","), text = paste(deparse(e, width.cutoff = 200L), collapse = " "))
    }
    for (a in as.list(e)) if (!missing(a)) walk(a, id)
  } else if (is.pairlist(e)) for (a in as.list(e)) if (!missing(a)) walk(a, id)
}
for (id in names(files)) {
  exprs <- tryCatch(parse(files[[id]], keep.source = FALSE), error = function(e) NULL)
  if (is.null(exprs)) next
  for (e in as.list(exprs)) walk(e, id)
}
cat("scripts", length(files), "\n")
jsonlite::write_json(rows, out, auto_unbox = TRUE)
