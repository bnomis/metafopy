# Runs the audited scripts in R as src/audit/r/runscripts.R does (Rd examples through Rd2ex, testthat files, test_that blocks flattened, the
# same stubs, set.seed(1234) and a 120 s limit per unit), with wrappers in the stubs environment that write one line (script, unit, form)
# whenever a script-level call takes the form of a shim case of agents/specs/r-library-shims-follow-r/spec.md. metafor's own code does not
# see the wrappers, so only the calls the transpiled Python makes through audit.rlib are counted. Shard k of n (k = 0 .. n-1):
#
#   LC_ALL=C.UTF-8 R_LIBS=.audit/Rlib Rscript agents/specs/r-library-shims-follow-r/reach-dynamic.R .audit/source/metafor <k> <n> <out.tsv>
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; shard <- as.integer(args[2]); nshards <- as.integer(args[3]); out <- args[4]
suppressPackageStartupMessages(library(metafor))
suppressPackageStartupMessages(library(metadat))
options(width = 80, warn = 1)
flatten_units <- function(exprs) {
  units <- list()
  for (e in as.list(exprs)) {
    if (is.call(e) && identical(e[[1]], as.name("test_that")) && length(e) >= 3L && is.call(e[[3]]) && identical(e[[3]][[1]], as.name("{"))) {
      for (stmt in as.list(e[[3]])[-1]) units[[length(units) + 1L]] <- stmt
    } else units[[length(units) + 1L]] <- e
  }
  units
}
stubs <- new.env(parent = globalenv())
first <- function(object, ...) object
for (nm in c("expect_equivalent", "expect_equal", "expect_identical", "expect_true", "expect_false", "expect_null", "expect_s3_class", "expect_length",
             "expect_warning", "expect_message", "expect_silent", "expect_gt", "expect_lt", "expect_gte", "expect_lte", "expect_match", "expect_no_error",
             "expect_no_warning", "expect_named", "expect_type", "expect_vector", "expect_setequal", "expect_output", "expect_invisible", "expect_condition"))
  assign(nm, first, envir = stubs)
assign("expect_error", function(object, ...) tryCatch({ force(object); FALSE }, error = function(e) TRUE), envir = stubs)
for (nm in c("skip_on_cran", "skip_if_not_installed", "skip", "context", "png", "pdf", "svg", "jpeg", "dev.off", "graphics.off", "Sys.setenv",
             "requireNamespace", "suppressPackageStartupMessages")) assign(nm, function(...) invisible(NULL), envir = stubs)
assign("library", function(...) invisible(NULL), envir = stubs)
assign("require", function(...) invisible(TRUE), envir = stubs)
assign(".vistest", function(...) TRUE, envir = stubs)
assign("test_that", function(desc, code) invisible(NULL), envir = stubs)

cur <- c(script = "", unit = "")
con <- file(out, open = "wt")
note <- function(tag) { writeLines(paste(cur[["script"]], cur[["unit"]], tag, sep = "\t"), con); flush(con) }
recall_base <- function(fn, cl, env) { cl[[1]] <- fn; eval(cl, env) }
w <- list(
  class = function(x) {
    if (is.null(attr(x, "class"))) {
      note(if (is.matrix(x)) "class:implicit-matrix" else if (is.array(x)) "class:implicit-array" else if (is.numeric(x)) "class:implicit-numeric"
           else paste0("class:implicit-", base::class(x)[1]))
    } else note(paste0("class:object-", paste(base::class(x), collapse = "/")))
    base::class(x)
  },
  inherits = function(x, what, which = FALSE) {
    note(paste0("inherits:", if (is.object(x)) "object" else "implicit", ":", paste(what, collapse = "|")))
    base::inherits(x, what, which)
  },
  length = function(x) { if (is.null(x)) note("length:null"); base::length(x) },
  is.na = function(x) { if (is.data.frame(x)) note("is.na:data.frame"); base::is.na(x) },
  tail = function(x, ...) { if (is.list(x) && !is.data.frame(x)) note(paste0("tail:list-", paste(base::class(x), collapse = "/"))); utils::tail(x, ...) },
  as.data.frame = function(x, ...) {
    if (is.matrix(x)) note(paste0("as.data.frame:matrix-", if (is.null(colnames(x))) "unnamed" else "named"))
    recall_base(quote(base::as.data.frame), sys.call(), parent.frame())
  },
  as.numeric = function(x, ...) { if (is.character(x)) note(paste0("as.numeric:character-length-", if (base::length(x) == 1L) "1" else "n")); base::as.numeric(x, ...) },
  seq = function(...) {
    nm <- names(sys.call())[-1]
    if (!is.null(nm) && all(nzchar(nm)) && "to" %in% nm && any(c("length.out", "length", "len") %in% nm) && !("from" %in% nm)) note("seq:to-length.out-no-from")
    recall_base(quote(base::seq), sys.call(), parent.frame())
  },
  matrix = function(...) { if ("dimnames" %in% names(sys.call())) note("matrix:dimnames"); recall_base(quote(base::matrix), sys.call(), parent.frame()) },
  cbind = function(..., deparse.level = 1) {
    nm <- names(sys.call())[-1]
    if (!is.null(nm) && any(nzchar(setdiff(nm, "deparse.level")))) note("cbind:named")
    recall_base(quote(base::cbind), sys.call(), parent.frame())
  },
  list = function(...) {
    nm <- names(sys.call())[-1]
    if (!is.null(nm) && any(nzchar(nm)) && any(!nzchar(nm))) note("list:mixed")
    recall_base(quote(base::list), sys.call(), parent.frame())
  }
)
for (nm in names(w)) assign(nm, w[[nm]], envir = stubs)
for (op in c("+", "-", "*", "/", "^", "%%", "%/%", "==", "!=", "<", ">", "<=", ">=")) local({
  f <- get(op, baseenv()); o <- op
  assign(op, function(e1, e2) {
    if (missing(e2)) return(f(e1))
    n1 <- base::length(e1); n2 <- base::length(e2)
    if (n1 > 1L && n2 > 1L && n1 != n2) {
      m <- is.matrix(e1) || is.matrix(e2) || is.data.frame(e1) || is.data.frame(e2)
      note(paste0("recycle:", if (m) "matrix-or-frame" else "vectors", ":", o))
    }
    f(e1, e2)
  }, envir = stubs)
})

script_file <- function(file, dest) {
  if (!grepl("\\.Rd$", file)) return(file.path(src, file))
  suppressWarnings(tools::Rd2ex(file.path(src, file), dest))
  if (file.exists(dest)) dest else NULL
}
jobs <- c(paste0("man/", sort(list.files(file.path(src, "man"), pattern = "\\.Rd$"))),
          paste0("tests/testthat/", sort(list.files(file.path(src, "tests", "testthat"), pattern = "^test_.*\\.[rR]$"))))
jobs <- jobs[seq_along(jobs) %% nshards == shard]
tmp <- tempfile(); dir.create(tmp)
for (j in jobs) {
  path <- script_file(j, file.path(tmp, sub("\\.Rd$", ".R", basename(j))))
  if (is.null(path)) next
  id <- if (grepl("\\.Rd$", j)) paste0("rd/", sub("\\.Rd$", "", basename(j))) else paste0("test/", sub("^test_", "", sub("\\.[rR]$", "", basename(j))))
  exprs <- tryCatch(parse(path, keep.source = FALSE), error = function(e) NULL)
  if (is.null(exprs)) next
  units <- flatten_units(exprs)
  env <- new.env(parent = stubs)
  old <- setwd(tmp)
  assign("source", function(file, ...) invisible(sys.source(file.path(dirname(path), basename(file)), envir = env)), envir = stubs)
  pdf(NULL)
  for (i in seq_along(units)) {
    cur[["script"]] <- id; cur[["unit"]] <- as.character(i - 1L)
    set.seed(1234)
    tryCatch({
      setTimeLimit(elapsed = 120, transient = TRUE)
      suppressWarnings(suppressMessages(eval(units[[i]], env)))
    }, error = function(e) NULL)
    setTimeLimit(elapsed = Inf)
  }
  graphics.off()
  setwd(old)
  cur[["unit"]] <- "-"; note("done")
}
close(con)
