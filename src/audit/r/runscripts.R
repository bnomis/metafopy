# Runs the scripts of the audit in R: the \examples of every help page of metafor and the files of its tests/testthat, one top-level expression
# at a time (the statements of a test_that block count as top-level), recording the value of each next to its parse tree.
#
#   Rscript runscripts.R <metafor-source-dir> <out.json> <shard> <shards> [timeout-seconds]
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
suppressPackageStartupMessages(library(metafor))
suppressPackageStartupMessages(library(metadat))
options(width = 80, warn = 1)
source(file.path(here, "drawn.R"))

ast <- function(e) {
  if (is.null(e)) return(list(t = "null"))
  if (is.name(e)) return(list(t = "sym", v = as.character(e)))
  if (is.character(e)) return(list(t = "str", v = e))
  if (is.logical(e)) return(if (is.na(e)) list(t = "na", kind = "lgl") else list(t = "lgl", v = e))
  if (is.integer(e)) return(if (is.na(e)) list(t = "na", kind = "int") else list(t = "int", v = e))
  if (is.numeric(e)) {
    if (is.na(e)) return(list(t = "na", kind = "num"))
    if (is.infinite(e)) return(list(t = "inf", v = if (e > 0) 1 else -1))
    return(list(t = "num", v = e))
  }
  if (is.call(e)) {
    parts <- as.list(e)
    nm <- names(parts)
    args <- lapply(seq_along(parts)[-1], function(i) {
      empty <- identical(parts[[i]], quote(expr = ))
      list(name = if (is.null(nm) || nm[i] == "") NULL else nm[i],
           v = if (empty) list(t = "empty") else ast(parts[[i]]))
    })
    return(list(t = "call", f = ast(parts[[1]]), args = args))
  }
  if (is.pairlist(e)) return(list(t = "pairlist", args = lapply(seq_along(e), function(i) {
    empty <- identical(e[[i]], quote(expr = ))
    list(name = names(e)[i], v = if (empty) list(t = "empty") else ast(e[[i]]))
  })))
  list(t = "other", v = paste(deparse(e), collapse = " "))
}


# the test_that blocks are flattened: their statements are units of the script, run in one shared environment
flatten_units <- function(exprs) {
  units <- list()
  for (e in as.list(exprs)) {
    if (is.call(e) && identical(e[[1]], as.name("test_that")) && length(e) >= 3L && is.call(e[[3]]) && identical(e[[3]][[1]], as.name("{"))) {
      for (stmt in as.list(e[[3]])[-1]) units[[length(units) + 1L]] <- stmt
    } else {
      units[[length(units) + 1L]] <- e
    }
  }
  units
}

stubs <- new.env(parent = globalenv())
first <- function(object, ...) object
for (nm in c("expect_equivalent", "expect_equal", "expect_identical", "expect_true", "expect_false", "expect_null", "expect_s3_class", "expect_length",
             "expect_warning", "expect_message", "expect_silent", "expect_gt", "expect_lt", "expect_gte", "expect_lte", "expect_match", "expect_no_error",
             "expect_no_warning", "expect_named", "expect_type", "expect_vector", "expect_setequal", "expect_output", "expect_invisible", "expect_condition")) {
  assign(nm, first, envir = stubs)
}
assign("expect_error", function(object, ...) tryCatch({ force(object); FALSE }, error = function(e) TRUE), envir = stubs)
for (nm in c("skip_on_cran", "skip_if_not_installed", "skip", "context", "png", "pdf", "svg", "jpeg", "dev.off", "graphics.off", "Sys.setenv", "library", "require",
             "requireNamespace", "suppressPackageStartupMessages")) {
  assign(nm, function(...) invisible(NULL), envir = stubs)
}
assign("library", function(...) invisible(NULL), envir = stubs)
assign("require", function(...) invisible(TRUE), envir = stubs)
assign(".vistest", function(...) TRUE, envir = stubs)
assign("test_that", function(desc, code) invisible(NULL), envir = stubs)

limit <- if (length(commandArgs(TRUE)) >= 5L) as.numeric(commandArgs(TRUE)[5]) else 120

run_unit <- function(expr, env) {
  warns <- character()
  drawn <<- character()
  t0 <- Sys.time()
  res <- withCallingHandlers(
    tryCatch({
      setTimeLimit(elapsed = limit, transient = TRUE)
      withVisible(eval(expr, env))
    }, error = function(e) e),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) invokeRestart("muffleMessage"))
  setTimeLimit(elapsed = Inf)
  out <- list(warnings = as.list(unique(warns)), secs = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  if (length(drawn)) out$drawn <- as.list(sort(unique(drawn)))
  # as in runcases.R: a try() inside metafor can catch the time limit and return an incomplete value
  if (!inherits(res, "error") && out$secs >= limit) res <- simpleError("reached elapsed time limit inside the call, which carried on with the step it interrupted counted as failed")
  if (inherits(res, "error")) {
    out$status <- "error"
    out$error <- conditionMessage(res)
  } else {
    out$status <- "ok"
    out$value <- tryCatch(node_of(res$value), error = function(e) list(k = "opaque", cls = "flatten failed"))
  }
  out
}

run_script <- function(path, id, kind, topic) {
  exprs <- tryCatch(parse(path, keep.source = FALSE), error = function(e) e)
  if (inherits(exprs, "error")) return(list(id = id, kind = kind, topic = topic, error = conditionMessage(exprs), units = list()))
  units <- flatten_units(exprs)
  env <- new.env(parent = stubs)
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)
  assign("source", function(file, ...) invisible(sys.source(file.path(dirname(path), basename(file)), envir = env)), envir = stubs)
  pdf(NULL)
  on.exit(graphics.off(), add = TRUE)
  results <- lapply(units, function(u) {
    set.seed(1234)
    r <- run_unit(u, env)
    r$src <- paste(deparse(u, width.cutoff = 200L), collapse = "\n")
    r$ast <- ast(u)
    r
  })
  list(id = id, kind = kind, topic = topic, units = results)
}

src <- args[1]
shard <- as.integer(args[3])
shards <- as.integer(args[4])
jobs <- list()

rd_dir <- file.path(src, "man")
tmp <- tempfile()
dir.create(tmp)
for (rd in sort(list.files(rd_dir, pattern = "\\.Rd$", full.names = TRUE))) {
  topic <- sub("\\.Rd$", "", basename(rd))
  dest <- file.path(tmp, paste0(topic, ".R"))
  suppressWarnings(tools::Rd2ex(rd, dest))
  if (file.exists(dest)) jobs[[length(jobs) + 1L]] <- list(path = dest, id = paste0("rd/", topic), kind = "rd", topic = topic)
}
tdir <- file.path(src, "tests", "testthat")
for (f in sort(list.files(tdir, pattern = "^test_.*\\.[rR]$", full.names = TRUE))) {
  topic <- sub("^test_", "", sub("\\.[rR]$", "", basename(f)))
  jobs[[length(jobs) + 1L]] <- list(path = f, id = paste0("test/", topic), kind = "test", topic = topic)
}
only <- Sys.getenv("AUDIT_ONLY", "")
if (nzchar(only)) {
  parts <- strsplit(only, "\x1f", fixed = TRUE)[[1]]
  jobs <- Filter(function(j) any(vapply(parts, function(p) grepl(p, j$id, fixed = TRUE), logical(1))), jobs)
}
jobs <- jobs[seq_along(jobs) %% shards == shard %% shards]
out <- lapply(jobs, function(j) run_script(j$path, j$id, j$kind, j$topic))
write_json(list(scripts = out), args[2])
