# Runs the scripts of the audit in R: the \examples of every help page of metafor and the files of its tests/testthat, one top-level expression
# at a time (the statements of a test_that block count as top-level), recording the value of each next to its parse tree.
#
#   Rscript runscripts.R list <metafor-source-dir> <out.json>
#   Rscript runscripts.R run <metafor-source-dir> <scripts.json> <out.jsonl> [timeout-seconds]
#
# list writes [{id, kind, topic, file}] for every script (AUDIT_ONLY filters by id). run executes the listed scripts and writes one JSON line per
# event, flushed each time: {"start": id}, one {"part": id, "unit": ...} per unit, then {"id", "kind", "topic"}, so that a crash of R costs only the
# rest of the script it died in.
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

limit <- if (identical(args[1], "run") && length(args) >= 5L) as.numeric(args[5]) else 120

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

# runs one script, calling emit(unit_record) as each unit finishes; returns list(error = ...) when the file does not parse, else list()
run_script <- function(path, emit) {
  exprs <- tryCatch(parse(path, keep.source = FALSE), error = function(e) e)
  if (inherits(exprs, "error")) return(list(error = conditionMessage(exprs)))
  units <- flatten_units(exprs)
  env <- new.env(parent = stubs)
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)
  assign("source", function(file, ...) invisible(sys.source(file.path(dirname(path), basename(file)), envir = env)), envir = stubs)
  pdf(NULL)
  on.exit(graphics.off(), add = TRUE)
  for (u in units) {
    set.seed(1234)
    r <- run_unit(u, env)
    r$src <- paste(deparse(u, width.cutoff = 200L), collapse = "\n")
    r$ast <- ast(u)
    emit(r)
  }
  list()
}

# the R file of a script: the examples of an Rd page (NULL when it has none), or the test file itself
script_file <- function(src, file, dest) {
  if (!grepl("\\.Rd$", file)) return(file.path(src, file))
  suppressWarnings(tools::Rd2ex(file.path(src, file), dest))
  if (file.exists(dest)) dest else NULL
}

mode <- args[1]
src <- args[2]
if (identical(mode, "list")) {
  jobs <- list()
  tmp <- tempfile()
  dir.create(tmp)
  for (rd in sort(list.files(file.path(src, "man"), pattern = "\\.Rd$"))) {
    topic <- sub("\\.Rd$", "", rd)
    if (!is.null(script_file(src, file.path("man", rd), file.path(tmp, paste0(topic, ".R")))))
      jobs[[length(jobs) + 1L]] <- list(id = paste0("rd/", topic), kind = "rd", topic = topic, file = file.path("man", rd))
  }
  for (f in sort(list.files(file.path(src, "tests", "testthat"), pattern = "^test_.*\\.[rR]$"))) {
    topic <- sub("^test_", "", sub("\\.[rR]$", "", f))
    jobs[[length(jobs) + 1L]] <- list(id = paste0("test/", topic), kind = "test", topic = topic, file = file.path("tests", "testthat", f))
  }
  only <- Sys.getenv("AUDIT_ONLY", "")
  if (nzchar(only)) {
    parts <- strsplit(only, "\x1f", fixed = TRUE)[[1]]
    jobs <- Filter(function(j) any(vapply(parts, function(p) grepl(p, j$id, fixed = TRUE), logical(1))), jobs)
  }
  write_json(jobs, args[3])
} else if (identical(mode, "run")) {
  jobs <- read_json(args[3])
  tmp <- tempfile()
  dir.create(tmp)
  con <- file(args[4], open = "wt")
  line <- function(x) {
    writeLines(toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = NA), con)
    flush(con)
  }
  for (j in jobs) {
    line(list(start = j$id))
    path <- script_file(src, j$file, file.path(tmp, paste0(j$topic, ".R")))
    res <- if (is.null(path)) list(error = "no examples") else run_script(path, function(r) line(list(part = j$id, unit = r)))
    line(c(list(id = j$id, kind = j$kind, topic = j$topic), res))
  }
  close(con)
} else {
  stop("usage: runscripts.R list|run ...")
}
