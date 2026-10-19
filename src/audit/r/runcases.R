# Runs the cases of a shard in R and writes the outcome of each: status, warnings, value tree and printed text.
#
#   Rscript runcases.R <fixtures.json> <cases.json> <out.jsonl> [timeout-seconds]
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
suppressPackageStartupMessages(library(metafor))
options(width = 80, warn = 1)
limit <- if (length(args) >= 4L) as.numeric(args[4]) else 120

fixtures <- read_json(args[1])
cases <- read_json(args[2])

env <- new.env()
fixture_errors <- list()
# the Python side seeds 1234 before it builds the fixtures, so fixtures that draw random numbers (hettest(), permutest()) must start from the same state
set.seed(1234)
for (fx in fixtures) {
  res <- tryCatch(run_call(fx, env), error = function(e) e)
  if (inherits(res, "error")) {
    fixture_errors[[fx$name]] <- conditionMessage(res)
  } else {
    env[[fx$name]] <- res
  }
}

pdf(NULL)
TIME_LIMIT_SWALLOWED <- "reached elapsed time limit inside the call, which carried on with the step it interrupted counted as failed"
run_case <- function(case) {
  warns <- character()
  out <- list(id = case$id, status = "ok")
  t0 <- Sys.time()
  set.seed(1234)
  # a case must not see the options that another one left behind (forest(), funnel() and baujat() set na.action and put it back on exit)
  options(na.action = "na.omit")
  value <- withCallingHandlers(
    tryCatch({
      setTimeLimit(elapsed = limit, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf), add = TRUE)
      missing_fx <- Filter(function(r) !is.null(fixture_errors[[r]]), unlist(case$needs))
      if (length(missing_fx)) stop("fixture failed: ", missing_fx[[1]], ": ", fixture_errors[[missing_fx[[1]]]])
      run_call(case$call, env)
    }, error = function(e) e),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) invokeRestart("muffleMessage"))
  out$secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (!identical(getOption("na.action"), "na.omit")) {
    warns <- c(warns, paste0("options(na.action) was left at ", getOption("na.action"), " by this case"))
    options(na.action = "na.omit")
  }
  out$warnings <- as.list(unique(warns))
  # the time limit raises an error wherever R is, and a try() of the code under test (permutest() refits in try()) records it as a failed step
  if (!inherits(value, "error") && out$secs >= limit) value <- simpleError(TIME_LIMIT_SWALLOWED)
  if (inherits(value, "error")) {
    out$status <- "error"
    out$error <- conditionMessage(value)
    return(out)
  }
  out$value <- tryCatch(node_of(value), error = function(e) list(k = "opaque", cls = paste("flatten failed:", conditionMessage(e))))
  if (isTRUE(case$print)) {
    out$text <- tryCatch(print_text(value), error = function(e) paste("print failed:", conditionMessage(e)))
  }
  out
}

# one JSON line per finished case, preceded by a marker line, so that a crash of R costs only the case it died on
con <- file(args[3], open = "wt")
for (case in cases) {
  writeLines(toJSON(list(start = case$id), auto_unbox = TRUE), con)
  flush(con)
  res <- run_case(case)
  writeLines(toJSON(res, auto_unbox = TRUE, null = "null", na = "null", digits = NA), con)
  flush(con)
}
close(con)
