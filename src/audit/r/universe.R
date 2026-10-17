# Dumps the universe the audit enumerates from: every function and S3 method of metafor with its formals, the choices of its enumerated arguments,
# and the arguments it reads from `...`, plus the datasets and the installed package versions.
#
#   Rscript universe.R <fixtures.json> <out.json>
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
suppressPackageStartupMessages(library(metafor))

ns <- asNamespace("metafor")

const_strings <- function(e) {
  if (is.call(e) && identical(e[[1]], as.name("c")) && length(e) > 1L) {
    parts <- as.list(e)[-1]
    if (all(vapply(parts, is.character, logical(1)))) return(unlist(parts))
  }
  if (is.character(e) && length(e) > 1L) return(e)
  NULL
}

# names read from `...`: X <- list(...) then X$name / X[["name"]], and .chkdots(X, c("a", "b")).
dots_names <- function(body) {
  holders <- character()
  found <- character()
  walk <- function(e) {
    if (is.call(e)) {
      head <- e[[1]]
      if (identical(head, as.name("<-")) || identical(head, as.name("="))) {
        rhs <- e[[3]]
        if (is.call(rhs) && identical(rhs[[1]], as.name("list")) && length(rhs) == 2L && identical(rhs[[2]], as.name("..."))) {
          if (is.name(e[[2]])) holders <<- c(holders, as.character(e[[2]]))
        }
      }
      if ((identical(head, as.name("$")) || identical(head, as.name("[["))) && length(e) == 3L && is.name(e[[2]]) && as.character(e[[2]]) %in% holders) {
        nm <- e[[3]]
        found <<- c(found, if (is.name(nm)) as.character(nm) else if (is.character(nm)) nm else character())
      }
      if (is.name(head) && as.character(head) == ".chkdots" && length(e) >= 3L) {
        found <<- c(found, if (is.character(e[[3]])) e[[3]] else character())
      }
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    } else if (is.pairlist(e) || is.expression(e)) {
      for (i in seq_along(e)) walk(e[[i]])
    }
  }
  walk(body)
  sort(unique(found))
}

# metafor functions the body hands its `...` to (addpoly.rma passes them on to addpoly.default): what those read is read for the caller too
dots_delegates <- function(body) {
  found <- character()
  walk <- function(e) {
    if (is.call(e)) {
      if (is.name(e[[1]]) && any(vapply(as.list(e)[-1], function(a) identical(a, as.name("...")), logical(1)))) found <<- c(found, as.character(e[[1]]))
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    }
  }
  walk(body)
  unique(found)
}

dots_all <- function(f, seen = character()) {
  own <- dots_names(body(f))
  for (d in dots_delegates(body(f))) {
    if (d %in% seen || !exists(d, envir = ns, inherits = FALSE)) next
    g <- get(d, envir = ns)
    if (is.function(g) && "UseMethod" %in% all.names(body(g)) && exists(paste0(d, ".default"), envir = ns, inherits = FALSE)) {
      d <- paste0(d, ".default")
      g <- get(d, envir = ns)
    }
    if (is.function(g) && "..." %in% names(formals(g))) own <- c(own, dots_all(g, c(seen, d)))
  }
  sort(unique(own))
}

graphics_calls <- c("plot", "par", "segments", "polygon", "lines", "points", "abline", "legend", "mtext", "axis", "rect", "arrows", "text", "title")

# the functions the body calls (a variable called legend, or mstyle$legend, is not a call of legend())
called_names <- function(body) {
  found <- character()
  walk <- function(e) {
    if (is.call(e)) {
      if (is.name(e[[1]])) found <<- c(found, as.character(e[[1]]))
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    }
  }
  walk(body)
  unique(found)
}

# .chkclass(class(x), must = "rma", notav = c("rma.glmm"), notap = ...): metafor's own declaration of the fit classes a function accepts.
chkclass_calls <- function(body) {
  found <- list()
  walk <- function(e) {
    if (is.call(e)) {
      if (is.name(e[[1]]) && as.character(e[[1]]) == ".chkclass") {
        args <- as.list(e)[-1]
        nm <- names(args)
        if (is.null(nm)) nm <- rep("", length(args))
        entry <- list()
        for (i in seq_along(args)) {
          if (nm[i] %in% c("must", "notap", "notav", "notav.error")) {
            v <- const_strings(args[[i]])
            if (is.null(v) && is.character(args[[i]])) v <- args[[i]]
            entry[[nm[i]]] <- as.list(v)
          }
        }
        found[[length(found) + 1L]] <<- entry
      }
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    }
  }
  walk(body)
  found
}

# string constants the body compares an argument with: is.element(arg, c(...)), arg %in% c(...), match.arg(arg, c(...)), switch(arg, a = , b = ).
body_choices <- function(body, argnames) {
  out <- list()
  add <- function(a, v) out[[a]] <<- sort(unique(c(out[[a]], v)))
  walk <- function(e) {
    if (is.call(e)) {
      head <- if (is.name(e[[1]])) as.character(e[[1]]) else ""
      if (head %in% c("is.element", "match.arg") && length(e) >= 3L && is.name(e[[2]]) && as.character(e[[2]]) %in% argnames) {
        v <- const_strings(e[[3]])
        if (!is.null(v)) add(as.character(e[[2]]), v)
      }
      if (head == "%in%" && length(e) == 3L && is.name(e[[2]]) && as.character(e[[2]]) %in% argnames) {
        v <- const_strings(e[[3]])
        if (!is.null(v)) add(as.character(e[[2]]), v)
      }
      if (head == "switch" && length(e) >= 3L && is.name(e[[2]]) && as.character(e[[2]]) %in% argnames) {
        labels <- names(as.list(e))[-(1:2)]
        labels <- labels[!is.na(labels) & labels != ""]
        if (length(labels)) add(as.character(e[[2]]), labels)
      }
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    }
  }
  walk(body)
  out
}

# options the body reads with getOption("name"): R's way to pass na.action and similar settings to a function
options_read <- function(body) {
  found <- character()
  walk <- function(e) {
    if (is.call(e)) {
      if (is.name(e[[1]]) && as.character(e[[1]]) == "getOption" && length(e) >= 2L && is.character(e[[2]])) found <<- c(found, e[[2]])
      for (i in seq_along(e)) try(walk(e[[i]]), silent = TRUE)
    }
  }
  walk(body)
  sort(unique(found))
}

describe <- function(f) {
  fm <- formals(f)
  list(
    draws = any(graphics_calls %in% called_names(body(f))),
    formals = lapply(names(fm), function(n) {
      miss <- identical(fm[[n]], quote(expr = ))
      list(name = n,
           default = if (miss) NULL else paste(deparse(fm[[n]], width.cutoff = 500L), collapse = " "),
           choices = if (miss) NULL else as.list(const_strings(fm[[n]])))
    }),
    dots = as.list(dots_all(f)),
    options_read = as.list(options_read(body(f))),
    chkclass = chkclass_calls(body(f)),
    body_choices = lapply(body_choices(body(f), names(fm)), as.list),
    uses_dots = "..." %in% names(fm))
}

exports <- sort(getNamespaceExports("metafor"))
exports <- exports[vapply(exports, function(n) is.function(get(n, envir = ns)), logical(1))]
functions <- setNames(lapply(exports, function(n) describe(get(n, envir = ns))), exports)

s3 <- getNamespaceInfo(ns, "S3methods")
methods <- lapply(seq_len(nrow(s3)), function(i) {
  f <- tryCatch(get(s3[i, 3], envir = ns), error = function(e) NULL)
  d <- if (is.function(f)) describe(f) else list(formals = list(), dots = list(), uses_dots = FALSE)
  c(list(generic = s3[i, 1], class = s3[i, 2], method = s3[i, 3]), d)
})

meta_data <- tryCatch(utils::data(package = "metadat")$results[, "Item"], error = function(e) character())
own_data <- tryCatch(utils::data(package = "metafor")$results[, "Item"], error = function(e) character())

fixtures <- read_json(args[1])
env <- new.env()
fixture_info <- list()
for (fx in fixtures) {
  res <- tryCatch(run_call(fx, env), error = function(e) e)
  if (inherits(res, "error")) {
    fixture_info[[fx$name]] <- list(error = conditionMessage(res))
  } else {
    env[[fx$name]] <- res
    fixture_info[[fx$name]] <- list(class = as.list(class(res)))
  }
}

out <- list(
  fixtures = fixture_info,
  r = paste(R.version$major, R.version$minor, sep = "."),
  metafor = as.character(packageVersion("metafor")),
  functions = functions,
  methods = methods,
  datasets = as.list(sort(unique(sub(" .*", "", c(meta_data, own_data))))))
write_json(out, args[2])
