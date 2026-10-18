# Shared helpers of the audit's R side: JSON in, canonical value trees out.
#
# A value tree is the language-neutral form both sides compare (src/audit/valuetree.py is the Python side):
#   {"k": "num"|"str"|"lgl", "v": [...], "n": [names] | null, "d": [dims] | null}
#   {"k": "df", "cols": [{"name": ..., "node": ...}], "rn": [...]}
#   {"k": "list", "items": [{"name": ..., "node": ...}], "cls": [...]}
#   {"k": "null"} and {"k": "opaque", "cls": "function"}
# Non-finite numbers are the strings "Inf" and "-Inf"; NA and NaN are null.

suppressPackageStartupMessages(library(jsonlite))

read_json <- function(path) fromJSON(path, simplifyVector = FALSE)

write_json <- function(x, path) {
  writeLines(toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = NA, pretty = FALSE), path)
}

scalar_or_null <- function(x) if (length(x) == 0L) NULL else x

num_cell <- function(x) {
  if (is.na(x)) return(NULL)
  if (is.infinite(x)) return(if (x > 0) "Inf" else "-Inf")
  as.numeric(x)
}

vec_node <- function(x) {
  kind <- if (is.logical(x)) "lgl" else if (is.character(x)) "str" else "num"
  nm <- names(x)
  d <- dim(x)
  v <- if (kind == "num") lapply(as.numeric(x), num_cell) else if (kind == "lgl") lapply(as.logical(x), function(e) if (is.na(e)) NULL else e)
       else lapply(as.character(x), function(e) if (is.na(e)) NULL else e)
  if (length(d) == 2L) {
    rn <- rownames(x)
    cn <- colnames(x)
    nm <- NULL
  } else {
    rn <- NULL
    cn <- NULL
  }
  list(k = kind, v = v, n = if (is.null(nm)) NULL else as.list(nm), d = if (is.null(d)) NULL else as.list(as.integer(d)),
       rn = if (is.null(rn)) NULL else as.list(rn), cn = if (is.null(cn)) NULL else as.list(cn))
}

node_of <- function(x, depth = 0L) {
  if (is.null(x)) return(list(k = "null"))
  if (depth > 6L) return(list(k = "opaque", cls = "depth"))
  if (is.function(x) || is.environment(x) || is.language(x) || inherits(x, "formula")) return(list(k = "opaque", cls = class(x)[1]))
  if (is.factor(x)) x <- as.character(x)
  if (is.data.frame(x)) {
    cols <- lapply(seq_along(x), function(i) list(name = names(x)[i], node = node_of(x[[i]], depth + 1L)))
    for (a in c("ni", "slab", "measure")) {
      v <- attr(x, a, exact = TRUE)
      if (!is.null(v) && is.atomic(v)) cols[[length(cols) + 1L]] <- list(name = paste0("@", a), node = vec_node(unclass(v)))
    }
    return(list(k = "df", cols = cols, rn = as.list(rownames(x)), cls = as.list(class(x))))
  }
  if (is.atomic(x)) {
    if (inherits(x, "Date") || inherits(x, "POSIXct")) x <- as.character(x)
    node <- vec_node(unclass(x))
    if (!is.null(oldClass(x))) node$cls <- as.list(oldClass(x))
    return(node)
  }
  if (is.list(x)) {
    nm <- names(x)
    items <- lapply(seq_along(x), function(i) list(name = if (is.null(nm) || is.na(nm[i]) || nm[i] == "") NULL else nm[i], node = node_of(x[[i]], depth + 1L)))
    return(list(k = "list", items = items, cls = as.list(class(x))))
  }
  list(k = "opaque", cls = class(x)[1])
}

# Argument values in a case: plain JSON, plus {"$ref": fixture}, {"$data": dataset}, {"$sym": column}, {"$formula": text}, {"$fn": name},
# {"$col": [fixture, column]}, {"$na": true}, {"$idx": [...]} (indices: 1-based here, 0-based in Python) and {"$inf": 1|-1} {"$matrix": [[row], ...]} and {"$lambda": {"args": "x1, x2", "body": "x1/x2"}}, {"$named": {"a": 1}} (a named vector) and {"$call": spec} (the value of another call).
resolve_value <- function(x, env) {
  if (is.list(x) && !is.null(names(x)) && length(x) == 1L && substr(names(x)[1], 1L, 1L) == "$") {
    tag <- names(x)[1]
    val <- x[[1]]
    return(switch(tag,
      "$ref" = env[[val]],
      "$data" = get_dataset(val),
      "$sym" = as.name(val),
      "$formula" = as.formula(val, env = globalenv()),
      "$fn" = get_function(val),
      "$col" = env[[val[[1]]]][[val[[2]]]],
      "$na" = NA,
      "$idx" = unlist(val),
      "$lambda" = eval(parse(text = paste0("function(", val$args, ") ", val$body))),
      "$matrix" = {
        rows <- if (!is.null(names(val)) && "data" %in% names(val)) val$data else val
        m <- do.call(rbind, lapply(rows, function(r) as.numeric(unlist(r))))
        if (!is.null(names(val)) && !is.null(val$rn)) rownames(m) <- unlist(val$rn)
        if (!is.null(names(val)) && !is.null(val$cn)) colnames(m) <- unlist(val$cn)
        m
      },
      "$named" = unlist(val),
      "$call" = run_call(val, env),
      "$inf" = if (val > 0) Inf else -Inf,
      stop("unknown value tag ", tag)))
  }
  if (is.list(x)) {
    out <- lapply(x, resolve_value, env = env)
    # c(1, NA, 3) is a numeric vector in R; the $na tag keeps it from being simplified before it is resolved
    has_na <- is.null(names(x)) && any(vapply(x, function(e) is.list(e) && identical(names(e), "$na"), logical(1)))
    if (has_na && all(vapply(out, function(e) is.atomic(e) && length(e) == 1L, logical(1)))) return(unlist(out))
    return(out)
  }
  x
}

dataset_cache <- new.env()
get_dataset <- function(name) {
  if (!is.null(dataset_cache[[name]])) return(dataset_cache[[name]])
  e <- new.env()
  utils::data(list = name, package = c("metadat", "metafor"), envir = e)
  dataset_cache[[name]] <- e[[name]]
  e[[name]]
}

get_function <- function(name) {
  if (exists(name, envir = asNamespace("metafor"), inherits = FALSE)) return(get(name, envir = asNamespace("metafor")))
  get(name, envir = globalenv())
}

# Simplify JSON arrays of scalars to atomic vectors, as R users write c(...).
simplify_args <- function(x) {
  if (is.list(x) && is.null(names(x)) && length(x) > 0L && all(vapply(x, function(e) is.atomic(e) && length(e) == 1L, logical(1)))) {
    return(unlist(lapply(x, function(e) if (is.null(e)) NA else e)))
  }
  if (is.list(x) && !is.null(names(x)) && !(length(x) == 1L && substr(names(x)[1], 1L, 1L) == "$")) return(lapply(x, simplify_args))
  if (is.list(x) && is.null(names(x))) return(lapply(x, simplify_args))
  x
}

# Run one call spec: {"fn": name, "target": value|null, "args": {...}}.
run_call <- function(spec, env) {
  if (identical(spec$op, "setna")) {
    x <- get_dataset(spec$from)
    for (cell in spec$cells) x[cell[[1]], cell[[2]]] <- NA
    return(x)
  }
  for (p in spec$pre) run_call(p, env)
  fn <- get_function(spec$fn)
  args <- if (is.null(spec$args)) list() else lapply(spec$args, function(a) resolve_value(simplify_args(a), env))
  if (!is.null(spec$target)) args <- c(list(resolve_value(spec$target, env)), args)
  if (!is.null(spec$options)) {
    old <- do.call(options, spec$options)
    on.exit(options(old), add = TRUE)
  }
  if (identical(spec$fn, "print")) return(utils::capture.output(do.call(fn, args)))
  do.call(fn, args)
}

print_text <- function(x) {
  paste(capture.output(print(x)), collapse = "\n")
}
