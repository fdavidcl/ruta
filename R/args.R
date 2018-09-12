test_function <- function(network = arg_network(), loss = arg_loss(), weight = 2e-4) {
  check_args_internal()
}

# %to% operator. Defines intervals for argument checks
`%to%` <- function(min, max) list(min = min, max = max)

arg_constructor <- function(.class, .values, .default = NULL) {
  structure(function(default = .default, get = FALSE) {
    if (get) {
      list(
        class = .class,
        values = .values
      )
    } else {
      default
    }
  }, class = ruta_arg)
}

as_arg <- function(x) UseMethod("as_arg", x)
as_arg.ruta_arg <- function(x) x
as_arg.default <- function(x) {
  arg_constructor(class(x), .default = x)
}

arg_network <- arg_constructor("ruta_network")

arg_loss <- arg_constructor(
  c("ruta_loss", "character"),
  c("mean_squared_error"),
  "mean_squared_error"
)

which_args <- function(f) {
  defaults <- formals(f)

  checks <- lapply(defaults, function(arg) {
    if (class(arg) == "call") {
      get(as.character(arg))(get = TRUE)
    } else {
      as_arg(arg)(get = TRUE)
    }
  })

  names(checks) <- names(defaults)
  checks
}

check_args_internal <- function() {
  call_l <- as.list(sys.call(sys.parent(1)))
  #print(call_l)
  checks <- which_args(as.character(call_l[[1]]))
  #print(checks)

  check_args(call_l[2:length(call_l)], checks)
}

check_args <- function(args, checks) {
  unnamed_args <- if (is.null(names(args))) seq_along(args) else which(names(args) == "")
  missing_args <- setdiff(names(checks), names(args)[-unnamed_args])
  names(args)[unnamed_args] <- missing_args[1:length(unnamed_args)]

  #print(unnamed_args)
  #print(missing_args)
  #print(args)

  for (name in names(args)) {
    check <- checks[[name]]

    if (!is.null(check)) {
      val <- eval(args[[name]])

      # Check type: class
      if (!is.null(check$class)) {
        if (length(intersect(class(val), check$class)) == 0) {
          stop(paste0(name, " does not have any allowed class (", paste(check$class, collapse = ", "), ")"))
        }
      }

      # Check type: values
      if (!is.null(check$values)) {
        # Value check with set
        if (is.atomic(check$values)) {
          if (!(val %in% check$values)) {
            stop(paste0(name, " does not equal any allowed value (", paste(check$values, collapse = ", "), ")"))
          }
        } else {
          # Value check in interval
          if (!is.null(check$values$min)) {
            if (val < check$values$min) {
              stop(paste0(name, " is lower than the minimum allowed value (", check$values$min, ")"))
            }
          }
          if (!is.null(check$values$max)) {
            if (val > check$values$max) {
              stop(paste0(name, " is higher than the maximum allowed value (", check$values$max, ")"))
            }
          }
        }
      }
    }
  }

  invisible(TRUE)
}


