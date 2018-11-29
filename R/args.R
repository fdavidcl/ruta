# %to% operator. Defines intervals for argument checks
`%to%` <- function(min, max) list(min = min, max = max)

# Constructor for argument descriptors
arg_constructor <- function(.class, .values = NULL, .default = NULL, .required = FALSE) {
  # A function which returns a default value for a parameter. If the `get`
  # argument is set to TRUE, a machine-readable description of
  # the parameter is returned instead.
  structure(function(default = .default, get = FALSE) {
    if (get) {
      list(
        class = .class,
        values = .values,
        required = .required
      )
    } else {
      default
    }
  }, class = ruta_arg)
}

as_arg <- function(x) UseMethod("as_arg", x)
as_arg.ruta_arg <- function(x) x

# Any value can be used to generate an argument descriptor for its class
as_arg.default <- function(x) {
  if (is.name(x) && x == "")
    arg_constructor(NULL, .required = TRUE)
  else
    arg_constructor(class(x), .default = x)
}

# Argument descriptor for a neural network object
arg_network <- arg_constructor("ruta_network", .required = TRUE)

# Argument descriptor for a loss function
arg_loss <- arg_constructor(
  c("ruta_loss", "character"),
  list_keras_objects("loss"),
  "mean_squared_error"
)

arg_activation <- arg_constructor(
  c("character"),
  list_keras_objects("activation"),
  "linear"
)

which_functions <- function() as.character(lsf.str("package:ruta"))

which_args <- function(f) {
  # Gets formal arguments for the function
  defaults <- formals(f)
  defaults$... <- NULL

  checks <- lapply(defaults, function(arg) {
    # Retrieves the argument descriptor and calls it
    if (class(arg) == "call") {
      get(as.character(arg))(get = TRUE)
    } else {
      as_arg(arg)(get = TRUE)
    }
  })

  names(checks) <- names(defaults)
  # Return descriptions for each argument
  structure(checks, class = "ruta_args")
}

print.ruta_args <- function(checks) {
  cat("Usage: \n")
  for (arg in names(checks)) {
    type <- if (is.null(checks[[arg]]$class) || checks[[arg]]$class == "NULL")
      "unknown"
    else
      paste0(checks[[arg]]$class, collapse = " or ")

    values <- if (!is.null(checks[[arg]]$values))
      c("allowed values {\n    ", paste0(checks[[arg]]$values, collapse = "\n    "), "\n  }, ")
    else
      ""

    required <- if (checks[[arg]]$required) "" else "not "

    cat(
      "  ", arg, ": type ", type, ", ", values, required, "required\n",
      sep = ""
    )
  }
}

check_args_internal <- function() {
  # Which function was called and with that arguments?
  call_l <- as.list(sys.call(sys.parent(1)))
  # What are the argument descriptions for this function?
  checks <- which_args(as.character(call_l[[1]]))

  check_args(call_l[2:length(call_l)], checks)
}

check_args <- function(args, checks) {
  # Detect positional arguments
  unnamed_args <- if (is.null(names(args))) seq_along(args) else which(names(args) == "")
  # Detect missing named arguments
  missing_args <- setdiff(names(checks), names(args)[-unnamed_args])
  # Pair positional arguments with missing named arguments
  names(args)[unnamed_args] <- missing_args[1:length(unnamed_args)]

  for (name in names(args)) {
    check <- checks[[name]]

    if (!is.null(check)) {
      val <- eval(args[[name]])
      validate_arg(name, val, check)
    }
  }

  invisible(TRUE)
}

validate_arg <- function(name, val, check) {

  # Check type: mandatory argument
  if (check$required && is.null(val)) {
    stop(paste0(name, " is a required argument"), call. = F)
  }

  # Check type: class
  if (!is.null(check$class)) {
    if (length(intersect(class(val), check$class)) == 0) {
      stop(paste0(name, " does not have any allowed class (", paste(check$class, collapse = ", "), ")"), call. = F)
    }
  }

  # Check type: values
  if (!is.null(check$values)) {
    # Value check with set
    if (is.atomic(check$values)) {
      if (!(val %in% check$values)) {
        stop(paste0(name, " does not equal any allowed value (", paste(check$values, collapse = ", "), ")"), call. = F)
      }
    } else {
      # Value check in interval
      if (!is.null(check$values$min)) {
        if (val < check$values$min) {
          stop(paste0(name, " is lower than the minimum allowed value (", check$values$min, ")"), call. = F)
        }
      }
      if (!is.null(check$values$max)) {
        if (val > check$values$max) {
          stop(paste0(name, " is higher than the maximum allowed value (", check$values$max, ")"), call. = F)
        }
      }
    }
  }
}

test_function <- function(network = arg_network(), loss = arg_loss(), activation = arg_activation(), weight = 2e-4) {
  check_args_internal()
}

