ARGS <- list()

def <- function(., ...) {
  arg_checks <- list(...)

  checked_f <- function() {
    # Do something with args


  }

  f_args <- vector("list", length(arg_checks))
  names(f_args) <- names(arg_checks)
  formals(checked_f) <- f_args
  body(checked_f) <- body(.)
  checked_f
}

example_argcheck <- def(
  number = arg_numeric(),
  network = arg_network(),
  # string = arg_string(),
  loss = arg_loss(),
  function() {
    print(number)
  }
)

arg_numeric <- arg(numeric = NULL)

check_args <- function(formal_args, arguments) {
  checks <- ARGS[]
  validate_call(as.list(arguments), checks)
}
