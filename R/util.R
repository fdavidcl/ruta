load_keras <- function() {
  keras::is_keras_available()
}

get_keras_object <- function(name, prefix = "", quiet = FALSE) {
  f_name <- if (length(prefix) > 0) paste0(prefix, "_", name) else name
  if (exists(f_name, where = asNamespace("keras"))) {
    get(f_name, envir = asNamespace("keras"))
  } else if (quiet) {
    FALSE
  } else {
    stop("There is no ", f_name, " function exported from keras.")
  }
}

print_line <- function(length = 40) {
  line <- paste0(rep("─", length), collapse = "")
  cat(line, "\n", sep = "")
}
