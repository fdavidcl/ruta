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

list_keras_objects <- function(prefix, rm_prefix = TRUE) {
  found <- ls(pattern = paste0("^", prefix), envir = asNamespace("keras"))
  if (rm_prefix) {
    gsub(paste0("^", prefix, "_"), "", found)
  } else {
    found
  }
}

print_line <- function(length = 40) {
  line <- paste0(rep("-", length), collapse = "")
  cat(line, "\n", sep = "")
}
