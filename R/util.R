load_keras <- function() {
  keras::is_keras_available()
}

to_keras <- function(x, ...) UseMethod("to_keras")

