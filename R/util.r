loadKeras <- function() {
  keras::is_keras_available()
}

toKeras <- function(x, ...) UseMethod("toKeras")
