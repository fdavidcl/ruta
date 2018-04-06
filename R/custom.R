ruta_custom <- function(to_keras_f) {
  structure(
    to_keras_f,
    class = ruta_custom
  )
}

to_keras.ruta_custom <- function(x, ...) {
  x(...)
}
