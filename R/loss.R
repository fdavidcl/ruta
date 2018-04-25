as_loss.character <- function(x) {
  structure(
    list(
      name = x
    ),
    class = c(ruta_loss_named, ruta_loss)
  )
}

as_loss.ruta_loss <- function(x) x

to_keras.ruta_loss_named <- function(x, ...) {
  get_keras_object(x$name, "loss")
}

