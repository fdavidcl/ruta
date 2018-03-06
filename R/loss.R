as_loss <- function(x) UseMethod("as_loss")

as_loss.character <- function(x) {
  structure(
    list(
      name = x
    ),
    class = c(ruta_loss, ruta_named_loss)
  )
}

as_loss.ruta_loss <- function(x) x

to_keras.ruta_named_loss <- function(x, ...) {
  if (exists(paste0("loss_", x$name), where = asNamespace("keras"))) {
    get(paste0("loss_", x$name), envir = asNamespace("keras"))
  } else {
    stop("There is no ", x$name, " loss in keras.")
  }
}
