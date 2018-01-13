#' @import kerasR
loadKeras <- function() {
  #package = "kerasR"
  #if (!requireNamespace(package, quietly = TRUE)) {
  #  stop(paste0("ruta: Package '", package, "' is not installed and is needed for autoencoder functionality"))
  #}
  keras_init()
  keras_available()
}
