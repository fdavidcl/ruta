#' Apply filters
#'
#' @description Apply a filter to input data, generally a noise filter in
#' order to train a denoising autoencoder. Users won't generally need to use
#' these functions
#'
#' @param filter Filter object to be applied
#' @param data Input data to be filtered
#' @param ... Other parameters
#'
#' @rdname filters
apply_filter <- function(filter, data, ...) UseMethod("apply_filter")

#' @rdname filters
#' @export
apply_filter.ruta_custom <- function(filter, data, ...) {
  filter(data, ...)
}
