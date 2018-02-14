sparsity <- function(expected_value, weight) {
  structure(
    list(
      expected_value = expected_value,
      weight = weight
    ),
    class = c(rutaRegularizer, rutaSparsity)
  )
}

#' @export
makeSparse <- function(learner, expected_value, weight = 0.2) {
  # TODO warn when sparsity expected value and encoding activation function
  # don't match (e.g. -0.8 for sigmoid or relu)
  learner$regularizers$sparsity <- sparsity(expected_value, weight)

  learner
}

toKeras.rutaSparsity <- function(x, ...) {
  regularizers = reticulate::import("keras.regularizers")
  K = reticulate::import("keras.backend")

}
