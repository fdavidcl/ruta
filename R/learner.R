#' Create a representation for a learning algorithm.
#'
#' @param cl A string. Type of learner.
#' @param id A string. A name for the learner.
#' @param ... Additional parameters for the learner.
#' @return An object for the learner containing the provided parameters.
#' @examples
#' \dontrun{
#' ruta.makeLearner("autoencoder", "ae1", hidden = c(4, 2, 4), activation = "relu")
#' }
#' @export
ruta.makeLearner <- function(cl, id = cl, ...) {
  if (cl == "autoencoder") {
    makeAutoencoder(id, ...)
  } else {
    stop(paste0("No corresponding function found for ", cl, " learner type"))
  }
}
