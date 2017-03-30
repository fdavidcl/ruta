#' @export
ruta.makeLearner <- function(cl, id = cl, ...) {
  if (cl == "autoencoder") {
    makeAutoencoder(id, ...)
  } else {
    stop(paste0("No corresponding function found for ", type, " learner type"))
  }
}

# ruta.train <- function(learner, task, subset) {
#   if (class(learner) == "character")
#     learner <- ruta.makeLearner(learner)
#
#   if (!("ruta.learner" %in% class(learner)))
#     stop("'learner' parameter is not of class 'ruta.learner'")
#
#
# }
