makeRBM <-
  function(id, hidden, activation = "sigm") {
    learner <- list(id = id,
                    parameters = list(
                      hidden = hidden,
                      activation = activation,
                      backend = "deepnet"
                    ))
    class(learner) <- c(rutaLearner, rutaRBM)

    learner
  }

#' \code{print} method for RBMs.
#' @param x \code{"rutaRBM"} object
#' @param ... Ignored
#' @export
print.rutaRBM <- function(x, ...) {
  cat(
    "# ruta Learner\n",
    "# Type: RBM\n",
    "# Backend: ", x$parameters$backend, "\n",
    "# Activation: ", x$parameters$activation, "\n",
    sep = ""
  )
}
