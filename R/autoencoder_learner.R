makeAutoencoder <-
  function(id, hidden, activation = NULL, sparse = FALSE, sparseness.penalty = NULL, ...) {
    learner <- list(id = id,
                    parameters = list(
                      hidden = hidden,
                      activation = activation
                    ))
    class(learner) <- c(rutaLearner, rutaAutoencoder)

    if (sparse) {
      if (is.null(activation) || activation != "sigmoid") {
        warning("Sparse autoencoders are only available with 'sigmoid' activation. Sparseness penalty has been disabled")
        learner$parameters$sparsenessPenalty = 0
      } else {
        learner$parameters$sparsenessPenalty = if (is.null(sparseness.penalty)) 0.001 else sparseness.penalty
        class(learner) <- c(class(learner), rutaSparseAutoencoder)
      }
    }

    tryRequire("mxnet")

    ## Use MXnet's symbolic functionality to build the neural network
    nn <- mxnet::mx.symbol.Variable("data")

    ## TODO some checks on hidden (size of first and last layer, etc.)
    encoding <- 1 + floor(length(hidden) / 2)
    learner$encodingLayer <- encoding
    learner$layers = list()

    for (l in 1:length(hidden)) {
      name = paste0("aelayer", l)
      learner$layers[[l]] = list(
        weight = paste0(name, "_weight"),
        bias = paste0(name, "_bias"),
        layerout = paste0(name, "_output"),
        out = if (sparse)
          paste0(name, "kl_output")
        else if (!is.null(activation))
          paste0(name, "act_output")
        else
          paste0(name, "_output")
      )
      nn <- autoencoderAddLayer(nn, hidden[l], learner$parameters$activation, learner$parameters$sparsenessPenalty, name = name)
    }

    # Add output layer and output
    # lastLayer <- length(x$layers) + 1
    #   x$layers[[lastLayer]] <- list(
    #
    # )
    # nn <- autoencoderAddLayer(x$nn, dim(trainX)[1], x$parameters$activation, x$parameters$sparseness.penalty, name = paste0("aelayer", lastLayer))
    nn <- mxnet::mx.symbol.LinearRegressionOutput(data = nn)

    learner$nn <- nn

    learner
  }

autoencoderAddLayer <- function(nn, hidden, activation = NULL, sparseness.penalty = NULL, name) {
  nn <- mxnet::mx.symbol.FullyConnected(data = nn, num_hidden = hidden, name = name)
  if (!is.null(activation)) {
    nn <- mxnet::mx.symbol.Activation(data = nn, act.type = activation, name = paste0(name, "act"))
  }
  if (!is.null(sparseness.penalty)) {
    nn <- mxnet::mx.symbol.IdentityAttachKLSparseReg(data = nn, penalty = sparseness.penalty, name = paste0(name, "kl"))
  }
  nn
}

#' \code{print} method for autoencoders.
#' @param x \code{"rutaAutoencoder"} object
#' @param ... Ignored
#' @export
print.rutaAutoencoder <- function(x, ...) {
  cat(
    "# ruta Learner\n",
    "# Type: Autoencoder\n",
    "# Backend: ", x$backend, "\n",
    "# Sparse: ",
    (if (rutaSparseAutoencoder %in% class(x))
      "Yes"
     else
       "No"),
    "\n",
    sep = ""
  )
}
