makeAutoencoder <-
  function(id, backend = "mxnet", ...) {
    learner <- list(id = id,
                    backend = backend)
    class(learner) <- c(rutaLearner, rutaAutoencoder)
    learner
  }


print.rutaAutoencoder <- function(x) {
  cat("# ruta Learner\n",
      "# Type: Autoencoder\n",
      "# Backend: ", x$backend, "\n",
      sep = "")
}

#' @export
train <- function(x, ...)
  UseMethod("train")

#' @export
train.rutaAutoencoder <- function(x, task, ...) {
  if (x$backend == "mxnet") {
    trainAutoencoderMXnet(x, ...)
  } else if (x$backend == "h2o") {
    trainAutoencoderH2o(x, ...)
  } else {
    stop("Invalid backend selected")
  }
}

trainAutoencoderMXnet <-
  function(x,
           dataset,
           class,
           layers,
           activation = NULL,
           epochs,
           optimizer,
           learning_rate,
           momentum,
           ...) {
    ruta.util.require("mxnet")
    train.x = t(data.matrix(dataset[-class]))

    nn <- mxnet::mx.symbol.Variable("data")

    for (l in layers) {
      nn <- mxnet::mx.symbol.FullyConnected(nn, num_hidden = l)
      if (!is.null(activation)) {
        nn <- mxnet::mx.symbol.Activation(nn, act.type = activation)
      }
    }

    nn <- mxnet::mx.symbol.LinearRegressionOutput(data = nn)

    optimizer <-
      mxnet::mx.opt.create(optimizer, learning.rate = learning_rate, ...)
    ## other optimizers:
    ## mx.opt.rmsprop
    ## mx.opt.adam
    ## mx.opt.adagrad
    ## mx.opt.adadelta
    ## source: https://github.com/dmlc/mxnet/blob/master/R-package/R/optimizer.R

    ae <-
      mxnet::mx.model.FeedForward.create(
        nn,
        X = train.x,
        y = train.x,
        num.round = epochs,
        momentum = momentum,
        eval.metric = mxnet::mx.metric.rmse,
        array.layout = "colmajor",
        optimizer = optimizer
      )


    ## para el sparse: mx.symbol.IdentityAttachKLSparseReg
  }

trainAutoencoderH2o <-
  function(x,
           dataset,
           class_col,
           layers,
           activation,
           epochs) {
    ruta.util.require("h2o")
    dataset.h2o <- h2o::as.h2o(dataset)
    inputs <- setdiff(1:ncol(dataset), class_col)
    dataset.h2o <- dataset.h2o[-class_col]

    ae_model <- h2o::h2o.deeplearning(
      x = inputs,
      training_frame = dataset.h2o,
      activation = activation,
      autoencoder = T,
      hidden = layers,
      epochs = epoch_num,
      ignore_const_cols = F,
      max_w2 = 10,
      l1 = 1e-5
    )

    model <- list(
      model   = ae_model,
      inputs  = dataset[inputs],
      classes = dataset[class_col],
      layers  = layer,
      backend = backend
    )
    class(model) <- ruta_model
    model
  }

#' @export
ruta.deep_features <- function(x, dataset) {
  if (x$backend == "mxnet") {

  } else if (x$backend == "h2o") {
    h2o::h2o.deepfeatures(x$model, dataset.h2o, layer = floor((length(layer) + 1) /
                                                                2))
  }
}
