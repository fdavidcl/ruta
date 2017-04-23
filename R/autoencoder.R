makeAutoencoder <-
  function(id, hidden, activation = NULL, sparse = FALSE, sparseness.penalty = NULL, ..., backend = "mxnet") {
    learner <- list(id = id,
                    backend = backend,
                    parameters = list(
                      hidden = hidden,
                      activation = activation
                    ))
    class(learner) <- c(rutaLearner, rutaAutoencoder)

    if (sparse) {
      learner$parameters$sparsenessPenalty = if (is.null(sparseness.penalty)) 0.001 else sparseness.penalty
      class(learner) <- c(class(learner), rutaSparseAutoencoder)
    }

    ## Use MXnet's symbolic functionality to build the neural network
    nn <- mxnet::mx.symbol.Variable("data")

    ## TODO some checks on hidden (size of first and last layer, etc.)
    innermost <- 1 + floor(length(hidden) / 2)
    learner$innermostLayer <- "innermost"

    for (l in 1:length(hidden)) {
      name = if(l != innermost) paste0("aelayer", l) else learner$innermostLayer
      nn <- autoencoderAddLayer(nn, hidden[l], learner$parameters$activation, learner$parameters$sparsenessPenalty, name = name)
    }

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

print.rutaAutoencoder <- function(x) {
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

#' @export
train <- function(x, ...)
  UseMethod("train")

#' @export
train.rutaAutoencoder <- function(x, task, ...) {
  if (x$backend == "mxnet") {
    trainAutoencoderMXnet(x, task, ...)
  } else if (x$backend == "h2o") {
    trainAutoencoderH2o(x, task, ...)
  } else {
    stop("Invalid backend selected")
  }
}

trainAutoencoderMXnet <-
  function(x,
           task,
           epochs,
           optimizer = "sgd",
           learning.rate = 0.01,
           momentum = 0.9,
           ...) {
    dataset <- task$data
    class <- task$cl

    tryRequire("mxnet")

    ## Remove class column if necessary, use a data structure supported
    ## by MXnet
    trainX <- taskToMxnet(task)

    # Add output layer and output
    nn <- autoencoderAddLayer(x$nn, dim(trainX)[1], x$parameters$activation, x$parameters$sparseness.penalty, name = "layerout")
    nn <- mxnet::mx.symbol.LinearRegressionOutput(data = nn)

    ## Create an optimizer
    optimizer <-
      mxnet::mx.opt.create(optimizer, learning.rate = learning.rate, ...)
    ## available optimizers:
    ## mx.opt.sgd
    ## mx.opt.rmsprop
    ## mx.opt.adam
    ## mx.opt.adagrad
    ## mx.opt.adadelta
    ## source: https://github.com/dmlc/mxnet/blob/master/R-package/R/optimizer.R

    ## Train the network
    mxmodel <-
      mxnet::mx.model.FeedForward.create(
        symbol = nn,
        X = trainX,
        y = trainX,
        num.round = epochs,
        momentum = momentum,
        eval.metric = mxnet::mx.metric.rmse,
        array.layout = "colmajor",
        optimizer = optimizer
      )

    model <-
      list(
        internal = mxmodel,
        backend = x$backend,
        parameters = list(
          epochs = epochs,
          optimizer = optimizer,
          learningRate = learning.rate,
          momentum = momentum
        ),
        learner = x
      )
  }

trainAutoencoderH2o <-
  function(x,
           dataset,
           class_col,
           layers,
           activation,
           epochs) {
    tryRequire("h2o")
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

predictInternal <- function(model, X, ctx=NULL, layer.prefix, array.batch.size=128, array.layout="auto") {
  # Copyright (c) 2017 by mxnet contributors, David Charte
  # All rights reserved.
  #
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions are met:
  #
  # * Redistributions of source code must retain the above copyright notice, this
  # list of conditions and the following disclaimer.
  #
  # * Redistributions in binary form must reproduce the above copyright notice,
  # this list of conditions and the following disclaimer in the documentation
  # and/or other materials provided with the distribution.
  #
  # * Neither the name of the copyright holder nor the names of its
  # contributors may be used to endorse or promote products derived from
  # this software without specific prior written permission.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  # initialization stuff I probably don't care about ---------------------------
  if (is.null(ctx)) ctx <- mxnet:::mx.ctx.default()
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mxnet:::mx.model.select.layout.predict(X, model)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  # end initialization ---------------------------------------------------------
  # iterator creation ----------------------------------------------------------
  ## X iterates through the batches of input data
  X <- mxnet:::mx.model.init.iter(X, NULL, batch.size=array.batch.size, is.train=FALSE)
  X$reset()
  if (!X$iter.next()) stop("Cannot predict on empty iterator")
  dlist = X$value()
  # end iterator creation ------------------------------------------------------
  # executor creation ----------------------------------------------------------
  ## mx.simple.bind defined in https://github.com/dmlc/mxnet/blob/master/R-package/R/executor.R#L5
  ## internally calls mx.symbol.bind, defined in https://github.com/dmlc/mxnet/blob/master/R-package/src/executor.cc#L191
  ### see also: https://github.com/dmlc/mxnet/blob/e7514fe1b3265aaf15870b124bb6ed0edd82fa76/R-package/demo/basic_executor.R
  internals <- model$symbol$get.internals()
  ## TODO layer.prefix only works for layer symbols (not for activations since we're using '_bias' later). Fix.
  layerIndex <- which(internals$outputs == paste0(layer.prefix, "_output"))
  print(layerIndex)
  pexec <- mxnet:::mx.simple.bind(internals[[layerIndex]], ctx=ctx, data=dim(dlist$data), grad.req="null")
  # end executor creation ------------------------------------------------------
  # set up arg arrays ----------------------------------------------------------
  argIndex <- which(names(model$arg.params) == paste0(layer.prefix, "_bias"))
  internalArgParams <- model$arg.params[1:argIndex]
  # print(names(internalArgParams))

  mxnet:::mx.exec.update.arg.arrays(pexec, internalArgParams, match.name=T)
  mxnet:::mx.exec.update.aux.arrays(pexec, model$aux.params, match.name=T)
  # end set up arg arrays ------------------------------------------------------
  # the rest is left untouched -------------------------------------------------
  packer <- mxnet:::mx.nd.arraypacker()
  X$reset()
  while (X$iter.next()) {
    dlist = X$value()
    mxnet:::mx.exec.update.arg.arrays(pexec, list(data=dlist$data), match.name=T)
    mxnet:::mx.exec.forward(pexec, is.train=FALSE)
    out.pred <- mxnet:::mx.nd.copyto(pexec$ref.outputs[[1]], mxnet:::mx.cpu())
    padded <- X$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
  }
  X$reset()
  return(packer$get())
}

#' @export
ruta.deepFeatures <- function(model, task, ...) {
  if (model$backend == "mxnet") {
    predX = taskToMxnet(task)
    predOut = predictInternal(model$internal, predX, layer.prefix = model$learner$innermostLayer, ...)
    t(predOut)
  } else if (x$backend == "h2o") {
    h2o::h2o.deepfeatures(x$model, dataset.h2o, layer = floor((length(layer) + 1) /
                                                                2))
  }
}
