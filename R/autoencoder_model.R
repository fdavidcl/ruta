#' Train an autoencoder.
#' @param x A \code{"rutaAutoencoder"} object. This is the learner that will be
#'   trained.
#' @param task A \code{"rutaUnsupervisedTask"} object. It contains the data that
#'   will be used to train the autoencoder. Class information, if present, will
#'   not be used.
#' @param ... Additional parameters for the MXNet optimizer
#' @return A \code{"rutaModel"} object containing the trained model.
#' @export
train.rutaAutoencoder <- function(x, task, ...) {
  trainAutoencoderMXnet(x, task, ...)
}

#' Pretrain autoencoders.
#'
#' Obtain initial weights for an autoencoder via stack of RBMs.
#'
#' @param x A \code{"rutaAutoencoder"} object.
#' @param task A \code{"rutaTask"} object.
#' @param epochs The number of epochs for the training process of each RBM.
#'
#' @import deepnet
#' @export
ruta.pretrain <- function(x, task, epochs = 10, ...) {
  ## Use stack of RBMs to initialize weights
  initialArguments <- list()
  n <- length(x$parameters$hidden)
  rbmTask <- task
  for (l in 1:x$encodingLayer) {
    print(rbmTask)
    #cat("Entrenando una RBM de ", length(rbmTask$data), " neuronas visibles y ", x$parameters$hidden[l + 1], " neuronas ocultas\n")
    rbm <- ruta.makeLearner("rbm", hidden = x$parameters$hidden[l + 1])
    rbmModel <- train(rbm, rbmTask, numepochs = epochs, ...)
    initialArguments[[x$layers[[l]]$weight]] = mxnet::mx.nd.array(ruta.getWeights(rbmModel, 2))
    initialArguments[[x$layers[[n - l]]$weight]] = mxnet::mx.nd.array(ruta.getWeights(rbmModel, 1))
    rbmTask <- ruta.makeTask("rbm", ruta.layerOutputs(rbmModel, rbmTask, layerInput = 1))
  }

  initialArguments
}

trainAutoencoderMXnet <-
  function(x,
           task,
           epochs,
           optimizer = "sgd",
           eval.metric = mxnet::mx.metric.rmse,
           initial.args = NULL,
           ...) {
    dataset <- task$data
    class <- task$cl

    ## Remove class column if necessary, use a data structure supported
    ## by MXnet
    trainX <- taskToMxnet(task)

    ## Create an optimizer
    optimizer <- mxnet::mx.opt.create(optimizer, ...)
    ## available optimizers and parameters:
    ## mx.opt.sgd - learning.rate, momentum, wd, rescale.grad, clip_gradient, lr_scheduler
    ## mx.opt.rmsprop - learning.rate, gamma1, gamma2, wd, rescale.grad, clip_gradient, lr_scheduler
    ## mx.opt.adam - learning.rate, beta1, beta2, epsilon, wd, rescale.grad, clip_gradient, lr_scheduler
    ## mx.opt.adagrad - learning.rate, epsilon, wd, rescale.grad, clip_gradient, lr_scheduler
    ## mx.opt.adadelta - rho, epsilon, wd, rescale.grad, clip_gradient
    ## source: https://github.com/dmlc/mxnet/blob/master/R-package/R/optimizer.R

    ## Train the network
    mxmodel <- if (!(rutaSparseAutoencoder %in% class(x)))
      mxnet::mx.model.FeedForward.create(
        symbol = x$nn,
        X = trainX,
        y = trainX,
        num.round = epochs,
        eval.metric = eval.metric,
        array.layout = "colmajor",
        optimizer = optimizer,
        arg.params = initial.args
      )
    else
      mxnet::mx.model.FeedForward.create(
        symbol = x$nn,
        X = trainX,
        y = trainX,
        num.round = epochs,
        eval.metric = eval.metric,
        array.layout = "colmajor",
        optimizer = optimizer,
        arg.params = initial.args#,
        #aux.params = list(aelayer1kl_moving_avg = 0.1, aelayer2kl_moving_avg = 0.1, aelayer3kl_moving_avg = 0.1)
      )

    model <-
      list(
        internal = mxmodel,
        backend = x$backend,
        parameters = list(
          epochs = epochs,
          optimizer = optimizer#,
          #learningRate = learning.rate,
          #momentum = momentum
        ),
        learner = x
      )
    class(model) <- c(rutaModel, rutaAutoencoderModel)
    model
  }

predictPartial <- function(model, X, ctx = NULL, output.layer, arg.limit, array.batch.size = 128, array.layout="auto") {
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
  if (is.null(ctx)) ctx <- mxnet::mx.ctx.default()
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
  ## Select the output layer from mxnet internals
  layerIndex <- which(internals$outputs == output.layer)
  cat(paste0("Extracting layer ", output.layer, " (output #", layerIndex, ")\n"))
  ## We *could* select all the layers and create a group symbol, thus obtaining all the outputs
  ## but it doesn't look necessary
  pexec <- mxnet::mx.simple.bind(internals[[layerIndex]], ctx=ctx, data=dim(dlist$data), grad.req="null")
  # end executor creation ------------------------------------------------------
  # set up arg arrays ----------------------------------------------------------
  argIndex <- which(names(model$arg.params) == arg.limit)
  cat(paste0("Setting arguments up to ", arg.limit, " (arg #", argIndex, ")\n"))

  internalArgParams <- model$arg.params[1:argIndex]
  # print(names(internalArgParams))

  mxnet::mx.exec.update.arg.arrays(pexec, internalArgParams, match.name=T)

  ## leave aux params untouched since we're still not using them
  mxnet::mx.exec.update.aux.arrays(pexec, model$aux.params, match.name=T)
  # end set up arg arrays ------------------------------------------------------
  # the rest is left untouched -------------------------------------------------
  packer <- mxnet:::mx.nd.arraypacker()
  X$reset()
  while (X$iter.next()) {
    dlist = X$value()
    mxnet::mx.exec.update.arg.arrays(pexec, list(data=dlist$data), match.name=T)
    mxnet::mx.exec.forward(pexec, is.train=FALSE)
    out.pred <- mxnet::mx.nd.copyto(pexec$ref.outputs[[1]], mxnet::mx.cpu())
    padded <- X$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
  }
  X$reset()
  return(packer$get())
}

predictInternal <- function(rutaModel, X, ctx = NULL, layer, array.batch.size = 128, array.layout="auto") {
  predictPartial(
    rutaModel$internal, X, ctx,
    output.layer = rutaModel$learner$layers[[layer]]$out,
    arg.limit = rutaModel$learner$layers[[layer]]$bias,
    array.batch.size,
    array.layout
  )
}

#' Predict outputs for trained models and new data.
#'
#' You should pass a \code{model} argument and a \code{task} argument at least.
#' The task can be the same input data the model was trained with, or new data
#' with the same shape.
#' Other arguments will be passed to the internal prediction function.
#'
#' @param object A \code{"rutaModel"} object.
#' @param ... Custom parameters for MXNet prediction function.
#' @return A matrix containing predictions for each instance in the given task.
#' @importFrom stats predict
#' @export
predict.rutaModel <- function(object, ...) {
  args <- list(...)
  task <- args$task
  args$task = NULL
  predX <- taskToMxnet(task)
  predOut <- predict(object$internal, predX, ... = args)
  t(predOut)
}

#' Get outputs from any layer from a trained model and new data.
#'
#' @param model A \code{"rutaModel"} object from an autoencoder learner.
#' @param task A \code{"rutaTask"} object.
#' @param layerInput Currently input is always injected into the first layer.
#' @param layerOutput An integer indicating the index of the layer to be
#'   obtained.
#' @param ... Custom parameters for internal prediction function.
#' @return A matrix containing layer outputs for each instance in the given task.
#' @export
ruta.layerOutputs.rutaAutoencoderModel <- function(model, task, layerInput = 1, layerOutput, ...) {
  predX <- taskToMxnet(task)
  predOut <- predictInternal(model, predX, layer = layerOutput, ...)
  t(predOut)
}

#' Get the deep features from a trained autoencoder model.
#'
#' @param model A \code{"rutaModel"} object from an autoencoder learner.
#' @param task A \code{"rutaTask"} object.
#' @param ... Custom parameters for internal prediction function.
#' @return A matrix containing the encoding output for each instance in the
#'   given task.
#' @export
ruta.deepFeatures <- function(model, task, ...) {
  ruta.layerOutputs(model, task, 1, model$learner$encodingLayer, ...)
}
