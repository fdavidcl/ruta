#' Create an autoencoder learner
#'
#' @param layers Layer construct of class \code{"rutaNetwork"}
#' @param loss Character string specifying a loss function
#'
#' @export
autoencoder <- function(layers, loss = "mse") {
  structure(
    list(
      layers = layers,
      loss = loss,
      regularizers = list()
    ),
    class = ruta_learner
  )
}

#' @import purrr
make_autoencoder <- function(learner, input_shape) {
  load_keras()

  network <- to_keras(learner$layers, input_shape)

  input_layer <- network[[1]]
  model <- keras::keras_model(input_layer, network[[length(network)]])

  encoder <- keras::keras_model(input_layer, network[[network %@% "encoding"]])

  encoding_dim <- learner$layers[[network %@% "encoding"]]$units
  encoded_input <- keras::layer_input(shape = encoding_dim)
  decoder_stack <- encoded_input

  for (lay_i in (network %@% "encoding" + 1):(length(network))) {
    decoder_stack <- model$layers[[lay_i]](decoder_stack)
  }

  decoder <- keras::keras_model(encoded_input, decoder_stack)

  structure(
    list(
      learner = learner,
      model = model,
      encoder = encoder,
      decoder = decoder
    ),
    class = ruta_autoencoder,
    trained = FALSE
  )
}

#' @rdname train.ruta_learner
#' @export
train <- function(learner, ...)
  UseMethod("train")

#' Train a learner object with data
#'
#' This function compiles the neural network described by the learner object
#' and trains it with the input data.
#' @param learner A \code{"rutaLearner"} object
#' @param data Training data: columns are attributes and rows are instances
#' @param validation_data Additional data.frame of data which will not be used
#' for training but the loss measure will be calculated against it
#' @param epochs The number of times data will pass through the network
#' @param ... Additional parameters for \code{keras::fit}
#' @export
train.ruta_learner <- function(learner, data, validation_data = NULL, epochs = 100, ...) {
  ae <- make_autoencoder(learner, input_shape = ncol(data))

  keras::compile(
    ae$model,
    optimizer = keras::optimizer_rmsprop(),
    loss = keras::loss_binary_crossentropy
  )
  keras::fit(
    ae$model,
    x = data,
    y = data,
    batch_size = 256,
    epochs = epochs,
    ...
  )

  attr(ae, "trained") <- TRUE
  ae
}

#' Retrieve encoding of data
#'
#' Extracts the encoding calculated by a trained autoencoder for the specified
#' data.
#' @param ae Autoencoder model
#' @param data data.frame to be encoded
#' @export
encode <- function(ae, data) {
  ae$encoder$predict(data)
}

#' Retrieve decoding of encoded data
#'
#' Extracts the decodification calculated by a trained autoencoder for the specified
#' data.
#' @param ae Autoencoder model
#' @param data data.frame to be encoded
#' @export
decode <- function(ae, data) {
  ae$decoder$predict(data)
}
