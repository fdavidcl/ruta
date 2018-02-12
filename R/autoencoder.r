#' Create an autoencoder learner
#'
#' @param layers Layer construct of class \code{"rutaNetwork"}
#' @param sparse Logical
#' @param contractive Logical
#'
#' @export
autoencoder <- function(layers, sparse = F, contractive = F) {
  learner = list(
    layers = layers,
    sparse = sparse,
    contractive = contractive
  )
  class(learner) = rutaLearner
  learner
}

#' @import kerasR
#' @importFrom reticulate tuple
#' @importFrom reticulate import
#' @import purrr
makeAutoencoder <- function(learner, input_shape) {
  loadKeras()

  keras_layers = import("keras.layers")
  keras_models = import("keras.models")

  network = toKeras(learner$layers, input_shape)

  input_layer = network[[1]]
  model = keras_models$Model(input_layer, network[[length(network)]])

  encoder = keras_models$Model(input_layer, network[[network %@% "encoding"]])

  encoding_dim = learner$layers[[network %@% "encoding"]]$units
  encoded_input = keras_layers$Input(shape = tuple(encoding_dim))
  decoder_stack = encoded_input

  for (lay_i in (network %@% "encoding" + 1):(length(network))) {
    decoder_stack = model$layers[[lay_i]](decoder_stack)
  }

  decoder = keras_models$Model(encoded_input, decoder_stack)

  structure(list(
    learner = learner,
    model = model,
    encoder = encoder,
    decoder = decoder),
    class = rutaAutoencoder
  )
}

#' @rdname train.rutaLearner
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
#' @param ... Additional parameters for \code{keras_fit}
#' @importFrom kerasR keras_compile
#' @importFrom kerasR keras_fit
#' @export
train.rutaLearner <- function(learner, data, validation_data = NULL, epochs = 100, ...) {
  ae <- makeAutoencoder(learner, input_shape = ncol(data))

  keras_compile(ae$model, optimizer = RMSprop(), loss = "binary_crossentropy")
  keras_fit(
    ae$model,
    x = data,
    y = data,
    batch_size = 256,
    epochs = epochs,
    ...
  )

  ae
}

#' Retrieve encoding of data
#'
#' Extracts the encoding calculated by a trained autoencoder for the specified
#' data.
#' @param model Autoencoder model
#' @param data data.frame to be encoded
#' @export
encode <- function(ae, data) {
  ae$encoder$predict(data)
}

#' Retrieve decoding of encoded data
#'
#' Extracts the decodification calculated by a trained autoencoder for the specified
#' data.
#' @param model Autoencoder model
#' @param data data.frame to be encoded
#' @export
decode <- function(ae, data) {
  ae$decoder$predict(data)
}
