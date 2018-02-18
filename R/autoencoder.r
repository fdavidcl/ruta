#' Create an autoencoder learner
#'
#' Internal function to create autoencoder objects. Instead, consider using
#' \code{\link{autoencoder}}.
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying a loss function
#' @return A construct of class \code{"ruta_autoencoder"}
#' @export
new_autoencoder <- function(network, loss) {
  structure(
    list(
      network = network,
      loss = loss
    ),
    class = c(ruta_learner, ruta_autoencoder)
  )
}

#' Create an autoencoder learner
#'
#' Represents a generic autoencoder network.
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying a loss function
#' @return A construct of class \code{"ruta_autoencoder"}
#' @seealso \code{\link{train.ruta_autoencoder}}
#' @export
autoencoder <- function(network, loss = "mse") {
  new_autoencoder(network, loss)
}

is_trained <- function(learner) {
  !is_null(learner$models)
}

#' Extract Keras models from an autoencoder wrapper
#'
#' @param learner Object of class \code{"ruta_autoencoder"}
#' @param input_shape Number of attributes in input data
#' @return A list with several Keras models: \itemize{
#' \item \code{autoencoder}: model from the input layer to the output layer
#' \item \code{encoder}: model from the input layer to the encoding layer
#' \item \code{decoder}: model from the encoding layer to the output layer
#' }
#' @import purrr
#' @export
to_keras.ruta_autoencoder <- function(learner, input_shape) {
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

  list(
    autoencoder = model,
    encoder = encoder,
    decoder = decoder
  )
}

#' @rdname train.ruta_autoencoder
#' @export
train <- function(learner, ...) {
  UseMethod("train")
}

#' Train a learner object with data
#'
#' This function compiles the neural network described by the learner object
#' and trains it with the input data.
#' @param learner A \code{"ruta_autoencoder"} object
#' @param data Training data: columns are attributes and rows are instances
#' @param validation_data Additional data.frame of data which will not be used
#' for training but the loss measure will be calculated against it
#' @param epochs The number of times data will pass through the network
#' @param ... Additional parameters for \code{keras::fit}
#' @return Same autoencoder passed as parameter, with trained internal models
#' @export
train.ruta_autoencoder <- function(learner, data, validation_data = NULL, epochs = 100, ...) {
  learner$models <- to_keras(learner, input_shape = ncol(data))

  keras::compile(
    learner$models$autoencoder,
    optimizer = keras::optimizer_rmsprop(),
    loss = keras::loss_binary_crossentropy
  )
  keras::fit(
    learner$models$autoencoder,
    x = data,
    y = data,
    batch_size = 256,
    epochs = epochs,
    ...
  )

  learner
}

#' Retrieve encoding of data
#'
#' Extracts the encoding calculated by a trained autoencoder for the specified
#' data.
#' @param learner Trained autoencoder model
#' @param data data.frame to be encoded
#' @return Matrix containing the encodings
#' @seealso \code{\link{decode}}
#' @export
encode <- function(learner, data) {
  if (!is_trained(learner)) {
    stop("Autoencoder is not trained")
  }

  learner$models$encoder$predict(data)
}

#' Retrieve decoding of encoded data
#'
#' Extracts the decodification calculated by a trained autoencoder for the specified
#' data.
#' @param learner Trained autoencoder model
#' @param data data.frame to be encoded
#' @return Matrix containing the decodifications
#' @seealso \code{\link{encode}}
#' @export
decode <- function(learner, data) {
  if (!is_trained(learner)) {
    stop("Autoencoder is not trained")
  }

  learner$models$decoder$predict(data)
}
