#' Create an autoencoder learner
#'
#' Internal function to create autoencoder objects. Instead, consider using
#' \code{\link{autoencoder}}.
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying a loss function
#' @return A construct of class \code{"ruta_autoencoder"}
#' @export
new_autoencoder <- function(network, loss, extra_class = NULL) {
  structure(
    list(
      network = as_network(network),
      loss = loss
    ),
    class = c(extra_class, ruta_autoencoder, ruta_learner)
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
autoencoder <- function(network, loss = "mean_squared_error") {
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
  len <- length(learner$network)
  model <- to_keras(learner$network, input_shape)

  input_layer <- keras::get_layer(model, index = 0)
  encoding_layer <- keras::get_layer(model, index = learner$network %@% "encoding" - 1)
  #output_layer <- model %>% keras::get_layer(index = len - 1)

  #model <- keras::keras_model(input_layer, output_layer)
  encoder <- keras::keras_model(input_layer$output, encoding_layer$output)

  encoding_dim <- learner$network[[learner$network %@% "encoding"]]$units
  encoded_input <- keras::layer_input(shape = encoding_dim)
  decoder_stack <- encoded_input

  for (lay_i in (learner$network %@% "encoding"):(len - 1)) {
    decoder_stack <- keras::get_layer(model, index = lay_i)(decoder_stack)
  }

  decoder <- keras::keras_model(encoded_input, decoder_stack)

  list(
    autoencoder = model,
    encoder = encoder,
    decoder = decoder
  )
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

  loss_f <- learner$loss %>% as_loss() %>% to_keras(learner$models$autoencoder)

  keras::compile(
    learner$models$autoencoder,
    optimizer = keras::optimizer_rmsprop(),
    loss = loss_f
  )

  input_data <- if (is.null(learner$filter)) {
    data
  } else {
    apply_filter(learner$filter, data)
  }

  keras::fit(
    learner$models$autoencoder,
    x = input_data,
    y = data,
    batch_size = 256,
    epochs = epochs,
    ...
  )

  learner
}

#' Automatically compute an encoding of a data matrix
#'
#' Trains an autoencoder adapted to the data and extracts its encoding for the
#'   same data matrix.
#' @param data Numeric matrix to be encoded
#' @param encoding_dim Number of variables to be used in the encoding
#' @return Matrix containing the encodings
#' @export
autoencode <- function(data, encoding_dim) {
  autoencoder(input() + dense(encoding_dim) + output()) %>%
    train(data) %>%
    encode(data)
}

#' Retrieve encoding of data
#'
#' Extracts the encoding calculated by a trained autoencoder for the specified
#' data.
#' @param learner Trained autoencoder model
#' @param data data.frame to be encoded
#' @return Matrix containing the encodings
#' @seealso \code{\link{decode}}, \code{\link{reconstruct}}
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
#' @param data data.frame to be decoded
#' @return Matrix containing the decodifications
#' @seealso \code{\link{encode}}, \code{\link{reconstruct}}
#' @export
decode <- function(learner, data) {
  if (!is_trained(learner)) {
    stop("Autoencoder is not trained")
  }

  learner$models$decoder$predict(data)
}


#' Retrieve reconstructions for input data
#'
#' Extracts the reconstructions calculated by a trained autoencoder for the specified
#' input data after encoding and decoding.
#' @param learner Trained autoencoder model
#' @param object Trained autoencoder model
#' @param data data.frame to be passed through the network
#' @return Matrix containing the reconstructions
#' @seealso \code{\link{encode}}, \code{\link{decode}}
#' @export
reconstruct <- function(learner, data) {
  if (!is_trained(learner)) {
    stop("Autoencoder is not trained")
  }

  learner$models$autoencoder$predict(data)
}

#' @rdname reconstruct
#' @export
predict.ruta_autoencoder <- function(object, ...) reconstruct(object, ...)
