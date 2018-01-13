#' Create an autoencoder learner
#'
#' @param layers
#' @param sparse
#' @param contractive
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
makeAutoencoder <- function(learner, input_shape) {
  loadKeras()

  input_shape = reticulate::tuple(as.integer(input_shape))
  encoding_dim = as.integer(32)
  layers = reticulate::import("keras.layers")
  models = reticulate::import("keras.models")

  input_img = layers$Input(shape = input_shape)
  encoded = layers$Dense(encoding_dim, activation = "tanh")(input_img)
  decoded = layers$Dense(input_shape[[0]], activation = "sigmoid")(encoded)

  model = models$Model(input_img, decoded)
  encoder = models$Model(input_img, encoded)

  encoded_input = layers$Input(shape = reticulate::tuple(encoding_dim))
  decoder_layer = model$layers[[3]]
  decoder = models$Model(encoded_input, decoder_layer(encoded_input))

  kerasR::plot_model(model)

  autoencoder_obj = list(
    learner = learner,
    model = model,
    encoder = encoder,
    decoder = decoder
  )
  class(autoencoder_obj) = rutaAutoencoder
  autoencoder_obj
}

#' @export
train <- function(learner, ...)
  UseMethod("train")

#' @import kerasR
#' @export
train.rutaLearner <- function(learner, data, validation_data = NULL, epochs = 100) {
  ae <- makeAutoencoder(learner, input_shape = 784)

  keras_compile(ae$model, optimizer = RMSprop(), loss = "binary_crossentropy")
  keras_fit(
    ae$model,
    x = data,
    y = data,
    batch_size = 256,
    epochs = epochs
  )

  ae
}

#' @import kerasR
#' @export
encode <- function(ae, data) {
  ae$encoder$predict(data)
}

