autoencoder_complexity <- function() {

}

to_keras.autoencoder_complexity <- function(object) {

}

example_autoencoder_complexity <- function() {
  library(keras)
  library(purrr)
  library(colorspace)
  library(scatterplot3d)

  input_size <- 30
  latent_dim <- 3

  enc_input <- layer_input(list(input_size))
  encoding <- layer_dense(enc_input, units = latent_dim)
  # decodification layer object
  decode_layer <- layer_dense(units = input_size, activation = "sigmoid")
  # decodification layer of the autoencoder (attached to the encoder)
  decodification <- decode_layer(encoding)

  # class input accepts ones and zeros
  class_pos <- layer_input(list(1))
  # calculate negative instances
  class_neg <- 1 - class_pos

  # count positive and negative instances
  amount_pos <- k_sum(class_pos)
  amount_neg <- k_sum(class_neg)

  # keeps the value of the encoding or zero according to each instance's class
  encoding_if_pos <- class_pos * encoding
  encoding_if_neg <- class_neg * encoding

  # sum each feature over all instances in the batch
  mean_pos <- k_sum(encoding_if_pos, axis = 1) / amount_pos
  mean_neg <- k_sum(encoding_if_neg, axis = 1) / amount_neg

  # similarly calculate variance as E[X^2] - E[X]^2
  variance_pos <- k_sum(k_square(encoding_if_pos), axis = 1) / amount_pos - k_square(mean_pos)
  variance_neg <- k_sum(k_square(encoding_if_neg), axis = 1) / amount_neg - k_square(mean_neg)

  fisher_ratios <- k_square(mean_pos - mean_neg) * (1 / (variance_pos + variance_neg))
  fisher_gain <- k_max(fisher_ratios)

  rec_loss <- loss_binary_crossentropy(enc_input, decodification)
  regularized_loss <- (rec_loss %>% k_mean()) - fisher_gain

  encoder <- keras_model(enc_input, encoding)
  #decoder_input <- layer_input(latent_dim)
  #decoder <- keras_model(decoder_input, decode_layer(decoder_input))
  ae <- keras_model(list(enc_input, class_pos), decodification)

  ae$add_loss(regularized_loss)
  ae$compile(optimizer_rmsprop(lr = 0.01))

  # ae %>% compile("rmsprop", "mean_squared_error")

  wdbc <- read.csv("wdbc.data")
  x_train <- wdbc[, 3:32]
  mx <- apply(x_train, 2, max)
  mn <- apply(x_train, 2, min)
  range <- mx - mn
  x_train <- t(apply(x_train, 1, function(x) (x - mn) / range))
  y_train <- ifelse(wdbc[, 2] == "M", 1, 0)

  fit(
    ae,
    list(x_train, y_train),
    epochs = 500,
    batch_size = 32
  )
  scatterplot3d((encoder %>% predict(x_train))[, c(2,1,3)], color = rainbow_hcl(2)[y_train + 1], pch = 20)

  ae %>% compile("rmsprop", "mean_squared_error")
  evaluate(ae, list(x_train, y_train), y = x_train)

  #### normal autoencoder

  library(ruta)
  codes <- x_train %>% autoencode(3, epochs = 500)
  scatterplot3d(codes, color = rainbow_hcl(2)[y_train + 1], pch = 20)
}
