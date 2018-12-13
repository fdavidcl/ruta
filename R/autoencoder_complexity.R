autoencoder_complexity <- function() {
  library(keras)
  library(purrr)

  input_size <- 30
  latent_dim <- 3

  enc_input <- layer_input(list(input_size))
  encoding <- layer_dense(enc_input, units = latent_dim)
  decode_layer <- layer_dense(units = input_size, activation = "sigmoid")
  decodification <- decode_layer(encoding)

  encoder <- keras_model(enc_input, encoding)
  decoder_input <- layer_input(latent_dim)
  decoder <- keras_model(decoder_input, decode_layer(decoder_input))

  # input accepts ones and zeros
  class_pos <- layer_input(list(1))
  class_neg <- 1 - class_pos

  # keeps the value of the encoding or zero according to each instance's class
  encoding_if_pos <- class_pos * encoding
  encoding_if_neg <- class_neg * encoding

  # count positive and negative instances
  amount_pos <- k_sum(class_pos)
  amount_neg <- k_sum(class_neg)

  # sum each feature over all instances in the batch
  mean_pos <- k_sum(encoding_if_pos, axis = 1) #/ amount_pos
  mean_neg <- k_sum(encoding_if_neg, axis = 1) #/ amount_neg

  # similarly calculate variance as E[X^2] - E[X]^2
  variance_pos <- k_sum(k_square(encoding_if_pos), axis = 1) #/ amount_pos - k_square(mean_pos)
  variance_neg <- k_sum(k_square(encoding_if_neg), axis = 1) #/ amount_neg - k_square(mean_neg)

  fisher_ratios <- k_square(mean_pos - mean_neg) #* k_pow(variance_pos + variance_neg, as.integer(-1))
  fisher_loss <- k_max(fisher_ratios)

  rec_loss <- loss_mean_squared_error(enc_input, decodification)
  regularized_loss <- 0.9 * rec_loss - 0.1 * fisher_loss

  ae <- keras_model(list(enc_input, class_pos), decodification)

  ae$add_loss(regularized_loss)

  # ae$compile("rmsprop")

  ae %>% compile("rmsprop", "mean_squared_error")

  object <- list(
    encoder = encoder,
    autoencoder = ae,
    decoder = decoder
  )

  train <- function(object, epochs = 10, batch_size = 32, seed = 4242) {
    latent_dim <- object$discriminator$input_shape[[2]]
    set.seed(seed)

    wdbc <- read.csv("wdbc.data")
    x_train <- as.matrix(wdbc[, 3:32])
    y_train <- array_reshape(ifelse(wdbc[, 2] == "M", 1, 0), list(dim(x_train)[1], 1))

    fit(object$autoencoder, list(x_train, y_train), y = x_train, epochs = epochs, batch_size = batch_size)
  }

  train(object, epochs = 10)


  mnist <- dataset_mnist()
  x_test <- array_reshape(
    mnist$test$x, c(dim(mnist$test$x)[1], 784)
  )
  x_test <- x_test / 255.0

  encodings <- object$encoder %>% predict(x_test)
  to_sample <- t(sapply(seq(0, 1, 0.1), function(c) encodings[1,] * c + encodings[3,] * (1 - c)))
  samples <- object$decoder %>% predict(to_sample)
  image(array_reshape(samples[4,], c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col = gray((255:0)/255))
}
