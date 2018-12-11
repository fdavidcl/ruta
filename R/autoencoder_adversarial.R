# Initial implementation following https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py#L46

autoencoder_adversarial <- function() {
  library(keras)
  library(purrr)

  input_size <- 784
  hidden_dim <- 256
  latent_dim <- 32

  dist_input <- layer_input(list(latent_dim))
  probability <- dist_input %>%
    layer_dense(units = hidden_dim) %>%
    layer_dense(units = 1, activation = "sigmoid")
  discriminator <- keras_model(dist_input, probability)
  compile(discriminator, optimizer_rmsprop(lr = 0.01), "binary_crossentropy")

  enc_input <- layer_input(list(input_size))
  encoding <- layer_dense(enc_input, units = latent_dim)
  decode_layer <- layer_dense(units = input_size, activation = "sigmoid")
  decodification <- decode_layer(encoding)
  validity <- discriminator(encoding)

  encoder <- keras_model(enc_input, encoding)
  decoder_input <- layer_input(latent_dim)
  decoder <- keras_model(decoder_input, decode_layer(decoder_input))
  ae <- keras_model(enc_input, list(decodification, validity))
  discriminator$trainable <- FALSE
  compile(
    ae,
    "rmsprop",
    list("binary_crossentropy", "binary_crossentropy"),
    loss_weights = list(0.995, 0.005)
  )

  object <- list(
    discriminator = discriminator,
    encoder = encoder,
    autoencoder = ae,
    decoder = decoder
  )

  train <- function(object, epochs = 10, batch_size = 128, seed = 4242) {
    latent_dim <- object$discriminator$input_shape[[2]]
    set.seed(seed)

    mnist <- dataset_mnist()
    x_train <- array_reshape(
      mnist$train$x, c(dim(mnist$train$x)[1], 784)
    )
    x_train <- x_train / 255.0
    # x_test <- array_reshape(
    #   mnist$test$x, c(dim(mnist$test$x)[1], 784)
    # )
    # x_test <- x_test / 255.0

    y_real <- rep(1, batch_size)
    y_fake <- rep(0, batch_size)

    for (epoch in 1:epochs) {
      batch_idx <- sample.int(dim(x_train)[1], size = batch_size)
      imgs <- x_train[batch_idx, ]

      latent_fake <- predict(object$encoder, imgs)
      latent_real <-
        matrix(rnorm(batch_size * latent_dim), nrow = batch_size)

      d_loss_real <- object$discriminator %>% train_on_batch(latent_real, y_real)
      d_loss_fake <- object$discriminator %>% train_on_batch(latent_fake, y_fake)
      d_loss <- 0.5 * (d_loss_fake + d_loss_real)

      ae_loss <- object$autoencoder %>% train_on_batch(imgs, list(imgs, y_real))

      if (epoch %% 100 == 0) {
        message(
          epoch,
          "/",
          epochs,
          " [D loss: ",
          d_loss,
          "] [AE loss: ",
          ae_loss[1],
          "]"
        )
      }
    }
  }

  train(object, epochs = 1000)


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
