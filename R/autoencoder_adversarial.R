autoencoder_adversarial <- function() {
  library(keras)

  mnist <- dataset_mnist()
  x_train <- array_reshape(
    mnist$train$x, c(dim(mnist$train$x)[1], 784)
  )
  x_train <- x_train / 255.0
  x_test <- array_reshape(
    mnist$test$x, c(dim(mnist$test$x)[1], 784)
  )
  x_test <- x_test / 255.0


  enc_input <- layer_input(list(784))
  encoder <- layer_dense(enc_input, units = 8)
  decoder <- layer_dense(encoder, units = 784)

  dist_input <-
    list(encoder, encoder) %>%
    layer_subtract() %>%
    layer_gaussian_noise(stddev = 1)
  discriminator <- layer_dense(units = 1, activation = "sigmoid")
  disc_fake <- discriminator(encoder)
  disc_real <- discriminator(dist_input)

  ae_loss <-
    loss_binary_crossentropy(enc_input, decoder)
  disc_loss <- loss_binary_crossentropy(disc_fake * 0, disc_fake) +
    loss_binary_crossentropy(disc_real * 0 + 1, disc_real)

  ae <- keras_model(enc_input, decoder)
  disc1 <- keras_model(enc_input, disc_fake)
  disc2 <- keras_model(enc_input, disc_real)

  ae$add_loss(ae_loss)
  disc1$add_loss(disc_loss)
  disc2$add_loss(disc_loss)

  ae$compile(optimizer = "rmsprop")
  ae$trainable <- FALSE
  disc1$compile(optimizer = optimizer_rmsprop(lr = 0.1))
  disc2$compile(optimizer = optimizer_rmsprop(lr = 0.1))
  for (epoch in 1:10) {
    cat("Epoch ", epoch, ":\n", sep = "")
    ae %>% fit(x_train, initial_epoch = epoch - 1, epochs = epoch, steps_per_epoch = 6)
    disc1 %>% fit(x_train, initial_epoch = epoch - 1, epochs = epoch, steps_per_epoch = 6)
    disc2 %>% fit(x_train, initial_epoch = epoch - 1, epochs = epoch, steps_per_epoch = 6)
  }
}
