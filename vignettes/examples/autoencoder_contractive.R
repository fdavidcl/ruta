#' **This example demonstrates the use of contractive autoencoders with the Ruta package.**
#'
#' Define a contractive autoencoder with 36-variable encoding.
library(magrittr)
library(keras)
library(ruta)

network <- input() + dense(36, "elu") + output("sigmoid")
learner <-
  network %>%
  autoencoder_contractive(
    loss = "binary_crossentropy",
    weight = 2e-4
  )

#' Load MNIST and normalize
mnist <- dataset_mnist()
x_train <- array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train <- x_train / 255.0
x_test <- array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0

#' Train
model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 64
)

#' Generate reconstructions from test data
decoded <- model %>% reconstruct(x_test)

#' Utility functions for plotting
plot_digit <- function(digit, ...) {
  image(array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray((255:0)/255), ...)
}

plot_sample <- function(digits_test, digits_dec, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:sample_size, (sample_size + 1):(2 * sample_size)), byrow = F, nrow = 2)
  )

  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_digit(digits_test[i, ])
    plot_digit(digits_dec[i, ])
  }
}

#' Plot reconstructions
plot_sample(x_test, decoded, 1:10)
