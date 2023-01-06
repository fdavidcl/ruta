#' **This example demonstrates a possible implementation of convolutional autoencoders with the Ruta package.**
#'
#' Convolutional layers are defined with `conv` indicating number of learned filters, size of the kernels and whether there are
#' max/average pooling or upsampling operations to be made.
library(magrittr)
library(keras)
library(ruta)

mnist <- dataset_mnist()
x_train <- mnist$train$x / 255.0
x_test <- mnist$test$x / 255.0
# convert to shape: (batch_size, 28, 28, 1)
x_train <- array_reshape(x_train, c(dim(x_train), 1))
x_test <- array_reshape(x_test, c(dim(x_test), 1))

net <- input() +
  conv(16, 3, activation = "relu", max_pooling = 2) +
  conv( 8, 3, activation = "relu", max_pooling = 2) +
  conv( 8, 3, activation = "relu", upsampling = 2) +
  conv(16, 3, activation = "relu", upsampling = 2) +
  conv(1, 3, activation = "sigmoid")
ae <- autoencoder(net, loss = "binary_crossentropy")

model <-
  ae %>% train(x_train, validation_data = x_test, epochs = 4)

evaluate_mean_squared_error(model, x_test)
enc <- model %>% encode(x_test)
decode <- model %>% reconstruct(x_test)
