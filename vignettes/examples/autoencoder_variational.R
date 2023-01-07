#' **This example demonstrates the use of variational autoencoders with the Ruta package.**
#'
#' Define a variational autoencoder with 3-variable latent space.
#' The encoding of a variational autoencoder is defined with `variational_block`.
library(keras)
library(ruta)

network <-
  input() +
  dense(256, "elu") +
  variational_block(3, seed = 42) +
  dense(256, "elu") +
  output("sigmoid")

learner <- autoencoder_variational(network, loss = "binary_crossentropy")

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
model <- learner |> train(x_train, epochs = 5)

#' Sample the trained model
samples <- model |> generate(dimensions = c(2, 3), fixed_values = 0.5)

#' Utility functions for plotting
plot_digit <- function(digit, ...) {
  image(array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray((255:0)/255), ...)
}

plot_matrix <- function(digits) {
  n <- dim(digits)[1]
  layout(
    matrix(1:n, byrow = F, nrow = sqrt(n))
  )

  for (i in 1:n) {
    par(mar = c(0,0,0,0) + .2)
    plot_digit(digits[i, ])
  }
}

#' Plot samples
plot_matrix(samples)

#' Creating an animation from a sampling
library(animation)

par(bg = "white")  # ensure the background color is white
plot(c(), type = "n")

ani.record(reset = T)

for (t in seq(from = 0.001, to = 0.999, length.out = 180)) {
  model |> generate(dimensions = c(2, 3), from = 0.001, to = 0.999, fixed_values = t) |> plot_matrix()
  ani.record()
}

saveHTML(ani.replay(), img.name = "record_plot")
