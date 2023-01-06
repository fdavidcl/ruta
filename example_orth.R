library(magrittr)
library(keras)
library(ruta)

plot_digit <- function(digit, ...) {
  image(array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col = gray((255:0)/255), ...)
}

plot_square <- function(square, ...) {
  side <- sqrt(length(square))
  image(array_reshape(square, c(side, side), "F")[, side:1], xaxt = "n", yaxt = "n", col = gray((255:0)/255), ...)
}

plot_sample <- function(digits_test, digits_enc, digits_dec, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:(3 * sample_size)), byrow = F, nrow = 3)
  )

  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_square(digits_test[i, ])
    plot_square(digits_enc[i, ])
    plot_square(digits_dec[i, ])
  }
}
mnist <- dataset_mnist()
x_train <- array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train <- x_train / 255.0
x_test <- array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0

network <- input() + dense(36, "relu") + output("sigmoid")
learner <- autoencoder_orthonormal(network, loss = "binary_crossentropy", weight = 1e-4)

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 32
)

#x_test_noisy <- apply_filter(noise_gaussian(sd = .1), x_test)
enc <- model %>% encode(x_test)
decode <- model %>% reconstruct(x_test)
#decoded <- modeld %>% reconstruct(x_test_noisy)
plot_sample(x_test, enc, decode, 11:20)
evaluate_mean_squared_error(model, x_test) # => 0.008661253
