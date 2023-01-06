library(ruta)
library(purrr)
library(colorspace)
library(scatterplot3d)

wdbc <- read.csv("wdbc.data")
x_train <- wdbc[, 3:32]
mx <- apply(x_train, 2, max)
mn <- apply(x_train, 2, min)
range <- mx - mn
x_train <- t(apply(x_train, 1, function(x) (x - mn) / range))
y_train <- ifelse(wdbc[, 2] == "M", 1, 0)

model <- autoencoder_reductive(input() + dense(3) + output("sigmoid"), loss = "binary_crossentropy", weight = 0.1)
model <- train(model, x_train, y_train, epochs = 40)

model %>% encode(x_train) %>% scatterplot3d(color = rainbow_hcl(2)[y_train + 1], pch = 20)
x_train %>% autoencode(3, epochs = 40) %>% scatterplot3d(color = rainbow_hcl(2)[y_train + 1], pch = 20)

# -----------------------------------------

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
y_train <- as.numeric(mnist$train$y == 1 | mnist$train$y == 2 | mnist$train$y == 9 | mnist$train$y == 4 | mnist$train$y == 6)
x_test <- array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0
y_test <- as.numeric(mnist$test$y == 1 | mnist$test$y == 2 | mnist$test$y == 9 | mnist$test$y == 4 | mnist$test$y == 6)

network <- input() + dense(16, "sigmoid") + output("sigmoid")
learner <- autoencoder_reductive(network, loss = contraction("binary_crossentropy"), weight = 0.01)

model <- train(
  learner,
  x_train,
  y_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 128
)

enc <- model %>% encode(x_test)
decode <- model %>% decode(enc)
plot_sample(x_test, enc, decode, 11:20)
evaluate_mean_squared_error(model, list(x_test, y_test)) # => does not work with reductive AE
