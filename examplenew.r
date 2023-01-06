#!/usr/bin/env Rscript

library(magrittr)
library(keras)
library(ruta)
library(ggplot2)

plot_digit <- function(digit, ...) {
  image(array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray((0:255)/255), ...)
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

mnist = dataset_mnist()
x_train = array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train = x_train / 255.0
x_test = array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test = x_test / 255.0

network = input() + dense(36, "tanh") + dense(2, "tanh") + dense(36, "tanh") + output("sigmoid")
net2 = network[c(1, 2, 5)]
print(net2)
plot(net2)

learner =
  net2 %>%
  autoencoder_robust(sigma = 0.2) %>%
  add_weight_decay()

learner =
  net2 %>%
  autoencoder("binary_crossentropy")

learner = autoencoder_variational(36, 36)

model = learner %>% train(x_train, epochs = 40)

encoded = model %>% encode(x_test)
decoded = model %>% decode(encoded)

plot_sample(x_test, decoded, 1:10)

mean(sapply(1:nrow(decoded), function(irow) {
  mean((decoded[irow, ] - x_test[irow, ]) ** 2)
}))
