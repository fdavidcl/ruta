xtrain <- quakes[1:750,] %>% as.matrix()
xtest <- quakes[751:1000,] %>% as.matrix()
encoding_dim <- 2
hidden_dim <- 6

# ============== Ruta  ==============
library(ruta)
library(purrr)

features <- autoencoder(c(hidden_dim, encoding_dim), "sigmoid") %>%
  train(xtrain) %>%
  encode(xtest)

# ============== Keras ==============
library(keras)

input_l <- layer_input(shape = 5)
encoded <- layer_dense(input_l, units = hidden_dim)
encoded <- layer_dense(encoded, units = encoding_dim, activation = "sigmoid")
decoded <- layer_dense(encoded, units = hidden_dim)
decoded <- layer_dense(decoded, units = 5)

autoe <- keras_model(input_l, decoded)
encoder <- keras_model(input_l, encoded)

compile(autoe, loss = "mean_squared_error", optimizer = "rmsprop")
fit(autoe, xtrain, xtrain)

features <- predict(encoder, xtest)
