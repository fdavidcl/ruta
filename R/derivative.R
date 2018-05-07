# This file defines the derivatives of activation functions available in Keras
# (except softmax). They are not exported since this computation could change
# in the future, in case Keras can automatically calculate derivatives for activations.

k_greater_int <- function(a, b) keras::k_cast(keras::k_greater(a, b), keras::k_floatx())
k_less_int <- function(a, b) keras::k_cast(keras::k_less(a, b), keras::k_floatx())

derivative <- function(activation = "linear") {
  switch(activation,
    elu = derivative_elu,
    hard_sigmoid = derivative_hard_sigmoid,
    linear = derivative_linear,
    relu = derivative_relu,
    selu = derivative_selu,
    sigmoid = derivative_sigmoid,
    softplus = derivative_softplus,
    softsign = derivative_softsign,
    tanh = derivative_tanh,
    stop("The activation function is not supported yet")
  )
}

# elu' assuming alpha = 1
derivative_elu <- function(x, y) 1 + k_less_int(x, 0) * y

derivative_hard_sigmoid <- function(x, y) 0.2 * k_greater_int(y, 0) * k_less_int(y, 1)

# tensor of ones of same shape than x
derivative_linear <- function(x, y) 1. + 0. * x

derivative_relu <- function(x, y) k_greater_int(x, 0)

derivative_selu <- function(x, y) {
  scale = 1.0507009873554804934193349852946
  alpha = 1.6732632423543772848170429916717
  greater = k_greater_int(x, 0)
  scale * (greater + (1 - greater) * alpha * keras::k_exp(x))
}

derivative_sigmoid <- function(x, y) y * (1 - y)

derivative_softplus <- function(x, y) keras::k_sigmoid(x)

derivative_softsign <- function(x, y) 1 / keras::k_square(keras::k_abs(x) + 1)

derivative_tanh <- function(x, y) 1 - y * y
