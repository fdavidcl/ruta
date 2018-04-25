#' Create a contractive autoencoder
#'
#' A contractive autoencoder adds a penalty term to the loss
#' function of a basic autoencoder which attempts to induce a contraction of
#' data in the latent space.
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying the reconstruction error part of the loss function
#' @param weight Weight assigned to the contractive loss
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#'
#' @references
#' - [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](https://dl.acm.org/citation.cfm?id=3104587)
#' - [A practical tutorial on autoencoders for nonlinear feature fusion](https://arxiv.org/abs/1801.01586)
#'
#' @family autoencoder variants
#' @export
autoencoder_contractive <- function(network, loss, weight) {
  autoencoder(network, loss) %>%
    make_contractive(weight)
}

#' Contractive loss
#'
#' @description This is a wrapper for a loss which induces a contraction in the
#' latent space.
#'
#' @param reconstruction_loss Original reconstruction error to be combined with the
#' contractive loss (e.g. `"binary_crossentropy"`)
#' @param weight Weight assigned to the contractive loss
#' @return A loss object which can be converted into a Keras loss
#'
#' @seealso `\link{autoencoder_contractive}`
#' @family loss functions
#' @export
contraction <- function(reconstruction_loss, weight) {
  structure(
    list(
      reconstruction_loss = reconstruction_loss,
      weight = weight
    ),
    class = c(ruta_loss_contraction, ruta_loss)
  )
}

#' Add contractive behavior to any autoencoder
#'
#' @description Converts an autoencoder into a contractive one by assigning a
#' contractive loss to it
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param weight Weight assigned to the contractive loss
#'
#' @return An autoencoder object which contains the contractive loss
#'
#' @seealso `\link{autoencoder_contractive}`
#' @export
make_contractive <- function(learner, weight) {
  if (!(ruta_loss_contraction %in% class(learner$loss))) {
    learner$loss = contraction(learner$loss, weight)
  }

  learner
}

#' Obtain a Keras contractive loss
#'
#' @description Builds the Keras loss function corresponding to the object received
#'
#' @param x A \code{"ruta_loss_contraction"} object
#' @param keras_model The keras autoencoder which will use the loss function
#' @return A function which returns the contractive loss for given true and
#' predicted values
#' @param ... Rest of parameters, ignored
#' @references
#' \href{https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/}{Deriving Contractive Autoencoder and Implementing it in Keras}
#'
#' @seealso `\link{autoencoder_contractive}`
#' @export
to_keras.ruta_loss_contraction <- function(x, keras_model, ...) {
  rec_err <- x$reconstruction_loss %>% as_loss() %>% to_keras()
  encoding_layer <- keras::get_layer(keras_model, name = "pre_encoding")
  encoding_activation <- keras::get_layer(keras_model, name = "encoding")
  act_f <- encoding_activation$activation

  # derivative of the activation function -- only tanh for now
  act_der <- function(h) 1 - h * h
  # derivative_activation <- function(t) {
  #   x = encoding_layer$input
  #   y = encoding_layer$activation(x)
  #   dydx = keras::k_gradients(y, list(x))
  #   df = keras::k_function(list(x), dydx)
  #   df(t)
  # }
  # h = encoding_activation$input
  # y = encoding_activation$output
  # dydx = keras::k_gradients(y, list(h))

  # h = keras::k_placeholder(shape = NULL, name = "der_input")
  # y = encoding_activation$activation(h)
  # dydh = keras::k_gradients(y, list(h))
  # df = keras::k_function(list(h), dydh)
  # dfl = keras::layer_lambda(encoding_activation$output, df, name = "derivative")

  gradv = function(enc_out)
    keras::k_map_fn(
      fn = function(e) {
        # h = keras::layer_input(
        #   shape = list(NULL),
        #   name = "der_input",
        #   tensor = k_constant(e, dtype = "float32")
        # )
        # y = act_f(e)
        dydh = keras::k_gradients(e, e)
        keras::k_eval(dydh[[1]])
      },
      elems = enc_out,
      name = "vec_grads",
      dtype = "float32"
    )

  derivative_activation <- function(keras_activation) {
    x = k_placeholder(shape = NULL)
    y = keras_activation(x)
    df = k_function(list(x), k_gradients(y, list(x)))

    function(t) {
      df(list(t))[[1]]
    }
  }

  jacobian <- function(y, x) {
    fn <- function(e) keras::k_gradients(e, x)[[1]]
    y_flat <- keras::k_flatten(y)
    keras::k_map_fn(fn, y_flat, name = "halou")
  }

  myjac <- jacobian(encoding_activation$output, encoding_activation$input)

  # contractive loss
  function(y_true, y_pred) {
    reconstruction <- rec_err(y_true, y_pred)

    sum_wt2 <-
      keras::k_variable(value = keras::get_weights(encoding_layer)[[1]]) %>%
      keras::k_transpose() %>%
      keras::k_square() %>%
      keras::k_sum(axis = 2)

    #dh2 <- keras::k_square(keras::k_gradients(encoding_activation$output, encoding_activation$input))
    dh2 <- keras::k_square(myjac)

    # df <- derivative_activation(encoding_activation$activation)
    # dh2 <- keras::k_map_fn(df, encoding_activation$output, name = "jacobian", dtype = "float32") %>% keras::k_square()

    contractive <- x$weight * keras::k_sum(dh2 * sum_wt2, axis = 2)

    #reconstruction + contractive

    # weights <- keras::k_variable(value = keras::get_weights(encoding_layer)[[1]])
    # grad <- keras::k_gradients(keras::k_sum(encoding_activation$output), encoding_activation$input)
    # jacob <- keras::k_batch_dot(keras::k_square(grad), keras::k_square(keras::k_sum(weights, axis = 1)))
    # frob_norm <- keras::k_sum(jacob) / 64 # replace with batch size
    # penalty <- x$weight * frob_norm

    contractive <-

    reconstruction + contractive
  }
}
