[Ruta](https://ruta.software/)
=================================================

*Software for unsupervised deep architectures*

[![R language](https://img.shields.io/badge/language-R-lightgrey.svg)](https://www.r-project.org/)
[![Downloads](https://cranlogs.r-pkg.org/badges/ruta)](https://cranlogs.r-pkg.org/downloads/total/last-month/ruta)
[![Travis](https://img.shields.io/travis/fdavidcl/ruta/master.svg)](https://travis-ci.org/fdavidcl/ruta)
[![license](https://img.shields.io/github/license/fdavidcl/ruta.svg)](https://www.gnu.org/licenses/gpl.html)

---

Get uncomplicated access to unsupervised deep neural networks, from building their architecture to their training and evaluation

[Get started](https://ruta.software/articles/examples/autoencoder_basic.html)

## Installation

### Dependencies

Ruta is based in the well known open source deep learning library [Keras](https://keras.io) and its [R interface](https://keras.rstudio.com). It has been developed to work with the [TensorFlow](https://www.tensorflow.org/) backend. In order to install these dependencies you will need the Python interpreter as well, and you can install them via the Python package manager *pip* or possibly your distro's package manager if you are running Linux.

```sh
$ sudo pip install tensorflow
$ sudo pip install keras
```

Otherwise, you can follow the official installation guides:

- [Installing TensorFlow](https://www.tensorflow.org/install/)
- [Keras installation](https://keras.io/#installation)

### Ruta package

```r
# Just get Ruta from the CRAN
install.packages("ruta")

# Or get the latest development version from GitHub
devtools::install_github("fdavidcl/ruta")
```

All R dependencies will be automatically installed. These include the Keras R interface and [`purrr`](https://purrr.tidyverse.org/). For convenience we also recommend installing and loading either [`magrittr`](https://magrittr.tidyverse.org/) or `purrr`, so that the pipe operator `%>%` is available.

## Usage

The easiest way to start working with Ruta is to use the `autoencode()` function. It allows for selecting a type of autoencoder and transforming the feature space of a data set onto another one with some desirable properties depending on the chosen type.

```r
iris[, 1:4] %>% as.matrix %>% autoencode(2, type = "denoising")
```

You can learn more about different variants of autoencoders by reading [*A practical tutorial on autoencoders for nonlinear feature fusion*](https://arxiv.org/abs/1801.01586).

Ruta provides the functionality to build diverse neural architectures (see `autoencoder()`), train them as autoencoders (see `train()`) and perform different tasks with the resulting models (see `reconstruct()`), including evaluation (see `evaluate_mean_squared_error()`). The following is a basic example of a natural pipeline with an autoencoder:

```r
library(ruta)
library(purrr)

# Shuffle and normalize dataset
x <- iris[, 1:4] %>% sample %>% as.matrix %>% scale
x_train <- x[1:100, ]
x_test <- x[101:150, ]

autoencoder(
  input() + dense(256) + dense(36, "tanh") + dense(256) + output("sigmoid"),
  loss = "mean_squared_error"
) %>%
  make_contractive(weight = 1e-4) %>%
  train(x_train, epochs = 40) %>%
  evaluate_mean_squared_error(x_test)
```

For more details, see [other examples](http://ruta.software/articles/examples) and [the documentation](http://ruta.software/reference/).

