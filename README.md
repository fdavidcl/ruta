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

## How to install

In order to develop Ruta models, you will need to install its dependencies first and then get the package from CRAN.

### Dependencies

Ruta is based in the well known open source deep learning library [Keras](https://keras.io) and its [R interface](https://cran.r-project.org/package=keras), which is integrated in [Tensorflow](https://www.tensorflow.org/). In order to install them easily, you can use the [`keras::install_keras()`](https://tensorflow.rstudio.com/reference/keras/install_keras) function. Depending on whether you want to use the system installation, a Conda environment or a Virtualenv, you may need to call [`use_condaenv()` or `use_virtualenv()` from `reticulate`](https://rstudio.github.io/reticulate/articles/versions.html).


Another straightforward way to install these dependencies is to use global system-wide (`sudo pip install`) or user-wide (`pip install --user`) installation with `pip`. This is generally not recommended unless you are sure you will not need alternative versions or clash with other packages. The following shell command would install all libraries expected by Keras:

```sh
$ pip install --user tensorflow tensorflow-hub tensorflow-datasets scipy requests pyyaml Pillow h5py pandas pydot
```

Otherwise, you can follow the official installation guides:

- [Installing TensorFlow](https://www.tensorflow.org/install/)

Check whether Keras is accesible from R by running:

```r
keras::is_keras_available() # should return TRUE
```

### Ruta package

From an R interpreter such as the R REPL or the RStudio console, run one of the following commands to get the Ruta package:

```r
# Just get Ruta from the CRAN
install.packages("ruta")

# Or get the latest development version from GitHub
devtools::install_github("fdavidcl/ruta")
```

All R dependencies will be automatically installed. These include the Keras R interface and [`purrr`](https://purrr.tidyverse.org/).

## First steps

The easiest way to start working with Ruta is to use the `autoencode()` function. It allows for selecting a type of autoencoder and transforming the feature space of a data set onto another one with some desirable properties depending on the chosen type.

```r
iris[, 1:4] |> as.matrix() |> autoencode(2, type = "denoising")
```

You can learn more about different variants of autoencoders by reading [*A practical tutorial on autoencoders for nonlinear feature fusion*](https://arxiv.org/abs/1801.01586).

Ruta provides the functionality to build diverse neural architectures (see `autoencoder()`), train them as autoencoders (see `train()`) and perform different tasks with the resulting models (see `reconstruct()`), including evaluation (see `evaluate_mean_squared_error()`). The following is a basic example of a natural pipeline with an autoencoder:

```r
library(ruta)

# Shuffle and normalize dataset
x <- iris[, 1:4] |> sample() |> as.matrix() |> scale()
x_train <- x[1:100, ]
x_test <- x[101:150, ]

autoencoder(
  input() + dense(256) + dense(36, "tanh") + dense(256) + output("sigmoid"),
  loss = "mean_squared_error"
) |>
  make_contractive(weight = 1e-4) |>
  train(x_train, epochs = 40) |>
  evaluate_mean_squared_error(x_test)
```

For more details, see [other examples](https://ruta.software/articles/examples/) and [the documentation](https://ruta.software/reference/).
