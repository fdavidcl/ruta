#' Save and load Ruta models
#'
#' Functions to save a trained or untrained Ruta learner into a file and load it
#'
#' @param learner The `"ruta_autoencoder"` object to be saved
#' @param file In `save`, filename with extension (usually `.tar.gz`) where the object will be
#'   saved. In `load`, path to the saved model
#' @param dir Directory where to save the file. Use `"."` to save in the current
#'   working directory or `tempdir()` to use a temporary one
#' @param compression Type of compression to be used, for R function `\link{tar}`
#' @return `save_as` returns the filename where the model has been saved,
#'   `load_from` returns the loaded model as a `"ruta_autoencoder"` object
#'
#' @examples
#' library(purrr)
#'
#' x <- as.matrix(iris[, 1:4])
#'
#' \donttest{
#' # Save a trained model
#' saved_file <-
#'   autoencoder(2) %>%
#'   train(x) %>%
#'   save_as("my_model.tar.gz", dir = tempdir())
#'
#' # Load and use the model
#' encoded <- load_from(saved_file) %>% encode(x)
#' }
#' @export
save_as <- function(learner, file = paste0(substitute(learner), ".tar.gz"), dir, compression = "gzip") {
  output_file <- file.path(dir, file)
  # Work in a temporary dir
  tmpdir <- tempdir()

  # Create an empty directory
  base_name <- file.path(tmpdir, "ruta")
  if (file.exists(base_name)) {
    unlink(base_name, recursive = TRUE)
  }
  save_dir <- dir.create(base_name)

  # Save Ruta learner
  files <- file.path(base_name, "model.rds")
  saveRDS(learner, file = files)

  # Save model weights
  if (is_trained(learner)) {
    files <- c(files, file.path(base_name, "weights.hdf5"))
    keras::save_model_weights_hdf5(learner$models$autoencoder, files[2])
  }

  # Create archive
  oldwd <- setwd(base_name)
  archive <- utils::tar(file, files = NULL, compression = compression)
  if (!is.null(oldwd)) setwd(oldwd)
  file.copy(file.path(base_name, file), output_file)

  output_file
}

#' @rdname save_as
#' @export
load_from <- function(file) {
  # Work in a temporary dir
  tmpdir <- tempdir()
  base_name <- file.path(tmpdir, "ruta")
  if (file.exists(base_name)) {
    unlink(base_name, recursive = TRUE)
  }

  utils::untar(file, exdir = base_name)

  learner <- readRDS(file = file.path(base_name, "model.rds"))

  weights_file <- file.path(base_name, "weights.hdf5")
  if (file.exists(weights_file)) {
    learner$models <- to_keras(learner, weights_file = weights_file)
  }

  learner
}
