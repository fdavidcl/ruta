#' Save and load Ruta models
#'
#' Functions to save a trained or untrained Ruta learner into a file and load it
#'
#' @param learner The `"ruta_autoencoder"` object to be saved
#' @param file File name with extension (usually `.tar.gz`) where the object will be
#'   saved to or loaded from
#' @param compression Type of compression to be used, for R function `\link{tar}`
#' @return `save_as` invisibly returns the filename where the model has been saved,
#'   `load_from` returns the loaded model as a `"ruta_autoencoder"` object
#'
#' @examples
#' \dontrun{
#' x <- as.matrix(iris[, 1:4])
#'
#' # Save a trained model
#' autoencoder(2) %>% train(x) %>% save_as("my_model.tar.gz")
#'
#' # Load and use the model
#' encoded <- load_from("my_model.tar.gz") %>% encode(x)
#' }
#' @export
save_as <- function(learner, file = paste0(substitute(learner), ".tar.gz"), compression = "gzip") {
  # Work in a temporary dir
  tmpdir <- tempdir()
  oldwd <- setwd(tmpdir)

  # Create an empty directory
  base_name <- "ruta"
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
  archive <- tar(file.path(oldwd, file), files = base_name, compression = compression)

  if (!is.null(oldwd)) setwd(oldwd)

  invisible(file)
}

#' @rdname save_as
#' @export
load_from <- function(file) {
  # Work in a temporary dir
  tmpdir <- tempdir()
  oldwd <- setwd(tmpdir)
  base_name <- "ruta"
  if (file.exists(base_name)) {
    unlink(base_name, recursive = TRUE)
  }

  untar(file.path(oldwd, file))

  learner <- readRDS(file = file.path("ruta", "model.rds"))

  weights_file <- file.path("ruta", "weights.hdf5")
  if (file.exists(weights_file)) {
    learner$models <- to_keras(learner, weights_file = weights_file)
  }

  if (!is.null(oldwd)) setwd(oldwd)

  learner
}
