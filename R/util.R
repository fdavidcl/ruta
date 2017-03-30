ruta.util.require <- function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
        stop(paste0("rutavis: Package '", package, "' is not installed and is needed for autoencoder functionality"))
    }
}
