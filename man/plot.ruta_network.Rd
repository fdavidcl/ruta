% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/network_plot.R
\name{plot.ruta_network}
\alias{plot.ruta_network}
\title{Draw a neural network}
\usage{
\method{plot}{ruta_network}(x, ...)
}
\arguments{
\item{x}{A \code{"ruta_network"} object}

\item{...}{Additional parameters for style. Available parameters: \itemize{
\item \code{bg}: Color for the text over layers
\item \code{fg}: Color for the background of layers
\item \code{log}: Use logarithmic scale
}}
}
\description{
Draw a neural network
}
\examples{
net <-
  input() +
  dense(1000, "relu") + dropout() +
  dense(100, "tanh") +
  dense(1000, "relu") + dropout() +
  output("sigmoid")
plot(net, log = TRUE, fg = "#30707a", bg = "#e0e6ea")
}
