% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/autoencoder_robust.R
\name{make_robust}
\alias{make_robust}
\title{Add robust behavior to any autoencoder}
\usage{
make_robust(learner, sigma = 0.2)
}
\arguments{
\item{learner}{The \code{"ruta_autoencoder"} object}

\item{sigma}{Sigma parameter in the kernel used for correntropy}
}
\value{
An autoencoder object which contains the correntropy loss
}
\description{
Converts an autoencoder into a robust one by assigning a
correntropy loss to it. Notice that this will replace the previous loss
function
}
\seealso{
\verb{\link{autoencoder_robust}}
}
