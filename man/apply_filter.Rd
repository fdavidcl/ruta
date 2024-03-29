% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/filter.R, R/generics.R
\name{apply_filter.ruta_noise_zeros}
\alias{apply_filter.ruta_noise_zeros}
\alias{apply_filter.ruta_noise_ones}
\alias{apply_filter.ruta_noise_saltpepper}
\alias{apply_filter.ruta_noise_gaussian}
\alias{apply_filter.ruta_noise_cauchy}
\alias{apply_filter}
\title{Apply filters}
\usage{
\method{apply_filter}{ruta_noise_zeros}(filter, data, ...)

\method{apply_filter}{ruta_noise_ones}(filter, data, ...)

\method{apply_filter}{ruta_noise_saltpepper}(filter, data, ...)

\method{apply_filter}{ruta_noise_gaussian}(filter, data, ...)

\method{apply_filter}{ruta_noise_cauchy}(filter, data, ...)

apply_filter(filter, data, ...)
}
\arguments{
\item{filter}{Filter object to be applied}

\item{data}{Input data to be filtered}

\item{...}{Other parameters}
}
\description{
Apply a filter to input data, generally a noise filter in
order to train a denoising autoencoder. Users won't generally need to use
these functions
}
\seealso{
\verb{\link{autoencoder_denoising}}
}
