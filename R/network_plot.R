get_ratios <- function(struct) {
  mx <- max(struct)
  struct / mx
}

# center ratios in the [0,1] interval
get_ys <- function(struct) {
  # fix layers with negative units
  amount <- length(struct)
  hm <- 1 / (amount / 2 + amount - 1)
  struct[struct < 0] <- hm / 2

  # calculate vertical margins and center
  marg <- (1 - struct) / 2
  list(lower = marg, upper = marg + struct)
}

# distribute layers in [0,1] interval
get_xs <- function(struct) {
  amount <- length(struct)
  # margin among layers is set to double the width of a layer
  marg <- 1 / (amount / 2 + amount - 1)
  lower <- seq(0, 1, marg * 1.5)

  list(lower = lower, upper = lower + marg / 2)
}

#' Draw a neural network
#'
#' @param x A \code{"ruta_network"} object
#' @param ... Additional parameters for style. Available parameters: \itemize{
#' \item \code{bg}: Color for the text over layers
#' \item \code{fg}: Color for the background of layers
#' \item \code{log}: Use logarithmic scale
#' }
#' @import graphics
#' @import purrr
#' @export
plot.ruta_network <- function(x, ...) {
  args <- list(...)
  bg <- args$bg %||% "#c0c0c0"
  fg <- args$fg %||% "black"
  log <- args$log %||% FALSE

  struct <- as.integer(
    sapply(x, function(n)
      if (!is.null(n$units)) {
        n$units
      } else if (ruta_layer_custom %in% class(n)) {
        0
      } else {
        -1
      })
  )

  for (i in 1:length(struct)) {
    if (struct[i] == 0 && i > 0) {
      struct[i] <- struct[i - 1]
    }
  }

  labels <- as.character(
    sapply(x, function(n)
      if (!is.null(n$units)) {
        n$units
      } else if (ruta_layer_custom %in% class(n)) {
        n$name
      } else {
        "⇩"
      })
  )

  if (log) {
    struct[struct > 0] <- log(struct[struct > 0])
  }

  ratios <- get_ratios(struct)

  ys <- get_ys(ratios)
  xs <- get_xs(ratios)

  plot(c(0, 1), c(0, 1), type = "n", xlab = "", ylab = "", axes = F)

  for (l in 1:length(ratios)) {
    rect(xs$lower[l], ys$lower[l], xs$upper[l], ys$upper[l], col = bg, border = fg)
    if (l > 1) {
      # connections between layers
      lines(
        x = c(xs$upper[l - 1], xs$lower[l]),
        y = ys$lower[(l - 1):l], col = fg
      )
      lines(
        x = c(xs$upper[l - 1], xs$lower[l]),
        y = ys$upper[(l - 1):l], col = fg
      )
      # crossing lines
      if (ruta_layer_dense %in% class(x[[l]])) {
        lines(
          x = c(xs$upper[l - 1], xs$lower[l]),
          y = c(ys$lower[l - 1], ys$upper[l]), col = bg
        )
        lines(
          x = c(xs$upper[l - 1], xs$lower[l]),
          y = c(ys$upper[l - 1], ys$lower[l]), col = bg
        )
      }
    }
    text(mean(c(xs$lower[l], xs$upper[l])), 0.5, labels[l], srt = 90, col = fg)
  }
}
