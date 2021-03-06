gradients <- reticulate::import("tensorflow.python.ops.parallel_for.gradients", delay_load = TRUE)

# inputs:
#  - y: shape (batch_size, output_length)
#  - x: shape (batch_size, input_length)
# outputs:
#  - J: shape (batch_size, output_length, input_length)
#    the jacobians of y with respect to inputs x
jacobian <- function(y, x) {
  gradients$batch_jacobian(y, x)
}
