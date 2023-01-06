context("test-layers.R")

test_that("layers can be joined", {
  expect_equal(length(input() + dense(1)), 2)
  expect_equal(length(input() + (dense(1) + output())), 3)
})

test_that("networks can be subsetted", {
  expect_equal(length((input() + dense(1))[1]), 1)
  expect_equal(length((input() + dense(1) + dropout() + output())[2:3]), 2)
})

test_that("layers have correct classes", {
  expect_s3_class(dense(1), ruta_network)
  expect_s3_class(dense(1)[1], ruta_network)
  expect_s3_class(dense(1)[[1]], ruta_layer)
  expect_s3_class((dense(1) + dense(1))[2], ruta_network)
})

test_that("networks can be built from integers", {
  expect_s3_class(as_network(c(30, 3, 1)), ruta_network)
})
