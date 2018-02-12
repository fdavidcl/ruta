context("test-layers.R")

test_that("layers can be joined", {
  expect_equal(length(input() + dense(1)), 2)
  expect_equal(length(input() + (dense(1) + output())), 3)
})

test_that("networks can be subsetted", {
  expect_equal(length((input() + dense(1))[1]), 1)
  expect_equal(length((input() + dense(1) + output())[2:3]), 2)
})

test_that("layers have correct classes", {
  expect_s3_class(dense(1), rutaNetwork)
  expect_s3_class(dense(1)[1], rutaNetwork)
  expect_s3_class((dense(1) + dense(1))[2], rutaNetwork)
})
