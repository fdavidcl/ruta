context("test-autoencoders.R")

test_that("can create autoencoders", {
  expect_s3_class(autoencoder(1), ruta_autoencoder)
  expect_s3_class(autoencoder_sparse(1), ruta_autoencoder)
  expect_s3_class(autoencoder_contractive(1), ruta_autoencoder)
  expect_s3_class(autoencoder_denoising(1), ruta_autoencoder)
  expect_s3_class(autoencoder_robust(1), ruta_autoencoder)
  expect_s3_class(autoencoder_variational(1), ruta_autoencoder_variational)
})

test_that("can convert autoencoders", {
  ae <- autoencoder(1)

  sae <- make_sparse(ae)
  cae <- make_contractive(ae)
  dae <- make_denoising(ae)
  rae <- make_robust(ae)

  expect_s3_class(sae$network[[2]]$activity_regularizer, ruta_sparsity)
  expect_s3_class(cae$loss, ruta_loss_contraction)
  expect_s3_class(dae$filter, ruta_noise)
  expect_s3_class(rae$loss, ruta_loss_correntropy)
})

test_that("can check autoencoders", {
  ae <- autoencoder(1)

  sae <- make_sparse(ae)
  cae <- make_contractive(ae)
  dae <- make_denoising(ae)
  rae <- make_robust(ae)
  vae <- autoencoder_variational(1)

  expect_true(is_sparse(sae))
  expect_false(is_sparse(cae))
  expect_false(is_sparse(dae))
  expect_false(is_sparse(rae))
  expect_false(is_sparse(vae))

  expect_true(is_contractive(cae))
  expect_false(is_contractive(sae))
  expect_false(is_contractive(dae))
  expect_false(is_contractive(rae))
  expect_false(is_contractive(vae))

  expect_true(is_denoising(dae))
  expect_false(is_denoising(sae))
  expect_false(is_denoising(cae))
  expect_false(is_denoising(rae))
  expect_false(is_denoising(vae))

  expect_true(is_robust(rae))
  expect_false(is_robust(sae))
  expect_false(is_robust(cae))
  expect_false(is_robust(dae))
  expect_false(is_robust(vae))

  expect_true(is_variational(vae))
  expect_false(is_variational(sae))
  expect_false(is_variational(cae))
  expect_false(is_variational(dae))
  expect_false(is_variational(rae))
})
