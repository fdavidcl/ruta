## Resubmission

This is a resubmission. Changes from the original:

- Omitted redundant "for R" in the title
- Improved description indicating the implemented neural networks and incorporating a reference.
- Replaced \dontrun by \donttest in examples which need >5s to execute.
- Ensured that functions and examples do not write to the user's filespace. The save_as() function only does so if the user specifies a directory.

## Test environments
* local Linux Mint install, R 3.4.4
* ubuntu 14.04 (on travis-ci), R 3.5.0 and devel
* win-builder (devel and release)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.
