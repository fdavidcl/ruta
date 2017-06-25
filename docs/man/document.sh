#!/bin/bash

cd ../../man
Rscript ../docs/man/Rd2Knitr2HTML.R ruta
mv *.html ../docs/man/
[ -f *.css ] && mv *.css ../docs/man/
cd ../docs/man