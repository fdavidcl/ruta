#!/usr/bin/Rscript
library(tools)
library(knitr)
library(sowsear)
opts_knit$set(progress = FALSE, verbose = FALSE)

## We need a list of files and a package to start.
args <- commandArgs(TRUE)
package <- args[[1]]
files <- args[-1]

message(sprintf("Using package '%s", package))
library(package, character.only=TRUE)

if ( length(files) == 0 ) {
  message("Using all Rd files")
  files <- list.files(pattern="\\.Rd$")
}

msg <- sprintf("Processing file(s) %s", paste(files, collapse=", "))
message(paste(strwrap(msg, exdent=4), collapse="\n"))

## Main processing function:
prettyRd <- function(file) {
  message(sprintf("*** Processing %s", file))
  base <- file_path_sans_ext(file)
  ## Names of files that will be created.
  file.html <- paste(base, ".html", sep="")
  file.ex.R <- paste(base, "-examples.R", sep="")
  file.ex.Rmd <- paste(base, "-examples.Rmd", sep="")
  file.ex.md <- paste(base, "-examples.md", sep="")
  file.ex.html <- paste(base, "-examples.html", sep="")

  ## R's native Rd -> HTML coversion for most of the file, saved as
  ## character variable 'html'.  The stylesheet argument is different
  ## to the default.
  con <- textConnection("html", "w")
  Rd2HTML(file, con, package=package, stylesheet="stylesheet.css")
  close(con)

  ## Extract the examples part of the file:
  Rd2ex(file, file.ex.R)
  if ( file.exists(file.ex.R) ) {
    ## Prettify the \notrun sections, by telling knitr not to evaluate
    ## them.  Ideally we would also put a <div> around them, perhaps?
    tmp <- readLines(file.ex.R)
    writeLines(cleanup.notrun(tmp), file.ex.R)

    ## Convert the example file to Rmd format
    sowsear(file.ex.R, "Rmd")
    ## Knit that to markdown
    knit(file.ex.Rmd, file.ex.md)
    ## Convert this into HTML with pandoc (assumed installed)
    system(sprintf("pandoc %s -o %s", file.ex.md, file.ex.html))
    ## And replace the example section in R's HTML output.
    html <- c(html[seq_len(grep("<h3>Examples</h3>", html, fixed=TRUE))],
              readLines(file.ex.html),
              html[(max(grep("</pre>", html, fixed=TRUE))+1):length(html)])

    ## Cleanup
    file.remove(file.ex.R, file.ex.Rmd, file.ex.md, file.ex.html)
  } 
  writeLines(html, file.html)

  ## Return the \name, \title, and output filename
  tmp <- parse_Rd(file)
  list(name=unlist(get.tag("\\name", tmp)),
       title=unlist(get.tag("\\title", tmp)),
       file=file.html)
}

## This would be better to get right in Rd2ex
cleanup.notrun <- function(str) {
  i0 <- which(str == "## Not run: ")
  i1 <- which(str == "## End(Not run)")
  if ( length(i0) != length(i1) || any(i1 < i0) )
    stop("Parse error") # not trying any clever stuff.

  for ( i in seq_along(i0) ) {
    idx <- (i0[i]+1):(i1[i]-1)
    tmp <- sub("^##D ", "", str[idx])
    type <- sowsear:::sowsear.classify(tmp)
    is.code <- which(type == "code")

    j <- is.code[!((is.code-1) %in% is.code)]
    tmp[j] <- sprintf("##+ eval=FALSE\n%s", tmp[j])
    str[idx] <- tmp
  }
  str[i0] <- "\n## *[This section is not run by default]*"
  str[i1] <- "## *[Ends not run section]*\n"
  
  str
}

## Pull contents of first matched tag from parsed Rd file
get.tag <- function(tag, rd) {
  for ( x in rd )
    if (attr(x, "Rd_tag") == tag)
      return(x)
  stop("didn't find tag")
}

## Actually do the processing.  This is not fast.
info <- lapply(files, prettyRd)

## If we did more than one file, make a *really* simple index.
if ( length(files) > 1 ) {
  contents <- sapply(info, function(x)
                     sprintf("* [%s](%s) %s", x$name, x$file,
                             paste(x$title, collapse="")))
  contents <- gsub('\n', ' ', contents)
  contents <- c(sprintf("# Help files for %s", package), contents)
  
  writeLines(paste(contents, collapse="\n\n"), "index.md")
  system("pandoc index.md -o index.html -c stylesheet.css --standalone")
  file.remove("index.md")
}

## Default stylesheet, from pandoc's tango theme, plus very minimal
## page css styling.  Will be saved as stylesheet.css iff it does not
## exist.
default.stylesheet <- "/* Highlighting from pandoc / tango */
table.sourceCode, tr.sourceCode, td.lineNumbers, td.sourceCode {
  margin: 0; padding: 0; vertical-align: baseline; border: none; }
table.sourceCode { width: 100%; background-color: #f8f8f8; }
td.lineNumbers { text-align: right; padding-right: 4px; padding-left: 4px; color: #aaaaaa; border-right: 1px solid #aaaaaa; }
td.sourceCode { padding-left: 5px; }
pre, code { background-color: #f8f8f8; }
code > span.kw { color: #204a87; font-weight: bold; }
code > span.dt { color: #204a87; }
code > span.dv { color: #0000cf; }
code > span.bn { color: #0000cf; }
code > span.fl { color: #0000cf; }
code > span.ch { color: #4e9a06; }
code > span.st { color: #4e9a06; }
code > span.co { color: #8f5902; font-style: italic; }
code > span.ot { color: #8f5902; }
code > span.al { color: #ef2929; }
code > span.fu { color: #000000; }
code > span.er { font-weight: bold; }

body { font-family: Helvetica, sans-serif;
       color: #333; 
       padding: 0 5px; 
       margin: 0 auto; 
       font-size: 14px;
       width: 80%;
       max-width: 60em; /* 960px */
       position: relative; 
       line-height: 1.5; 
     }

/* Hide caption */
p.caption { display:none }
"

if ( !file.exists(default.stylesheet) )
  writeLines(default.stylesheet, "stylesheet.css")
