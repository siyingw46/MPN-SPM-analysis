# Parse scripts without executing analyses; test fabricated labels only.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
root <- normalizePath(args[1],mustWork=TRUE)
files <- list.files(file.path(root,'scripts','r'),pattern='[.]R$',full.names=TRUE)
for (path in files) parse(file=path)
source(file.path(root,'scripts','r','income_recode.R'))
stopifnot(is.na(parse_income('Unknown/missing/no match/Not 1990-2024')),
          is.na(parse_income('')),is.na(parse_income(NA_character_)),
          parse_income('$55,000 - $59,999')<60000,
          parse_income('$60,000 - $64,999')==60000,
          inherits(try(parse_income('unrecognised fabricated label'),silent=TRUE),'try-error'))
cat('PASS: parsed',length(files),'R scripts; fabricated income labels classified correctly.\n')
