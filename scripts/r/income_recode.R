# Proposed replacement for parse_income() in both conventional R scripts.
# Does not execute an analysis or write/overwrite any results.
# Use exact exported labels; never extract a year from an Unknown label.
parse_income <- function(x) {
  z <- trimws(as.character(x))
  income_map <- c(
    "< $40,000" = 40000,
    "$40,000 - $44,999" = 40000,
    "$45,000 - $49,999" = 45000,
    "$50,000 - $54,999" = 50000,
    "$55,000 - $59,999" = 55000,
    "$60,000 - $64,999" = 60000,
    "$65,000 - $69,999" = 65000,
    "$70,000 - $74,999" = 70000,
    "$75,000 - $79,999" = 75000,
    "$80,000 - $84,999" = 80000,
    "$85,000 - $89,999" = 85000,
    "$90,000 - $94,999" = 90000,
    "$95,000 - $99,999" = 95000,
    "$100,000 - $109,999" = 100000,
    "$110,000 - $119,999" = 110000,
    "$120,000+" = 120000
  )
  unknown <- is.na(z) | z == "" |
    z == "Unknown/missing/no match/Not 1990-2024"
  unexpected <- !unknown & !(z %in% names(income_map))
  if (any(unexpected)) {
    stop("Unmapped SEER income categories: ",
         paste(unique(z[unexpected]), collapse = "; "))
  }
  values <- unname(income_map[z])
  values[unknown] <- NA_real_
  # These numbers are category boundaries for grouping, not patient income.
  values
}

stopifnot(
  is.na(parse_income("Unknown/missing/no match/Not 1990-2024")),
  parse_income("< $40,000") < 60000,
  parse_income("$55,000 - $59,999") < 60000,
  parse_income("$60,000 - $64,999") == 60000,
  parse_income("$120,000+") == 120000
)
