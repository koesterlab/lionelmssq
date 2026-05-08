# -*- coding: utf-8 -*-
"""Module for mass-related variables."""

import importlib.resources

import polars as pl

# Maximum variance for intact mass
MAX_VARIANCE = 1

# Set the number of decimal places up to which to consider nucleoside masses
DECIMAL_PLACES = 3

# Set precision for calculations
PRECISION = 10 ** (-DECIMAL_PLACES)


# Build dict with elemental masses
elements = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}
