import importlib.resources

import polars as pl

# TODO: Currently, the list of unmodified bases is only defined for RNA;
#  make it universally applicable
UNMODIFIED_BASES = ["A", "C", "G", "U"]

# Set default value for intensity cutoff
DEFAULT_INTENSITY_CUTOFF = 115000

# Maximum variance for intact mass
MAX_VARIANCE = 1

# Set number of binary-compressed masses per integer cell in traceback matrix
COMPRESSION_RATE = 32


# Set the number of decimal places up to which to consider nucleoside masses
DECIMAL_PLACES = 3

# Set precision for calculations
PRECISION = 10 ** (-DECIMAL_PLACES)

# Set relative tolerance such that we consider
# abs(sum(masses)/target_mass - 1) < TOLERANCE for matching
# Note that the error is on the higher side than would be for a good
# calibrated machine (10 ppm), but in the absence of an experimental
# measurement of this error, this conservative value works well
TOLERANCE = 10e-6


# Build dict with elemental masses
elements = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}
