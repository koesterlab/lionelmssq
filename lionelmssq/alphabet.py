from lionelmssq.masses import UNIQUE_MASSES
import polars as pl

DIFF_VALUE = 0.1

# TODO: the -2.0 should be informed by the variance in the measurements
_MIN_PLAUSIBLE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").min()).item() - DIFF_VALUE
)
MAX_PLAUSILE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").max()).item() + DIFF_VALUE
)


def is_similar(mass_a, mass_b):
    """Return whether two masses are similar enough to be considered the same nucleoside."""
    # TODO choose threshold correctly
    # TODO: This is obviously problematic since, C and U differ by only 1 unit. Modified bases further complicate this situation.
    return abs(mass_a - mass_b) < DIFF_VALUE
    # TODO: NOTE: This change creates a problem in the testcase test_01, where U gets switched with C!
