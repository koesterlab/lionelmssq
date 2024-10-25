from lionelmssq.masses import UNIQUE_MASSES
import polars as pl


# TODO: the -2.0 should be informed by the variance in the measurements
_MIN_PLAUSIBLE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").min()).item() - 2.0
)
MAX_PLAUSILE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").max()).item() + 2.0
)


def is_similar(mass_a, mass_b):
    """Return whether two masses are similar enough to be considered the same nucleoside."""
    # TODO choose threshold correctly
    return abs(mass_a - mass_b) < 2.0
