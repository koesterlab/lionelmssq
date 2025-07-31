import polars as pl
import numpy as np

from lionelmssq.mass_explanation import is_valid_mass
from lionelmssq.mass_table import DynamicProgrammingTable


MAX_VARIANCE = 10


# METHOD: For each breakage option that yields a valid mass (i.e. one that
# can be explained by any valid composition) for a given fragment, duplicate
# the fragment and determine its breakage-independent standard-unit mass by
# subtracting the weight imposed by the breakage.


def classify_fragments(
    fragment_masses,
    dp_table: DynamicProgrammingTable,
    breakage_dict: dict,
    output_file_path=None,
    intensity_cutoff=0.5e6,
    seq_mass=None,
    mass_cutoff=50000,
) -> pl.DataFrame:
    # If no intensity is given, set it so that all fragments pass the filter
    if "intensity" not in fragment_masses.columns:
        fragment_masses = fragment_masses.with_columns(
            pl.lit(intensity_cutoff * 1.1).alias("intensity"),
        )

    # Rename 'neutral_mass' values from deisotoping to 'observed_mass'
    if "neutral_mass" in fragment_masses.columns:
        fragment_masses = fragment_masses.rename({"neutral_mass": "observed_mass"})

    # Index fragments
    fragment_masses = fragment_masses.with_row_index("fragment_index")

    # Copy each fragment for each unique breakage weights and set standard-unit mass
    fragments = pl.concat(
        [
            fragment_masses.with_columns(
                (
                    pl.col("observed_mass") - (breakage_weight * dp_table.precision)
                ).alias("standard_unit_mass"),
                pl.lit(breakages[0]).alias("breakage"),
            )
            for (breakage_weight, breakages) in breakage_dict.items()
        ]
    )

    # Filter out all fragments without any explanations
    fragments = (
        fragments.with_columns(
            pl.struct("observed_mass", "standard_unit_mass")
            .map_elements(
                lambda x: is_valid_mass(
                    mass=x["standard_unit_mass"],
                    dp_table=dp_table,
                    threshold=dp_table.tolerance * x["observed_mass"],
                ),
                return_dtype=bool,
            )
            .alias("is_valid")
        )
        .filter(pl.col("is_valid"))
        .drop("is_valid")
    )

    # Determine all fragments that may be singletons
    fragments = fragments.with_columns(
        pl.struct("observed_mass", "standard_unit_mass")
        .map_elements(
            lambda x: is_singleton(
                mass=x["standard_unit_mass"],
                integer_masses=[mass.mass for mass in dp_table.masses],
                dp_table=dp_table,
                threshold=dp_table.tolerance * x["observed_mass"],
            ),
            return_dtype=bool,
        )
        .alias("is_singleton")
    )

    # Filter out fragments that have a too high mass or too low intensity
    fragments = (
        fragments.sort(pl.col("standard_unit_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(pl.col("observed_mass") < mass_cutoff)
    )

    if seq_mass is not None:
        # Select highest valid SU mass, i.e. sequence mass without START_END breakage
        mass_cutoff = (
            seq_mass
            - [
                mass * dp_table.precision
                for mass in breakage_dict
                if "START_END" in breakage_dict[mass]
            ][0]
        )

        # Filter fragments based on mass cutoff
        fragments = filter_by_sequence_mass(mass_cutoff, fragments)

    # Write terminal fragments to file if file name is given
    if output_file_path is not None:
        fragments.write_csv(output_file_path, separator="\t")

    return fragments


def is_singleton(mass, integer_masses, dp_table, threshold=None):
    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    # Check whether a singleton mass could be found
    for value in range(target - threshold, target + threshold + 1):
        if value in integer_masses:
            return True
    return False


def filter_by_sequence_mass(
    mass_cutoff: float, fragments: pl.DataFrame
) -> pl.DataFrame:
    # Filter out fragments that have a too high SU mass (within variance)
    fragments = fragments.filter(
        pl.col("standard_unit_mass") < mass_cutoff + MAX_VARIANCE
    )

    return fragments
