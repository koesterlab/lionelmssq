import polars as pl
import numpy as np

from lionelmssq.mass_explanation import is_valid_mass
from lionelmssq.mass_table import DynamicProgrammingTable


def classify_fragments(
    fragment_masses,
    dp_table: DynamicProgrammingTable,
    breakage_dict: dict,
    output_file_path=None,
    intensity_cutoff=0.5e6,
    mass_cutoff=50000,
) -> pl.DataFrame:
    if "observed_mass" in fragment_masses.columns:
        fragment_masses = fragment_masses.with_columns(
            pl.lit(intensity_cutoff * 1.1).alias("intensity"),
        )
    else:
        fragment_masses = fragment_masses.with_columns(
            pl.col("neutral_mass").alias("observed_mass"),
        )

    fragment_masses = fragment_masses.with_row_index("fragment_index")

    # Copy each fragment for each unique breakage weights
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
