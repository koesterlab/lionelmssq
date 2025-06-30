from dataclasses import dataclass
from typing import List
from platformdirs import user_cache_dir

import polars as pl
import pathlib
import numpy as np
import os

from lionelmssq.linear_program import UNMODIFIED_BASES


# Set OS-independent cache directory for DP tables
TABLE_DIR = user_cache_dir(
    appname="lionelmssq/dp_table", version="1.0", ensure_exists=True
)


@dataclass
class NucleotideMass:
    mass: int
    names: List[str]
    is_modification: bool
    modification_rate: float


@dataclass
class DynamicProgrammingTable:
    table: np.ndarray
    compression_per_cell: int
    precision: float
    tolerance: float
    masses: List[NucleotideMass]

    def __init__(
        self,
        nucleotide_df,
        compression_rate,
        tolerance,
        precision,
        reduced_table=False,
        reduced_set=False,
    ):
        self.compression_per_cell = compression_rate
        self.precision = precision
        self.tolerance = tolerance
        self.masses = initialize_nucleotide_masses(nucleotide_df)
        self.table = load_dp_table(
            table_path=set_table_path(
                reduced_table, reduced_set, precision, compression_rate
            ),
            reduce_table=reduced_table,
            integer_masses=[mass.mass for mass in self.masses],
        )


    def adapt_individual_modification_rates_by_universal_one(
            self, universal_rate
    ):
        for nucleotide_mass in self.masses:
            if not nucleotide_mass.is_modification:
                continue
            if nucleotide_mass.modification_rate > universal_rate:
                nucleotide_mass.modification_rate = universal_rate


def set_table_path(reduce_table, reduce_set, precision, compression_rate):
    # Set path for DP table
    path = (
        f"{TABLE_DIR}/{'reduced' if reduce_table else 'full'}_table."
        f"{'reduced' if reduce_set else 'full'}_set/"
        f"tol_{precision:.0E}.{compression_rate}_per_cell"
    )

    # Create directory for DP table if it does not already exist
    subdir = "/".join(path.split("/")[:-1])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    return path


def initialize_nucleotide_masses(nucleotide_df):
    # Get list of integer masses
    integer_masses = nucleotide_df.get_column("tolerated_integer_masses").to_list()

    # Add a default weight for easier initialization
    integer_masses += [0]

    # Ensure unique and sorted entries after tolerance correction
    integer_masses = sorted(set(integer_masses))

    # Create dict with all associated nucleotide names for each mass
    names = {
        mass: pl.DataFrame({"tolerated_integer_masses": mass})
        .join(
            nucleotide_df,
            on="tolerated_integer_masses",
            how="left",
        )
        .get_column("nucleoside")
        .to_list()
        for mass in nucleotide_df.get_column("tolerated_integer_masses").to_list()
    }

    # Create dict with indicator whether each mass is associated with a modified base
    is_mod = {
        mass: any(base not in UNMODIFIED_BASES for base in names[mass])
        for mass in nucleotide_df.get_column("tolerated_integer_masses").to_list()
    }

    # Create dict with the largest associated modification rate for each mass
    rates = {
        mass: max(
            pl.DataFrame({"tolerated_integer_masses": mass})
            .join(
                nucleotide_df,
                on="tolerated_integer_masses",
                how="left",
            )
            .get_column("modification_rate")
            .to_list()
        )
        for mass in nucleotide_df.get_column("tolerated_integer_masses").to_list()
    }

    # Return list of NucleotideMass instances
    return [
        NucleotideMass(mass, names[mass], is_mod[mass], rates[mass])
        if mass != 0
        else NucleotideMass(0, [], False, 0.0)
        for mass in integer_masses
    ]


def set_up_bit_table(integer_masses, max_mass: int, compression_rate: int):
    """
    Calculate complete bit-representation mass table with dynamic programming.
    """
    settings = select_table_building_settings(compression_rate)

    # Initialize bit-representation numpy table
    max_col = int(np.ceil((max_mass + 1) / compression_rate))
    dp_table = np.zeros((len(integer_masses), max_col), dtype=settings["type"])
    dp_table[0, 0] = settings["init"]

    # Fill DP table row-wise
    for i in range(1, len(integer_masses)):
        # Case: Start new row (i.e. move on to new nucleotide) by initializing reachable cells from before
        dp_table[i] = [
            ((val | (val >> 1)) & settings["alt_sec"]) for val in dp_table[i - 1]
        ]

        # Define number of cells to move (step) and bit shift in a cell (shift)
        step = int(integer_masses[i] / compression_rate)
        shift = integer_masses[i] % compression_rate

        # Case: Add more of current nucleotide
        for j in range(max_col):
            # Consider cell defined by step
            if step + j < max_col:
                dp_table[i, j + step] |= settings["alt_first"] & (
                    (dp_table[i, j] >> (2 * shift) << 1)
                    | (dp_table[i, j] >> (2 * shift))
                )

            # If shift is needed, consider the next cell as well
            if shift != 0 and j + step + 1 < max_col:
                dp_table[i, j + step + 1] |= settings["alt_first"] & (
                    (dp_table[i, j] << 2 * (compression_rate - shift) << 1)
                    | (dp_table[i, j] << 2 * (compression_rate - shift))
                )

    # Adjust last column for unused cells
    dp_table[:, -1] &= settings["full"] << 2 * (max_col - (max_mass + 1) % max_col)

    return dp_table


def select_table_building_settings(compression_rate: int):
    match compression_rate:
        case 4:
            return {
                "type": np.uint8,
                "init": 0xC0,
                "alt_first": 0xAA,
                "alt_sec": 0x55,
                "full": np.uint8(0xFF),
            }
        case 8:
            return {
                "type": np.uint16,
                "init": 0xC000,
                "alt_first": 0xAAAA,
                "alt_sec": 0x5555,
                "full": np.uint16(0xFFFF),
            }
        case 16:
            return {
                "type": np.uint32,
                "init": 0xC0000000,
                "alt_first": 0xAAAAAAAA,
                "alt_sec": 0x55555555,
                "full": np.uint32(0xFFFFFFFF),
            }
        case 32:
            return {
                "type": np.uint64,
                "init": 0xC000000000000000,
                "alt_first": 0xAAAAAAAAAAAAAAAA,
                "alt_sec": 0x5555555555555555,
                "full": np.uint64(0xFFFFFFFFFFFFFFFF),
            }
        case _:
            raise ValueError(
                f"The compression rate {compression_rate} is "
                f"not compatible with the table setup."
            )


def set_up_mass_table(integer_masses, max_mass):
    """
    Calculate complete mass table with dynamic programming.
    """
    # Initialize numpy table
    dp_table = np.zeros((len(integer_masses), max_mass + 1), dtype=np.uint8)
    dp_table[0, 0] = 3.0

    # Fill DP table row-wise
    for i in range(1, len(integer_masses)):
        # Case: Start new row (i.e. move on to new nucleoside) by initializing
        # reachable cells from before
        dp_table[i] = [int(val != 0.0) for val in dp_table[i - 1]]

        # Case: Add more of current nucleoside
        for j in range(max_mass + 1):
            # If cell is not reachable, skip it
            if dp_table[i, j] == 0.0:
                continue

            # Add another nucleoside if possible
            if integer_masses[i] + j <= max_mass:
                dp_table[i, j + integer_masses[i]] += 2.0

    return dp_table


def load_dp_table(table_path, reduce_table, integer_masses):
    """
    Load dynamic-programming table if it exists and compute it otherwise.
    """
    # Select compression rate from path string
    compression_rate = int(table_path.split(".")[-1].rstrip("_per_cell"))

    # Select maximum integer mass for which table should be built
    max_mass = max(integer_masses) * (12 if reduce_table else 35)

    # Compute and save bit-representation DP table if not existing
    if not pathlib.Path(f"{table_path}.npy").is_file():
        print("Table not found")
        dp_table = (
            set_up_mass_table(integer_masses, max_mass)
            if compression_rate == 1
            else (set_up_bit_table(integer_masses, max_mass, compression_rate))
        )
        np.save(table_path, dp_table)

    # Read DP table
    return np.load(f"{table_path}.npy")
