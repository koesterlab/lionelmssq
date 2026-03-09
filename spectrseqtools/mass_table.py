from dataclasses import dataclass
from typing import List
from platformdirs import user_cache_dir

import polars as pl
import pathlib
import numpy as np
import os

from spectrseqtools.masses import NUCLEOTIDE_DF, UNMODIFIED_BASES


# Set OS-independent cache directory for DP table
TABLE_DIR = user_cache_dir(
    appname="spectrseqtools/dp_table", version="1.4", ensure_exists=True
)
# Set maximum sequence length to be represented in DP table
MAX_SEQ_LENGTH = 35


@dataclass
class SequenceInformation:
    max_len: int
    su_mass: float
    obs_mass: float
    modification_rate: float


@dataclass
class NucleotideMass:
    mass: int
    names: List[str]
    is_modification: bool
    modification_rate: float

    def __eq__(self, other):
        return self.mass == other.mass

    def __le__(self, other):
        return self.mass <= other.mass

    def __lt__(self, other):
        return self.mass < other.mass

    def __ge__(self, other):
        return self.mass >= other.mass

    def __gt__(self, other):
        return self.mass > other.mass


@dataclass
class DynamicProgrammingTable:
    table: np.ndarray
    compression_per_cell: int
    precision: float
    tolerance: float
    seq: SequenceInformation
    masses: List[NucleotideMass]

    def __init__(
        self,
        nucleotide_df: pl.DataFrame,
        compression_rate: int,
        tolerance: float,
        precision: float,
        seq: SequenceInformation,
    ):
        self.compression_per_cell = compression_rate
        self.tolerance = tolerance
        self.precision = precision
        self.seq = seq
        self.masses = initialize_nucleotide_masses(nucleotide_df)
        self.table = None

        # Adapt individual modification rates to universal one
        self._adapt_individual_modification_rates_by_universal_one()

        # Initialize table form file (for no alphabet reduction)
        if self.table is None:
            self.table = load_dp_table(
                table_path=set_table_path(precision, compression_rate),
                integer_masses=[mass.mass for mass in self.masses],
            )

    def _adapt_individual_modification_rates_by_universal_one(self):
        for nucleotide_mass in self.masses:
            if not nucleotide_mass.is_modification:
                continue
            if nucleotide_mass.modification_rate > self.seq.modification_rate:
                nucleotide_mass.modification_rate = self.seq.modification_rate
        self._reduce_nucleotide_list()

    def adapt_individual_modification_rates_by_alphabet_reduction(self, alphabet):
        for nucleotide_mass in self.masses:
            if not nucleotide_mass.is_modification:
                continue
            if all(name not in alphabet for name in nucleotide_mass.names):
                nucleotide_mass.modification_rate = 0.0
        self._reduce_nucleotide_list()

    def _reduce_nucleotide_list(self):
        new_masses = [
            mass
            for mass in self.masses
            if mass.mass == 0.0 or mass.modification_rate > 0.0
        ]

        # Return if nucleotide list was not reduced
        if len(new_masses) == len(self.masses):
            return

        # Recompute table
        self.table = set_up_bit_table(
            integer_masses=[mass.mass for mass in new_masses],
            max_mass=max([mass.mass for mass in new_masses]) * MAX_SEQ_LENGTH,
            compression_rate=self.compression_per_cell,
        )

        # Update nucleotide list
        self.masses = new_masses

    def print_masses(self):
        mass_names = []
        for mass in self.masses:
            mass_names += mass.names
        masses = NUCLEOTIDE_DF.sort("nucleoside_mass").filter(
            pl.col("representative").is_in(mass_names)
        )

        print(
            masses.replace_column(
                masses.get_column_index("modification_rate"),
                pl.Series(
                    "modification_rate",
                    [mass.modification_rate for mass in self.masses[1:]],
                ),
            )
        )


def set_table_path(precision, compression_rate):
    # Set path for DP table
    path = f"{TABLE_DIR}/tol_{precision:.0E}.{compression_rate}_per_cell"

    # Create directory for DP table if it does not already exist
    subdir = "/".join(path.split("/")[:-1])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    return path


def initialize_nucleotide_masses(nucleotide_df):
    # Get list of integer masses
    integer_masses = nucleotide_df.get_column("integer_mass").to_list()

    # Add a default weight for easier initialization
    integer_masses += [0]

    # Ensure unique and sorted entries after tolerance correction
    integer_masses = sorted(set(integer_masses))

    # Create dict with all associated nucleotide names for each mass
    names = {
        mass: pl.DataFrame({"integer_mass": mass})
        .join(
            nucleotide_df,
            on="integer_mass",
            how="left",
        )
        .get_column("representative")
        .to_list()
        for mass in nucleotide_df.get_column("integer_mass").to_list()
    }

    # Create dict with indicator whether each mass is associated with a modified base
    is_mod = {
        mass: any(base not in UNMODIFIED_BASES for base in names[mass])
        for mass in nucleotide_df.get_column("integer_mass").to_list()
    }

    # Create dict with the largest associated modification rate for each mass
    rates = {
        mass: max(
            pl.DataFrame({"integer_mass": mass})
            .join(
                nucleotide_df,
                on="integer_mass",
                how="left",
            )
            .get_column("modification_rate")
            .to_list()
        )
        for mass in nucleotide_df.get_column("integer_mass").to_list()
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


def load_dp_table(table_path, integer_masses):
    """
    Load dynamic-programming table if it exists and compute it otherwise.
    """
    # Select compression rate from path string
    compression_rate = int(table_path.split(".")[-1].rstrip("_per_cell"))

    # Select maximum integer mass for which table should be built
    max_mass = max(integer_masses) * MAX_SEQ_LENGTH

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


def compute_sequence_length_bound(dp_table: DynamicProgrammingTable, dir: str) -> int:
    """
    Return bound on length for any sequence that could explain the given mass.
    """
    # Set compression rate
    compression_rate = dp_table.compression_per_cell

    # Set maximum number of modifications
    max_modifications = round(dp_table.seq.modification_rate * dp_table.seq.max_len)

    # Convert the target to an integer for easy operations
    target = int(round(dp_table.seq.su_mass / dp_table.precision, 0))

    # Convert the threshold to integer
    threshold = int(
        np.ceil(dp_table.tolerance * dp_table.seq.obs_mass / dp_table.precision)
    )

    # Initialize memorization dict
    memo = {}

    # Select default value based on desired bound
    match dir:
        case "lower":
            default_bound = dp_table.seq.max_len + 1
        case "upper":
            default_bound = -1
        case _:
            raise NotImplementedError(f"Support for '{dir}' is currently not given.")

    def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
        current_weight = dp_table.masses[current_idx].mass

        # If the result for this state is already computed, return it
        if (total_mass, current_idx) in memo:
            return memo[(total_mass, current_idx)]

        # Return default value for cells outside of table
        if total_mass < 0:
            return default_bound

        # Initialize new counter for valid start in table
        if total_mass == 0:
            return 0

        # Raise error if mass is not in table (due to its size)
        if total_mass >= len(dp_table.table[0]) * compression_rate:
            raise NotImplementedError(
                f"The value {value} is not in the DP table. Extend its "
                f"size if you want to compute larger masses."
            )

        current_value = (
            dp_table[current_idx, total_mass]
            if compression_rate == 1
            else dp_table.table[current_idx, total_mass // compression_rate]
            >> 2 * (compression_rate - 1 - total_mass % compression_rate)
        )

        # Return default value for unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            return default_bound

        # Initialize list of possible bounds
        bounds = [default_bound]

        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            bounds.append(
                backtrack(
                    total_mass,
                    current_idx - 1,
                    max_mods_all,
                    round(
                        dp_table.seq.max_len
                        * dp_table.masses[current_idx - 1].modification_rate
                    ),
                )
            )

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not dp_table.masses[current_idx].is_modification or (
                max_mods_all > 0 and max_mods_ind > 0
            ):
                # Adjust number of still allowed modifications if necessary
                if dp_table.masses[current_idx].is_modification:
                    max_mods_all -= 1
                    max_mods_ind -= 1

                bounds.append(
                    backtrack(
                        total_mass - current_weight,
                        current_idx,
                        max_mods_all,
                        max_mods_ind,
                    )
                    + 1
                )

        # Select result based on desired bound
        match dir:
            case "lower":
                result = min(bounds)
            case "upper":
                result = max(bounds)
            case _:
                raise NotImplementedError(
                    f"Support for '{dir}' is currently not given."
                )

        # Store result in memo
        memo[(total_mass, current_idx)] = result

        return result

    # Compute bounds for all masses within the threshold interval
    solutions = []
    for value in range(
        target - threshold,
        target + threshold + 1,
    ):
        solutions.append(
            backtrack(
                value,
                len(dp_table.masses) - 1,
                max_modifications,
                round(dp_table.seq.max_len * dp_table.masses[-1].modification_rate),
            )
        )

    # Return solution based on desired bound and replace default value if selected
    match dir:
        case "lower":
            opt_len = min(solutions)
            if opt_len == default_bound:
                opt_len = 1
        case "upper":
            opt_len = max(solutions)
            if opt_len == default_bound:
                opt_len = dp_table.seq.max_len
        case _:
            raise NotImplementedError(f"Support for '{dir}' is currently not given.")

    return opt_len
