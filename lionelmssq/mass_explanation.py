from dataclasses import dataclass
from typing import Set, Tuple, List
from itertools import product, combinations_with_replacement, chain
from platformdirs import user_cache_dir

import polars as pl
import pathlib
import numpy as np
import os

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import MATCHING_THRESHOLD
from lionelmssq.masses import MAX_MASS
from lionelmssq.masses import BREAKAGES
from lionelmssq.masses import COMPRESSION_RATE


# Set OS-independent cache directory for DP tables
TABLE_DIR = user_cache_dir(
    appname="lionelmssq/dp_table", version="1.0", ensure_exists=True
)

UNMODIFIED_BASES = ["A", "C", "G", "U"]

match COMPRESSION_RATE:
    case 4:
        settings = {
            "type": np.uint8,
            "init": 0xC0,
            "alt_first": 0xAA,
            "alt_sec": 0x55,
            "full": np.uint8(0xFF),
        }
    case 8:
        settings = {
            "type": np.uint16,
            "init": 0xC000,
            "alt_first": 0xAAAA,
            "alt_sec": 0x5555,
            "full": np.uint16(0xFFFF),
        }
    case 16:
        settings = {
            "type": np.uint32,
            "init": 0xC0000000,
            "alt_first": 0xAAAAAAAA,
            "alt_sec": 0x55555555,
            "full": np.uint32(0xFFFFFFFF),
        }
    case 32:
        settings = {
            "type": np.uint64,
            "init": 0xC000000000000000,
            "alt_first": 0xAAAAAAAAAAAAAAAA,
            "alt_sec": 0x5555555555555555,
            "full": np.uint64(0xFFFFFFFFFFFFFFFF),
        }


@dataclass
class MassExplanations:
    breakage: str
    explanations: Set[Tuple[str]]


@dataclass
class NucleotideMass:
    mass: int
    names: List[str]
    is_modification: bool


@dataclass
class DynamicProgrammingTable:
    table: np.ndarray
    compression_per_cell: int
    precision: float
    tolerance: float
    masses: List[NucleotideMass]

    def __init__(
        self,
        nucleotide_dataframe,
        reduced_table=False,
        reduced_set=False,
        compression_rate=COMPRESSION_RATE,
        tolerance=MATCHING_THRESHOLD,
        precision=TOLERANCE,
    ):
        self.compression_per_cell = compression_rate
        self.precision = precision
        self.tolerance = tolerance
        self.masses = initialize_nucleotide_masses(nucleotide_dataframe)
        self.table = load_dp_table(
            compression_rate,
            table_path=set_table_path(reduced_table, reduced_set, precision),
            integer_masses=[mass.mass for mass in self.masses],
        )


def set_table_path(reduce_table, reduce_set, precision):
    # Set path for DP table
    path = (
        f"{TABLE_DIR}/{'reduced' if reduce_table else 'full'}_table."
        f"{'reduced' if reduce_set else 'full'}_set/"
        f"tol_{precision:.0E}"
    )

    # Create directory for DP table if it does not already exist
    subdir = "/".join(path.split("/")[:-1])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    return path


def initialize_nucleotide_masses(nucleotide_df=EXPLANATION_MASSES):
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
        mass: any(
            base not in UNMODIFIED_BASES
            for base in pl.DataFrame({"tolerated_integer_masses": mass})
            .join(
                nucleotide_df,
                on="tolerated_integer_masses",
                how="left",
            )
            .get_column("nucleoside")
            .to_list()
        )
        for mass in nucleotide_df.get_column("tolerated_integer_masses").to_list()
    }

    # Return list of NucleotideMass instances
    return [
        NucleotideMass(mass, names[mass], is_mod[mass])
        if mass != 0
        else NucleotideMass(0, [], False)
        for mass in integer_masses
    ]


MASS_NAMES = {
    mass: pl.DataFrame({"tolerated_integer_masses": mass})
    .join(
        EXPLANATION_MASSES,
        on="tolerated_integer_masses",
        how="left",
    )
    .get_column("nucleoside")
    .to_list()
    for mass in EXPLANATION_MASSES.get_column("tolerated_integer_masses").to_list()
}

IS_MOD = {
    mass: any(
        base not in UNMODIFIED_BASES
        for base in pl.DataFrame({"tolerated_integer_masses": mass})
        .join(
            EXPLANATION_MASSES,
            on="tolerated_integer_masses",
            how="left",
        )
        .get_column("nucleoside")
        .to_list()
    )
    for mass in EXPLANATION_MASSES.get_column("tolerated_integer_masses").to_list()
}


def set_up_bit_table(integer_masses):
    """
    Calculate complete bit-representation mass table with dynamic programming.
    """
    # Initialize bit-representation numpy table
    max_col = int(np.ceil((MAX_MASS + 1) / COMPRESSION_RATE))
    dp_table = np.zeros((len(integer_masses), max_col), dtype=settings["type"])
    dp_table[0, 0] = settings["init"]

    # Fill DP table row-wise
    for i in range(1, len(integer_masses)):
        # Case: Start new row (i.e. move on to new nucleotide) by initializing reachable cells from before
        dp_table[i] = [
            ((val | (val >> 1)) & settings["alt_sec"]) for val in dp_table[i - 1]
        ]

        # Define number of cells to move (step) and bit shift in a cell (shift)
        step = int(integer_masses[i] / COMPRESSION_RATE)
        shift = integer_masses[i] % COMPRESSION_RATE

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
                    (dp_table[i, j] << 2 * (COMPRESSION_RATE - shift) << 1)
                    | (dp_table[i, j] << 2 * (COMPRESSION_RATE - shift))
                )

    # Adjust last column for unused cells
    dp_table[:, -1] &= settings["full"] << 2 * (max_col - (MAX_MASS + 1) % max_col)

    return dp_table


def set_up_mass_table(integer_masses):
    """
    Calculate complete mass table with dynamic programming.
    """
    # Initialize numpy table
    dp_table = np.zeros((len(integer_masses), MAX_MASS + 1), dtype=np.uint8)
    dp_table[0, 0] = 3.0

    # Fill DP table row-wise
    for i in range(1, len(integer_masses)):
        # Case: Start new row (i.e. move on to new nucleoside) by initializing
        # reachable cells from before
        dp_table[i] = [int(val != 0.0) for val in dp_table[i - 1]]

        # Case: Add more of current nucleoside
        for j in range(MAX_MASS + 1):
            # If cell is not reachable, skip it
            if dp_table[i, j] == 0.0:
                continue

            # Add another nucleoside if possible
            if integer_masses[i] + j <= MAX_MASS:
                dp_table[i, j + integer_masses[i]] += 2.0

    return dp_table


def load_dp_table(compression_rate, table_path, integer_masses):
    """
    Load dynamic-programming table if it exists and compute it otherwise.
    """
    # Compute and save bit-representation DP table if not existing
    if not pathlib.Path(f"{table_path}.{compression_rate}_per_cell.npy").is_file():
        print("Table not found")
        dp_table = (
            set_up_mass_table(integer_masses)
            if compression_rate == 1
            else (set_up_bit_table(integer_masses))
        )
        np.save(f"{table_path}.{compression_rate}_per_cell", dp_table)

    # Read DP table
    return np.load(f"{table_path}.{compression_rate}_per_cell.npy")


def is_valid_mass(
    mass: float,
    dp_table,
    breakages=BREAKAGES,
    compression_rate=COMPRESSION_RATE,
    threshold=MATCHING_THRESHOLD,
) -> bool:
    # Ensure that all breakage weights have a associated breakage
    breakages = {
        breakage_weight: breakage
        for breakage_weight, breakage in breakages.items()
        if len(breakage) > 0
    }

    # Convert the target to an integer for easy operations
    target = int(round(mass / TOLERANCE, 0))

    # Set matching threshold based on target mass
    threshold = int(np.ceil(threshold * target))

    current_idx = len(dp_table) - 1
    for breakage_weight in breakages:
        for value in range(
            target - breakage_weight - threshold,
            target - breakage_weight + threshold + 1,
        ):
            # Skip non-positive masses
            if value <= 0:
                continue

            # Raise error if mass is not in table (due to its size)
            if value >= len(dp_table[0]) * compression_rate:
                raise NotImplementedError(
                    f"The value {value} is not in the DP table. Extend its "
                    f"size if you want to compute larger masses."
                )

            current_value = (
                dp_table[current_idx, value]
                if compression_rate == 1
                else dp_table[current_idx, value // compression_rate]
                >> 2 * (compression_rate - 1 - value % compression_rate)
            )

            # Skip unreachable cells
            if compression_rate != 1 and current_value % compression_rate == 0.0:
                continue

            # Return True when mass corresponds to valid entry in table
            if current_value % 2 == 1 or (current_value >> 1) % 2 == 1:
                return True
    return False


def explain_mass_with_dp(
    mass: float,
    dp_table: DynamicProgrammingTable,
    with_memo: bool,
    max_modifications=np.inf,
    compression_rate=None,
    threshold=None,
) -> list[MassExplanations]:
    """
    Return all possible combinations of nucleosides that could sum up to the given mass.
    """
    if threshold is None:
        threshold = dp_table.tolerance

    if compression_rate is None:
        compression_rate = dp_table.compression_per_cell

    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set matching threshold based on target mass
    threshold = int(np.ceil(threshold * target))

    memo = {}

    def backtrack_with_memo(total_mass, current_idx, max_mods):
        current_weight = dp_table.masses[current_idx].mass

        # If the result for this state is already computed, return it
        if (total_mass, current_idx) in memo:
            return memo[(total_mass, current_idx)]

        # Return empty list for cells outside of table
        if total_mass < 0:
            return []

        # Initialize a new nucleoside set for a valid start in table
        if total_mass == 0:
            return [[]]

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

        # Return empty list for unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack_with_memo(total_mass, current_idx - 1, max_mods)

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not dp_table.masses[current_idx].is_modification or max_mods > 0:
                # Adjust number of still allowed modifications if necessary
                if dp_table.masses[current_idx].is_modification:
                    max_mods -= 1

                solutions += [
                    entry + [current_weight]
                    for entry in backtrack_with_memo(
                        total_mass - current_weight, current_idx, max_mods
                    )
                ]

        # Store result in memo
        memo[(total_mass, current_idx)] = solutions

        return solutions

    def backtrack(total_mass, current_idx, max_mods):
        current_weight = dp_table.masses[current_idx].mass

        # Return empty list for cells outside of table
        if total_mass < 0:
            return []

        # Initialize a new nucleoside set for a valid start in table
        if total_mass == 0:
            return [[]]

        current_value = (
            dp_table.table[current_idx, total_mass]
            if compression_rate == 1
            else dp_table.table[current_idx, total_mass // compression_rate]
            >> 2 * (compression_rate - 1 - total_mass % compression_rate)
        )

        # Return empty list for unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack(total_mass, current_idx - 1, max_mods)

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not dp_table.masses[current_idx].is_modification or max_mods > 0:
                # Adjust number of still allowed modifications if necessary
                if dp_table.masses[current_idx].is_modification:
                    max_mods -= 1

                solutions += [
                    entry + [current_weight]
                    for entry in backtrack(
                        total_mass - current_weight, current_idx, max_mods
                    )
                ]

        return solutions

    solution_tolerated_integer_masses = {}
    for breakage_weight in BREAKAGES:
        # Compute all valid solutions within the threshold interval
        solutions = []
        for value in range(
            target - breakage_weight - threshold,
            target - breakage_weight + threshold + 1,
        ):
            solutions += (
                backtrack_with_memo(value, len(dp_table.masses) - 1, max_modifications)
                if with_memo
                else backtrack(value, len(dp_table.masses) - 1, max_modifications)
            )

        # Add valid solutions to dictionary of breakpoint options
        for breakage in BREAKAGES[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    # Convert the DP table masses to their respective nucleoside names
    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
        solution_names = set()
        if len(solution_tolerated_integer_masses[breakage]) > 0:
            for combo in solution_tolerated_integer_masses[breakage]:
                if len(combo) == 0:
                    continue
                solution_names.update(
                    [
                        tuple(chain.from_iterable(entry))
                        for entry in list(
                            product(
                                *[
                                    list(
                                        combinations_with_replacement(
                                            MASS_NAMES[mass], combo.count(mass)
                                        )
                                    )
                                    for mass in [
                                        combo[idx]
                                        for idx in range(len(combo))
                                        if idx == 0 or combo[idx - 1] != combo[idx]
                                    ]
                                ]
                            )
                        )
                    ]
                )
        # Add desired Dataclass object to list
        explanations.append(
            MassExplanations(breakage=breakage, explanations=solution_names)
        )

    # Return list of explanations
    return explanations


def explain_mass(
    mass: float,
    max_modifications=np.inf,
    matching_threshold=MATCHING_THRESHOLD,
) -> MassExplanations:
    """
    Returns all the possible combinations of nucleosides that could sum up to the given mass.
    """

    tolerated_integer_masses = EXPLANATION_MASSES.get_column(
        "tolerated_integer_masses"
    ).to_list()

    # Convert the target to an integer for easy operations
    target = int(round(mass / TOLERANCE, 0))

    # Set matching threshold based on target mass
    matching_threshold = int(np.ceil(matching_threshold * target))

    # Ensure unique and sorted entries after tolerance correction
    tolerated_integer_masses = sorted(set(tolerated_integer_masses))

    # Memoization dictionary to store results for a given target
    memo = {}

    def dp(remaining, start, used_mods):
        # If too many modifications are used, return empty list
        if used_mods > max_modifications:
            return []

        # If the result for this state is already computed, return it
        if (remaining, start) in memo:
            return memo[(remaining, start)]

        # Base case: if abs(target) is less than threshold, return a list with one empty combination
        if abs(remaining) <= matching_threshold:
            return [[]]

        # Base case: if target is zero, return a list with one empty combination
        if remaining == 0:
            return [[]]

        # Base case: if target is negative, no combinations possible
        if remaining < 0:
            return []

        # List to store all combinations for this state
        combinations = []

        # Try each tolerated_integer_mass starting from the current position to avoid duplicates
        for i in range(start, len(tolerated_integer_masses)):
            tolerated_integer_mass = tolerated_integer_masses[i]
            # Recurse with reduced target and the current tolerated_integer_mass
            sub_combinations = dp(
                remaining - tolerated_integer_mass,
                i,
                used_mods + 1 if IS_MOD[tolerated_integer_mass] else used_mods,
            )
            # Add current tolerated_integer_mass to all sub-combinations
            for combo in sub_combinations:
                combinations.append([tolerated_integer_mass] + combo)

        # Store result in memo
        memo[(remaining, start)] = combinations

        return combinations

    solution_tolerated_integer_masses = {}
    for breakage_weight in BREAKAGES:
        # Start with the full target and all tolerated_integer_masses
        solutions = dp(target - breakage_weight, 0, 0)
        for breakage in BREAKAGES[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    # Convert the tolerated_integer_masses to the respective nucleoside names
    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
        solution_names = set()
        if len(solution_tolerated_integer_masses[breakage]) > 0:
            for combo in solution_tolerated_integer_masses[breakage]:
                if len(combo) == 0:
                    continue
                solution_names.update(
                    [
                        tuple(chain.from_iterable(entry))
                        for entry in list(
                            product(
                                *[
                                    list(
                                        combinations_with_replacement(
                                            MASS_NAMES[mass], combo.count(mass)
                                        )
                                    )
                                    for mass in [
                                        combo[idx]
                                        for idx in range(len(combo))
                                        if idx == 0 or combo[idx - 1] != combo[idx]
                                    ]
                                ]
                            )
                        )
                    ]
                )
        # Add desired Dataclass object to list
        explanations.append(
            MassExplanations(breakage=breakage, explanations=solution_names)
        )

    # Return list of explanations
    return explanations
