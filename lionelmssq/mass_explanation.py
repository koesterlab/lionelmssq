from dataclasses import dataclass
from typing import Set, Tuple
from itertools import product

# from lionelmssq.masses import MASSES
import polars as pl
import pathlib
import numpy as np

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import TABLE_PATH
from lionelmssq.masses import MATCHING_THRESHOLD
from lionelmssq.masses import MAX_MASS
from lionelmssq.masses import BREAKAGES
from lionelmssq.masses import COMPRESSION_RATE

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


def set_up_bit_table():
    """
    Calculate complete bit-representation mass table with dynamic programming.
    """
    integer_masses = pl.Series(
        EXPLANATION_MASSES.select(pl.col("tolerated_integer_masses"))
    ).to_list()

    # Add a default weight for easier initialization
    integer_masses += [0]

    # Ensure unique entries after tolerance correction
    integer_masses = list(set(integer_masses))

    # Sort the tolerated_integer_masses for better overview
    integer_masses.sort()

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


def set_up_mass_table():
    """
    Calculate complete mass table with dynamic programming.
    """
    integer_masses = pl.Series(
        EXPLANATION_MASSES.select(pl.col("tolerated_integer_masses"))
    ).to_list()

    # Add a default weight for easier initialization
    integer_masses += [0]

    # Ensure unique entries after tolerance correction
    integer_masses = list(set(integer_masses))

    # Sort the tolerated_integer_masses for better overview
    integer_masses.sort()

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


def explain_mass_with_dp(mass: float, with_memo: bool) -> list[MassExplanations]:
    """
    Return all possible combinations of nucleosides that could sum up to the given mass.
    """
    tolerated_integer_masses = pl.Series(
        EXPLANATION_MASSES.select(pl.col("tolerated_integer_masses"))
    ).to_list()

    # Convert the targets and tolerated_integer_masses to integers for easy operations
    target = int(round(mass / TOLERANCE, 0))

    # Add a default weight for easier initialization
    tolerated_integer_masses += [0]

    # Ensure unique entries after tolerance correction
    tolerated_integer_masses = list(set(tolerated_integer_masses))

    # Sort the tolerated_integer_masses, makes life easier
    tolerated_integer_masses.sort()

    # Compute and save bit-representation DP table if not existing
    if not pathlib.Path(f"{TABLE_PATH}.{COMPRESSION_RATE}_per_cell.npy").is_file():
        print("Table not found")
        dp_table = set_up_mass_table() if COMPRESSION_RATE == 1 else (
            set_up_bit_table())
        np.save(f"{TABLE_PATH}.{COMPRESSION_RATE}_per_cell", dp_table)

    # Read DP table
    dp_table = np.load(
        f"{TABLE_PATH}.{COMPRESSION_RATE}_per_cell.npy"
    )

    memo = {}

    def backtrack_with_memo(total_mass, current_idx):
        current_weight = tolerated_integer_masses[current_idx]

        # If the result for this state is already computed, return it
        if (total_mass, current_idx) in memo:
            return memo[(total_mass, current_idx)]

        # Return empty list for cells outside of table
        if total_mass < 0:
            return []

        # Initialize a new nucleoside set for a valid start in table
        if total_mass == 0:
            return [[]]

        current_value = (
            dp_table[current_idx, total_mass]
            if COMPRESSION_RATE == 1 else
            dp_table[current_idx, total_mass // COMPRESSION_RATE]
            >> 2 * (COMPRESSION_RATE - 1 - total_mass % COMPRESSION_RATE)
        )

        # Return empty list for unreachable cells
        if current_value % COMPRESSION_RATE == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack_with_memo(total_mass, current_idx - 1)

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            solutions += [
                entry + [current_weight]
                for entry in backtrack_with_memo(
                    total_mass - current_weight, current_idx
                )
            ]

        # Store result in memo
        memo[(total_mass, current_idx)] = solutions

        return solutions

    def backtrack(total_mass, current_idx):
        current_weight = tolerated_integer_masses[current_idx]

        # Return empty list for cells outside of table
        if total_mass < 0:
            return []

        # Initialize a new nucleoside set for a valid start in table
        if total_mass == 0:
            return [[]]

        current_value = (
            dp_table[current_idx, total_mass]
            if COMPRESSION_RATE == 1 else
            dp_table[current_idx, total_mass // COMPRESSION_RATE]
            >> 2 * (COMPRESSION_RATE - 1 - total_mass % COMPRESSION_RATE)
        )

        # Return empty list for unreachable cells
        if current_value % COMPRESSION_RATE == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack(total_mass, current_idx - 1)

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            solutions += [
                entry + [current_weight]
                for entry in backtrack(total_mass - current_weight, current_idx)
            ]

        return solutions

    solution_tolerated_integer_masses = {}
    for breakage_weight in BREAKAGES:
        # Compute all valid solutions within an interval of MATCHING_THRESHOLD
        solutions = []
        for value in range(
            target - breakage_weight - MATCHING_THRESHOLD,
            target - breakage_weight + MATCHING_THRESHOLD,
        ):
            solutions += (
                backtrack_with_memo(value, len(tolerated_integer_masses) - 1)
                if with_memo
                else backtrack(value, len(tolerated_integer_masses) - 1)
            )

        # Add valid solutions to dictionary of breakpoint options
        for breakage in BREAKAGES[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Convert the tolerated_integer_masses to the respective nucleoside names
        solutions = set()

        # Store the nucleoside names (as tuples of strings (alphabets)) for the given tolerated_integer_masses in the set solution_names
        for combo in solution_tolerated_integer_masses[breakage]:
            combo_df = pl.DataFrame({"tolerated_integer_masses": combo})

            # Determines the type of data in the column nucleosides
            if isinstance(
                EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.List
            ):
                # When using "agg function" in the group_by (masses.py), we can get all the nucleosides with the same mass, then the following needs to be used:
                solutions.update(
                    product(
                        *combo_df.join(
                            EXPLANATION_MASSES,
                            on="tolerated_integer_masses",
                            how="left",
                        )
                        .get_column("nucleoside")
                        .to_list()
                    )
                )

            elif isinstance(
                EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.String
            ):
                # When using the "first" function in the group_by (masses.py), we can only get the first nucleoside with the given mass, then the following needs to be used:
                solutions.add(
                    tuple(
                        combo_df.join(
                            EXPLANATION_MASSES,
                            on="tolerated_integer_masses",
                            how="left",
                        )
                        .get_column("nucleoside")
                        .to_list()
                    )
                )
        # Add desired Dataclass object to list
        explanations.append(MassExplanations(breakage=breakage, explanations=solutions))

    # Return list of explanations
    return explanations


def explain_mass(mass: float) -> list[MassExplanations]:
    """
    Returns all the possible combinations of nucleosides that could sum up to the given mass.
    """

    tolerated_integer_masses = pl.Series(
        EXPLANATION_MASSES.select(pl.col("tolerated_integer_masses"))
    ).to_list()

    # Convert the targets and tolerated_integer_masses to integers for easy operations
    target = int(round(mass / TOLERANCE, 0))

    # Sort the tolerated_integer_masses, makes life easier
    tolerated_integer_masses.sort()

    # Memoization dictionary to store results for a given target
    memo = {}

    def dp(remaining, start):
        # TODO: Can check if there are any possible ways to achieve remaining using tolerated_integer_masses[start:], if NOT, don't execute the dict check, nor store the values (empty list) in the dictionary!

        # If the result for this state is already computed, return it
        if (remaining, start) in memo:
            return memo[(remaining, start)]

        # Base case: if abs(target) is less than MATCHING_THRESHOLD, return a list with one empty combination
        if abs(remaining) < MATCHING_THRESHOLD:
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
            sub_combinations = dp(remaining - tolerated_integer_mass, i)
            # Add current tolerated_integer_mass to all sub-combinations
            for combo in sub_combinations:
                combinations.append([tolerated_integer_mass] + combo)

        # Store result in memo
        memo[(remaining, start)] = combinations

        return combinations

    solution_tolerated_integer_masses = {}
    for breakage_weight in BREAKAGES:
        # Start with the full target and all tolerated_integer_masses
        solutions = dp(target - breakage_weight, 0)
        for breakage in BREAKAGES[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Convert the tolerated_integer_masses to the respective nucleoside names
        solutions = set()

        # Store the nucleoside names (as tuples of strings (alphabets)) for the given tolerated_integer_masses in the set solution_names
        for combo in solution_tolerated_integer_masses[breakage]:
            combo_df = pl.DataFrame({"tolerated_integer_masses": combo})

            # Determines the type of data in the column nucleosides
            if isinstance(
                EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.List
            ):
                # When using "agg function" in the group_by (masses.py), we can get all the nucleosides with the same mass, then the following needs to be used:
                solutions.update(
                    product(
                        *combo_df.join(
                            EXPLANATION_MASSES,
                            on="tolerated_integer_masses",
                            how="left",
                        )
                        .get_column("nucleoside")
                        .to_list()
                    )
                )

            elif isinstance(
                EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.String
            ):
                # When using the "first" function in the group_by (masses.py), we can only get the first nucleoside with the given mass, then the following needs to be used:
                solutions.add(
                    tuple(
                        combo_df.join(
                            EXPLANATION_MASSES,
                            on="tolerated_integer_masses",
                            how="left",
                        )
                        .get_column("nucleoside")
                        .to_list()
                    )
                )
        # Add desired Dataclass object to list
        explanations.append(MassExplanations(breakage=breakage, explanations=solutions))

    # Return list of explanations
    return explanations
