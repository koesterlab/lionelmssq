from dataclasses import dataclass
from typing import Set, Tuple
from itertools import product

# from lionelmssq.masses import MASSES
import polars as pl
import pathlib
import math
import numpy as np

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import TABLE_PATH
from lionelmssq.masses import MATCHING_THRESHOLD
from lionelmssq.masses import MAX_MASS

@dataclass
class MassExplanations:
    explanations: Set[Tuple[str]]


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
    dp_table = np.zeros((len(integer_masses), MAX_MASS+1), dtype=np.uint8)
    dp_table[0, 0] = 3.0

    # Fill DP table row-wise
    for i in range(1, len(integer_masses)):
        # Case: Start new row (i.e. move on to new nucleoside) by initializing
        # reachable cells from before
        dp_table[i] = [int(val != 0.0) for val in dp_table[i-1]]

        # Case: Add more of current nucleoside
        for j in range(MAX_MASS+1):
            # If cell is not reachable, skip it
            if dp_table[i, j] == 0.0:
                continue

            # Add another nucleoside if possible
            if integer_masses[i]+j <= MAX_MASS:
                dp_table[i, j + integer_masses[i]] += 2.0

    return dp_table


def explain_mass_with_dp(mass: float, with_memo: bool) -> MassExplanations:
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

    # Compute and save DP table if not existing
    if not pathlib.Path(f"{TABLE_PATH}.npy").is_file():
        print('Table not found')
        dp_table = set_up_mass_table()
        np.save(TABLE_PATH, dp_table)

    # Read DP table
    dp_table = np.load(f"{TABLE_PATH}.npy")

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

        # Return empty list for unreachable cells
        current_value = dp_table[current_idx, total_mass]
        if current_value == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack_with_memo(total_mass, current_idx-1)

        # Backtrack to the next left-side column if possible
        if current_value >= 2.0:
            solutions += [[current_weight]+entry for entry in
                          backtrack_with_memo(total_mass-current_weight,
                                        current_idx)]

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

        # Return empty list for unreachable cells
        current_value = dp_table[current_idx, total_mass]
        if current_value == 0.0:
            return []

        solutions = []
        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            solutions += backtrack(total_mass, current_idx-1)

        # Backtrack to the next left-side column if possible
        if current_value >= 2.0:
            solutions += [[current_weight]+entry for entry in
                          backtrack(total_mass-current_weight, current_idx)]

        return solutions

    # Compute all valid solutions within an interval of MATCHING_THRESHOLD
    solution_tolerated_integer_masses = []
    for value in range(target-MATCHING_THRESHOLD, target+MATCHING_THRESHOLD):
        solution_tolerated_integer_masses += backtrack_with_memo(
            value, len(tolerated_integer_masses)-1) if with_memo \
            else backtrack(value, len(tolerated_integer_masses)-1)

    # Convert the tolerated_integer_masses to the respective nucleoside names
    solution_names = set()

    # Store the nucleoside names (as tuples of strings (alphabets)) for the given tolerated_integer_masses in the set solution_names
    for combo in solution_tolerated_integer_masses:
        combo_df = pl.DataFrame({"tolerated_integer_masses": combo})

        # Determines the type of data in the column nucleosides
        if isinstance(
            EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.List
        ):
            # When using "agg function" in the group_by (masses.py), we can get all the nucleosides with the same mass, then the following needs to be used:
            solution_names.update(
                product(
                    *combo_df.join(
                        EXPLANATION_MASSES, on="tolerated_integer_masses", how="left"
                    )
                    .get_column("nucleoside")
                    .to_list()
                )
            )

        elif isinstance(
            EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.String
        ):
            # When using the "first" function in the group_by (masses.py), we can only get the first nucleoside with the given mass, then the following needs to be used:
            solution_names.add(
                tuple(
                    combo_df.join(
                        EXPLANATION_MASSES, on="tolerated_integer_masses", how="left"
                    )
                    .get_column("nucleoside")
                    .to_list()
                )
            )

    # Return the desired Dataclass object
    return MassExplanations(explanations=solution_names)


def explain_mass(mass: float) -> MassExplanations:
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

    # Start with the full target and all tolerated_integer_masses
    solution_tolerated_integer_masses = dp(target, 0)

    # Convert the tolerated_integer_masses to the respective nucleoside names
    solution_names = set()

    # Store the nucleoside names (as tuples of strings (alphabets)) for the given tolerated_integer_masses in the set solution_names
    for combo in solution_tolerated_integer_masses:
        combo_df = pl.DataFrame({"tolerated_integer_masses": combo})

        # Determines the type of data in the column nucleosides
        if isinstance(
            EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.List
        ):
            # When using "agg function" in the group_by (masses.py), we can get all the nucleosides with the same mass, then the following needs to be used:
            solution_names.update(
                product(
                    *combo_df.join(
                        EXPLANATION_MASSES, on="tolerated_integer_masses", how="left"
                    )
                    .get_column("nucleoside")
                    .to_list()
                )
            )

        elif isinstance(
            EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0], pl.String
        ):
            # When using the "first" function in the group_by (masses.py), we can only get the first nucleoside with the given mass, then the following needs to be used:
            solution_names.add(
                tuple(
                    combo_df.join(
                        EXPLANATION_MASSES, on="tolerated_integer_masses", how="left"
                    )
                    .get_column("nucleoside")
                    .to_list()
                )
            )

            # Alternate old solution (with loop):
            # temp_list  = []
            # for tolerated_integer_mass_val in combo:
            #   temp_list.append(EXPLANATION_MASSES.filter(pl.col("tolerated_integer_masses") == tolerated_integer_mass_val).select(pl.col("nucleoside")).item())
            # solution_names.add(tuple(temp_list))

    # Return the desired Dataclass object
    return MassExplanations(explanations=solution_names)
