from dataclasses import dataclass
from typing import Set, Tuple
from itertools import product

# from lionelmssq.masses import MASSES
import polars as pl
import pathlib
import math

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import MATCHING_THRESHOLD


MAX_MASS = EXPLANATION_MASSES.select(pl.col(
    "tolerated_integer_masses")).max().item() * 35

@dataclass
class MassExplanations:
    explanations: Set[Tuple[str]]


def set_up_mass_table():
    """
    Calculates complete mass table with dynamic programming.
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

    print(integer_masses)

    # Initialize DP table
    dp_table = pl.DataFrame(
        {"name": [0], "0": [[0]],
        **{str(val): None for val in range(1, MAX_MASS+1)}}).with_columns(pl.all().exclude("name").cast(pl.List(pl.Int64), strict=False))

    # print(dp_table)

    for idx in range(1, len(integer_masses)):
        new_row = pl.DataFrame({"name": integer_masses[idx],
                                **{str(val): None for val in
                                   range(MAX_MASS+1)}})
        last_row = dp_table.filter(
            pl.col("name") == integer_masses[idx-1])
        valid_entries = last_row.select(
            pl.col("*").is_not_null().exclude("name")).unpivot().filter(
            pl.col.value).to_series().cast(pl.Int64)
        for multiplication_factor in range(
                math.floor(MAX_MASS / integer_masses[idx])+1):
            if multiplication_factor > 0:
                valid_entries = valid_entries.filter(
                    valid_entries <= MAX_MASS-multiplication_factor *
                    integer_masses[idx])
            new_row = new_row.with_columns(
                pl.col(valid_entries.map_elements(function=lambda x: str(
                    x+multiplication_factor * integer_masses[idx]),
                                                  return_dtype=pl.String))
                .map_elements(function=lambda x: [
                    multiplication_factor] if x is None else x+[
                    multiplication_factor], skip_nulls=False,
                              return_dtype=pl.List(pl.Int64)))
        dp_table = dp_table.vstack(new_row)

    return dp_table


def explain_mass_with_dp(mass: float) -> MassExplanations:
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

    # Compute and save DP table if not existing
    table_path = f"dp_table_with_tolerance_{TOLERANCE}.json"
    # print(table_path)

    if not pathlib.Path(table_path).is_file():
        print('Table not found')
        dp_table = set_up_mass_table()
        dp_table.write_json(table_path)

    # print("Table build")

    # Read DP table
    dp_table = pl.read_json(table_path)
    dp_table.select(pl.col("name", *[str(val) for val in range(MAX_MASS+1)]))

    print(dp_table.null_count().pipe(sum).item())
    print(dp_table.null_count().pipe(sum).item()+dp_table.select(
        pl.col("*").exclude("name")).count().pipe(sum).item())
    dp_table.null_count().pipe(sum).item() / (
                dp_table.select(pl.col("*").exclude("name")).count().pipe(
                    sum).item()+dp_table.null_count().pipe(sum).item())

    def backtrack(total_mass, current_idx):
        current_weight = tolerated_integer_masses[current_idx]
        if total_mass == 0:
            return [[]]
        if dp_table.select(pl.col(str(total_mass)).filter(pl.col(
                "name")==current_weight)).item() is None:
            return []
        valid_factors = dp_table.select(pl.col(str(total_mass)).filter(
            pl.col("name") == current_weight)).item().to_list()
        solutions = []
        for factor in valid_factors:
            solutions += [[current_weight] * factor+entry for entry in
                          backtrack(
                              total_mass-factor * current_weight,
                              current_idx-1)]
        return solutions

    # Compute all valid solutions within an interval of MATCHING_THRESHOLD
    solution_tolerated_integer_masses = []
    for value in range(target-MATCHING_THRESHOLD, target+MATCHING_THRESHOLD):
        solution_tolerated_integer_masses += backtrack(
            value, len(tolerated_integer_masses)-1)

    print(solution_tolerated_integer_masses)

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

    # Convert the targets and tolerated_integer_masses to integers for easy opearations
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

        # Determines the type of data in the column nucleosdies
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
