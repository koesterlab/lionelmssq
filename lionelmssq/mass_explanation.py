from dataclasses import dataclass
from typing import Set, Tuple
from itertools import product

# from lionelmssq.masses import MASSES
import polars as pl

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import MATCHING_THRESHOLD


@dataclass
class MassExplanations:
    explanations: Set[Tuple[str]]


def explain_mass(
    mass: float, explanation_masses=EXPLANATION_MASSES
) -> MassExplanations:
    """
    Returns all the possible combinations of nucleosides that could sum up to the given mass.
    """

    tolerated_integer_masses = pl.Series(
        explanation_masses.select(pl.col("tolerated_integer_masses"))
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
        # if abs(remaining) < MATCHING_THRESHOLD:
        # Base case: if the relative error between the target and our estimate is less than the MATCHING_THRESHOLD, return a list with one empty combination
        if abs(remaining / target) < MATCHING_THRESHOLD:
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
            explanation_masses.select(pl.col("nucleoside")).dtypes[0], pl.List
        ):
            # When using "agg function" in the group_by (masses.py), we can get all the nucleosides with the same mass, then the following needs to be used:
            solution_names.update(
                product(
                    *combo_df.join(
                        explanation_masses, on="tolerated_integer_masses", how="left"
                    )
                    .get_column("nucleoside")
                    .to_list()
                )
            )

        elif isinstance(
            explanation_masses.select(pl.col("nucleoside")).dtypes[0], pl.String
        ):
            # When using the "first" function in the group_by (masses.py), we can only get the first nucleoside with the given mass, then the following needs to be used:
            solution_names.add(
                tuple(
                    combo_df.join(
                        explanation_masses, on="tolerated_integer_masses", how="left"
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
