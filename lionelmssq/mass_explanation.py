from dataclasses import dataclass
from typing import List, Set, Tuple
from itertools import product, combinations_with_replacement, chain

import polars as pl
import numpy as np

from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import EXPLANATION_MASSES, UNMODIFIED_BASES


@dataclass
class MassExplanations:
    explanations: Set[Tuple[str]]


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


def is_valid_mass(
    mass: float,
    dp_table: DynamicProgrammingTable,
    threshold: float = None,
) -> bool:
    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    compression_rate = dp_table.compression_per_cell

    current_idx = len(dp_table.table) - 1
    for value in range(target - threshold, target + threshold + 1):
        # Skip non-positive masses
        if value <= 0:
            continue

        # Raise error if mass is not in table (due to its size)
        if value >= len(dp_table.table[0]) * compression_rate:
            raise NotImplementedError(
                f"The value {value} is not in the DP table. Extend its "
                f"size if you want to compute larger masses."
            )

        current_value = (
            dp_table.table[current_idx, value]
            if compression_rate == 1
            else dp_table.table[current_idx, value // compression_rate]
            >> 2 * (compression_rate - 1 - value % compression_rate)
        )

        # Skip unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            continue

        # Return True when mass corresponds to valid entry in table
        if current_value % 2 == 1 or (current_value >> 1) % 2 == 1:
            return True
    return False


def explain_mass_with_table(
    mass: float,
    dp_table: DynamicProgrammingTable,
    seq_len: int,
    max_modifications=np.inf,
    compression_rate=None,
    threshold=None,
    with_memo=True,
) -> MassExplanations:
    """
    Return all possible combinations of nucleosides that could sum up to the given mass.
    """
    if compression_rate is None:
        compression_rate = dp_table.compression_per_cell

    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    memo = {}

    def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
        current_weight = dp_table.masses[current_idx].mass

        # If the result for this state is already computed, return it
        if with_memo and (total_mass, current_idx) in memo:
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
            solutions += backtrack(
                total_mass,
                current_idx - 1,
                max_mods_all,
                round(seq_len * dp_table.masses[current_idx - 1].modification_rate),
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

                solutions += [
                    entry + [current_weight]
                    for entry in backtrack(
                        total_mass - current_weight,
                        current_idx,
                        max_mods_all,
                        max_mods_ind,
                    )
                ]

        # Store result in memo
        if with_memo:
            memo[(total_mass, current_idx)] = solutions

        return solutions

    # Compute all valid solutions within the threshold interval
    solutions = []
    for value in range(
        target - threshold,
        target + threshold + 1,
    ):
        solutions += backtrack(
            value,
            len(dp_table.masses) - 1,
            max_modifications,
            round(seq_len * dp_table.masses[-1].modification_rate),
        )

    return convert_nucleotide_masses_to_names(solutions=solutions)


def explain_mass_with_recursion(
    mass: float,
    dp_table: DynamicProgrammingTable,
    seq_len: int,
    max_modifications=np.inf,
    threshold=None,
) -> MassExplanations:
    """
    Returns all the possible combinations of nucleosides that could sum up to the given mass.
    """
    tolerated_integer_masses = [mass.mass for mass in dp_table.masses]

    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    # Memoization dictionary to store results for a given target
    memo = {}

    def dp(remaining, start, used_mods_all, used_mods_ind):
        # If too many modifications are used, return empty list
        if used_mods_all > max_modifications or used_mods_ind > round(
            seq_len * dp_table.masses[start].modification_rate
        ):
            return []

        # If the result for this state is already computed, return it
        if (remaining, start) in memo:
            return memo[(remaining, start)]

        # Base case: if abs(target) is less than threshold, return a list with one empty combination
        if abs(remaining) <= threshold:
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
                used_mods_all + 1 if IS_MOD[tolerated_integer_mass] else used_mods_all,
                0
                if i != start
                else (
                    used_mods_ind + 1
                    if IS_MOD[tolerated_integer_mass]
                    else used_mods_ind
                ),
            )
            # Add current tolerated_integer_mass to all sub-combinations
            for combo in sub_combinations:
                combinations.append([tolerated_integer_mass] + combo)

        # Store result in memo
        memo[(remaining, start)] = combinations

        return combinations

    # Compute all solutions for the full target and all allowed masses (except 0.0)
    solutions = dp(target, 1, 0, 0)

    return convert_nucleotide_masses_to_names(solutions=solutions)


def convert_nucleotide_masses_to_names(solutions: List[List[int]]) -> MassExplanations:
    # Store the nucleotide names (as tuples) for the given masses in a set
    solution_names = set()
    # Return None if no explanation is found
    if len(solutions) == 0:
        return MassExplanations(None)
    # Convert the masses to their respective nucleotide names
    for solution in solutions:
        if len(solution) == 0:
            continue
        solution_names.update(
            [
                tuple(chain.from_iterable(entry))
                for entry in list(
                    product(
                        *[
                            list(
                                combinations_with_replacement(
                                    MASS_NAMES[mass], solution.count(mass)
                                )
                            )
                            for mass in [
                                solution[idx]
                                for idx in range(len(solution))
                                if idx == 0 or solution[idx - 1] != solution[idx]
                            ]
                        ]
                    )
                )
            ]
        )

    # Return list of explanations
    return MassExplanations(solution_names)
