from dataclasses import dataclass
from typing import Set, Tuple
from itertools import product, combinations_with_replacement, chain

import polars as pl
import numpy as np

from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import (
    EXPLANATION_MASSES,
    BREAKAGES,
    COMPRESSION_RATE,
    UNMODIFIED_BASES,
)


@dataclass
class MassExplanations:
    breakage: str
    explanations: Set[Tuple[str]]


@dataclass
class MassExplanation:
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


def is_valid_su_mass(
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


def is_valid_mass(
    mass: float,
    dp_table: DynamicProgrammingTable,
    threshold=None,
    breakages=BREAKAGES,
    compression_rate=COMPRESSION_RATE,
) -> bool:
    # Ensure that all breakage weights have a associated breakage
    breakages = {
        breakage_weight: breakage
        for breakage_weight, breakage in breakages.items()
        if len(breakage) > 0
    }

    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    current_idx = len(dp_table.table) - 1
    for breakage_weight in breakages:
        for value in range(
            target - breakage_weight - threshold,
            target - breakage_weight + threshold + 1,
        ):
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


def explain_mass_with_dp(
    mass: float,
    dp_table: DynamicProgrammingTable,
    with_memo: bool,
    seq_len: int,
    max_modifications=np.inf,
    compression_rate=None,
    threshold=None,
    breakage_dict=BREAKAGES,
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

    def backtrack_with_memo(total_mass, current_idx, max_mods_all, max_mods_ind):
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
            solutions += backtrack_with_memo(
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
                    for entry in backtrack_with_memo(
                        total_mass - current_weight,
                        current_idx,
                        max_mods_all,
                        max_mods_ind,
                    )
                ]

        # Store result in memo
        memo[(total_mass, current_idx)] = solutions

        return solutions

    def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
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

        return solutions

    solution_tolerated_integer_masses = {}
    for breakage_weight in breakage_dict:
        # Compute all valid solutions within the threshold interval
        solutions = []
        for value in range(
            target - breakage_weight - threshold,
            target - breakage_weight + threshold + 1,
        ):
            solutions += (
                backtrack_with_memo(
                    value,
                    len(dp_table.masses) - 1,
                    max_modifications,
                    round(seq_len * dp_table.masses[-1].modification_rate),
                )
                if with_memo
                else backtrack(
                    value,
                    len(dp_table.masses) - 1,
                    max_modifications,
                    round(seq_len * dp_table.masses[-1].modification_rate),
                )
            )

        # Add valid solutions to dictionary of breakpoint options
        for breakage in breakage_dict[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    # Convert the DP table masses to their respective nucleoside names
    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Return None if no explanation is found
        if len(solution_tolerated_integer_masses[breakage]) == 0:
            solution_names = None
        # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
        else:
            solution_names = set()
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
    dp_table: DynamicProgrammingTable,
    seq_len: int,
    max_modifications=np.inf,
    threshold=None,
    breakage_dict=BREAKAGES,
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

    solution_tolerated_integer_masses = {}
    for breakage_weight in breakage_dict:
        # Start with the full target and all tolerated_integer_masses (except 0.0)
        solutions = dp(target - breakage_weight, 1, 0, 0)
        for breakage in breakage_dict[breakage_weight]:
            solution_tolerated_integer_masses[breakage] = solutions

    # Convert the tolerated_integer_masses to the respective nucleoside names
    explanations = []
    for breakage in solution_tolerated_integer_masses.keys():
        # Return None if no explanation is found
        if len(solution_tolerated_integer_masses[breakage]) == 0:
            solution_names = None
        # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
        else:
            solution_names = set()
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


def explain_mass_without_breakage(
    mass: float,
    dp_table: DynamicProgrammingTable,
    seq_len: int,
    max_modifications=np.inf,
    compression_rate=None,
    threshold=None,
) -> MassExplanation:
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

    # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
    solution_names = set()
    # Return None if no explanation is found
    if len(solutions) == 0:
        return MassExplanation(None)
    # Convert the DP table masses to their respective nucleoside names
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
    return MassExplanation(solution_names)


def explain_mass_recursively_without_breakage(
    mass: float,
    dp_table: DynamicProgrammingTable,
    seq_len: int,
    max_modifications=np.inf,
    threshold=None,
) -> MassExplanation:
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

    # Start with the full target and all tolerated_integer_masses (except 0.0)
    solutions = dp(target, 1, 0, 0)

    # Store the nucleoside names (as tuples) for the given tolerated_integer_masses in the set solution_names
    solution_names = set()
    # Return None if no explanation is found
    if len(solutions) == 0:
        return MassExplanation(None)
    # Convert the DP table masses to their respective nucleoside names
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
    return MassExplanation(solution_names)
