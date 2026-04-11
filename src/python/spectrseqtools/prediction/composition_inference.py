from dataclasses import dataclass
from typing import List, Set, Tuple
from itertools import product, combinations_with_replacement, chain

import polars as pl
import numpy as np

from spectrseqtools.masses import UNMODIFIED_BASES
from spectrseqtools.nucleotide_alphabet import NUCLEOTIDE_DF
from spectrseqtools.prediction.traceback_matrix import CompositionInferrer


@dataclass
class MassCompositions:
    compositions: Set[Tuple[str]]


MASS_NAMES = {
    mass: pl.DataFrame({"integer_mass": mass})
    .join(
        NUCLEOTIDE_DF,
        on="integer_mass",
        how="left",
    )
    .get_column("representative")
    .to_list()
    for mass in NUCLEOTIDE_DF.get_column("integer_mass").to_list()
}

IS_MOD = {
    mass: any(
        base not in UNMODIFIED_BASES
        for base in pl.DataFrame({"integer_mass": mass})
        .join(
            NUCLEOTIDE_DF,
            on="integer_mass",
            how="left",
        )
        .get_column("representative")
        .to_list()
    )
    for mass in NUCLEOTIDE_DF.get_column("integer_mass").to_list()
}


def is_valid_mass(
    mass: float,
    inferrer: CompositionInferrer,
    threshold: float = None,
) -> bool:
    # Convert the target to an integer for easy operations
    target = int(round(mass / inferrer.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = inferrer.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / inferrer.precision))

    compression_rate = inferrer.compression_per_cell

    current_idx = len(inferrer.matrix) - 1
    for value in range(target - threshold, target + threshold + 1):
        # Skip non-positive masses
        if value <= 0:
            continue

        # Raise error if mass is not in matrix (due to its size)
        if value >= len(inferrer.matrix[0]) * compression_rate:
            raise NotImplementedError(
                f"The value {value} is not in the traceback matrix. "
                f"Extend its size if you want to compute larger masses."
            )

        current_value = (
            inferrer.matrix[current_idx, value]
            if compression_rate == 1
            else inferrer.matrix[current_idx, value // compression_rate]
            >> 2 * (compression_rate - 1 - value % compression_rate)
        )

        # Skip unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            continue

        # Return True when mass corresponds to valid entry in matrix
        if current_value % 2 == 1 or (current_value >> 1) % 2 == 1:
            return True
    return False


def infer_compositions_with_matrix(
    mass: float,
    inferrer: CompositionInferrer,
    max_modifications=np.inf,
    compression_rate=None,
    threshold=None,
    with_memo=True,
) -> MassCompositions:
    """
    Return all possible nucleotide compositions that could sum up to the given mass.
    """
    if compression_rate is None:
        compression_rate = inferrer.compression_per_cell

    # Convert the target to an integer for easy operations
    target = int(round(mass / inferrer.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = inferrer.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / inferrer.precision))

    # Memoization dictionary to store results for a given target
    memo = {}

    def backtrack(target_mass, current_idx, max_mods_all, max_mods_ind):
        current_mass = inferrer.alphabet[current_idx].mass

        # If the result for this state is already computed, return it
        if with_memo and (target_mass, current_idx) in memo:
            return memo[(target_mass, current_idx)]

        # Return empty list for cells outside of matrix
        if target_mass < 0:
            return []

        # Initialize a new composition for a valid start in matrix
        if target_mass == 0:
            return [[]]

        # Raise error if mass is not in matrix (due to its size)
        if target_mass >= len(inferrer.matrix[0]) * compression_rate:
            raise NotImplementedError(
                f"The value {value} is not in the traceback matrix. "
                f"Extend its size if you want to compute larger masses."
            )

        current_value = (
            inferrer.matrix[current_idx, target_mass]
            if compression_rate == 1
            else inferrer.matrix[current_idx, target_mass // compression_rate]
            >> 2 * (compression_rate - 1 - target_mass % compression_rate)
        )

        # Return empty list for unreachable cells
        if compression_rate != 1 and current_value % compression_rate == 0.0:
            return []

        # Initialize list to store all compositions for this state
        compositions = []

        # Backtrack to the next row above if possible
        if current_value % 2 == 1:
            compositions += backtrack(
                target_mass,
                current_idx - 1,
                max_mods_all,
                round(
                    inferrer.seq.max_len
                    * inferrer.alphabet[current_idx - 1].modification_rate
                ),
            )

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not inferrer.alphabet[current_idx].is_modification or (
                max_mods_all > 0 and max_mods_ind > 0
            ):
                # Adjust number of still allowed modifications if necessary
                if inferrer.alphabet[current_idx].is_modification:
                    max_mods_all -= 1
                    max_mods_ind -= 1

                compositions += [
                    entry + [current_mass]
                    for entry in backtrack(
                        target_mass - current_mass,
                        current_idx,
                        max_mods_all,
                        max_mods_ind,
                    )
                ]

        # Store result in memo
        if with_memo:
            memo[(target_mass, current_idx)] = compositions

        return compositions

    # Compute all valid solutions within the threshold interval
    solutions = []
    for value in range(target - threshold, target + threshold + 1):
        solutions += backtrack(
            value,
            len(inferrer.alphabet) - 1,
            max_modifications,
            round(inferrer.seq.max_len * inferrer.alphabet[-1].modification_rate),
        )

    return convert_nucleotide_masses_to_names(solutions=solutions)


def infer_compositions_with_recursion(
    mass: float,
    inferrer: CompositionInferrer,
    max_modifications=np.inf,
    threshold=None,
) -> MassCompositions:
    """
    Returns all possible nucleotide compositions that could sum up to the given mass.
    """
    mass_list = [mass.mass for mass in inferrer.alphabet]

    # Convert the target to an integer for easy operations
    target = int(round(mass / inferrer.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = inferrer.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / inferrer.precision))

    # Memoization dictionary to store results for a given target
    memo = {}

    def backtrack(target_mass, current_idx, used_mods_all, used_mods_ind):
        # If too many modifications are used, return empty list
        if used_mods_all > max_modifications or used_mods_ind > round(
            inferrer.seq.max_len * inferrer.alphabet[current_idx].modification_rate
        ):
            return []

        # If the result for this state is already computed, return it
        if (target_mass, current_idx) in memo:
            return memo[(target_mass, current_idx)]

        # Initialize a new composition for a valid start (i.e. target within threshold)
        if abs(target_mass) <= threshold:
            return [[]]

        # Return empty list for a negative target mass outside of threshold
        if target_mass < 0:
            return []

        # Initialize list to store all compositions for this state
        compositions = []

        # Try each mass starting from the current position to avoid duplicates
        for i in range(current_idx, len(mass_list)):
            current_mass = mass_list[i]

            # Add compositions for recursion with reduced target and current mass
            compositions += [
                [current_mass] + entry
                for entry in backtrack(
                    target_mass - current_mass,
                    i,
                    used_mods_all + 1 if IS_MOD[current_mass] else used_mods_all,
                    0
                    if i != current_idx
                    else (used_mods_ind + 1 if IS_MOD[current_mass] else used_mods_ind),
                )
            ]

        # Store result in memo
        memo[(target_mass, current_idx)] = compositions

        return compositions

    # Compute all solutions for the full target and all allowed masses (except 0.0)
    solutions = backtrack(target, 1, 0, 0)

    return convert_nucleotide_masses_to_names(solutions=solutions)


def convert_nucleotide_masses_to_names(solutions: List[List[int]]) -> MassCompositions:
    # Store the nucleotide names (as tuples) for the given masses in a set
    solution_names = set()
    # Return None if no composition is found
    if len(solutions) == 0:
        return MassCompositions(None)
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

    # Return composition set
    return MassCompositions(solution_names)
