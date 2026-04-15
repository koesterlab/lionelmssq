from dataclasses import dataclass
from itertools import chain, combinations_with_replacement, product
from typing import List, Set, Tuple

import numpy as np
import polars as pl

from spectrseqtools.masses import UNMODIFIED_BASES
from spectrseqtools.nucleotide_alphabet import NUCLEOTIDE_DF
from spectrseqtools.prediction.traceback_matrix import (
    TracebackMatrix,
    set_matrix_path,
)


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
class SequenceInformation:
    max_len: int
    su_mass: float
    obs_mass: float
    modification_rate: float


@dataclass
class CompositionInferrer:
    matrix: TracebackMatrix
    precision: float
    tolerance: float
    seq: SequenceInformation
    alphabet: List[NucleotideMass]

    def __init__(
        self,
        nucleotide_df: pl.DataFrame,
        compression_rate: int,
        tolerance: float,
        precision: float,
        seq: SequenceInformation,
    ):
        self.tolerance = tolerance
        self.precision = precision
        self.seq = seq
        self.alphabet = initialize_nucleotide_alphabet(nucleotide_df)
        self.matrix = None

        # Adapt individual modification rates to universal one
        self._adapt_individual_modification_rates_by_universal_one(
            compression_rate=compression_rate
        )

        # Initialize matrix from file (for no alphabet reduction)
        if self.matrix is None:
            self.matrix = TracebackMatrix.load(
                path=set_matrix_path(precision, compression_rate),
                integer_masses=[mass.mass for mass in self.alphabet],
            )

    def _adapt_individual_modification_rates_by_universal_one(
        self, compression_rate: int
    ):
        for nucleotide_mass in self.alphabet:
            if not nucleotide_mass.is_modification:
                continue
            if nucleotide_mass.modification_rate > self.seq.modification_rate:
                nucleotide_mass.modification_rate = self.seq.modification_rate
        self._reduce_nucleotide_alphabet(compression_rate=compression_rate)

    def adapt_individual_modification_rates_by_alphabet_reduction(self, alphabet):
        for nucleotide_mass in self.alphabet:
            if not nucleotide_mass.is_modification:
                continue
            if all(name not in alphabet for name in nucleotide_mass.names):
                nucleotide_mass.modification_rate = 0.0
        self._reduce_nucleotide_alphabet()

    def _reduce_nucleotide_alphabet(self, compression_rate: int = None):
        new_alphabet = [
            mass
            for mass in self.alphabet
            if mass.mass == 0.0 or mass.modification_rate > 0.0
        ]

        # Return if alphabet was not reduced
        if len(new_alphabet) == len(self.alphabet):
            return
        if self.matrix is not None:
            compression_rate = self.matrix.compression_rate

        # Recompute matrix
        self.matrix = TracebackMatrix.set_up_bit_matrix(
            integer_masses=[mass.mass for mass in new_alphabet],
            compression_rate=compression_rate,
        )

        # Update nucleotide alphabet
        self.alphabet = new_alphabet

    def print_alphabet(self):
        mass_names = []
        for mass in self.alphabet:
            mass_names += mass.names
        masses = NUCLEOTIDE_DF.sort("nucleoside_mass").filter(
            pl.col("representative").is_in(mass_names)
        )

        print(
            masses.replace_column(
                masses.get_column_index("modification_rate"),
                pl.Series(
                    "modification_rate",
                    [mass.modification_rate for mass in self.alphabet[1:]],
                ),
            )
        )


def initialize_nucleotide_alphabet(nucleotide_df):
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

    # Return alphabet of NucleotideMass instances
    return [
        NucleotideMass(mass, names[mass], is_mod[mass], rates[mass])
        if mass != 0
        else NucleotideMass(0, [], False, 0.0)
        for mass in integer_masses
    ]


def compute_sequence_length_bound(inferrer: CompositionInferrer, dir: str) -> int:
    """
    Return bound on length for any sequence that could explain the given mass.
    """
    # Set maximum number of modifications
    max_modifications = round(inferrer.seq.modification_rate * inferrer.seq.max_len)

    # Convert the target to an integer for easy operations
    target = int(round(inferrer.seq.su_mass / inferrer.precision, 0))

    # Convert the threshold to integer
    threshold = int(
        np.ceil(inferrer.tolerance * inferrer.seq.obs_mass / inferrer.precision)
    )

    # Initialize memorization dict
    memo = {}

    # Select default value based on desired bound
    match dir:
        case "lower":
            default_bound = inferrer.seq.max_len + 1
        case "upper":
            default_bound = -1
        case _:
            raise NotImplementedError(f"Support for '{dir}' is currently not given.")

    def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
        current_weight = inferrer.alphabet[current_idx].mass

        # If the result for this state is already computed, return it
        if (total_mass, current_idx) in memo:
            return memo[(total_mass, current_idx)]

        # Return default value for cells outside of matrix
        if total_mass < 0:
            return default_bound

        # Initialize new counter for valid start in matrix
        if total_mass == 0:
            return 0

        # Assert that total mass is in matrix
        inferrer.matrix.assert_in_matrix(mass=total_mass)

        # Get current value
        current_value = inferrer.matrix.get_entry(mass=total_mass, nuc_idx=current_idx)

        # Return default value for unreachable cells
        if inferrer.matrix.is_unreachable(value):
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
                        inferrer.seq.max_len
                        * inferrer.alphabet[current_idx - 1].modification_rate
                    ),
                )
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
                len(inferrer.alphabet) - 1,
                max_modifications,
                round(inferrer.seq.max_len * inferrer.alphabet[-1].modification_rate),
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
                opt_len = inferrer.seq.max_len
        case _:
            raise NotImplementedError(f"Support for '{dir}' is currently not given.")

    return opt_len


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

    for value in range(target - threshold, target + threshold + 1):
        # Skip non-positive masses
        if value <= 0:
            continue

        # Assert that value is in matrix
        inferrer.matrix.assert_in_matrix(mass=value)

        # Get current value
        current_value = inferrer.matrix.get_entry(mass=value, nuc_idx=-1)

        # Skip unreachable cells
        if inferrer.matrix.is_unreachable(value=current_value):
            continue

        # Return True when mass corresponds to valid entry in matrix
        if current_value % 2 == 1 or (current_value >> 1) % 2 == 1:
            return True
    return False


def infer_compositions_with_matrix(
    mass: float,
    inferrer: CompositionInferrer,
    max_modifications=np.inf,
    threshold=None,
    with_memo=True,
) -> MassCompositions:
    """
    Return all possible nucleotide compositions that could sum up to the given mass.
    """
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

        # Assert that target mass is in matrix
        inferrer.matrix.assert_in_matrix(mass=target_mass)

        # Get current value
        current_value = inferrer.matrix.get_entry(mass=target_mass, nuc_idx=current_idx)

        # Return empty list for unreachable cells
        if inferrer.matrix.is_unreachable(value=current_value):
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
