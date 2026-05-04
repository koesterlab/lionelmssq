from dataclasses import dataclass
from typing import List, Self, Set, Tuple

import numpy as np

from spectrseqtools.masses import MAX_VARIANCE
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.traceback_matrix import TracebackMatrix
from spectrseqtools.sequence import SkeletonSequence


@dataclass
class MassCompositions:
    """Class for set of compositions to explain a given mass."""

    compositions: Set[Tuple[str]] = None

    @classmethod
    def from_indices(
        cls, solutions: List[List[int]], alphabet: NucleotideAlphabet
    ) -> Self:
        """
        Initialize composition list from index lists.

        Parameters
        ----------
        solutions : List[List[int]]
            List of nucleotide index lists (representing compositions).
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        # Return default if no composition is found
        if len(solutions) == 0:
            return MassCompositions()

        # Store the representative tuples for the given indices in a set
        solution_names = set()

        # Convert the masses to their respective representative
        for solution in solutions:
            if len(solution) == 0:
                continue
            solution_names.update([(alphabet.get_rep(entry) for entry in solution)])

        # Return composition set
        return cls(solution_names)


@dataclass
class SequenceInformation:
    max_len: int
    su_mass: float
    obs_mass: float
    modification_rate: float

    def validate_sequence(self, seq: SkeletonSequence, nuc_masses: dict) -> bool:
        """
        Validate sequence length by mass.

        Parameters
        ----------
        seq : SkeletonSequence
            Skeleton sequence.
        nuc_masses : dict
            Dictionary assigning masses to each representative in alphabet.

        Returns
        -------
        bool
            Flag whether sequence length is valid.

        """
        # Check whether mass interval defined by skeleton contains sequence mass
        # Use MAX_VARIANCE to accommodate for uncertainty in sequence mass selection
        return (
            seq.min_mass(nuc_masses=nuc_masses) - MAX_VARIANCE
            <= self.su_mass
            <= seq.max_mass(nuc_masses=nuc_masses) + MAX_VARIANCE
        )


@dataclass
class CompositionInferrer:
    matrix: TracebackMatrix
    tolerance: float
    seq: SequenceInformation
    alphabet: NucleotideAlphabet

    def __init__(
        self,
        alphabet: NucleotideAlphabet,
        compression_rate: int,
        tolerance: float,
        seq: SequenceInformation,
    ):
        self.tolerance = tolerance
        self.seq = seq
        self.matrix = None

        self.alphabet = alphabet
        self._reduce_nucleotide_alphabet(compression_rate=compression_rate)

        # Initialize matrix from file (for no alphabet reduction)
        if self.matrix is None:
            self.matrix = TracebackMatrix.load(
                alphabet=self.alphabet, compression_rate=compression_rate
            )

    def adapt_individual_modification_rates_by_alphabet_reduction(self, alphabet):
        self.alphabet.adapt_individual_modification_rates_by_alphabet(alphabet=alphabet)
        self._reduce_nucleotide_alphabet()

    def _reduce_nucleotide_alphabet(self, compression_rate: int = None):
        # Get current alphabet size
        alphabet_size = self.alphabet.size

        # Reduce nucleotide alphabet (if possible)
        self.alphabet.reduce()

        # Return if alphabet was not reduced
        if self.alphabet.size == alphabet_size:
            return

        if self.matrix is not None:
            compression_rate = self.matrix.compression_rate

        # Recompute matrix
        self.matrix = TracebackMatrix.set_up_bit_matrix(
            alphabet=self.alphabet,
            compression_rate=compression_rate,
        )

    def print_alphabet(self) -> None:
        print(self.alphabet)

    def set_target(self, mass: float, threshold: float = None) -> Tuple[int, int]:
        # Convert the target to an integer for easy operations
        target = self.alphabet.set_target(value=mass)

        # Set relative threshold if not given
        if threshold is None:
            threshold = self.tolerance * mass

        # Convert the threshold to integer
        threshold = self.alphabet.set_threshold(value=threshold)

        return target, threshold


def compute_sequence_length_bound(inferrer: CompositionInferrer, dir: str) -> int:
    """
    Return bound on length for any sequence that could explain the given mass.
    """
    # Set maximum number of modifications
    max_modifications = round(inferrer.seq.modification_rate * inferrer.seq.max_len)

    target, threshold = inferrer.set_target(
        mass=inferrer.seq.su_mass, threshold=inferrer.tolerance * inferrer.seq.obs_mass
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
                        * inferrer.alphabet.get_rate(current_idx - 1)
                    ),
                )
            )

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not inferrer.alphabet.is_mod(current_idx) or (
                max_mods_all > 0 and max_mods_ind > 0
            ):
                # Adjust number of still allowed modifications if necessary
                if inferrer.alphabet.is_mod(current_idx):
                    max_mods_all -= 1
                    max_mods_ind -= 1

                bounds.append(
                    backtrack(
                        total_mass - inferrer.alphabet.get_mass(current_idx),
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
                inferrer.alphabet.size - 1,
                max_modifications,
                round(inferrer.seq.max_len * inferrer.alphabet.get_rate(-1)),
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
    target, threshold = inferrer.set_target(mass=mass, threshold=threshold)

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
    target, threshold = inferrer.set_target(mass=mass, threshold=threshold)

    # Memoization dictionary to store results for a given target
    memo = {}

    def backtrack(target_mass, current_idx, max_mods_all, max_mods_ind):
        current_mass = inferrer.alphabet.get_mass(current_idx)

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
                    inferrer.seq.max_len * inferrer.alphabet.get_rate(current_idx - 1)
                ),
            )

        # Backtrack to the next left-side column if possible
        if (current_value >> 1) % 2 == 1:
            if not inferrer.alphabet.is_mod(current_idx) or (
                max_mods_all > 0 and max_mods_ind > 0
            ):
                # Adjust number of still allowed modifications if necessary
                if inferrer.alphabet.is_mod(current_idx):
                    max_mods_all -= 1
                    max_mods_ind -= 1

                compositions += [
                    entry + [current_idx]
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
            inferrer.alphabet.size - 1,
            max_modifications,
            round(inferrer.seq.max_len * inferrer.alphabet.get_rate(-1)),
        )

    return MassCompositions.from_indices(
        solutions=solutions, alphabet=inferrer.alphabet
    )


def infer_compositions_with_recursion(
    mass: float,
    inferrer: CompositionInferrer,
    max_modifications=np.inf,
    threshold=None,
) -> MassCompositions:
    """
    Returns all possible nucleotide compositions that could sum up to the given mass.
    """
    target, threshold = inferrer.set_target(mass=mass, threshold=threshold)

    # Memoization dictionary to store results for a given target
    memo = {}

    def backtrack(target_mass, current_idx, used_mods_all, used_mods_ind):
        # If too many modifications are used, return empty list
        if used_mods_all > max_modifications or used_mods_ind > round(
            inferrer.seq.max_len * inferrer.alphabet.get_rate(current_idx)
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
        for idx in range(current_idx, inferrer.alphabet.size):
            current_mass = inferrer.alphabet.get_mass(idx=idx)
            is_mod = inferrer.alphabet.is_mod(idx=idx)

            # Add compositions for recursion with reduced target and current mass
            compositions += [
                [idx] + entry
                for entry in backtrack(
                    target_mass - current_mass,
                    idx,
                    used_mods_all + 1 if is_mod else used_mods_all,
                    0
                    if idx != current_idx
                    else (used_mods_ind + 1 if is_mod else used_mods_ind),
                )
            ]

        # Store result in memo
        memo[(target_mass, current_idx)] = compositions

        return compositions

    # Compute all solutions for the full target and all allowed masses (except 0.0)
    solutions = backtrack(target, 1, 0, 0)

    return MassCompositions.from_indices(
        solutions=solutions, alphabet=inferrer.alphabet
    )
