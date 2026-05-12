# -*- coding: utf-8 -*-
"""Module for composition inference."""

from dataclasses import dataclass
from typing import Tuple

from spectrseqtools.compositions import CompositionList
from spectrseqtools.dataclasses import LengthBoundary, SequenceInformation
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.traceback_matrix import TracebackMatrix


@dataclass
class CompositionInferrer:
    """Class to infer compositions."""

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
        """Adapt modification rates for each nucleotide based on new alphabet."""
        mapping = self.alphabet.adapt_individual_modification_rates_by_alphabet(
            alphabet=alphabet
        )
        self._reduce_nucleotide_alphabet()
        return mapping

    def _reduce_nucleotide_alphabet(self, compression_rate: int = None):
        """Reduce alphabet by removing nucleotides that can never occur."""
        # Get current alphabet size
        alphabet_size = len(self.alphabet)

        # Reduce nucleotide alphabet (if possible)
        self.alphabet.reduce()

        # Return if alphabet was not reduced
        if len(self.alphabet) == alphabet_size:
            return

        if self.matrix is not None:
            compression_rate = self.matrix.compression_rate

        # Recompute matrix
        self.matrix = TracebackMatrix.set_up_bit_matrix(
            alphabet=self.alphabet,
            compression_rate=compression_rate,
        )

    def print_alphabet(self) -> None:
        """Print alphabet."""
        print(self.alphabet)

    def set_target(self, mass: float, threshold: float = None) -> Tuple[int, int]:
        """
        Return precision-adjusted target and threshold (as integers).

        Parameters
        ----------
        mass : float
            Given target mass.
        threshold : float
            Given threshold.

        Returns
        -------
        mass : int
            Precision-adapted inference target.
        threshold : int
            Precision-adapted inference threshold.

        """
        # Convert the target to an integer for easy operations
        target = self.alphabet.set_target(value=mass)

        # Set relative threshold if not given
        if threshold is None:
            threshold = self.tolerance * mass

        # Convert the threshold to integer
        threshold = self.alphabet.set_threshold(value=threshold)

        return target, threshold

    def infer_length_bound(self, bound: LengthBoundary) -> int:
        """
        Return bound on length for any composition of the given mass.

        Parameters
        ----------
        bound : LengthBoundary
            Bound direction.

        Returns
        -------
        opt_len : int
            Tightest bound on composition length in given direction.

        """
        # Set maximum number of modifications
        max_modifications = self.seq.max_modifications

        target, threshold = self.set_target(
            mass=self.seq.su_mass, threshold=self.tolerance * self.seq.obs_mass
        )

        # Initialize memorization dict
        memo = {}

        def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
            # If the result for this state is already computed, return it
            if (total_mass, current_idx) in memo:
                return memo[(total_mass, current_idx)]

            # Return default value for cells outside of matrix
            if total_mass < 0 or current_idx < 0:
                return bound.default_value

            # Initialize new counter for valid start in matrix
            if total_mass == 0:
                return 0

            # Assert that total mass is in matrix
            self.matrix.assert_in_matrix(mass=total_mass)

            # Get current value
            current_value = self.matrix.get_entry(mass=total_mass, nuc_idx=current_idx)

            # Return default value for unreachable cells
            if self.matrix.is_unreachable(value):
                return bound.default_value

            # Initialize list of possible bounds
            bounds = [bound.default_value]

            # Backtrack to the next row above if possible
            if current_value % 2 == 1:
                bounds.append(
                    backtrack(
                        total_mass,
                        current_idx - 1,
                        max_mods_all,
                        round(
                            self.seq.max_len
                            * self.alphabet.get(current_idx - 1).modification_rate
                        ),
                    )
                )

            # Backtrack to the next left-side column if possible
            if (current_value >> 1) % 2 == 1:
                current_nuc = self.alphabet.get(current_idx)

                if not current_nuc.is_modification or (
                    max_mods_all > 0 and max_mods_ind > 0
                ):
                    # Adjust number of still allowed modifications if necessary
                    if current_nuc.is_modification:
                        max_mods_all -= 1
                        max_mods_ind -= 1

                    bounds.append(
                        backtrack(
                            total_mass - current_nuc.mass,
                            current_idx,
                            max_mods_all,
                            max_mods_ind,
                        )
                        + 1
                    )

            # Select result based on desired bound
            result = bound.select_best(bounds=bounds)

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
                    len(self.alphabet) - 1,
                    max_modifications,
                    round(self.seq.max_len * self.alphabet.get(-1).modification_rate),
                )
            )

        # Return solution based on desired bound and replace default value if selected
        return bound.select_best(bounds=solutions, replace_default=True)

    def is_valid_mass(
        self,
        mass: float,
        threshold: float = None,
    ) -> bool:
        """
        Check whether a given mass has any valid composition.

        Parameters
        ----------
        mass : float
            Given mass.
        threshold : float
            Given threshold.

        Returns
        -------
        bool
            Flag whether a valid composition exists.

        """
        target, threshold = self.set_target(mass=mass, threshold=threshold)

        for value in range(target - threshold, target + threshold + 1):
            # Skip non-positive masses
            if value <= 0:
                continue

            # Assert that value is in matrix
            self.matrix.assert_in_matrix(mass=value)

            # Get current value
            current_value = self.matrix.get_entry(mass=value, nuc_idx=-1)

            # Skip unreachable cells
            if self.matrix.is_unreachable(value=current_value):
                continue

            # Return True when mass corresponds to valid entry in matrix
            if current_value % 2 == 1 or (current_value >> 1) % 2 == 1:
                return True
        return False

    def infer_compositions(
        self,
        mass: float,
        threshold: float = None,
        with_memo: bool = True,
    ) -> CompositionList:
        """
        Return all possible nucleotide compositions that could sum up to the given mass.

        Parameters
        ----------
        mass : float
            Given mass.
        threshold : float
            Given threshold.
        with_memo : bool
            Flag whether memorization is used.

        Returns
        -------
        compositions : CompositionList
            List of valid composition of the given mass.

        """
        # Set maximum number of modifications
        max_modifications = self.seq.max_modifications

        target, threshold = self.set_target(mass=mass, threshold=threshold)

        # Memoization dictionary to store results for a given target
        memo = {}

        def backtrack(target_mass, current_idx, max_mods_all, max_mods_ind):
            # If the result for this state is already computed, return it
            if with_memo and (target_mass, current_idx) in memo:
                return memo[(target_mass, current_idx)]

            # Return empty list for cells outside of matrix
            if target_mass < 0 or current_idx < 0:
                return []

            # Initialize a new composition for a valid start in matrix
            if target_mass == 0:
                return [[]]

            # Assert that target mass is in matrix
            self.matrix.assert_in_matrix(mass=target_mass)

            # Get current value
            current_value = self.matrix.get_entry(mass=target_mass, nuc_idx=current_idx)

            # Return empty list for unreachable cells
            if self.matrix.is_unreachable(value=current_value):
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
                        self.seq.max_len
                        * self.alphabet.get(current_idx - 1).modification_rate
                    ),
                )

            # Backtrack to the next left-side column if possible
            if (current_value >> 1) % 2 == 1:
                current_nuc = self.alphabet.get(current_idx)

                if not current_nuc.is_modification or (
                    max_mods_all > 0 and max_mods_ind > 0
                ):
                    # Adjust number of still allowed modifications if necessary
                    if current_nuc.is_modification:
                        max_mods_all -= 1
                        max_mods_ind -= 1

                    compositions += [
                        entry + [current_idx]
                        for entry in backtrack(
                            target_mass - current_nuc.mass,
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
                len(self.alphabet) - 1,
                max_modifications,
                round(self.seq.max_len * self.alphabet.get(-1).modification_rate),
            )

        return CompositionList.from_list(compositions=list(solutions))


def infer_compositions_with_recursion(
    mass: float,
    inferrer: CompositionInferrer,
    threshold=None,
) -> CompositionList:
    """
    Return all possible nucleotide compositions that could sum up to the given mass.

    Parameters
    ----------
    mass : float
        Given mass.
    inferrer : CompositionInferrer
        CompositionInferrer.
    threshold : float
        Given threshold.

    Returns
    -------
    compositions : CompositionList
        List of valid composition of the given mass.

    """
    # Set maximum number of modifications
    max_modifications = inferrer.seq.max_modifications

    target, threshold = inferrer.set_target(mass=mass, threshold=threshold)

    # Memoization dictionary to store results for a given target
    memo = {}

    def backtrack(target_mass, current_idx, used_mods_all, used_mods_ind):
        # If too many modifications are used, return empty list
        if used_mods_all > max_modifications or used_mods_ind > round(
            inferrer.seq.max_len * inferrer.alphabet.get(current_idx).modification_rate
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
        for idx in range(current_idx, len(inferrer.alphabet)):
            current_nuc = inferrer.alphabet.get(idx=idx)

            # Add compositions for recursion with reduced target and current mass
            compositions += [
                [idx] + entry
                for entry in backtrack(
                    target_mass - current_nuc.mass,
                    idx,
                    used_mods_all + 1 if current_nuc.is_modification else used_mods_all,
                    0
                    if idx != current_idx
                    else (
                        used_mods_ind + 1
                        if current_nuc.is_modification
                        else used_mods_ind
                    ),
                )
            ]

        # Store result in memo
        memo[(target_mass, current_idx)] = compositions

        return compositions

    # Compute all solutions for the full target and all allowed masses
    solutions = backtrack(target, 0, 0, 0)

    return CompositionList.from_list(compositions=list(solutions))
