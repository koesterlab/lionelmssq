# -*- coding: utf-8 -*-
"""Module for composition inference."""

from dataclasses import dataclass
from typing import List, Set

from spectrseqtools.compositions import CompositionList
from spectrseqtools.dataclasses import (
    LengthBoundary,
    LowerLengthBound,
    SequenceInformation,
    UpperLengthBound,
)
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.traceback_matrix import TracebackMatrix


@dataclass
class CompositionInferrer:
    """Class to infer compositions."""

    alphabet: NucleotideAlphabet
    error: ErrorCalculator
    matrix: TracebackMatrix
    seq: SequenceInformation

    def reduce_alphabet(self, new_alphabet: Set[int]) -> dict:
        """
        Reduce alphabet by removing nucleotides not new alphabet list.

        Parameters
        ----------
        new_alphabet : Set[int]
            Set of nucleotide indices to keep in new alphabet.

        Returns
        -------
        dict
            Mapping between old and new indexing.

        """
        # Get current alphabet size
        alphabet_size = len(self.alphabet)

        # Reduce nucleotide alphabet (if possible)
        mapping = self.alphabet.reduce(new_alphabet=new_alphabet)

        # Recompute matrix if alphabet was reduced
        if len(self.alphabet) != alphabet_size:
            self.matrix.rebuild(alphabet=self.alphabet)

        return mapping

    def print_alphabet(self) -> None:
        """Print alphabet."""
        print(self.alphabet)

    def update_sequence_length(self, seq_len: int = None) -> None:
        """Update lower and upper bound for sequence length."""
        if seq_len is None:
            self.seq.min_len = self.infer_length_bound(
                bound=LowerLengthBound(max_len=self.seq.max_len)
            )
            self.seq.max_len = self.infer_length_bound(
                bound=UpperLengthBound(max_len=self.seq.max_len)
            )
        else:
            self.seq.min_len = seq_len
            self.seq.max_len = seq_len

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

        target, threshold = self.error.set_target(
            su_mass=self.seq.su_mass,
            obs_masses=[self.seq.obs_mass],
        )

        # Initialize memorization dict
        memo = {}

        def backtrack(total_mass, current_idx, max_mods_all, max_mods_ind):
            # If the result for this state is already computed, return it
            if (total_mass, current_idx) in memo:
                return memo[(total_mass, current_idx)]

            # Initialize new counter for valid start in matrix
            if total_mass == 0:
                return 0

            # Determine possible movements
            vertical_move, horizontal_move = self.matrix.allowed_movement(
                mass=total_mass, nuc_idx=current_idx
            )

            # Return default value if no movement is possible
            if not (vertical_move or horizontal_move):
                return bound.default_value

            # Initialize list of possible bounds
            bounds = [bound.default_value]

            # Backtrack to the next row above if possible
            if vertical_move:
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
            if horizontal_move:
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
        obs_masses: List[float] = None,
    ) -> bool:
        """
        Check whether a given mass has any valid composition.

        Parameters
        ----------
        mass : float
            Given SU-mass.
        obs_masses : List[float]
            Given observed masses.

        Returns
        -------
        bool
            Flag whether a valid composition exists.

        """
        target, threshold = self.error.set_target(su_mass=mass, obs_masses=obs_masses)

        for value in range(target - threshold, target + threshold + 1):
            # Skip non-positive masses
            if value <= 0:
                continue

            vertical_move, horizontal_move = self.matrix.allowed_movement(
                mass=value, nuc_idx=len(self.matrix) - 1
            )

            # Return True when mass corresponds to valid entry in matrix
            if vertical_move or horizontal_move:
                return True
        return False

    def infer_compositions(
        self,
        mass: float,
        obs_masses: List[float] = None,
        with_memo: bool = True,
    ) -> CompositionList:
        """
        Return all possible nucleotide compositions that could sum up to the given mass.

        Parameters
        ----------
        mass : float
            Given SU-mass.
        obs_masses : List[float]
            Given observed masses.
        with_memo : bool
            Flag whether memorization is used. Default: True.

        Returns
        -------
        compositions : CompositionList
            List of valid composition of the given mass.

        """
        # Set maximum number of modifications
        max_modifications = self.seq.max_modifications

        target, threshold = self.error.set_target(su_mass=mass, obs_masses=obs_masses)

        # Memoization dictionary to store results for a given target
        memo = {}

        def backtrack(target_mass, current_idx, max_mods_all, max_mods_ind):
            # If the result for this state is already computed, return it
            if with_memo and (target_mass, current_idx) in memo:
                return memo[(target_mass, current_idx)]

            # Initialize a new composition for a valid start in matrix
            if target_mass == 0:
                return [[]]

            vertical_move, horizontal_move = self.matrix.allowed_movement(
                mass=target_mass, nuc_idx=current_idx
            )

            # Return empty list if no movement is possible
            if not (vertical_move or horizontal_move):
                return []

            # Initialize list to store all compositions for this state
            compositions = []

            # Backtrack to the next row above if possible
            if vertical_move:
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
            if horizontal_move:
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
    obs_masses: List[float] = None,
) -> CompositionList:
    """
    Return all possible nucleotide compositions that could sum up to the given mass.

    Parameters
    ----------
    mass : float
        Given SU-mass.
    inferrer : CompositionInferrer
        CompositionInferrer.
    obs_masses : List[float]
        Given observed masses.

    Returns
    -------
    compositions : CompositionList
        List of valid composition of the given mass.

    """
    # Set maximum number of modifications
    max_modifications = inferrer.seq.max_modifications

    target, threshold = inferrer.error.set_target(su_mass=mass, obs_masses=obs_masses)

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
