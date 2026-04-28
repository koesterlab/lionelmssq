# -*- coding: utf-8 -*-
"""Building of sequence skeletons."""

from dataclasses import dataclass
from itertools import chain, groupby
from typing import List, Optional, Set, Tuple

import numpy as np

from spectrseqtools.common import (
    Composition,
    calculate_compositions,
    calculate_error_threshold,
)
from spectrseqtools.dataclasses import SolverParameters
from spectrseqtools.fragments import MAX_VARIANCE, StandardUnitFragments
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
    compute_sequence_length_bound,
)
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance


@dataclass
class SkeletonBuilder:
    """Class to build skeleton sequence."""

    compositions: dict
    inferrer: CompositionInferrer

    def build_skeleton(
        self, fragments: StandardUnitFragments, solver_params: SolverParameters
    ) -> Tuple[List[Set[str]], StandardUnitFragments]:
        """
        Build skeleton from given fragments.

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments to build skeleton.
        solver_params : SolverParameters
            Solver parameter.

        Returns
        -------
        List[Set[str]]
            Skeleton sequence.
        StandardUnitFragments
            SU-fragments after skeleton building.

        """
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments = self._predict_skeleton(
            fragments=fragments.start,
            skeleton_seq=[set() for _ in range(self.inferrer.seq.max_len)],
        )
        print("Skeleton sequence (5'-end)\t= ", start_skeleton)

        # Build skeleton sequence from 3'-end and reverse it
        end_skeleton, end_fragments = self._predict_skeleton(
            fragments=fragments.end,
            skeleton_seq=[set() for _ in range(self.inferrer.seq.max_len)],
        )
        end_skeleton = end_skeleton[::-1]
        print("Skeleton sequence (3'-end)\t= ", end_skeleton)

        # Select best sequence length with LP
        seq_len = self.select_sequence_length_with_lp(
            start_fragments=start_fragments,
            end_fragments=end_fragments,
            start_skeleton=start_skeleton,
            end_skeleton=end_skeleton,
            solver_params=solver_params,
        )
        if seq_len < 1:
            # Use Jaccard-based method as backup (in case LP does not work)
            seq_len = self.select_sequence_length_with_jaccard(
                start_skeleton=start_skeleton,
                end_skeleton=end_skeleton,
            )

        # Combine both skeleton sequences
        skeleton_seq = combine_skeleton_sequences(
            seq_len=seq_len,
            start_skeleton=start_skeleton,
            end_skeleton=end_skeleton,
        )
        print("Skeleton sequence (combined)\t= ", skeleton_seq)

        # Combine all fragments into one list
        fragments = StandardUnitFragments.from_fragment_classes(
            start_fragments=start_fragments,
            end_fragments=end_fragments,
            internal_fragments=fragments.internal,
            seq_len=len(skeleton_seq),
        )

        # Return skeleton and fragments
        return skeleton_seq, fragments

    def _predict_skeleton(
        self,
        fragments: StandardUnitFragments,
        skeleton_seq: Optional[List[Set[str]]] = None,
    ) -> Tuple[List[Set[str]], StandardUnitFragments]:
        """
        Predict directional skeleton from given fragments.

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments to build skeleton.
        skeleton_seq : List[Set[str]]
            Skeleton sequence.

        Returns
        -------
        List[Set[str]]
            Directional skeleton sequence.
        StandardUnitFragments
            Terminal SU-fragments used for skeleton building.

        """
        # Initialize skeleton sequence (if not already given)
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.inferrer.seq.max_len)]

        # METHOD: Reject fragments which are not explained well by mass
        # differences. While iterating through the fragments, bin them
        # to keep track of similar masses and reject them in bulk.

        pos = {0}

        invalid_list = []
        last_valid_bin = None
        bins = fragments.bin(tolerance=self.inferrer.tolerance)
        for bin_idx, current_bin in enumerate(bins):
            # Stop if no positions are left to fill
            if len(pos) == 0:
                invalid_list += current_bin.invalidate()
                continue

            # TODO: This condition imitates bug found in previous code; remove it
            if (bin_idx + 1 == len(bins)) & (len(current_bin) == 1):
                continue

            compositions = self.infer_compositions_for_bin_differences(
                prev_bin=last_valid_bin,
                current_bin=current_bin,
            )

            # Skip bins with no valid compositions
            if compositions is None:
                invalid_list += current_bin.invalidate()
            else:
                # Continue skeleton building
                pos, skeleton_seq = self.update_skeleton_for_given_compositions(
                    compositions=compositions,
                    pos=pos,
                    skeleton_seq=skeleton_seq,
                )

                # Update information on end index
                current_bin.update_end_indices(pos=pos)

                # Update information for previous bin
                last_valid_bin = current_bin

        return skeleton_seq, StandardUnitFragments.from_bins(
            bins=bins, invalid_list=invalid_list
        )

    def select_sequence_length_with_lp(
        self,
        start_skeleton: List[Set[str]],
        end_skeleton: List[Set[str]],
        start_fragments: StandardUnitFragments,
        end_fragments: StandardUnitFragments,
        solver_params: SolverParameters,
    ) -> int:
        """
        Select sequence length based on LP score.

        Parameters
        ----------
        start_skeleton : List[Set[str]]
            Skeleton in 5'-direction.
        end_skeleton : List[Set[str]]
            Skeleton in 3'-direction.
        start_fragments : StandardUnitFragments
            List of terminal fragments in 5'-direction.
        end_fragments : StandardUnitFragments
            List of terminal fragments in 3'-direction.
        solver_params : SolverParameters
            Solver parameter.

        Returns
        -------
        int
            Selected sequence length.

        """
        # Reduce nucleotide alphabet based on skeleton parts
        nucleotides = {
            nuc
            for skeleton_pos in start_skeleton + end_skeleton
            for nuc in skeleton_pos
        }
        self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotides
        )

        # Initialize nucleotide mass dict
        nucleotide_masses = self.inferrer.alphabet.to_dict()

        # Determine lower and upper bound
        min_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="lower")
        max_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="upper")

        # Determine sequence length with the best LP score
        best_len = -1
        best_val = np.inf
        for len_cand in range(min_len, max_len + 1):
            seq = combine_skeleton_sequences(
                seq_len=len_cand,
                start_skeleton=start_skeleton,
                end_skeleton=end_skeleton,
            )

            fragments = StandardUnitFragments.from_terminals(
                start_fragments=start_fragments,
                end_fragments=end_fragments,
                seq_len=len_cand,
            )

            # Determine LP score for terminal-fragment alignment
            value = self.determine_lp_score(
                terminal_fragments=fragments,
                skeleton_seq=seq,
                solver_params=solver_params,
            )

            # Update best found sequence length if needed
            if value < best_val and self.validate_sequence_length_by_mass(
                start_skeleton=start_skeleton[:len_cand],
                end_skeleton=end_skeleton[len(end_skeleton) - len_cand :],
                nuc_masses=nucleotide_masses,
            ):
                best_val = value
                best_len = len_cand

        return best_len

    def determine_lp_score(
        self,
        terminal_fragments: StandardUnitFragments,
        skeleton_seq: list,
        solver_params: SolverParameters,
    ) -> float:
        """

        Parameters
        ----------
        terminal_fragments : StandardUnitFragments
            Terminal SU-fragments.
        skeleton_seq : list
            Skeleton sequence.
        solver_params : SolverParameters
            Solver parameter.

        Returns
        -------
        float
            Score of linear program solution.

        """
        # Initialize LP instance for terminal fragment
        try:
            lp_instance = LinearProgramInstance(
                fragments=terminal_fragments.fragments,
                inferrer=self.inferrer,
                skeleton_seq=skeleton_seq,
            )
        except Exception:
            return np.inf

        # Return minimum error when fragments can feasibly be aligned to skeleton
        return lp_instance.minimize_error(solver_params=solver_params)

    def validate_sequence_length_by_mass(
        self,
        start_skeleton: List[Set[str]],
        end_skeleton: List[Set[str]],
        nuc_masses: dict,
    ) -> bool:
        """
        Validate sequence length by mass.

        Parameters
        ----------
        start_skeleton : List[Set[str]]
            Skeleton in 5'-direction.
        end_skeleton : List[Set[str]]
            Skeleton in 3'-direction.
        nuc_masses : dict
            Dictionary assigning masses to each representative in alphabet.

        Returns
        -------
        bool
            Flag whether sequence length is valid.

        """
        min_mass = 0
        max_mass = 0
        for start_nucs, end_nucs in zip(start_skeleton, end_skeleton):
            min_mass += min(
                (nuc_masses[nuc] for nuc in (start_nucs | end_nucs)), default=0
            )
            max_mass += max(
                (nuc_masses[nuc] for nuc in (start_nucs | end_nucs)), default=0
            )

        # Check whether mass interval defined by skeleton contains sequence mass
        # Use MAX_VARIANCE to accommodate for uncertainty in sequence mass selection
        return (
            min_mass - MAX_VARIANCE
            <= self.inferrer.seq.su_mass
            <= max_mass + MAX_VARIANCE
        )

    def select_sequence_length_with_jaccard(
        self, start_skeleton: List[Set[str]], end_skeleton: List[Set[str]]
    ) -> int:
        """
        Select sequence length based on Jaccard index.

        Parameters
        ----------
        start_skeleton : List[Set[str]]
            Skeleton in 5'-direction.
        end_skeleton : List[Set[str]]
            Skeleton in 3'-direction.

        Returns
        -------
        int
            Selected sequence length.

        """
        # Reduce nucleotide alphabet based on skeleton parts
        nucleotides = {
            nuc
            for skeleton_pos in start_skeleton + end_skeleton
            for nuc in skeleton_pos
        }
        self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotides
        )

        # Initialize nucleotide mass dict
        nucleoside_masses = self.inferrer.alphabet.to_dict()

        # Determine lower and upper bound
        min_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="lower")
        max_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="upper")

        # Determine sequence length with the highest similarity between skeleton parts
        best_len = min_len
        best_val = -1
        for len_cand in range(min_len, max_len + 1):
            # Determine normalized sum of Jaccard similarity in each position
            value = (
                sum(
                    map(
                        jaccard_index,
                        zip(
                            start_skeleton[:len_cand],
                            end_skeleton[len(end_skeleton) - len_cand :],
                        ),
                    )
                )
                / len_cand
            )

            # Update best found sequence length if needed
            if value > best_val and self.validate_sequence_length_by_mass(
                start_skeleton=start_skeleton[:len_cand],
                end_skeleton=end_skeleton[len(end_skeleton) - len_cand :],
                nuc_masses=nucleoside_masses,
            ):
                best_val = value
                best_len = len_cand

        if best_val < 0:
            raise Exception(
                "No sequence length fitting the given sequence mass could be estimated."
            )

        return best_len

    def infer_compositions_for_bin_differences(
        self,
        prev_bin: StandardUnitFragments,
        current_bin: StandardUnitFragments,
    ) -> List[Composition]:
        """
        Infer compositions between two bins.

        Parameters
        ----------
        prev_bin : StandardUnitFragments
            Previous SU-fragment bin.
        current_bin : StandardUnitFragments
            Current SU-fragment bin.

        Returns
        -------
        List[Composition]
            List of compositions.

        """
        current_bin = current_bin.fragments
        # Collect compositions for first bin
        if prev_bin is None:
            compositions = [
                self.infer_compositions_for_mass_difference(
                    diff=row["standard_unit_mass"],
                    prev_mass=0.0,
                    current_mass=row["observed_mass"],
                )
                for row in current_bin.rows(named=True)
            ]

        # Collect compositions between previous and current bin
        else:
            prev_bin = prev_bin.fragments
            compositions = [
                self.infer_compositions_for_mass_difference(
                    diff=current_row["standard_unit_mass"]
                    - prev_row["standard_unit_mass"],
                    prev_mass=prev_row["observed_mass"],
                    current_mass=current_row["observed_mass"],
                )
                for prev_row in prev_bin.rows(named=True)
                for current_row in current_bin.rows(named=True)
            ]

        # If no valid composition was found, return None
        if all(comp is None for comp in compositions):
            return None

        # Flatten composition list
        compositions = [
            comp
            for comp_list in compositions
            if comp_list is not None
            for comp in comp_list
            if comp is not None
        ]

        # Remove duplicates from composition list
        unique_compositions = []
        for comp in compositions:
            if comp not in unique_compositions:
                unique_compositions.append(comp)

        return unique_compositions

    def infer_compositions_for_mass_difference(
        self,
        diff: float,
        prev_mass: float,
        current_mass: float,
    ) -> List[Composition]:
        """
        Infer compositions between two masses.

        Parameters
        ----------
        diff : float
            Difference between SU-masses.
        prev_mass : float
            Previous observed mass.
        current_mass : float
            Current observed mass.

        Returns
        -------
        List[Composition]
            List of compositions.

        """
        if diff in self.compositions:
            return self.compositions.get(diff, [])
        threshold = calculate_error_threshold(
            prev_mass,
            current_mass,
            self.inferrer.tolerance,
        )
        return calculate_compositions(
            diff,
            threshold,
            self.inferrer,
        )

    def update_skeleton_for_given_compositions(
        self,
        compositions: List[Composition],
        pos: Set[int],
        skeleton_seq: List[Set[str]],
    ) -> Tuple[Set[int], List[Set[str]]]:
        """
        Update skeleton for given compositions.

        Parameters
        ----------
        compositions : List[Composition]
            List of compositions.
        pos : Set[int]
            Set of possible follow-up indices.
        skeleton_seq : List[Set[str]]
            Skeleton sequence.

        Returns
        -------
        Set[int]
            Updated set of follow-up indices.
        List[Set[str]]
            Updated skeleton sequence.

        """
        next_pos = set()
        for p in pos:
            # Group compositions by length in dict
            alphabet_per_len = {
                comp_len: set(chain(*comps))
                for comp_len, comps in groupby(
                    [
                        comp
                        for comp in compositions
                        if 0 <= p + len(comp) - 1 < self.inferrer.seq.max_len
                    ],
                    len,
                )
            }

            # Constrain current sets in range of compositions by the new nucleotides
            for comp_len, alphabet in alphabet_per_len.items():
                for i in range(comp_len):
                    possible_nucleotides = skeleton_seq[p + i]

                    # Clear nucleotide set if the new composition sharpens it
                    if possible_nucleotides.issuperset(alphabet):
                        possible_nucleotides.clear()

                    # Add all nucleotides in current composition to set
                    for j in alphabet:
                        possible_nucleotides.add(j)
                        # TODO: We need to do this better.
                        #  Instead of adding just the letters, we somehow
                        #  need to keep a track of the possibilities to be
                        #  able to constrain the LP!

            # Update possible follow-up positions
            next_pos.update(p + comp_len for comp_len in alphabet_per_len)
        return next_pos, skeleton_seq


def jaccard_index(input_tuple: Tuple[Set[str], Set[str]]) -> float:
    """
    Determine Jaccard index.

    Parameters
    ----------
    input_tuple : Tuple[Set[str], Set[str]]
        Tuple of candidates for nucleotides at any skeleton position.

    Returns
    -------
    float
        Jaccard index.

    """
    # Return score for perfect similarity if one set is empty
    if len(input_tuple[0]) == 0 or len(input_tuple[1]) == 0:
        return 1

    # Return Jaccard score
    return len(input_tuple[0].intersection(input_tuple[1])) / len(
        input_tuple[0].union(input_tuple[1])
    )


def combine_skeleton_sequences(
    seq_len: int,
    start_skeleton: List[Set[str]],
    end_skeleton: List[Set[str]],
) -> List[Set[str]]:
    """
    Combine directional skeleton sequences into final one.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    start_skeleton : List[Set[str]]
        Skeleton in 5'-direction.
    end_skeleton : List[Set[str]]
        Skeleton in 3'-direction.

    Returns
    -------
    List[Set[str]]
        Skeleton sequence.

    """
    # Adapt directed skeleton parts to have correct length
    start_skeleton = start_skeleton[:seq_len]
    end_skeleton = end_skeleton[len(end_skeleton) - seq_len :]

    skeleton_seq = [set() for _ in range(seq_len)]
    for i in range(seq_len):
        # Preferentially consider nucleotides where start and end agree
        skeleton_seq[i] = start_skeleton[i].intersection(end_skeleton[i])

        # If the intersection is empty, use the union instead
        if not skeleton_seq[i]:
            skeleton_seq[i] = start_skeleton[i].union(end_skeleton[i])

    # TODO: Its more complicated, since if two positions are ambiguous,
    #  they are not independent. If one nucleotide is selected this way,
    #  then the same nucleotide cannot be selected in the other position!

    return skeleton_seq
