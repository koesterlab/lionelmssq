# -*- coding: utf-8 -*-
"""Building of sequence skeletons."""

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np

from spectrseqtools.common import calculate_error_threshold
from spectrseqtools.compositions import CompositionList
from spectrseqtools.dataclasses import SolverParameters
from spectrseqtools.fragments import StandardUnitFragments
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
    compute_sequence_length_bound,
    infer_compositions_with_matrix,
)
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.sequence import SkeletonSequence


@dataclass
class SkeletonBuilder:
    """Class to build skeleton sequence."""

    compositions: dict
    inferrer: CompositionInferrer

    def build_skeleton(
        self, fragments: StandardUnitFragments, solver_params: SolverParameters
    ) -> Tuple[SkeletonSequence, StandardUnitFragments]:
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
        SkeletonSequence
            Skeleton sequence.
        StandardUnitFragments
            SU-fragments after skeleton building.

        """
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments = self._predict_skeleton(
            fragments=fragments.start,
            skeleton_seq=SkeletonSequence.empty(seq_len=self.inferrer.seq.max_len),
        )

        # Build skeleton sequence from 3'-end and reverse it
        end_skeleton, end_fragments = self._predict_skeleton(
            fragments=fragments.end,
            skeleton_seq=SkeletonSequence.empty(seq_len=self.inferrer.seq.max_len),
        )
        end_skeleton = end_skeleton.reverse

        # Reduce nucleotide alphabet based on skeleton parts
        mapping = (
            self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
                alphabet=start_skeleton.nucleotides.union(end_skeleton.nucleotides)
            )
        )
        start_skeleton.update_indexing(mapping=mapping)
        end_skeleton.update_indexing(mapping=mapping)

        print("Skeleton sequence (5'-end)\t= ", start_skeleton)
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
        skeleton_seq = start_skeleton.merge(other=end_skeleton, seq_len=seq_len)
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
        skeleton_seq: Optional[SkeletonSequence] = None,
    ) -> Tuple[SkeletonSequence, StandardUnitFragments]:
        """
        Predict directional skeleton from given fragments.

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments to build skeleton.
        skeleton_seq : SkeletonSequence
            Skeleton sequence.

        Returns
        -------
        SkeletonSequence
            Directional skeleton sequence.
        StandardUnitFragments
            Terminal SU-fragments used for skeleton building.

        """
        # Initialize skeleton sequence (if not already given)
        if skeleton_seq is None:
            skeleton_seq = SkeletonSequence.empty(seq_len=self.inferrer.seq.max_len)

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
            if len(compositions) == 0:
                invalid_list += current_bin.invalidate()
            else:
                # Continue skeleton building
                pos = skeleton_seq.update_with_compositions(compositions, pos)

                # Update information on end index
                current_bin.update_end_indices(pos=pos)

                # Update information on last valid bin
                last_valid_bin = current_bin

        return skeleton_seq, StandardUnitFragments.from_bins(
            bins=bins, invalid_list=invalid_list
        )

    def select_sequence_length_with_lp(
        self,
        start_skeleton: SkeletonSequence,
        end_skeleton: SkeletonSequence,
        start_fragments: StandardUnitFragments,
        end_fragments: StandardUnitFragments,
        solver_params: SolverParameters,
    ) -> int:
        """
        Select sequence length based on LP score.

        Parameters
        ----------
        start_skeleton : SkeletonSequence
            Skeleton in 5'-direction.
        end_skeleton : SkeletonSequence
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
        # Determine sequence length with the best LP score
        best_len = -1
        best_val = np.inf
        for len_cand in range(
            compute_sequence_length_bound(inferrer=self.inferrer, dir="lower"),
            compute_sequence_length_bound(inferrer=self.inferrer, dir="upper") + 1,
        ):
            # Skip candidates resulting in invalid sequences
            # TODO: Use merged sequence for additional tightening of bounds
            if not self.inferrer.seq.validate_sequence(
                seq=start_skeleton.combine(other=end_skeleton, seq_len=len_cand),
                alphabet=self.inferrer.alphabet,
            ):
                continue

            # Merge directional skeletons
            seq = start_skeleton.merge(other=end_skeleton, seq_len=len_cand)

            # Combine directional terminal-fragment lists
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
            if value < best_val:
                best_val = value
                best_len = len_cand

        return best_len

    def determine_lp_score(
        self,
        terminal_fragments: StandardUnitFragments,
        skeleton_seq: SkeletonSequence,
        solver_params: SolverParameters,
    ) -> float:
        """

        Parameters
        ----------
        terminal_fragments : StandardUnitFragments
            Terminal SU-fragments.
        skeleton_seq : SkeletonSequence
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

    def select_sequence_length_with_jaccard(
        self, start_skeleton: SkeletonSequence, end_skeleton: SkeletonSequence
    ) -> int:
        """
        Select sequence length based on Jaccard index.

        Parameters
        ----------
        start_skeleton : SkeletonSequence
            Skeleton in 5'-direction.
        end_skeleton : SkeletonSequence
            Skeleton in 3'-direction.

        Returns
        -------
        int
            Selected sequence length.

        """
        # Determine lower and upper bound
        min_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="lower")
        max_len = compute_sequence_length_bound(inferrer=self.inferrer, dir="upper")

        # Determine sequence length with the highest similarity between skeleton parts
        best_len = min_len
        best_val = -1
        for len_cand in range(min_len, max_len + 1):
            # Skip candidates resulting in invalid sequences
            if not self.inferrer.seq.validate_sequence(
                seq=start_skeleton.combine(other=end_skeleton, seq_len=len_cand),
                alphabet=self.inferrer.alphabet,
            ):
                continue

            # Determine normalized sum of Jaccard similarity in each position
            value = (
                sum(
                    map(
                        jaccard_index,
                        zip(
                            start_skeleton.sequence[:len_cand],
                            end_skeleton.sequence[len(end_skeleton) - len_cand :],
                        ),
                    )
                )
                / len_cand
            )

            # Update best found sequence length if needed
            if value > best_val:
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
    ) -> CompositionList:
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
        CompositionList
            List of compositions.

        """
        # Collect compositions for first bin
        current_bin = current_bin.fragments
        if prev_bin is None:
            return sum(
                (
                    self.infer_compositions_for_mass_difference(
                        diff=row["standard_unit_mass"],
                        prev_mass=0.0,
                        current_mass=row["observed_mass"],
                    )
                    for row in current_bin.rows(named=True)
                ),
                start=CompositionList(),
            )

        # Collect compositions between previous and current bin
        prev_bin = prev_bin.fragments
        return sum(
            (
                self.infer_compositions_for_mass_difference(
                    diff=current_row["standard_unit_mass"]
                    - prev_row["standard_unit_mass"],
                    prev_mass=prev_row["observed_mass"],
                    current_mass=current_row["observed_mass"],
                )
                for prev_row in prev_bin.rows(named=True)
                for current_row in current_bin.rows(named=True)
            ),
            start=CompositionList(),
        )

    def infer_compositions_for_mass_difference(
        self,
        diff: float,
        prev_mass: float,
        current_mass: float,
    ) -> CompositionList:
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
        CompositionList
            List of compositions.

        """
        if diff in self.compositions:
            return self.compositions.get(diff, CompositionList())

        threshold = calculate_error_threshold(
            prev_mass,
            current_mass,
            self.inferrer.tolerance,
        )
        return infer_compositions_with_matrix(
            mass=diff,
            inferrer=self.inferrer,
            max_modifications=round(
                self.inferrer.seq.modification_rate * self.inferrer.seq.max_len
            ),
            threshold=threshold,
        )


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
