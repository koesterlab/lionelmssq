# -*- coding: utf-8 -*-
"""Building of sequence skeletons."""

from dataclasses import dataclass
from typing import Optional, Tuple

from spectrseqtools.compositions import CompositionList
from spectrseqtools.fragments import StandardUnitFragments
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.sequence import SkeletonSequence
from spectrseqtools.sequence_length import SequenceLengthEstimator


@dataclass
class SkeletonBuilder:
    """Class to build skeleton sequence."""

    compositions: dict
    inferrer: CompositionInferrer

    def build_skeleton(
        self,
        fragments: StandardUnitFragments,
        estimator: SequenceLengthEstimator,
    ) -> Tuple[SkeletonSequence, StandardUnitFragments]:
        """
        Build skeleton from given fragments.

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments to build skeleton.
        estimator: SequenceLengthEstimator
            Sequence length estimator.

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
        mapping = self.inferrer.reduce_alphabet(
            new_alphabet=start_skeleton.nucleotides.union(end_skeleton.nucleotides)
        )
        self.inferrer.update_sequence_length()
        start_skeleton.update_indexing(mapping=mapping)
        end_skeleton.update_indexing(mapping=mapping)

        print("Skeleton sequence (5'-end)\t= ", start_skeleton)
        print("Skeleton sequence (3'-end)\t= ", end_skeleton)

        # Select best sequence length with LP
        seq_len = estimator.select_sequence_length(
            start_fragments=start_fragments,
            end_fragments=end_fragments,
            start_skeleton=start_skeleton,
            end_skeleton=end_skeleton,
        )
        print("Estimated sequence length:", seq_len)

        # Combine both skeleton sequences
        skeleton_seq = start_skeleton.merge(other=end_skeleton, seq_len=seq_len)
        self.inferrer.update_sequence_length(seq_len=len(skeleton_seq))
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
        bins = fragments.bin(error=self.inferrer.error)
        for current_bin in bins:
            # Stop if no positions are left to fill
            if len(pos) == 0:
                invalid_list += current_bin.invalidate()
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

        return self.inferrer.infer_compositions(
            mass=diff, obs_masses=[prev_mass, current_mass]
        )
