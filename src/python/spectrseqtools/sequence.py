# -*- coding: utf-8 -*-
"""Module for sequence classes."""

from dataclasses import dataclass
from itertools import chain, groupby
from typing import List, Self, Set


@dataclass
class SkeletonSequence:
    """Class for skeleton sequence."""

    sequence: List[Set[str]]

    def __iter__(self):
        return self.sequence.__iter__()

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return self.sequence.__repr__()

    @classmethod
    def empty(cls, seq_len) -> Self:
        """Return empty skeleton sequence of given length."""
        return cls(sequence=[set() for _ in range(seq_len)])

    @property
    def reverse(self) -> Self:
        """Return reverse skeleton sequence."""
        return SkeletonSequence(self.sequence[::-1])

    @property
    def nucleotides(self) -> Set:
        """Return set of nucleotides found at any position in skeleton sequence."""
        return {nuc for seq_pos in self.sequence for nuc in seq_pos}

    def min_mass(self, nuc_masses: dict) -> float:
        """Return minimum mass a sequence from the skeleton could possibly have."""
        total_mass = 0
        for nucs in self.sequence:
            total_mass += min((nuc_masses[nuc] for nuc in nucs), default=0)
        return total_mass

    def max_mass(self, nuc_masses: dict) -> float:
        """Return maximum mass a sequence from the skeleton could possibly have."""
        total_mass = 0
        for nucs in self.sequence:
            total_mass += max((nuc_masses[nuc] for nuc in nucs), default=0)
        return total_mass

    def update_with_compositions(
        self,
        compositions,
        pos: Set[int],
    ) -> Set[int]:
        """
        Update skeleton based on given compositions.

        Parameters
        ----------
        compositions : List[Composition]
            List of compositions.
        pos : Set[int]
            Set of possible follow-up indices.

        Returns
        -------
        Set[int]
            Updated set of follow-up indices.

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
                        if 0 <= p + len(comp) - 1 < len(self.sequence)
                    ],
                    len,
                )
            }

            # Constrain current sets in range of compositions by the new nucleotides
            for comp_len, alphabet in alphabet_per_len.items():
                for i in range(comp_len):
                    # Clear nucleotide set if the new composition sharpens it
                    if self.sequence[p + i].issuperset(alphabet):
                        self.sequence[p + i].clear()

                    # Add all nucleotides in current composition to set
                    for j in alphabet:
                        self.sequence[p + i].add(j)
                        # TODO: We need to do this better.
                        #  Instead of adding just the letters, we somehow
                        #  need to keep a track of the possibilities to be
                        #  able to constrain the LP!

            # Update possible follow-up positions
            next_pos.update(p + comp_len for comp_len in alphabet_per_len)
        return next_pos

    def merge(self, other: Self, seq_len: int) -> Self:
        """
        Merge with other skeleton sequence.

        Parameters
        ----------
        other : SkeletonSequence
            Other skeleton sequence.
        seq_len : int
            Sequence length.

        """
        # Adapt directed skeleton parts to have correct length
        start_skeleton = self.sequence[:seq_len]
        end_skeleton = other.sequence[len(other) - seq_len :]

        merged_skeleton = [set() for _ in range(seq_len)]
        for i in range(seq_len):
            # Preferentially consider nucleotides where start and end agree
            merged_skeleton[i] = start_skeleton[i].intersection(end_skeleton[i])

            # If the intersection is empty, use the union instead
            if not merged_skeleton[i]:
                merged_skeleton[i] = start_skeleton[i].union(end_skeleton[i])

        # TODO: Its more complicated, since if two positions are ambiguous,
        #  they are not independent. If one nucleotide is selected this way,
        #  then the same nucleotide cannot be selected in the other position!

        return SkeletonSequence(sequence=merged_skeleton)

    def combine(self, other: Self, seq_len: int) -> Self:
        """
        Combine with other skeleton sequence without merging.

        Parameters
        ----------
        other : SkeletonSequence
            Other skeleton sequence.
        seq_len : int
            Sequence length.

        """
        # Adapt directed skeleton parts to have correct length
        start_skeleton = self.sequence[:seq_len]
        end_skeleton = other.sequence[len(other) - seq_len :]

        combined_skeleton = [set() for _ in range(seq_len)]
        for i in range(seq_len):
            combined_skeleton[i] = start_skeleton[i].union(end_skeleton[i])

        return SkeletonSequence(sequence=combined_skeleton)
