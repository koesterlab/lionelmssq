from dataclasses import dataclass
from itertools import chain, groupby
from typing import List, Optional, Set, Tuple
from loguru import logger

from lionelmssq.common import (
    Explanation,
    Side,
    calculate_diff_dp,
    calculate_diff_errors,
)
from lionelmssq.mass_table import DynamicProgrammingTable


@dataclass
class TerminalFragment:
    index: int  # Fragment index
    min_end: int  # Minimum length of fragment (negative for end fragments)
    max_end: int  # Maximum length of fragment (negative for end fragments)


@dataclass
class SkeletonBuilder:
    fragments_side: dict
    mass_diffs: dict
    explanations: list[Explanation]
    seq_len: int
    dp_table: DynamicProgrammingTable

    def build_skeleton(
        self, modification_rate: float, breakage_dict: dict
    ) -> Tuple[
        List[Set[str]],
        List[TerminalFragment],
        List[TerminalFragment],
        List[int],
        List[int],
    ]:
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments, non_start_fragments = self._predict_skeleton(
            modification_rate=modification_rate,
            breakage_dict=breakage_dict,
            fragments=self.fragments_side[Side.START],
            mass_diffs=self.mass_diffs[Side.START],
            skeleton_seq=[set() for _ in range(self.seq_len)],
        )
        print("Skeleton sequence start = ", start_skeleton)

        # Build skeleton sequence from 3'-end
        end_skeleton, end_fragments, non_end_fragments = self._predict_skeleton(
            modification_rate=modification_rate,
            breakage_dict=breakage_dict,
            fragments=self.fragments_side[Side.END],
            mass_diffs=self.mass_diffs[Side.END],
            skeleton_seq=[set() for _ in range(self.seq_len)],
        )
        end_skeleton = end_skeleton[::-1]
        end_fragments = adjust_end_fragment_info(end_fragments)
        print("Skeleton sequence end = ", end_skeleton)

        # Combine both skeleton sequences
        skeleton_seq = self._align_skeletons(start_skeleton, end_skeleton)
        print("Skeleton sequence = ", skeleton_seq)

        # Return skeleton and fragments (divided by sequence end and validity)
        return (
            skeleton_seq,
            start_fragments,
            end_fragments,
            non_start_fragments,
            non_end_fragments,
        )

    def _predict_skeleton(
        self,
        modification_rate,
        breakage_dict,
        fragments,
        mass_diffs,
        skeleton_seq: Optional[List[Set[str]]] = None,
    ) -> Tuple[List[Set[str]], List[TerminalFragment], List[int]]:
        # Initialize skeleton sequence (if not already given)
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.seq_len)]

        # METHOD: Reject fragments which are not explained well by mass
        # differences. While iterating through the fragments,
        # the "carry_over_mass" keeps a track of the rejected mass
        # difference (to add to the next considered difference) and
        # "last_valid_mass" keeps a track of the last not-rejected mass
        pos = {0}
        carry_over_mass = 0.0
        last_valid_mass = 0.0

        # Get indices from fragments dataframe
        idx_observed_mass = fragments.get_column_index("observed_mass")
        idx_su_mass = fragments.get_column_index("standard_unit_mass")
        idx_breakage = fragments.get_column_index("breakage")
        idx_index = fragments.get_column_index("index")

        fragments_valid = []
        fragments_invalid = []
        for diff, frag in zip(mass_diffs, fragments.rows()):
            diff += carry_over_mass
            assert pos

            explanations = self.explain_difference(
                diff=diff,
                prev_mass=last_valid_mass,
                current_mass=frag[idx_observed_mass],
                modification_rate=modification_rate,
                breakage_dict=breakage_dict,
            )

            if explanations is None:
                is_valid = False
            else:
                next_pos, skeleton_seq = self.update_skeleton_for_given_explanations(
                    explanations=explanations,
                    pos=pos,
                    skeleton_seq=skeleton_seq,
                )
                is_valid = len(next_pos) != 0

            if is_valid:
                fragments_valid.append(
                    TerminalFragment(
                        index=frag[idx_index],
                        min_end=min(next_pos, default=None),
                        max_end=max(next_pos, default=None),
                    )
                )
                pos = next_pos
                last_valid_mass = frag[idx_observed_mass]
                carry_over_mass = 0.0
            else:
                logger.warning(
                    f"Skipping {frag[idx_breakage]} fragment {frag[idx_index]} "
                    f"with observed mass {frag[idx_observed_mass]}/SU mass "
                    f"{frag[idx_su_mass]} because no explanations were found."
                )
                carry_over_mass = diff

                # Consider the skipped fragments as internal fragments! Add back the terminal mass to this fragments!
                fragments_invalid.append(frag[idx_index])

        return skeleton_seq, fragments_valid, fragments_invalid

    def _align_skeletons(
        self, start_skeleton: List[Set[str]], end_skeleton: List[Set[str]]
    ) -> List[Set[str]]:
        skeleton_seq = [set() for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            # Preferentially consider nucleotides where start and end agree
            skeleton_seq[i] = start_skeleton[i].intersection(end_skeleton[i])
            # If the intersection is empty, use the union instead
            if not skeleton_seq[i]:
                skeleton_seq[i] = start_skeleton[i].union(end_skeleton[i])

        # TODO: Its more complicated, since if two positions are ambiguous,
        #  they are not independent. If one nucleotide is selected this way,
        #  then the same nucleotide cannot be selected in the other position!

        return skeleton_seq

    def explain_difference(
        self, diff, prev_mass, current_mass, modification_rate, breakage_dict
    ):
        if diff in self.explanations:
            return self.explanations.get(diff, [])
        else:
            threshold = calculate_diff_errors(
                prev_mass,
                current_mass,
                self.dp_table.tolerance,
            )
            return calculate_diff_dp(
                diff,
                threshold,  # * diff,
                modification_rate,
                self.seq_len,
                self.dp_table,
                breakage_dict,
                su_mode=True,
            )

    def update_skeleton_for_given_explanations(
        self,
        explanations: List[Explanation],
        pos: Set[int],
        skeleton_seq: List[Set[str]],
    ):
        next_pos = set()
        for p in pos:
            # Group explanations by length in dict
            alphabet_per_expl_len = {
                expl_len: set(chain(*expls))
                for expl_len, expls in groupby(
                    [
                        expl
                        for expl in explanations
                        if 0 <= p + len(expl) < self.seq_len
                    ],
                    len,
                )
            }

            # Constrain current sets in range of explanation by the new nucleotides
            for expl_len, alphabet in alphabet_per_expl_len.items():
                for i in range(expl_len):
                    possible_nucleotides = skeleton_seq[p + i]

                    # Clear nucleotide set if the new explanation sharpens it
                    if possible_nucleotides.issuperset(alphabet):
                        possible_nucleotides.clear()

                    # Add all nucleotides in current explanation to set
                    for j in alphabet:
                        possible_nucleotides.add(j)
                        # TODO: We need to do this better.
                        #  Instead of adding just the letters, we somehow
                        #  need to keep a track of the possibilities to be
                        #  able to constrain the LP!

            # Update possible follow-up positions
            next_pos.update(p + expl_len for expl_len in alphabet_per_expl_len)
        return next_pos, skeleton_seq


def adjust_end_fragment_info(fragments):
    return [
        TerminalFragment(fragment.index, -1 - fragment.max_end, -1 - fragment.min_end)
        for fragment in fragments
    ]
