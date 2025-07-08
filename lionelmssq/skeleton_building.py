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
    index: int  # fragment index
    min_end: int  # minimum length of fragment (negative for end fragments)
    max_end: int  # maximum length of fragment (negative for end fragments)


@dataclass
class SkeletonBuilder:
    fragment_masses: dict
    fragments_side: dict
    mass_diffs: dict
    explanations: list[Explanation]
    seq_len: int
    matching_threshold: float
    mass_tags: dict
    dp_table: DynamicProgrammingTable

    # def __init__(self, fragment_masses, fragments_side, seq_len):
    #     pass


    def build_skeleton(
        self, modification_rate
    ) -> Tuple[
        List[Set[str]],
        List[TerminalFragment],
        List[TerminalFragment],
        List[int],
        List[int],
    ]:
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments, non_start_fragments = (
            self._predict_skeleton(Side.START, modification_rate)
        )
        print("Skeleton sequence start = ", start_skeleton)

        # Build skeleton sequence from 3'-end
        end_skeleton, end_fragments, non_end_fragments = self._predict_skeleton(
            Side.END, modification_rate
        )
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
        side: Side,
        modification_rate,
        skeleton_seq: Optional[List[Set[str]]] = None,
        fragment_masses=None,
        candidate_fragments=None,
    ) -> Tuple[List[Set[str]], List[TerminalFragment], List[int]]:
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.seq_len)]

        factor = 1 if side == Side.START else -1

        if not fragment_masses:
            fragment_masses = self.fragment_masses[side]

        if not candidate_fragments:
            candidate_fragments = (
                self.fragments_side[side].get_column("index").to_list()
            )

        def get_possible_nucleosides(pos: int, i: int) -> Set[str]:
            return skeleton_seq[pos + factor * i]

        def is_valid_pos(pos: int, ext: int) -> bool:
            pos = pos + factor * ext
            return (
                # 0 <= pos <= self.seq_len
                0 <= pos < self.seq_len
                if side == Side.START
                # else -(self.seq_len + 1) <= pos < 0
                else -self.seq_len <= pos < 0
            )

        pos = {0} if side == Side.START else {-1}

        # We reject some masses/some fragments which are not explained well by mass differences.
        # While iterating through the fragments, the "carry_over_mass" keeps a track of the rejected mass difference.
        # This is added to "diff" to get the next mass_difference.
        carry_over_mass = 0

        # "last_mass_valid" keeps a track of the last mass which was NOT rejected.
        # This is useful for calculating the difference threshold
        last_mass_valid = 0.0

        valid_terminal_fragments = []
        invalid_fragments = []
        for fragment_index, (diff, mass) in enumerate(
            zip(self.mass_diffs[side], fragment_masses)
        ):
            diff += carry_over_mass
            assert pos

            is_valid = True
            if diff in self.explanations:
                explanations = self.explanations.get(diff, [])
            else:
                threshold = calculate_diff_errors(
                    last_mass_valid + self.mass_tags[side],
                    last_mass_valid + diff + self.mass_tags[side],
                    self.matching_threshold,
                )
                explanations = calculate_diff_dp(
                    diff, threshold, modification_rate, self.seq_len,
                    self.dp_table
                )

            # METHOD: if there is only one single nucleoside explanation, we can
            # directly assign the nucleoside if there are tuples, we have to assign a
            # possibility to the current and next position
            # and take care of expressing that both permutations are possible
            # Further, we consider multiple possible start positions, depending on
            # whether the previous explanations had different lengths.
            if explanations:
                next_pos = set()
                if side is Side.START:
                    min_p = min(pos)
                    max_p = max(pos)
                else:
                    min_p = max(pos)
                    max_p = min(pos)
                min_fragment_end = None
                max_fragment_end = None

                for p in pos:
                    p_specific_explanations = [
                        expl for expl in explanations if is_valid_pos(p, len(expl))
                    ]
                    alphabet_per_expl_len = {
                        expl_len: set(chain(*expls))
                        for expl_len, expls in groupby(p_specific_explanations, len)
                    }

                    if p_specific_explanations:
                        if p == min_p:
                            min_fragment_end = min_p + factor * min(
                                expl_len for expl_len in alphabet_per_expl_len
                            )
                        elif p == max_p:
                            max_fragment_end = max_p + factor * max(
                                expl_len for expl_len in alphabet_per_expl_len
                            )

                    # constrain already existing sets in the range of the expl
                    # by the nucs that are given in the explanations
                    for expl_len, alphabet in alphabet_per_expl_len.items():
                        for i in range(expl_len):
                            possible_nucleosides = get_possible_nucleosides(p, i)

                            if possible_nucleosides.issuperset(alphabet):
                                # If the current explanation sharpens the list of possibilities, clear all
                                # prior possibilities before the new explanation will add the sharpened ones below
                                possible_nucleosides.clear()

                            for j in alphabet:
                                possible_nucleosides.add(j)
                                # TODO: We need to do this better.
                                # Instead of adding just the letters, we somehow need to keep a track of the possibilities to be able to constrain the LP!
                                # We also then probably need part of the code immediately below!

                    # add possible follow up positions
                    next_pos.update(
                        p + factor * expl_len for expl_len in alphabet_per_expl_len
                    )
                if max_fragment_end is None:
                    max_fragment_end = min_fragment_end
                if min_fragment_end is None:
                    min_fragment_end = max_fragment_end
                if min_fragment_end is None:
                    # still None => both are None
                    # TODO can we stop early in building the ladder?
                    is_valid = False
            elif (
                abs(diff) <= self.matching_threshold * abs(mass + self.mass_tags[side])
                # Problem! The above approach might blow up if the masses are very close, i.e. diff is very close to zero!
            ):
                if side == Side.START:
                    min_fragment_end = min(pos)
                    max_fragment_end = max(pos)
                else:
                    min_fragment_end = max(pos)
                    max_fragment_end = min(pos)
                next_pos = pos
            else:
                is_valid = False

            if is_valid:
                valid_terminal_fragments.append(
                    TerminalFragment(
                        index=candidate_fragments[fragment_index],
                        min_end=min_fragment_end,
                        max_end=max_fragment_end,
                    )
                )
                pos = next_pos
                last_mass_valid = mass
                carry_over_mass = 0.0
            else:
                logger.warning(
                    f"Skipping {side} fragment {fragment_index} with observed mass {mass} because no "
                    "explanations are found for the mass difference."
                )
                carry_over_mass = diff

                # Consider the skipped fragments as internal fragments! Add back the terminal mass to this fragments!
                if fragment_index and candidate_fragments:
                    invalid_fragments.append(candidate_fragments[fragment_index])

        return skeleton_seq, valid_terminal_fragments, invalid_fragments



    def _align_skeletons(self, skeleton_seq_start, skeleton_seq_end) -> List[Set[str]]:
        # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
        # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

        # Align the skeletons of the start and end fragments to get the final skeleton sequence!
        # Wherever there is no ambiguity, that nucleotide is preferrentially considered!

        skeleton_seq = [set() for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            skeleton_seq[i] = skeleton_seq_start[i].intersection(skeleton_seq_end[i])
            if not skeleton_seq[i]:
                skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])

        # TODO: Its more complicated, since if two positions are ambiguous,
        #  they are not independent. If one nucleotide is selected this way,
        #  then the same nucleotide cannot be selected in the other position!

        return skeleton_seq
