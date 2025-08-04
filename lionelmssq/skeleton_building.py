from dataclasses import dataclass
from itertools import chain, groupby
from typing import List, Optional, Set, Tuple
from loguru import logger
import polars as pl

from lionelmssq.common import (
    Explanation,
    calculate_error_threshold,
    calculate_explanations,
)
from lionelmssq.mass_table import DynamicProgrammingTable


@dataclass
class SkeletonBuilder:
    explanations: list[Explanation]
    seq_len: int
    dp_table: DynamicProgrammingTable

    def build_skeleton(
        self, modification_rate: float, fragments: pl.DataFrame
    ) -> Tuple[List[Set[str]], pl.DataFrame]:
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments = self._predict_skeleton(
            modification_rate=modification_rate,
            fragments=fragments.filter(pl.col("breakage").str.contains("START")),
            skeleton_seq=[set() for _ in range(self.seq_len)],
        )
        print("Skeleton sequence start = ", start_skeleton)

        # Build skeleton sequence from 3'-end
        end_skeleton, end_fragments = self._predict_skeleton(
            modification_rate=modification_rate,
            fragments=fragments.filter(pl.col("breakage").str.contains("END")),
            skeleton_seq=[set() for _ in range(self.seq_len)],
        )
        # Reverse skeleton from END fragments
        end_skeleton = end_skeleton[::-1]

        # Adapt end indices to reverse indexation of END fragments
        end_fragments = end_fragments.with_columns(
            pl.struct("min_end", "max_end")
            .map_elements(lambda x: -1 - x["max_end"], return_dtype=int)
            .alias("min_end"),
            pl.struct("min_end", "max_end")
            .map_elements(lambda x: -1 - x["min_end"], return_dtype=int)
            .alias("max_end"),
        )

        # Ensure fragments only occur once
        end_fragments = end_fragments.filter(
            ~pl.col("index").is_in(start_fragments.get_column("index").to_list())
        )

        print("Skeleton sequence end = ", end_skeleton)

        # Combine both skeleton sequences
        skeleton_seq = self._align_skeletons(start_skeleton, end_skeleton)
        print("Skeleton sequence = ", skeleton_seq)

        # Return skeleton and valid terminal fragments
        return (
            skeleton_seq,
            pl.concat([start_fragments, end_fragments]),
        )

    def _predict_skeleton(
        self,
        modification_rate,
        fragments,
        skeleton_seq: Optional[List[Set[str]]] = None,
    ) -> Tuple[List[Set[str]], pl.DataFrame]:
        # Initialize skeleton sequence (if not already given)
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.seq_len)]

        # METHOD: Reject fragments which are not explained well by mass
        # differences. While iterating through the fragments, bin them
        # to keep track of similar masses and reject them in bulk.
        pos = {0}
        last_valid_bin = None

        invalid_list = []
        current_bin = [0]
        for frag_idx in range(1, len(fragments)):
            # Stop if no positions are left to fill
            if len(pos) == 0:
                invalid_list.append(fragments.item(frag_idx, "index"))
                continue

            # Define mass difference and threshold between neighbouring fragments
            neighbour_diff = fragments.item(
                frag_idx, "standard_unit_mass"
            ) - fragments.item(frag_idx - 1, "standard_unit_mass")
            neighbour_threshold = calculate_error_threshold(
                fragments.item(frag_idx - 1, "observed_mass"),
                fragments.item(frag_idx, "observed_mass"),
                self.dp_table.tolerance,
            )

            # Bin fragments with similar mass together
            if neighbour_diff <= neighbour_threshold:
                current_bin.append(frag_idx)
                continue

            explanations = self.explain_bin_differences(
                prev_bin=last_valid_bin,
                current_bin=current_bin,
                fragments=fragments,
                modification_rate=modification_rate,
            )

            # Skip bins without any explanation
            if explanations is None:
                for idx in current_bin:
                    # Add a warning in the log for the skipped fragment
                    logger.warning(
                        f"Skipping {fragments.item(idx, 'breakage')} fragment "
                        f"{fragments.item(idx, 'index')} with observed mass "
                        f"{fragments.item(idx, 'observed_mass'):.4f} and SU "
                        f"mass {fragments.item(idx, 'standard_unit_mass'):.4f}"
                        f" because no explanations were found."
                    )

                    invalid_list.append(fragments.item(idx, "index"))
            else:
                # Continue skeleton building
                pos, skeleton_seq = self.update_skeleton_for_given_explanations(
                    explanations=explanations,
                    pos=pos,
                    skeleton_seq=skeleton_seq,
                )

                # Adapt information on end index for given bin
                for idx in current_bin:
                    fragments[idx, "min_end"] = min(pos, default=0)
                    fragments[idx, "max_end"] = max(pos, default=-1)

                # Update information for previous bin
                last_valid_bin = current_bin

            # Update information for current bin
            current_bin = [frag_idx]

        # Filter out all invalid fragments
        fragments = fragments.filter(~pl.col("index").is_in(invalid_list))

        return skeleton_seq, fragments

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

    def explain_bin_differences(
        self,
        prev_bin: list,
        current_bin: list,
        fragments: pl.DataFrame,
        modification_rate: float,
    ) -> List[Explanation]:
        # Collect mass explanations for first bin
        if prev_bin is None:
            explanations = [
                self.explain_mass_difference(
                    diff=fragments.item(idx, "standard_unit_mass"),
                    prev_mass=0.0,
                    current_mass=fragments.item(idx, "observed_mass"),
                    modification_rate=modification_rate,
                )
                for idx in current_bin
            ]
        # Collect mass explanations between previous and current bin
        else:
            explanations = [
                self.explain_mass_difference(
                    diff=fragments.item(current_idx, "standard_unit_mass")
                    - fragments.item(prev_idx, "standard_unit_mass"),
                    prev_mass=fragments.item(prev_idx, "observed_mass"),
                    current_mass=fragments.item(current_idx, "observed_mass"),
                    modification_rate=modification_rate,
                )
                for prev_idx in prev_bin
                for current_idx in current_bin
            ]

        return (
            None
            if all(expl is None for expl in explanations)
            else [
                expl
                for expl_list in explanations
                if expl_list is not None
                for expl in expl_list
                if expl is not None
            ]
        )

    def explain_mass_difference(
        self,
        diff: float,
        prev_mass: float,
        current_mass: float,
        modification_rate: float,
    ) -> List[Explanation]:
        if diff in self.explanations:
            return self.explanations.get(diff, [])
        else:
            threshold = calculate_error_threshold(
                prev_mass,
                current_mass,
                self.dp_table.tolerance,
            )
            return calculate_explanations(
                diff,
                threshold,
                modification_rate,
                self.seq_len,
                self.dp_table,
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
