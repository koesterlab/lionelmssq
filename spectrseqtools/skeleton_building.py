from dataclasses import dataclass
from itertools import chain, groupby
from typing import List, Optional, Set, Tuple
from loguru import logger
import numpy as np
import polars as pl

from spectrseqtools.common import (
    Composition,
    calculate_error_threshold,
    calculate_compositions,
)
from spectrseqtools.fragment_classification import MAX_VARIANCE
from spectrseqtools.sequence_inference import LinearProgramInstance
from spectrseqtools.traceback_matrix import (
    CompositionInferrer,
    compute_sequence_length_bound,
)


@dataclass
class SkeletonBuilder:
    compositions: dict
    inferrer: CompositionInferrer

    def build_skeleton(
        self, fragments: pl.DataFrame, solver_params: dict
    ) -> Tuple[List[Set[str]], pl.DataFrame]:
        # Build skeleton sequence from 5'-end
        start_skeleton, start_fragments = self._predict_skeleton(
            fragments=fragments.filter(pl.col("fragmentation").str.contains("START")),
            skeleton_seq=[set() for _ in range(self.inferrer.seq.max_len)],
        )
        print("Skeleton sequence (5'-end) = ", start_skeleton)

        # Build skeleton sequence from 3'-end and reverse it
        end_skeleton, end_fragments = self._predict_skeleton(
            fragments=fragments.filter(pl.col("fragmentation").str.contains("END")),
            skeleton_seq=[set() for _ in range(self.inferrer.seq.max_len)],
        )
        end_skeleton = end_skeleton[::-1]
        print("Skeleton sequence (3'-end) = ", end_skeleton)

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
        print("Skeleton sequence = ", skeleton_seq)

        # Ensure fragments only occur once
        end_fragments = end_fragments.filter(
            ~pl.col("index").is_in(start_fragments.get_column("index").to_list())
        )

        # Remove indexing of the next pos for START fragments
        start_fragments = start_fragments.with_columns(
            (pl.col("min_end") - 1).alias("min_end"),
            (pl.col("max_end") - 1).alias("max_end"),
        )

        # Remove reverse indexing for END fragments
        end_fragments = end_fragments.with_columns(
            (len(skeleton_seq) - pl.col("min_end")).alias("min_end"),
            (len(skeleton_seq) - pl.col("max_end")).alias("max_end"),
        )

        frag_terminal = pl.concat([start_fragments, end_fragments])

        # Remove all "internal" fragment duplicates that are truly terminal fragments
        frag_internal = fragments.filter(
            ~pl.col("fragmentation").str.contains("START")
            & ~pl.col("fragmentation").str.contains("END")
        ).filter(
            ~pl.col("fragment_index").is_in(
                frag_terminal.get_column("fragment_index").to_list()
            )
        )

        # Rebuild fragment dataframe from internal and terminal fragments
        fragments = frag_internal.vstack(frag_terminal).sort("index")

        # Ensure all end indices match estimated sequence length
        fragments = fragments.with_columns(
            pl.when((pl.col("min_end") < 0) | (pl.col("min_end") >= len(skeleton_seq)))
            .then(pl.lit(len(skeleton_seq) - 1))
            .otherwise(pl.col("min_end"))
            .alias("min_end"),
            pl.when((pl.col("max_end") < 0) | (pl.col("max_end") >= len(skeleton_seq)))
            .then(pl.lit(len(skeleton_seq) - 1))
            .otherwise(pl.col("max_end"))
            .alias("max_end"),
        )

        # Return skeleton and fragments
        return skeleton_seq, fragments

    def _predict_skeleton(
        self,
        fragments: pl.DataFrame,
        skeleton_seq: Optional[List[Set[str]]] = None,
    ) -> Tuple[List[Set[str]], pl.DataFrame]:
        # Initialize skeleton sequence (if not already given)
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.inferrer.seq.max_len)]

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

            # Define mass difference and threshold between neighboring fragments
            neighbor_diff = fragments.item(
                frag_idx, "standard_unit_mass"
            ) - fragments.item(frag_idx - 1, "standard_unit_mass")
            neighbor_threshold = calculate_error_threshold(
                fragments.item(frag_idx - 1, "observed_mass"),
                fragments.item(frag_idx, "observed_mass"),
                self.inferrer.tolerance,
            )

            # Bin fragments with similar mass together
            if neighbor_diff <= neighbor_threshold:
                current_bin.append(frag_idx)

                # Only process bin immediately if there are no unbinned fragments left
                if frag_idx + 1 < len(fragments):
                    continue

            compositions = self.infer_compositions_for_bin_differences(
                prev_bin=last_valid_bin,
                current_bin=current_bin,
                fragments=fragments,
            )

            # Skip bins with no valid compositions
            if compositions is None:
                for idx in current_bin:
                    # Add a warning in the log for the skipped fragment
                    logger.warning(
                        f"Skipping {fragments.item(idx, 'fragmentation')} "
                        f"fragment {fragments.item(idx, 'index')} with observed "
                        f"mass {fragments.item(idx, 'observed_mass'):.4f} and "
                        f"SU mass {fragments.item(idx, 'standard_unit_mass'):.4f}"
                        f" because no valid compositions were found."
                    )

                    invalid_list.append(fragments.item(idx, "index"))
            else:
                # Continue skeleton building
                pos, skeleton_seq = self.update_skeleton_for_given_compositions(
                    compositions=compositions,
                    pos=pos,
                    skeleton_seq=skeleton_seq,
                )

                # Adapt information on end index for given bin
                for idx in current_bin:
                    fragments[idx, "min_end"] = min(pos, default=1)
                    fragments[idx, "max_end"] = max(pos, default=0)

                # Update information for previous bin
                last_valid_bin = current_bin

            # Update information for current bin
            current_bin = [frag_idx]

        # Filter out all invalid fragments
        fragments = fragments.filter(~pl.col("index").is_in(invalid_list))

        return skeleton_seq, fragments

    def select_sequence_length_with_lp(
        self,
        start_skeleton: List[Set[str]],
        end_skeleton: List[Set[str]],
        start_fragments: pl.DataFrame,
        end_fragments: pl.DataFrame,
        solver_params: dict,
    ) -> int:
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
        nucleotide_masses = {
            mass.names[0]: mass.mass * self.inferrer.precision
            for mass in self.inferrer.alphabet[1:]
        }

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
            # Determine LP score for terminal-fragment alignment
            value = self.determine_lp_score(
                start_fragments=start_fragments.clone(),
                end_fragments=end_fragments.clone(),
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
        start_fragments: pl.DataFrame,
        end_fragments: pl.DataFrame,
        skeleton_seq: list,
        solver_params: dict,
    ) -> pl.DataFrame:
        # Ensure fragments only occur once
        end_fragments = end_fragments.filter(
            ~pl.col("index").is_in(start_fragments.get_column("index").to_list())
        )

        # Remove indexing of the next pos for START fragments
        start_fragments = start_fragments.with_columns(
            (pl.col("min_end") - 1).alias("min_end"),
            (pl.col("max_end") - 1).alias("max_end"),
        )

        # Remove reverse indexing for END fragments
        end_fragments = end_fragments.with_columns(
            (len(skeleton_seq) - pl.col("min_end")).alias("min_end"),
            (len(skeleton_seq) - pl.col("max_end")).alias("max_end"),
        )

        # Initialize LP instance for terminal fragment
        try:
            lp_instance = LinearProgramInstance(
                fragments=pl.concat([start_fragments, end_fragments]),
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
        min_mass = 0
        max_mass = 0
        for start_nucs, end_nucs in zip(start_skeleton, end_skeleton):
            min_mass += min(
                [nuc_masses[nuc] for nuc in (start_nucs | end_nucs)], default=0
            )
            max_mass += max(
                [nuc_masses[nuc] for nuc in (start_nucs | end_nucs)], default=0
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
        nucleoside_masses = {
            mass.names[0]: mass.mass * self.inferrer.precision
            for mass in self.inferrer.alphabet[1:]
        }

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
        prev_bin: list,
        current_bin: list,
        fragments: pl.DataFrame,
    ) -> List[Composition]:
        # Collect compositions for first bin
        if prev_bin is None:
            compositions = [
                self.infer_compositions_for_mass_difference(
                    diff=fragments.item(idx, "standard_unit_mass"),
                    prev_mass=0.0,
                    current_mass=fragments.item(idx, "observed_mass"),
                )
                for idx in current_bin
            ]

        # Collect compositions between previous and current bin
        else:
            compositions = [
                self.infer_compositions_for_mass_difference(
                    diff=fragments.item(current_idx, "standard_unit_mass")
                    - fragments.item(prev_idx, "standard_unit_mass"),
                    prev_mass=fragments.item(prev_idx, "observed_mass"),
                    current_mass=fragments.item(current_idx, "observed_mass"),
                )
                for prev_idx in prev_bin
                for current_idx in current_bin
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
    ):
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


def jaccard_index(input: Tuple[Set[str], Set[str]]) -> float:
    # Return score for perfect similarity if one set is empty
    if len(input[0]) == 0 or len(input[1]) == 0:
        return 1

    # Return Jaccard score
    return len(input[0].intersection(input[1])) / len(input[0].union(input[1]))


def combine_skeleton_sequences(
    seq_len: int,
    start_skeleton: List[Set[str]],
    end_skeleton: List[Set[str]],
) -> List[Set[str]]:
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
