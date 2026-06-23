# -*- coding: utf-8 -*-
"""Module for fragment-related classes."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Set

import numpy as np
import polars as pl
from loguru import logger

from spectrseqtools.common import calculate_error_threshold
from spectrseqtools.dataclasses import SequenceInformation
from spectrseqtools.masses import PRECISION
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.sequence import SkeletonSequence

NUC_WEIGHT_FACTOR = 1


@dataclass
class StandardUnitFragments:
    """Class for SU-fragments."""

    fragments: pl.DataFrame

    def __len__(self) -> int:
        """Return length of fragment list."""
        return len(self.fragments)

    @classmethod
    def default(cls) -> Self:
        """Return empty fragments dataframe."""
        return cls(
            fragments=pl.DataFrame(
                schema={
                    "orig_index": pl.UInt32,
                    "observed_mass": pl.Float64,
                    "standard_unit_mass": pl.Float64,
                    "fragmentation": pl.String,
                    "intensity": pl.Float64,
                    "min_end": pl.UInt32,
                    "max_end": pl.UInt32,
                }
            ),
        )

    @classmethod
    def from_bins(cls, bins: List[Self], invalid_list: List) -> Self:
        """
        Initialize SU-fragments from list of binned fragments.

        Parameters
        ----------
        bins : List[StandardUnitFragments]
            List of binned SU-fragments.
        invalid_list : List
            List of indices for invalid fragments.

        """
        # Concatenate bins
        fragments = pl.concat(frag_bin.fragments for frag_bin in bins)

        # Filter out all invalid fragments
        return cls(fragments.filter(~pl.col("index").is_in(invalid_list)))

    @classmethod
    def from_terminals(
        cls, start_fragments: Self, end_fragments: Self, seq_len: int
    ) -> Self:
        """
        Initialize SU-fragments from terminal fragments.

        Parameters
        ----------
        start_fragments : StandardUnitFragments
            List of terminal fragments in 5'-direction.
        end_fragments : StandardUnitFragments
            List of terminal fragments in 3'-direction.
        seq_len : int
            Sequence length.

        """
        # Clone fragments from classes
        start_fragments = start_fragments.fragments.clone()
        end_fragments = end_fragments.fragments.clone()

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
            (seq_len - pl.col("min_end")).alias("min_end"),
            (seq_len - pl.col("max_end")).alias("max_end"),
        )

        return cls(fragments=start_fragments.vstack(end_fragments).sort("index"))

    @classmethod
    def from_fragment_classes(
        cls,
        start_fragments: Self,
        end_fragments: Self,
        internal_fragments: Self,
        seq_len: int,
    ) -> Self:
        """
        Initialize SU-fragments from fragment classes.

        Parameters
        ----------
        start_fragments : StandardUnitFragments
            List of terminal fragments in 5'-direction.
        end_fragments : StandardUnitFragments
            List of terminal fragments in 3'-direction.
        internal_fragments : StandardUnitFragments
            List of internal fragments.
        seq_len : int
            Sequence length.

        """
        frag_terminal = cls.from_terminals(
            start_fragments, end_fragments, seq_len=seq_len
        ).fragments

        # TODO: Move filter outside of skeleton building
        # Remove all "internal" fragment duplicates that are truly terminal fragments
        frag_internal = internal_fragments.fragments.filter(
            ~pl.col("fragment_index").is_in(
                frag_terminal.get_column("fragment_index").to_list()
            )
        )

        # Rebuild fragment dataframe from internal and terminal fragments
        fragments = frag_internal.vstack(frag_terminal).sort("index")

        # Ensure all end indices match estimated sequence length
        fragments = fragments.with_columns(
            pl.when((pl.col("min_end") < 0) | (pl.col("min_end") >= seq_len))
            .then(pl.lit(seq_len - 1))
            .otherwise(pl.col("min_end"))
            .alias("min_end"),
            pl.when((pl.col("max_end") < 0) | (pl.col("max_end") >= seq_len))
            .then(pl.lit(seq_len - 1))
            .otherwise(pl.col("max_end"))
            .alias("max_end"),
        )

        return cls(fragments=fragments)

    @property
    def start(self) -> Self:
        """Return terminal fragments in 5'-direction."""
        return StandardUnitFragments(
            fragments=self.fragments.filter(
                pl.col("fragmentation").str.contains("START")
            )
        )

    @property
    def end(self) -> Self:
        """Return terminal fragments in 3'-direction."""
        return StandardUnitFragments(
            fragments=self.fragments.filter(pl.col("fragmentation").str.contains("END"))
        )

    @property
    def internal(self) -> Self:
        """Return internal fragments."""
        return StandardUnitFragments(
            fragments=self.fragments.filter(
                ~pl.col("fragmentation").str.contains("START")
                & ~pl.col("fragmentation").str.contains("END")
            )
        )

    def filter_by_intact_mass(self, seq_info: SequenceInformation) -> None:
        """
        Filter SU-fragments by intact mass.

        Within variance, filter out fragments whose SU-mass is either
        1) higher than intact mass or
        2) lower than the intact mass while being intact fragment.

        Parameters
        ----------
        seq_info : SequenceInformation
            General sequence information.

        """
        # Filter out fragments that have a too high SU mass (within variance)
        self.fragments = self.fragments.filter(
            pl.col("standard_unit_mass") < seq_info.upper_intact_mass_bound
        )

        # Filter out all intact fragments with a too low SU mass (within variance)
        self.fragments = self.fragments.filter(
            (pl.col("standard_unit_mass") > seq_info.lower_intact_mass_bound)
            | ~(
                pl.col("fragmentation").str.contains("START")
                & pl.col("fragmentation").str.contains("END")
            )
        )

    def filter_with_traceback_matrix(self, inferrer: CompositionInferrer) -> None:
        """
        Filter out all fragments with no valid composition

        Parameters
        ----------
        inferrer : CompositionInferrer
            Composition inferrer, i.e., traceback matrix.

        """
        self.fragments = (
            self.fragments.with_columns(
                pl.struct("observed_mass", "standard_unit_mass")
                .map_elements(
                    lambda x: inferrer.is_valid_mass(
                        mass=x["standard_unit_mass"],
                        threshold=inferrer.tolerance * x["observed_mass"],
                    ),
                    return_dtype=bool,
                )
                .alias("is_valid")
            )
            .filter(pl.col("is_valid"))
            .drop("is_valid")
        )

    def index(self) -> None:
        """Index fragments."""
        self.fragments = (
            self.fragments.with_row_index(name="orig_index")
            .sort("standard_unit_mass")
            .with_row_index(name="index")
        )

    def save(self, output_path) -> None:
        """
        Save SU-fragments to file.

        Parameters
        ----------
        output_path : Path
            Path to output file in TSV format.

        """
        self.fragments.write_csv(output_path, separator="\t")

    def collect_singleton_compositions(self, inferrer: CompositionInferrer) -> dict:
        """
        Collect compositions of singletons in fragment list.

        Parameters
        ----------
        inferrer : CompositionInferrer
            Composition inferrer.

        Returns
        -------
        dict
            Dictionary of singleton masses and their corresponding compositions.

        """
        # Determine all fragments that may be singletons
        fragments = self.fragments.filter(
            pl.col("standard_unit_mass") <= 1.1 * inferrer.alphabet.max.nucleotide_mass
        )

        # Collect singleton masses
        compositions = {}
        for frag in fragments.rows(named=True):
            # Compute mass compositions
            comps = inferrer.infer_compositions(
                mass=frag["standard_unit_mass"],
                threshold=inferrer.tolerance * frag["observed_mass"],
            )

            # Ensure mass corresponds to true singleton
            if comps.contains_singleton():
                compositions[frag["standard_unit_mass"]] = comps

        return compositions

    def bin(self, tolerance: float) -> List[Self]:
        """
        Split fragments into bins within mass tolerance.

        Parameters
        ----------
        tolerance : float
            Relative mass tolerance.

        """
        # Sort fragments for consecutive binning
        self.fragments = self.fragments.sort("standard_unit_mass")

        bins = []
        start_idx = 0
        for frag_idx in range(1, len(self)):
            # Define mass difference and threshold between neighboring fragments
            neighbor_diff = self.fragments.item(
                frag_idx, "standard_unit_mass"
            ) - self.fragments.item(frag_idx - 1, "standard_unit_mass")
            neighbor_threshold = calculate_error_threshold(
                self.fragments.item(frag_idx - 1, "observed_mass"),
                self.fragments.item(frag_idx, "observed_mass"),
                tolerance,
            )

            # Continue filling bin for fragments with similar mass
            if neighbor_diff <= neighbor_threshold:
                continue

            # Close bin and update information for new one
            bins.append(StandardUnitFragments(self.fragments[start_idx:frag_idx]))
            start_idx = frag_idx

        # Add last bin to list
        bins.append(StandardUnitFragments(self.fragments[start_idx:]))

        return bins

    def invalidate(self) -> Self:
        """Invalidate all fragments."""
        # Add a warning in the log for the skipped fragment
        for frag in self.fragments.rows(named=True):
            logger.warning(
                f"Skipping {frag['fragmentation']} "
                f"fragment {frag['index']} with observed "
                f"mass {frag['observed_mass']:.4f} and "
                f"SU mass {frag['standard_unit_mass']:.4f}"
                f" because no valid compositions were found."
            )

        return self.fragments.get_column("index").to_list()

    def update_end_indices(self, pos: Set[int]) -> None:
        """
        Update minimum and maximum end index, respectively.

        Parameters
        ----------
        pos : Set[int]
            Set of possible follow-up indices.

        """
        # Adapt information on end index for given bin
        self.fragments = self.fragments.with_columns(
            pl.lit(min(pos, default=1), dtype=pl.Int64).alias("min_end"),
            pl.lit(max(pos, default=1), dtype=pl.Int64).alias("max_end"),
        )

    def collect_mass_difference_compositions(
        self, inferrer: CompositionInferrer
    ) -> dict:
        """
        Collect compositions of mass differences between fragments in list.

        Parameters
        ----------
        inferrer : CompositionInferrer
            Composition inferrer.

        Returns
        -------
        dict
            Dictionary of differences and their corresponding compositions.

        """
        su_masses = self.fragments.get_column("standard_unit_mass").to_list()
        observed_masses = self.fragments.get_column("observed_mass").to_list()
        start = 0
        end = 1

        compositions = {}
        max_weight = inferrer.alphabet.max.nucleotide_mass * NUC_WEIGHT_FACTOR
        while end < len(self):
            # Skip singletons
            if (end - start) <= 0:
                end += 1
                continue

            # Determine mass difference between fragments
            diff = su_masses[end] - su_masses[start]

            # If mass difference > maximum allowed weight, drop 1st fragment in window
            if diff > max_weight:
                start += 1
                end = start + 1
                continue

            diff_error = calculate_error_threshold(
                observed_masses[start],
                observed_masses[end],
                inferrer.tolerance,
            )
            comp = inferrer.infer_compositions(mass=diff, threshold=diff_error)
            if len(comp) > 0:
                compositions[diff] = comp
            if end == len(self) - 1:
                start += 1
            else:
                end += 1

        return compositions

    def print_warning(self) -> None:
        """Print warning if no directional terminal fragments are present."""
        if len(self.start) == 0:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if len(self.end) == 0:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

    def filter_by_index_list(self, invalid_indices: list) -> None:
        """Filter out fragments whose index is in given list."""
        self.fragments = self.fragments.filter(~pl.col("index").is_in(invalid_indices))

    def filter_with_linear_optimization(
        self,
        inferrer: CompositionInferrer,
        skeleton_seq: SkeletonSequence,
        solver_params,
    ) -> None:
        """
        Filter out fragments that the LP cannot individually fit into sequence.

        Parameters
        ----------
        inferrer : CompositionInferrer
            Composition inferrer.
        skeleton_seq : SkeletonSequence
            Skeleton sequence.
        solver_params : SolverParameters
            Solver parameter.

        """
        is_invalid = []
        for fragment in self.fragments.rows(named=True):
            # TODO: Add terminal-fragment filter based on LP output of
            #  sequence-length estimation and reuse the below (for speed-up)
            # # Skip terminal (i.e. non-internal) fragments
            # if (
            #     "START" in fragment["fragmentation"]
            #     or "END" in fragment["fragmentation"]
            # ):
            #     continue

            # Initialize LP instance for a singular fragment
            filter_instance = LinearProgramInstance(
                fragments=pl.DataFrame(fragment),
                alphabet=inferrer.alphabet,
                seq=inferrer.seq,
                skeleton_seq=skeleton_seq,
            )

            # Check whether fragment can feasibly be aligned to skeleton
            if (
                filter_instance.minimize_error(solver_params=solver_params)
                > inferrer.tolerance * fragment["observed_mass"]
            ):
                is_invalid.append(fragment["index"])

        # Return only valid fragments
        self.filter_by_index_list(invalid_indices=is_invalid)


@dataclass
class RawFragments:
    """Class for predicted fragments."""

    fragments: pl.DataFrame

    @classmethod
    def from_file(cls, input_path: Path) -> Self:
        """
        Initialize raw fragments from file.

        Parameters
        ----------
        input_path : Path
            Path to input file in TSV format.

        """
        # Read raw fragments from file
        fragments = pl.read_csv(input_path, separator="\t")

        # If no intensity is given, set it to -1 by default
        if "intensity" not in fragments.columns:
            fragments = fragments.with_columns(pl.lit(-1).alias("intensity"))

        # Rename 'neutral_mass' values from deisotoping to 'observed_mass'
        if "neutral_mass" in fragments.columns:
            fragments = fragments.rename({"neutral_mass": "observed_mass"})

        # Index fragments
        fragments = fragments.with_row_index("fragment_index")

        return cls(fragments=fragments)

    @classmethod
    def default(cls) -> Self:
        """Return empty fragments dataframe."""
        return cls(
            fragments=pl.DataFrame(
                schema={
                    "fragment_index": pl.Int64,
                    "observed_mass": pl.Float64,
                    "intensity": pl.Float64,
                }
            ),
        )

    def filter_by_intensity(
        self, cutoff_percentile: int, intensity_cutoff: float = None
    ) -> None:
        """
        Filter out fragments with too low intensity.

        Parameters
        ----------
        cutoff_percentile : int
            Intensity cutoff percentile.
        intensity_cutoff : float, optional
            Fixed intensity cutoff. Default: None.

        """
        if intensity_cutoff is None:
            # Get intensity cutoffs for all percentiles (in increments of 5%)
            percentile_df = self.fragments.get_column("intensity").describe(
                percentiles=np.linspace(0, 0.95, 20),
                interpolation="midpoint",
            )

            # Set intensity cutoff (if not given in metadata) based on desired percentile
            intensity_cutoff = percentile_df.filter(
                pl.col("statistic") == f"{cutoff_percentile}%"
            )["value"].to_list()[0]

        if self.fragments.select("intensity").min().item() > -1:
            self.fragments = self.fragments.filter(
                pl.col("intensity") >= intensity_cutoff
            )

    def standardize(self, fragmentation_dict: dict) -> StandardUnitFragments:
        """
        Obtain SU-fragments for each considered fragmentation type.

        Parameters
        ----------
        fragmentation_dict : dict
            Dictionary with masses of all considered fragmentation types.

        Returns
        -------
        StandardUnitFragments
            SU-fragments.

        """
        # Copy each fragment for each unique fragmentation weights and set standard-unit mass
        fragments = pl.concat(
            [
                self.fragments.with_columns(
                    (pl.col("observed_mass") - (weight * PRECISION)).alias(
                        "standard_unit_mass"
                    ),
                    pl.lit(fragmentation[0]).alias("fragmentation"),
                )
                for (weight, fragmentation) in fragmentation_dict.items()
            ]
        )

        # Initialize sequence boundaries
        fragments = fragments.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("min_end"),
            pl.lit(-1, dtype=pl.Int64).alias("max_end"),
        )

        # Sort fragments
        return StandardUnitFragments(
            fragments=fragments.sort(pl.col("standard_unit_mass"))
        )
