# -*- coding: utf-8 -*-
"""Module for fragment-related classes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl

from spectrseqtools.common import (
    calculate_compositions,
    calculate_error_threshold,
)
from spectrseqtools.masses import PRECISION
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
    is_valid_mass,
)

MAX_VARIANCE = 1


@dataclass
class PredictedFragments:
    """Class for predicted fragments."""

    fragments: pl.DataFrame

    @classmethod
    def from_file(cls, input_path: Path) -> Self:
        """
        Initialize predicted fragments from file.

        Parameters
        ----------
        input_path : Path
            Path to input file in TSV format.

        """
        return cls(fragments=pl.read_csv(input_path, separator="\t"))

    @classmethod
    def default(cls) -> Self:
        """Return empty fragments dataframe."""
        return cls(
            fragments=pl.DataFrame(
                schema={
                    "left": pl.Int64,
                    "right": pl.Int64,
                    "observed_mass": pl.Float64,
                    "standard_unit_mass": pl.Float64,
                    "predicted_mass": pl.Float64,
                    "predicted_diff": pl.Float64,
                    "predicted_seq": pl.String,
                    "orig_index": pl.UInt32,
                    "intensity": pl.Float64,
                }
            ),
        )

    def save(self, output_path) -> None:
        """
        Save predicted fragments to file.

        Parameters
        ----------
        output_path : Path
            Path to output file in TSV format.

        """
        self.fragments.write_csv(output_path, separator="\t")


@dataclass
class StandardUnitFragments:
    """Class for SU-fragments."""

    fragments: pl.DataFrame

    def __len__(self):
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

    def filter_by_intact_mass(self, intact_mass) -> None:
        """
        Filter SU-fragments by intact mass.

        Within variance, filter out fragments whose SU-mass is either
        1) higher than intact mass or
        2) lower than the intact mass while being intact fragment.

        Parameters
        ----------
        intact_mass : float
            Intact sequence mass.

        """
        # Filter out fragments that have a too high SU mass (within variance)
        self.fragments = self.fragments.filter(
            pl.col("standard_unit_mass") < intact_mass + MAX_VARIANCE
        )

        # Filter out all intact fragments with a too low SU mass (within variance)
        self.fragments = self.fragments.filter(
            (pl.col("standard_unit_mass") > intact_mass - MAX_VARIANCE)
            | ~(
                pl.col("fragmentation").str.contains("START")
                & pl.col("fragmentation").str.contains("END")
            )
        )

    def filter_with_traceback_matrix(self, inferrer) -> None:
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
                    lambda x: is_valid_mass(
                        mass=x["standard_unit_mass"],
                        inferrer=inferrer,
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
            pl.col("standard_unit_mass") <= 1.1 * inferrer.alphabet.max
        )

        # Collect singleton masses
        compositions = {}
        for frag in fragments.rows(named=True):
            # Compute mass compositions
            comps = calculate_compositions(
                diff=frag["standard_unit_mass"],
                threshold=inferrer.tolerance * frag["observed_mass"],
                inferrer=inferrer,
            )

            # Ensure mass corresponds to true singleton
            if comps is not None and any(len(comp) == 1 for comp in comps):
                compositions[frag["standard_unit_mass"]] = comps

        return compositions

    def collect_compositions_from_ladder_differences(
        self, inferrer: CompositionInferrer, max_weight: float, direction: str
    ) -> dict:
        """
        Collect compositions of singletons in fragment list.

        Parameters
        ----------
        inferrer : CompositionInferrer
            Composition inferrer.
        max_weight : float
            Maximum nucleotide weight to consider.
        direction : str
            Ladder direction (START or END).

        Returns
        -------
        dict
            Dictionary of differences and their corresponding compositions.

        """
        fragments = self.fragments.filter(
            pl.col("fragmentation").str.contains(direction)
        )
        su_masses = fragments.get_column("standard_unit_mass").to_list()
        observed_masses = fragments.get_column("observed_mass").to_list()
        start = 0
        end = 1

        compositions = {}
        while end < len(fragments):
            # Skip singletons
            if (end - start) <= 0:
                end += 1
                continue

            # Determine mass difference between fragments
            diff = su_masses[end] - su_masses[start]

            # TODO: Set max_weight to be the maximum nucleotide mass in alphabet * factor
            # If mass difference > any nucleotide mass, drop 1st fragment in window
            if diff > max_weight:
                start += 1
                end = start + 1
                continue

            diff_error = calculate_error_threshold(
                observed_masses[start],
                observed_masses[end],
                inferrer.tolerance,
            )
            comp = calculate_compositions(
                diff=diff,
                threshold=diff_error,
                inferrer=inferrer,
            )
            if comp is not None and len(comp) >= 1:
                compositions[diff] = comp
            if end == len(fragments) - 1:
                start += 1
            else:
                end += 1

        return compositions


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

    def filter_by_intensity(self, cutoff: float = 0.5e6) -> None:
        """
        Filter out fragments with too low intensity.

        Parameters
        ----------
        cutoff : float, optional
            Intensity cutoff. Default: 0.5e6.

        """
        if self.fragments.select("intensity").min().item() > -1:
            self.fragments = self.fragments.filter(pl.col("intensity") > cutoff)

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
