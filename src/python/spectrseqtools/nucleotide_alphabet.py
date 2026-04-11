# -*- coding: utf-8 -*-
"""Module for nucleotide alphabet."""

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import polars as pl

from spectrseqtools.masses import (
    DECIMAL_PLACES,
    ELEMENT_MASSES,
    UNMODIFIED_BASES,
)

_COLS = [
    "id",
    "canonical_name",
    "monoisotopic_mass",
    "modification_rate",
    "encoding",
]


@dataclass
class NucleotideAlphabet:
    """Class for considered nucleotide alphabet."""

    nucleotides: pl.DataFrame

    @classmethod
    def from_file(cls, input_path: Path = None):
        """
        Initialize nucleotide alphabet from file.

        Parameters
        ----------
        input_path : Path
            Path to file with nucleoside information.

        """
        # If input path is None, set default
        if input_path is None:
            input_path = (
                importlib.resources.files(__package__) / "assets" / "masses.tsv"
            )

        # Read nucleoside masses from file
        masses = pl.read_csv(input_path, separator="\t")
        assert masses.columns == _COLS

        # Round nucleoside masses, we consider DECIMAL_PLACES+1 for since
        # rounding errors propagate at the last decimal digit
        masses = masses.with_columns(
            pl.col("monoisotopic_mass").round(DECIMAL_PLACES + 1)
        ).rename({"monoisotopic_mass": "nucleoside_mass"})

        # Group nucleosides by their mass, select a representative for each
        # group, and aggregate them into a list of equal-mass nucleosides
        masses = masses.group_by("nucleoside_mass", maintain_order=True).agg(
            pl.col("id").first().alias("representative"),
            pl.col("id").unique().alias("id_list"),
            pl.col("modification_rate").max(),
        )

        # Set mass for phosphate link between bases
        phosphate_link = (
            ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"] - ELEMENT_MASSES["H+"]
        )

        # Add phosphate backbone to gain nucleotide masses (also rounded)
        masses = masses.with_columns(
            pl.col("nucleoside_mass")
            .add(phosphate_link)
            # .round(DECIMAL_PLACES + 1)
            .alias("nucleotide_mass")
        )

        # Add new columns for singleton m/z values (subtract one proton
        # from nucleotide) and integer masses for the DP algorithm
        masses = masses.with_columns(
            pl.col("nucleotide_mass").add(-ELEMENT_MASSES["H+"]).alias("singleton_mz"),
            (pl.col("nucleotide_mass") * 10**DECIMAL_PLACES)
            .round(0)
            .cast(pl.Int64)
            .alias("integer_mass"),
        )

        return NucleotideAlphabet(nucleotides=masses)

    def filter_by_singleton_selection(self, singleton_path: Path) -> None:
        """Filter out nucleotides not found during singleton identification.

        Parameters
        ----------
        singleton_path : Path
            Path to file with singletons identified during preprocessing.

        """
        # Read singletons if given
        singletons = None
        if os.path.isfile(singleton_path):
            singletons = pl.read_csv(singleton_path, separator="\t")

        print("Singletons identified during preprocessing:", singletons)
        print()

        # Filter by singletons
        if singletons is not None:
            # Map singletons to their mass representative
            singletons = singletons.with_columns(
                pl.col("id")
                .replace_strict(
                    {
                        nuc: row["representative"]
                        for row in self.nucleotides.rows(named=True)
                        for nuc in row["id_list"]
                    }
                )
                .alias("id")
            )

            # Select only bases found in singletons
            self.nucleotides = self.nucleotides.with_columns(
                pl.when(
                    pl.col("representative").is_in(
                        singletons.get_column("id").to_list()
                    )
                )
                .then(pl.col("modification_rate"))
                .otherwise(pl.lit(0.0))
                .alias("modification_rate")
            )

        # Ensure modification rates of unmodified bases are set to 1
        self.nucleotides = self.nucleotides.with_columns(
            pl.when(~pl.col("representative").is_in(UNMODIFIED_BASES))
            .then(pl.col("modification_rate"))
            .otherwise(pl.lit(1.0))
            .alias("modification_rate")
        )

    def min_mass(self) -> float:
        """Return smallest nucleotide mass in alphabet."""
        return (
            self.nucleotides.filter(pl.col("modification_rate") > 0.0)
            .select("nucleotide_mass")
            .min()
            .item()
        )

    def get_alternatives(self, representative: str) -> List[str]:
        """
        Return alternatives for given representative.

        Parameters
        ----------
        representative : str
            Nucleotide for which to find mass-equivalent alternatives.

        Returns
        -------
        List[str]
            List of alternatives.

        """
        return (
            self.nucleotides.filter(pl.col("representative") == representative)
            .select("id_list")
            .item()
            .to_list()
        )


NUCLEOTIDE_DF = NucleotideAlphabet.from_file().nucleotides
