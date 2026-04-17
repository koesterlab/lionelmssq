# -*- coding: utf-8 -*-
"""Module for nucleotide alphabet."""

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self

import numpy as np
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
    """Class for considered nucleotide alphabet as Polars dataframe."""

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


@dataclass
class NucleotideMass:
    """Class for nucleotide masses."""

    mass: int
    names: List[str]
    is_modification: bool
    modification_rate: float

    def __eq__(self, other):
        return self.mass == other.mass

    def __le__(self, other):
        return self.mass <= other.mass

    def __lt__(self, other):
        return self.mass < other.mass

    def __ge__(self, other):
        return self.mass >= other.mass

    def __gt__(self, other):
        return self.mass > other.mass


@dataclass
class NucleotideAlphabetReduced:
    """Class for considered nucleotide alphabet."""

    alphabet: List[NucleotideMass]
    precision: float

    def __repr__(self) -> str:
        masses = NUCLEOTIDE_DF.sort("nucleoside_mass").filter(
            pl.col("representative").is_in(self.names())
        )
        masses = masses.replace_column(
            masses.get_column_index("modification_rate"),
            pl.Series(
                "modification_rate",
                [mass.modification_rate for mass in self.alphabet[1:]],
            ),
        )

        return masses.__repr__()

    @classmethod
    def from_dataframe(
        cls, nucleotide_df: pl.DataFrame, modification_rate: float, precision: float
    ) -> Self:
        """
        Initialize nucleotide alphabet from file.

        Parameters
        ----------
        nucleotide_df : polars.DataFrame
            Polars dataframe containing nucleoside information.
        modification_rate : float
            Maximum percentage of modification in sequence.
        precision : float
            Precision used for (nucleotide) masses.

        """
        # Get list of integer masses
        integer_masses = nucleotide_df.get_column("integer_mass").to_list()

        # Add a default weight for easier initialization
        integer_masses += [0]

        # Ensure unique and sorted entries after tolerance correction
        integer_masses = sorted(set(integer_masses))

        # Create dict with all associated nucleotide names for each mass
        names = {
            mass: pl.DataFrame({"integer_mass": mass})
            .join(
                nucleotide_df,
                on="integer_mass",
                how="left",
            )
            .get_column("representative")
            .to_list()
            for mass in nucleotide_df.get_column("integer_mass").to_list()
        }

        # Create dict with indicator whether each mass is associated with a modified base
        is_mod = {
            mass: any(base not in UNMODIFIED_BASES for base in names[mass])
            for mass in nucleotide_df.get_column("integer_mass").to_list()
        }

        # Create dict with the largest associated modification rate for each mass
        rates = {
            mass: max(
                pl.DataFrame({"integer_mass": mass})
                .join(
                    nucleotide_df,
                    on="integer_mass",
                    how="left",
                )
                .get_column("modification_rate")
                .to_list()
            )
            for mass in nucleotide_df.get_column("integer_mass").to_list()
        }

        # Return alphabet of NucleotideMass instances
        nucleotides = list(
            NucleotideMass(mass, names[mass], is_mod[mass], rates[mass])
            if mass != 0
            else NucleotideMass(0, [], False, 0.0)
            for mass in integer_masses
        )

        # Adapt individual modification rates to universal one
        for nucleotide_mass in nucleotides:
            if not nucleotide_mass.is_modification:
                continue
            nucleotide_mass.modification_rate = min(
                nucleotide_mass.modification_rate, modification_rate
            )

        return cls(alphabet=nucleotides, precision=precision)

    @property
    def size(self) -> int:
        """Return alphabet size."""
        return len(self.alphabet)

    @property
    def max(self) -> int:
        """Return highest mass in alphabet."""
        return max(mass.mass for mass in self.alphabet)

    def get_mass(self, idx: int) -> int:
        """Return mass at index in alphabet."""
        return self.alphabet[idx].mass

    def get_rate(self, idx: int) -> float:
        """Return modification rate at index in alphabet."""
        return self.alphabet[idx].modification_rate

    def is_mod(self, idx: int) -> bool:
        """Return whether nucleotide at index in alphabet is modification."""
        return self.alphabet[idx].is_modification

    def names(self) -> List[str]:
        """Return list of all nucleotide names in alphabet."""
        return list(name for mass in self.alphabet for name in mass.names)

    def reps(self) -> List[str]:
        """Return list of all representative names in alphabet."""
        return list(mass.names[0] for mass in self.alphabet[1:])

    def to_dict(self) -> dict:
        """Return dictionary assigning masses to each representative."""
        return {mass.names[0]: mass.mass * self.precision for mass in self.alphabet[1:]}

    def set_threshold(self, value: float) -> int:
        """Return precision-adapted inference threshold."""
        return int(np.ceil(value / self.precision))

    def set_target(self, value: float) -> int:
        """Return precision-adapted inference target."""
        return int(round(value / self.precision, 0))

    def adapt_individual_modification_rates_by_alphabet(self, alphabet: List) -> None:
        """
        Set individual modification rate to 0 if nucleotide not in new alphabet.

        Parameters
        ----------
        alphabet : List
            List of nucleotide names in new alphabet.

        """
        for nucleotide_mass in self.alphabet:
            if not nucleotide_mass.is_modification:
                continue
            if all(name not in alphabet for name in nucleotide_mass.names):
                nucleotide_mass.modification_rate = 0.0

    def reduce(self) -> None:
        """Reduce alphabet by removing nucleotides that cannot be in sequence."""
        self.alphabet = [
            mass
            for mass in self.alphabet
            if mass.mass == 0.0 or mass.modification_rate > 0.0
        ]
