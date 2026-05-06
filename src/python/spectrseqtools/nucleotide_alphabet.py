# -*- coding: utf-8 -*-
"""Module for nucleotide alphabet."""

import importlib.resources
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Self

import numpy as np
import polars as pl

from spectrseqtools.masses import (
    DECIMAL_PLACES,
    ELEMENT_MASSES,
    PRECISION,
    UNMODIFIED_BASES,
)

_DF_COLS = [
    "id",
    "canonical_name",
    "monoisotopic_mass",
    "modification_rate",
    "encoding",
]


_ALPHABET_COLS = [
    "integer_mass",
    "nucleoside_mass",
    "nucleotide_mass",
    "singleton_mz",
    "names",
    "modification_rate",
    "is_modification",
]


@dataclass
class NucleotideMass:
    """Class for nucleotide masses."""

    integer_mass: int = 0
    nucleoside_mass: float = 0.0
    nucleotide_mass: float = 0.0
    singleton_mz: float = 0.0
    names: List[str] = field(default_factory=list)
    modification_rate: float = 0.0
    is_modification: bool = False

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

    @property
    def mass(self) -> int:
        """Return integer mass."""
        return self.integer_mass

    @property
    def representative(self) -> str:
        """Return name of representative nucleotide."""
        return self.names[0]

    def fmt(self) -> str:
        """Return nucleotide names formatted to string."""
        if len(self.names) == 1:
            return self.names[0]
        return "[" + "|".join(self.names) + "]"


@dataclass
class NucleotideAlphabet:
    """Class for considered nucleotide alphabet."""

    alphabet: List[NucleotideMass]
    precision: float

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    @classmethod
    def from_file(
        cls,
        precision: float = PRECISION,
        modification_rate: float = 0.5,
        input_path: Path = None,
    ) -> Self:
        """
        Initialize nucleotide alphabet from file.

        Parameters
        ----------
        modification_rate : float
            Maximum percentage of modification in sequence.
        precision : float
            Precision used for (nucleotide) masses.
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
        assert masses.columns == _DF_COLS

        # TODO: Round other masses in DF (not just nucleoside one)

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

        # Add modification flag
        masses = masses.with_columns(
            ~pl.col("representative").is_in(UNMODIFIED_BASES).alias("is_modification")
        )

        return cls.from_dataframe(
            nucleotide_df=masses,
            modification_rate=modification_rate,
            precision=precision,
        )

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
        new_df = (
            nucleotide_df.sort("integer_mass")
            .rename({"id_list": "names"})
            .drop("representative")
            .with_columns(
                pl.when(pl.col("is_modification"))
                .then(pl.col("modification_rate").clip(upper_bound=modification_rate))
                .otherwise(pl.col("modification_rate"))
            )
        )
        return cls(
            alphabet=[NucleotideMass(**row) for row in new_df.rows(named=True)],
            precision=precision,
        )

    def __len__(self) -> int:
        """Return alphabet size."""
        return len(self.alphabet)

    @property
    def min(self) -> float:
        """Return lowest nucleotide mass in alphabet."""
        return min(mass.nucleotide_mass for mass in self.alphabet)

    @property
    def max(self) -> float:
        """Return highest nucleotide mass in alphabet."""
        return max(mass.nucleotide_mass for mass in self.alphabet)

    @property
    def min_mz(self) -> float:
        """Return lowest singleton m/z in alphabet."""
        return min(mass.singleton_mz for mass in self.alphabet)

    @property
    def max_mz(self) -> float:
        """Return highest singleton m/z in alphabet."""
        return max(mass.singleton_mz for mass in self.alphabet)

    @property
    def max_integer(self) -> int:
        """Return highest integer mass in alphabet."""
        return max(mass.mass for mass in self.alphabet)

    def get_mass(self, idx: int) -> int:
        """Return mass at index in alphabet."""
        return self.alphabet[idx].mass

    def get_nuc_mass(self, idx: int) -> float:
        """Return nucleotide mass at index in alphabet."""
        return self.alphabet[idx].mass * self.precision

    def get_rep(self, idx: int) -> str:
        """Return representative nucleotide at index in alphabet."""
        return self.alphabet[idx].representative

    def get_rate(self, idx: int) -> float:
        """Return modification rate at index in alphabet."""
        return self.alphabet[idx].modification_rate

    def is_mod(self, idx: int) -> bool:
        """Return whether nucleotide at index in alphabet is modification."""
        return self.alphabet[idx].is_modification

    def fmt(self, rep: str) -> str:
        """Return formatted nucleotide by representative in alphabet."""
        for nuc in self.alphabet:
            if rep == nuc.names[0]:
                return nuc.fmt()
        return ""

    def get_idx(self, rep: str) -> int:
        """Return nucleotide index by representative in alphabet."""
        for idx, nuc in enumerate(self.alphabet):
            if rep == nuc.names[0]:
                return idx
        return -1

    def set_threshold(self, value: float) -> int:
        """Return precision-adapted inference threshold."""
        return int(np.ceil(value / self.precision))

    def set_target(self, value: float) -> int:
        """Return precision-adapted inference target."""
        return int(round(value / self.precision, 0))

    def filter_by_singletons(self, singleton_path: Path) -> None:
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
        if singletons is None:
            return

        # Select nucleotide names for all singletons
        singleton_names = set(singletons.get_column("id").to_list())

        # Select only bases found in singletons
        for nuc in self.alphabet:
            if len(set(nuc.names) & singleton_names) == 0 and nuc.is_modification:
                nuc.modification_rate = 0.0

    def adapt_individual_modification_rates_by_alphabet(self, alphabet: List) -> dict:
        """
        Set individual modification rate to 0 if nucleotide not in new alphabet.

        Parameters
        ----------
        alphabet : List
            List of nucleotide names in new alphabet.

        Returns
        -------
        dict
            Mapping between old and new indexing.

        """
        mapping = {idx: idx for idx in range(len(self))}
        for idx, nucleotide_mass in enumerate(self.alphabet):
            if nucleotide_mass.is_modification and idx not in alphabet:
                mapping = {
                    key: value - 1 if idx < key else value
                    for (key, value) in (mapping.items())
                    if key != idx
                }
                nucleotide_mass.modification_rate = 0.0
        return mapping

    def get_seq_weight(self, seq: tuple) -> float:
        """
        Determine weight of given sequence.

        Parameters
        ----------
        seq : tuple
            Given sequence consisting only of nucleotide representatives.

        Returns
        -------
        float
            Sequence weight.

        """
        mass_dict = {
            nuc_id: nuc.nucleotide_mass for nuc in self.alphabet for nuc_id in nuc.names
        }

        return round(sum(mass_dict[nuc] for nuc in seq), 5)

    def reduce(self) -> None:
        """Reduce alphabet by removing nucleotides that cannot be in sequence."""
        self.alphabet = [mass for mass in self.alphabet if mass.modification_rate > 0.0]

    def to_dataframe(self) -> pl.DataFrame:
        """Return nucleotide alphabet as Polars dataframe."""
        return pl.DataFrame(
            {
                col: [mass.__dict__[col] for mass in self.alphabet]
                for col in _ALPHABET_COLS
            }
        )
