# -*- coding: utf-8 -*-
"""Module for nucleotide alphabet."""

import importlib.resources
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Self, Set

import polars as pl

from spectrseqtools.error_calculator import ErrorCalculator

# TODO: Currently, the list of unmodified bases is only defined for RNA;
#  make it universally applicable
UNMODIFIED_BASES = ["A", "C", "G", "U"]

# Build dict with elemental masses
elements = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}

DEFAULT_ALPHABET_PATH = importlib.resources.files(__package__) / "assets" / "masses.tsv"

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

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    @classmethod
    def from_file(
        cls,
        error: ErrorCalculator,
        modification_rate: float = 0.5,
        input_path: Path = None,
    ) -> Self:
        """
        Initialize nucleotide alphabet from file.

        Parameters
        ----------
        modification_rate : float
            Maximum percentage of modification in sequence.
        error : ErrorCalculator
            Error calculator.
        input_path : Path | None
            Path to file with nucleoside information.

        """
        # If input path is None or non-existent, set default
        if input_path is None or not os.path.isfile(input_path):
            if input_path is not None:
                print("No valid alphabet path detected. Proceeding with default.")
            input_path = DEFAULT_ALPHABET_PATH

        # Read nucleoside masses from file
        masses = pl.read_csv(input_path, separator="\t")
        masses = masses.select(_DF_COLS)
        assert masses.columns == _DF_COLS

        # Set mass for phosphate link between bases
        phosphate_link = (
            ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"] - ELEMENT_MASSES["H+"]
        )

        # Rename monoisotopic mass to nucleoside mass
        masses = masses.rename({"monoisotopic_mass": "nucleoside_mass"})

        # Add phosphate backbone to gain nucleotide masses
        masses = masses.with_columns(
            pl.col("nucleoside_mass").add(phosphate_link).alias("nucleotide_mass")
        )

        # Add new columns for singleton m/z values (subtract one proton
        # from nucleotide) and integer masses for the DP algorithm
        masses = masses.with_columns(
            pl.col("nucleotide_mass").add(-ELEMENT_MASSES["H+"]).alias("singleton_mz"),
            (pl.col("nucleotide_mass") / error.precision)
            .round(0)
            .cast(pl.Int64)
            .alias("integer_mass"),
        )

        # Round masses
        masses = masses.with_columns(
            pl.col("nucleoside_mass").round(error.decimal_places),
            pl.col("nucleotide_mass").round(error.decimal_places),
            pl.col("singleton_mz").round(error.decimal_places),
        )

        # Group nucleotides by their mass, select a representative for each
        # group, and aggregate them into a list of equal-mass nucleotides
        masses = masses.group_by("integer_mass", maintain_order=True).agg(
            pl.col("id").first().alias("representative"),
            pl.col("nucleoside_mass").max(),
            pl.col("nucleotide_mass").max(),
            pl.col("singleton_mz").max(),
            pl.col("id").unique().alias("id_list"),
            pl.col("modification_rate").max(),
        )

        # Add modification flag
        masses = masses.with_columns(
            ~pl.col("representative").is_in(UNMODIFIED_BASES).alias("is_modification")
        )

        return cls.from_dataframe(
            nucleotide_df=masses,
            modification_rate=modification_rate,
        )

    @classmethod
    def from_dataframe(
        cls, nucleotide_df: pl.DataFrame, modification_rate: float
    ) -> Self:
        """
        Initialize nucleotide alphabet from file.

        Parameters
        ----------
        nucleotide_df : polars.DataFrame
            Polars dataframe containing nucleoside information.
        modification_rate : float
            Maximum percentage of modification in sequence.

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

        # Return alphabet over all nucleotides that can occur in a sequence
        return cls(
            alphabet=[
                NucleotideMass(**row)
                for row in new_df.filter(pl.col("modification_rate") > 0).rows(
                    named=True
                )
            ],
        )

    def __len__(self) -> int:
        """Return alphabet size."""
        return len(self.alphabet)

    @property
    def decimal_places(self) -> int:
        """Determine number of decimal places from first nucleotide in alphabet."""
        value = self.alphabet[0]
        return len(str(value.mass)) - len(str(int(value.nucleotide_mass)))

    @property
    def min(self) -> NucleotideMass:
        """Return lightest nucleotide in alphabet."""
        return self.alphabet[0]

    @property
    def max(self) -> NucleotideMass:
        """Return heaviest nucleotide in alphabet."""
        return self.alphabet[-1]

    @property
    def is_default(self) -> bool:
        """Return whether default alphabet was used."""
        return self == NucleotideAlphabet.from_file(
            error=ErrorCalculator.with_metric(decimal_places=self.decimal_places)
        )

    def get(self, idx: int) -> NucleotideMass:
        """Return nucleotide at index in alphabet."""
        return self.alphabet[idx]

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

    def reduce(self, new_alphabet: Set[int]) -> dict:
        """
        Reduce alphabet by removing nucleotides not in new alphabet.

        Parameters
        ----------
        new_alphabet : Set[int]
            Set of nucleotide indices to keep in new alphabet.

        Returns
        -------
        dict
            Mapping between old and new indexing.

        """
        mapping = {idx: idx for idx in range(len(self))}

        # Set modification rate of all superfluous nucleotides to 0 and adapt mapping
        for idx, nucleotide_mass in enumerate(self.alphabet):
            if nucleotide_mass.is_modification and idx not in new_alphabet:
                mapping = {
                    key: value - 1 if idx < key else value
                    for (key, value) in (mapping.items())
                    if key != idx
                }
                nucleotide_mass.modification_rate = 0.0

        # Remove all modifications that cannot occur in sequence
        self.alphabet = [mass for mass in self.alphabet if mass.modification_rate > 0.0]

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

    def to_dataframe(self) -> pl.DataFrame:
        """Return nucleotide alphabet as Polars dataframe."""
        return pl.DataFrame(
            {
                col: [mass.__dict__[col] for mass in self.alphabet]
                for col in _ALPHABET_COLS
            }
        )
