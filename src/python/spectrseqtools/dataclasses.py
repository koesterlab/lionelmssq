# -*- coding: utf-8 -*-
"""Module with dataclasses."""

import importlib.resources
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self

import polars as pl

from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet

_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")
MASSES = pl.read_csv(
    (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
    separator="\t",
)


@dataclass
class SolverParameters:
    """Class for parameters used to solve optimization problems."""

    solver: str
    threads: int
    msg: bool
    time_limit_short: int
    time_limit_long: int

    def to_dict(self, filter_only: bool = False) -> dict:
        """Return dictionary of solver parameters.

        Parameters
        ----------
        filter_only : bool
            Flag whether solver is only used as a filter (not prediction).

        Returns
        -------
        dict
            Dictionary containing solver parameters.

        """
        # Retrieve parameters from class
        params = self.__dict__.copy()

        # Set time limit based on flag
        if filter_only:
            params["timeLimit"] = params.pop("time_limit_short")
            params.pop("time_limit_long")
        else:
            params["timeLimit"] = params.pop("time_limit_long")
            params.pop("time_limit_short")

        return params


@dataclass
class Sequence:
    """Class for (predicted) sequence."""

    sequence: List[str]

    def __repr__(self) -> str:
        return self.sequence.__repr__()

    @classmethod
    def from_file(cls, input_path: Path) -> Self:
        """Initialize predicted sequence from file."""
        with open(input_path, mode="r", encoding="utf-8") as f:
            # Read only lines pertaining sequence in short format (consisting
            # only of representatives without mass-silent alternatives)
            head, seq = f.readlines()[:2]
            assert head.startswith(">")

        cls.from_str(input_seq=seq)

    @classmethod
    def from_str(cls, input_seq: str) -> Self:
        """Initialize predicted sequence from string."""
        return cls(_NUCLEOSIDE_RE.findall(input_seq.strip()))

    @classmethod
    def default(cls) -> Self:
        """Return empty sequence"""
        return cls(sequence=[])

    def to_str(self) -> str:
        """Format sequence to string."""
        return "".join(self.sequence)

    def to_full_str(self, nucleotide_alphabet: NucleotideAlphabet) -> str:
        """
        Format sequence to full string (i.e. include alternate nucleotides).

        Parameters
        ----------
        nucleotide_alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        Returns
        -------
        str
            Sequence with all alternate nucleotides.

        """
        output = ""
        for nuc in self.sequence:
            alt_nucs = nucleotide_alphabet.get_alternatives(representative=nuc)
            if len(alt_nucs) == 1:
                output += nuc
            else:
                output += "[" + "|".join(alt_nucs) + "]"
        return output

    def to_encoding(self) -> List[str]:
        """Format sequence to encoded ."""
        return [
            MASSES.row(named=True, by_predicate=pl.col("id") == val)["encoding"]
            for val in self.sequence
        ]

    def save(
        self, output_path: Path, sequence_name: str, alphabet: NucleotideAlphabet
    ) -> None:
        """
        Save predicted sequence to file.

        Parameters
        ----------
        output_path : Path
            Path to output file in FASTA format.
        sequence_name : str
            Name of sequence in header.
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        with open(output_path, mode="w", encoding="utf-8") as f:
            print(f">{sequence_name}", file=f)
            print("".join(self.sequence), file=f)
            print(f">{sequence_name}_full", file=f)
            print(self.to_full_str(nucleotide_alphabet=alphabet), file=f)
