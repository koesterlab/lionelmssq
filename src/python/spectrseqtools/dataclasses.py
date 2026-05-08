# -*- coding: utf-8 -*-
"""Module for dataclasses."""

import importlib.resources
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self

import polars as pl

from spectrseqtools.masses import MAX_VARIANCE
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.sequence import SkeletonSequence

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
class SequenceInformation:
    """Class for general information related to the sequence."""

    max_len: int
    su_mass: float
    obs_mass: float
    modification_rate: float

    def validate_sequence(
        self, seq: SkeletonSequence, alphabet: NucleotideAlphabet
    ) -> bool:
        """
        Validate sequence length by mass.

        Parameters
        ----------
        seq : SkeletonSequence
            Skeleton sequence.
        alphabet : NucleotideAlphabet
            Nucleotide alphabet.

        Returns
        -------
        bool
            Flag whether sequence length is valid.

        """
        # Check whether mass interval defined by skeleton contains sequence mass
        # Use MAX_VARIANCE to accommodate for uncertainty in sequence mass selection
        return (
            seq.min_mass(alphabet=alphabet) - MAX_VARIANCE
            <= self.su_mass
            <= seq.max_mass(alphabet=alphabet) + MAX_VARIANCE
        )


@dataclass
class Sequence:
    """Class for (predicted) sequence."""

    sequence: List[str]

    def __repr__(self) -> str:
        return self.sequence.__repr__()

    @classmethod
    def from_file(cls, input_path: Path) -> Self:
        """
        Initialize predicted sequence from file.

        Parameters
        ----------
        input_path : Path
            Path to input file in FASTA format.

        """
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

    def fmt(self, nucleotide_alphabet: NucleotideAlphabet) -> str:
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
        return "".join(nucleotide_alphabet.fmt(rep=nuc) for nuc in self.sequence)

    def to_encoding(self) -> List[str]:
        """Format sequence to use encoding."""
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
            print(self.fmt(nucleotide_alphabet=alphabet), file=f)


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
class Prediction:
    """Class for prediction results."""

    sequence: Sequence
    fragments: PredictedFragments

    @classmethod
    def from_files(cls, sequence_path: Path, fragments_path: Path) -> Self:
        """
        Initialize prediction result from files.

        Parameters
        ----------
        sequence_path : Path
            Path to sequence file in FASTA format.
        fragments_path : Path
            Path to fragments file in TSV format.

        """
        return Prediction(
            sequence=Sequence.from_file(input_path=sequence_path),
            fragments=PredictedFragments.from_file(input_path=fragments_path),
        )

    @classmethod
    def default(cls) -> Self:
        """Return empty prediction."""
        return Prediction(
            sequence=Sequence.default(),
            fragments=PredictedFragments.default(),
        )

    def save(
        self,
        fragment_path: Path,
        sequence_path: Path,
        sequence_name: str,
        alphabet: NucleotideAlphabet,
    ) -> None:
        """
        Save prediction results to file.

        Parameters
        ----------
        fragment_path : Path
            Path to output file for fragments in TSV format.
        sequence_path : Path
            Path to output file for sequence in FASTA format.
        sequence_name : str
            Name of sequence in FASTA header.
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        # Save fragment predictions
        self.fragments.save(output_path=fragment_path)

        # Save predicted sequence
        self.sequence.save(
            output_path=sequence_path,
            sequence_name=sequence_name,
            alphabet=alphabet,
        )
