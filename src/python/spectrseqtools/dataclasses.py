# -*- coding: utf-8 -*-
"""Module for dataclasses."""

import importlib.resources
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Self

import numpy as np
import polars as pl
import yaml

from spectrseqtools.masses import ELEMENT_MASSES, MAX_VARIANCE, PRECISION
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.sequence import SkeletonSequence

_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")
MASSES = pl.read_csv(
    (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
    separator="\t",
)

# Set fragmentation dict mode (full vs only c/y)
REDUCED_FRAGMENTATION_DICT = True


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
    fragmentation: dict

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        modification_rate: float,
        alphabet: NucleotideAlphabet,
        reduced_fragmentation: bool = REDUCED_FRAGMENTATION_DICT,
    ) -> Self:
        """
        Initialize sequence information from meta file.

        Parameters
        ----------
        file_path : Path
            Path to meta file.
        modification_rate : float
            Maximum percentage of modification in sequence.
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.
        reduced_fragmentation : bool
            Flag whether to use a reduced fragmentation dict (i.e. only c/y).

        """
        with open(file_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        # Read additional parameter from meta file
        start_tag = meta.setdefault("5_prime_tag", 555.1294)
        end_tag = meta.setdefault("3_prime_tag", 455.1491)

        # Build fragmentation dict
        fragmentation_dict = build_fragmentation_dict(
            start_tag=start_tag, end_tag=end_tag, reduced=reduced_fragmentation
        )

        # Standardize intact sequence mass by removing START_END fragmentation to gain SU mass
        seq_mass_obs = meta["intact_mass"]
        seq_mass_su = (
            seq_mass_obs
            - [
                mass * PRECISION
                for mass in fragmentation_dict
                if "START_END" in fragmentation_dict[mass]
            ][0]
        )

        return cls(
            max_len=int(seq_mass_su / alphabet.min),
            su_mass=seq_mass_su,
            obs_mass=seq_mass_obs,
            modification_rate=modification_rate,
            fragmentation=fragmentation_dict,
        )

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


# METHOD: Precompute all weight changes caused by fragmentation and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible fragmentation options.


def build_fragmentation_dict(
    start_tag: float, end_tag: float, reduced: bool
) -> dict:
    element_masses = ELEMENT_MASSES

    # Initialize dict with masses for 5'-end of fragments
    start_dict = {
        # Remove O from SU and add START tag (without H)
        "START": start_tag - element_masses["O"] - element_masses["H+"],
        # Add H to SU to achieve neutral charge
        "c/y": element_masses["H+"],
    }

    # Initialize dict with masses for 3'-end of fragments
    end_dict = {
        # Remove PO3H from SU and add END tag (without H)
        "END": end_tag
        - element_masses["P"]
        - 3 * element_masses["O"]
        - 2 * element_masses["H+"],
        # Remove H from SU to achieve neutral charge
        "c/y": -element_masses["H+"],
    }

    # Add a/w-, b/x-, and d/z-fragmentation for full dict version
    if not reduced:
        # Add PO3H2 to SU to achieve neutral charge
        start_dict["a/w"] = (
            element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
        )
        # Add P2O to SU to achieve neutral charge
        start_dict["b/x"] = element_masses["P"] + 2 * element_masses["O"]
        # Remove OH from SU to achieve neutral charge
        start_dict["d/z"] = -(element_masses["O"] + element_masses["H+"])

        # Remove PO3H2 from SU to achieve neutral charge
        end_dict["a/w"] = -(
            element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
        )
        # Remove P2O from SU to achieve neutral charge
        end_dict["b/x"] = -(element_masses["P"] + 2 * element_masses["O"])
        # Add OH to SU to achieve neutral charge
        end_dict["d/z"] = element_masses["O"] + element_masses["H+"]

    # Collect all unique fragmentation-related mass combinations in dict
    fragmentation_dict = {}
    for start, end in list(product(start_dict, end_dict)):
        val = int((start_dict[start] + end_dict[end]) / PRECISION)
        if val not in fragmentation_dict:
            fragmentation_dict[val] = []
        fragmentation_dict[val] += [f"{start}_{end}"]

    return fragmentation_dict
