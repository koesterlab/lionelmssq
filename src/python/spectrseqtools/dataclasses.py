# -*- coding: utf-8 -*-
"""Module with dataclasses."""

import importlib.resources
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self

import polars as pl

from spectrseqtools.masses import PRECISION
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.composition_inference import is_valid_mass

_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")
MASSES = pl.read_csv(
    (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
    separator="\t",
)
MAX_VARIANCE = 1


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
class StandardUnitFragments:
    """Class for SU-fragments."""

    fragments: pl.DataFrame

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

    def save(self, output_path) -> None:
        """
        Save SU-fragments to file.

        Parameters
        ----------
        output_path : Path
            Path to output file in TSV format.

        """
        self.fragments.write_csv(output_path, separator="\t")


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

        # Sort fragments
        return StandardUnitFragments(
            fragments=fragments.sort(pl.col("standard_unit_mass"))
        )


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
            print(self.to_full_str(nucleotide_alphabet=alphabet), file=f)


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
