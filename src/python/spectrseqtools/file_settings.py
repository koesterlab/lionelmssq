# -*- coding: utf-8 -*-
"""Module for file-related classes, functions, and global variables."""

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from platformdirs import user_cache_dir

# Set OS-independent cache directory for traceback matrix
MATRIX_CACHE_DIR = Path(
    user_cache_dir(
        appname="spectrseqtools/traceback_matrix", version="1.0", ensure_exists=True
    )
)

# Set default file paths
DEFAULT_ALPHABET_PATH = importlib.resources.files(__package__) / "assets" / "masses.tsv"
DEFAULT_ELEMENT_PATH = (
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv"
)

# Build dict with elemental masses
elements = pl.read_csv(DEFAULT_ELEMENT_PATH, separator="\t")
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}

_ALPHABET_DF_COLS = [
    "id",
    "canonical_name",
    "monoisotopic_mass",
    "modification_rate",
    "is_modification",
    "encoding",
]


def load_alphabet(input_path: Path | None = None) -> pl.DataFrame:
    """Return valid alphabet dataframe (default if path is invalid).

    Parameters
    ----------
    input_path : Path | None
        Candidate for valid alphabet path.

    Returns
    -------
    pl.DataFrame
        Dataframe containing valid alphabet.

    """
    # If input path is None or non-existent, set default
    if input_path is None or not os.path.isfile(input_path):
        if input_path is not None:
            print("No valid alphabet path detected. Proceeding with default.")
        input_path = DEFAULT_ALPHABET_PATH

    alphabet = pl.read_csv(input_path, separator="\t")
    masses = alphabet.select(_ALPHABET_DF_COLS)
    assert masses.columns == _ALPHABET_DF_COLS
    return alphabet


def set_matrix_path(num_places: int, compression_rate: int) -> Path:
    """
    Set path to traceback matrix.

    Parameters
    ----------
    num_places : int
        Number of decimal places used for rounding of (nucleotide) masses.
    compression_rate : int
        Compression per matrix cell.

    Returns
    -------
    path : Path
        Path to traceback matrix.

    """
    # Set path for traceback matrix
    path = MATRIX_CACHE_DIR / f"{num_places}_decimal_places.{compression_rate}_per_cell"

    # Create directory for traceback matrix if it does not already exist
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)

    return path


@dataclass
class FileSettings:
    """Class for file-related settings."""

    input_path: Path | pl.DataFrame
    meta_path: Path
    alphabet_path: Path | pl.DataFrame | None = None
    output_dir: Path | None = None

    def __post_init__(self):
        path = self.input_path.resolve()
        if self.output_dir is None:
            self.output_dir = path.parent

    @property
    def file_prefix(self):
        """Return file prefix (not including directory path)."""
        return self.input_path.stem


@dataclass
class PreprocessingFileSettings(FileSettings):
    """Class for file-related settings used during preprocessing phase."""

    @property
    def updated_meta_path(self) -> Path:
        """Return path for updated metafile."""
        return self.output_dir / f"{self.file_prefix}.preprocessed.meta.yaml"

    @property
    def fragment_path(self) -> Path:
        """Return path for file containing raw fragments."""
        return self.output_dir / f"{self.file_prefix}.tsv"

    @property
    def updated_alphabet_path(self) -> Path:
        """Return path for file containing updated nucleotide alphabet (singletons)."""
        return self.output_dir / f"{self.file_prefix}.singletons.tsv"


@dataclass
class PredictionFileSettings(FileSettings):
    """Class for file-related settings used during prediction phase."""

    predicted_fragment_path: Path | None = None
    sequence_path: Path | None = None
    sequence_header: str | None = None

    @property
    def raw_fragment_path(self) -> Path:
        """Return path for file containing raw fragments."""
        return self.input_path

    @property
    def su_fragment_path(self) -> Path:
        """Return path for file containing SU-fragments."""
        return self.output_dir / f"{self.file_prefix}.standard_unit_fragments.tsv"
