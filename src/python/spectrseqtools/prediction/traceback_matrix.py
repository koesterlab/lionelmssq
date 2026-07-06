# -*- coding: utf-8 -*-
"""Module for traceback matrix."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Tuple

import numpy as np

from spectrseqtools.file_settings import set_matrix_path
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet

# Set maximum sequence length to be represented in traceback matrix
MAX_SEQ_LENGTH = 35


# METHOD: Use dynamic programming to build a traceback matrix that implicitly
# contains all inferable compositions up to a specific target mass (based on
# an underlying nucleotide alphabet).


@dataclass
class TracebackMatrix:
    """Class for traceback matrix implicitly containing all compositions."""

    matrix: np.ndarray

    def __len__(self):
        return len(self.matrix)

    @classmethod
    def load_with_compression(
        cls, alphabet: NucleotideAlphabet, compression_rate: int
    ) -> Self:
        """
        Load traceback matrix from file if it exists and compute it otherwise.

        Parameters
        ----------
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.
        compression_rate : int
            Compression per matrix cell.

        """
        # Set matrix cache path
        path = set_matrix_path(
            num_places=alphabet.decimal_places, compression_rate=compression_rate
        )
        match compression_rate:
            case 1:
                return cls.load(alphabet=alphabet, path=path)
            case 4:
                return MatrixWith4PerCellCompression.load(alphabet=alphabet, path=path)
            case 8:
                return MatrixWith8PerCellCompression.load(alphabet=alphabet, path=path)
            case 16:
                return MatrixWith16PerCellCompression.load(alphabet=alphabet, path=path)
            case 32:
                return MatrixWith32PerCellCompression.load(alphabet=alphabet, path=path)
            case _:
                raise ValueError(
                    f"The compression rate {compression_rate} is "
                    f"not compatible with the matrix setup."
                )

    @classmethod
    def load(cls, alphabet: NucleotideAlphabet, path: Path) -> Self:
        """
        Load traceback matrix if it exists and compute it otherwise.

        Parameters
        ----------
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.
        path : Path
            Path to cached matrix file.

        """
        # For non-default alphabet, build matrix directly
        if not alphabet.is_default:
            print("Custom alphabet detected. Setting up new traceback matrix.\n")
            matrix = cls(matrix=None)
            matrix.rebuild(alphabet=alphabet)
            return matrix

        # Build and save bit-representation matrix if not existing
        print("Default alphabet detected. Trying to load traceback matrix.\n")
        if not Path(f"{path}.npy").is_file():
            print("Matrix not found. Building matrix...")
            matrix = cls(matrix=None)
            matrix.rebuild(alphabet=alphabet)
            print("Building complete. Save default matrix.")
            matrix.save(file_path=path)
            return matrix

        # Read traceback matrix
        return cls(matrix=np.load(f"{path}.npy"))

    def rebuild(self, alphabet: NucleotideAlphabet) -> None:
        """
        Rebuild complete matrix with dynamic programming.

        Parameters
        ----------
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        # Select maximum integer mass for which matrix should be built
        max_mass = alphabet.max.integer_mass * MAX_SEQ_LENGTH

        # Initialize matrix as numpy table (+ additional first row for easier indexing)
        matrix = np.zeros((len(alphabet) + 1, max_mass + 1), dtype=np.uint8)
        matrix[0, 0] = 3.0

        # Fill traceback matrix row-wise
        for i in range(1, len(alphabet) + 1):
            # Case: Start new row (i.e. move on to new nucleoside) by initializing
            # reachable cells from before
            matrix[i] = [int(val != 0.0) for val in matrix[i - 1]]

            # Case: Add more of current nucleoside
            for j in range(max_mass + 1):
                # If cell is not reachable, skip it
                if matrix[i, j] == 0.0:
                    continue

                # Add another nucleoside if possible
                if alphabet.get(i - 1).mass + j <= max_mass:
                    matrix[i, j + alphabet.get(i - 1).mass] += 2.0

        # Remove first row (as it is no longer needed)
        self.matrix = np.delete(matrix, 0, axis=0)

    def save(self, file_path: Path) -> None:
        """Save matrix in given file path."""
        np.save(file=file_path, arr=self.matrix)

    def allowed_movement(self, mass: int, nuc_idx: int) -> Tuple[bool, bool]:
        """
        Return flags indicating allowed movements for the given mass and nucleotide.

        Parameters
        ----------
        mass : int
            Given integer mass value (i.e. column index).
        nuc_idx : int
            Given nucleotide index (i.e. row index).

        Returns
        -------
        vertical_move : bool
            Flag whether a vertical move is possible.
        horizontal_move : bool
            Flag whether a horizontal move is possible.

        """
        # Raise error if mass is not in matrix (due to its size)
        if mass >= len(self.matrix[0]):
            raise NotImplementedError(
                f"The value {mass} is not in the traceback matrix. "
                f"Extend its size if you want to compute larger masses."
            )

        # Allow no movement for cells outside of matrix
        if mass < 0 or nuc_idx < 0:
            return False, False

        # Get current value
        current_value = self.matrix[nuc_idx, mass]

        # Determine possible movements from current value
        vertical_move = current_value % 2 == 1
        horizontal_move = (current_value >> 1) % 2 == 1

        return vertical_move, horizontal_move


class CompressedTracebackMatrix(TracebackMatrix, ABC):
    """Abstract class for compressed traceback matrix."""

    @property
    @abstractmethod
    def compression_rate(self) -> int:
        """Return compression rate used in traceback matrix."""

    @property
    @abstractmethod
    def build_settings(self) -> dict:
        """Return dictionary with settings for matrix building."""

    def rebuild(self, alphabet: NucleotideAlphabet) -> None:
        """
        Rebuild complete bit-representation matrix with dynamic programming.

        Parameters
        ----------
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        settings = self.build_settings

        # Select maximum integer mass for which matrix should be built
        max_mass = alphabet.max.integer_mass * MAX_SEQ_LENGTH

        # Initialize bit-representation matrix as numpy table (+ additional first row
        # for easier indexing)
        max_col = int(np.ceil((max_mass + 1) / self.compression_rate))
        matrix = np.zeros((len(alphabet) + 1, max_col), dtype=settings["dtype"])
        matrix[0, 0] = settings["initial_mask"]

        # Fill traceback matrix row-wise
        for i in range(1, len(alphabet) + 1):
            # Case: Start new row (i.e. move on to new nucleotide)
            # by initializing reachable cells from before
            matrix[i] = [
                ((val | (val >> 1)) & settings["alt_mask_second"])
                for val in matrix[i - 1]
            ]

            # Define number of cells to move (step) and bit shift in a cell (shift)
            step = int(alphabet.get(i - 1).mass / self.compression_rate)
            shift = alphabet.get(i - 1).mass % self.compression_rate

            # Case: Add more of current nucleotide
            for j in range(max_col):
                # Consider cell defined by step
                if step + j < max_col:
                    matrix[i, j + step] |= settings["alt_mask_first"] & (
                        (matrix[i, j] >> (2 * shift) << 1)
                        | (matrix[i, j] >> (2 * shift))
                    )

                # If shift is needed, consider the next cell as well
                if shift != 0 and j + step + 1 < max_col:
                    matrix[i, j + step + 1] |= settings["alt_mask_first"] & (
                        (matrix[i, j] << 2 * (self.compression_rate - shift) << 1)
                        | (matrix[i, j] << 2 * (self.compression_rate - shift))
                    )

        # Adjust last column for unused cells
        matrix[:, -1] &= settings["full_mask"] << 2 * (
            max_col - (max_mass + 1) % max_col
        )

        # Remove first row (as it is no longer needed)
        self.matrix = np.delete(matrix, 0, axis=0)

    def allowed_movement(self, mass: int, nuc_idx: int) -> Tuple[bool, bool]:
        # Raise error if mass is not in matrix (due to its size)
        if mass >= len(self.matrix[0]) * self.compression_rate:
            raise NotImplementedError(
                f"The value {mass} is not in the traceback matrix. "
                f"Extend its size if you want to compute larger masses."
            )

        # Allow no movement for cells outside of matrix
        if mass < 0 or nuc_idx < 0:
            return False, False

        # Get current value
        current_value = self.matrix[nuc_idx, mass // self.compression_rate] >> 2 * (
            self.compression_rate - 1 - mass % self.compression_rate
        )

        # Determine possible movements from current value
        vertical_move = current_value % 2 == 1
        horizontal_move = (current_value >> 1) % 2 == 1

        return vertical_move, horizontal_move


class MatrixWith4PerCellCompression(CompressedTracebackMatrix):
    """Class for compressed traceback matrix with 4 entries per cell."""

    @property
    def compression_rate(self):
        return 4

    @property
    def build_settings(self) -> dict:
        return {
            "dtype": np.uint8,
            "initial_mask": 0xC0,
            "alt_mask_first": 0xAA,
            "alt_mask_second": 0x55,
            "full_mask": np.uint8(0xFF),
        }


class MatrixWith8PerCellCompression(CompressedTracebackMatrix):
    """Class for compressed traceback matrix with 8 entries per cell."""

    @property
    def compression_rate(self):
        return 8

    @property
    def build_settings(self) -> dict:
        return {
            "dtype": np.uint16,
            "initial_mask": 0xC000,
            "alt_mask_first": 0xAAAA,
            "alt_mask_second": 0x5555,
            "full_mask": np.uint16(0xFFFF),
        }


class MatrixWith16PerCellCompression(CompressedTracebackMatrix):
    """Class for compressed traceback matrix with 16 entries per cell."""

    @property
    def compression_rate(self):
        return 16

    @property
    def build_settings(self) -> dict:
        return {
            "dtype": np.uint32,
            "initial_mask": 0xC0000000,
            "alt_mask_first": 0xAAAAAAAA,
            "alt_mask_second": 0x55555555,
            "full_mask": np.uint32(0xFFFFFFFF),
        }


class MatrixWith32PerCellCompression(CompressedTracebackMatrix):
    """Class for compressed traceback matrix with 32 entries per cell."""

    @property
    def compression_rate(self):
        return 32

    @property
    def build_settings(self) -> dict:
        return {
            "dtype": np.uint64,
            "initial_mask": 0xC000000000000000,
            "alt_mask_first": 0xAAAAAAAAAAAAAAAA,
            "alt_mask_second": 0x5555555555555555,
            "full_mask": np.uint64(0xFFFFFFFFFFFFFFFF),
        }
