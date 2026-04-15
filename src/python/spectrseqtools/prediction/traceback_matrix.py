# -*- coding: utf-8 -*-
"""Module for traceback matrix."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self

import numpy as np
from platformdirs import user_cache_dir

# Set OS-independent cache directory for traceback matrix
MATRIX_DIR = user_cache_dir(
    appname="spectrseqtools/traceback_matrix", version="1.0", ensure_exists=True
)

# Set maximum sequence length to be represented in traceback matrix
MAX_SEQ_LENGTH = 35


# METHOD: Use dynamic programming to build a traceback matrix that implicitly
# contains all inferable compositions up to a specific target mass (based on
# an underlying nucleotide alphabet).


@dataclass
class TracebackMatrix:
    """Class for traceback matrix implicitly containing all compositions."""

    matrix: np.ndarray
    compression_rate: int

    @classmethod
    def load(cls, path: str, integer_masses: List[int]) -> Self:
        """
        Load traceback matrix if it exists and compute it otherwise.

        Parameters
        ----------
        path : str
            Path to traceback matrix.
        integer_masses : List[int]
            List of integer nucleotide masses.

        """
        # Select compression rate from path string
        compression_rate = int(path.split(".")[-1].rstrip("_per_cell"))

        # Compute and save bit-representation matrix if not existing
        if not Path(f"{path}.npy").is_file():
            print("Matrix not found")
            matrix = (
                cls.set_up_matrix(integer_masses)
                if compression_rate == 1
                else cls.set_up_bit_matrix(integer_masses, compression_rate)
            )
            np.save(path, matrix)

        # Read traceback matrix
        return cls(matrix=np.load(f"{path}.npy"), compression_rate=compression_rate)

    @classmethod
    def set_up_matrix(cls, integer_masses: List[int]) -> Self:
        """
        Calculate complete matrix with dynamic programming.

        Parameters
        ----------
        integer_masses : List[int]
            List of integer nucleotide masses.

        """
        # Select maximum integer mass for which matrix should be built
        max_mass = max(integer_masses) * MAX_SEQ_LENGTH

        # Initialize matrix as numpy table
        matrix = np.zeros((len(integer_masses), max_mass + 1), dtype=np.uint8)
        matrix[0, 0] = 3.0

        # Fill traceback matrix row-wise
        for i in range(1, len(integer_masses)):
            # Case: Start new row (i.e. move on to new nucleoside) by initializing
            # reachable cells from before
            matrix[i] = [int(val != 0.0) for val in matrix[i - 1]]

            # Case: Add more of current nucleoside
            for j in range(max_mass + 1):
                # If cell is not reachable, skip it
                if matrix[i, j] == 0.0:
                    continue

                # Add another nucleoside if possible
                if integer_masses[i] + j <= max_mass:
                    matrix[i, j + integer_masses[i]] += 2.0

        return cls(matrix=matrix, compression_rate=1)

    @classmethod
    def set_up_bit_matrix(
        cls, integer_masses: List[int], compression_rate: int
    ) -> Self:
        """
        Calculate complete bit-representation matrix with dynamic programming.

        Parameters
        ----------
        integer_masses : List[int]
            List of integer nucleotide masses.
        compression_rate : int
            Compression per matrix cell.

        """
        settings = select_matrix_building_settings(compression_rate)

        # Select maximum integer mass for which matrix should be built
        max_mass = max(integer_masses) * MAX_SEQ_LENGTH

        # Initialize bit-representation matrix as numpy table
        max_col = int(np.ceil((max_mass + 1) / compression_rate))
        matrix = np.zeros((len(integer_masses), max_col), dtype=settings["type"])
        matrix[0, 0] = settings["init"]

        # Fill traceback matrix row-wise
        for i in range(1, len(integer_masses)):
            # Case: Start new row (i.e. move on to new nucleotide)
            # by initializing reachable cells from before
            matrix[i] = [
                ((val | (val >> 1)) & settings["alt_sec"]) for val in matrix[i - 1]
            ]

            # Define number of cells to move (step) and bit shift in a cell (shift)
            step = int(integer_masses[i] / compression_rate)
            shift = integer_masses[i] % compression_rate

            # Case: Add more of current nucleotide
            for j in range(max_col):
                # Consider cell defined by step
                if step + j < max_col:
                    matrix[i, j + step] |= settings["alt_first"] & (
                        (matrix[i, j] >> (2 * shift) << 1)
                        | (matrix[i, j] >> (2 * shift))
                    )

                # If shift is needed, consider the next cell as well
                if shift != 0 and j + step + 1 < max_col:
                    matrix[i, j + step + 1] |= settings["alt_first"] & (
                        (matrix[i, j] << 2 * (compression_rate - shift) << 1)
                        | (matrix[i, j] << 2 * (compression_rate - shift))
                    )

        # Adjust last column for unused cells
        matrix[:, -1] &= settings["full"] << 2 * (max_col - (max_mass + 1) % max_col)

        return cls(matrix=matrix, compression_rate=compression_rate)

    def assert_in_matrix(self, mass: int) -> None:
        """
        Raise error if given mass is not in traceback matrix.

        Parameters
        ----------
        mass : int
            Given mass.

        """
        # Raise error if mass is not in matrix (due to its size)
        if mass >= len(self.matrix[0]) * self.compression_rate:
            raise NotImplementedError(
                f"The value {mass} is not in the traceback matrix. "
                f"Extend its size if you want to compute larger masses."
            )

    def get_entry(self, mass: int, nuc_idx: int) -> int:
        """
        Get matrix entry corresponding to given mass and nucleotide index.

        Parameters
        ----------
        mass : int
            Given mass (i.e. column in matrix).
        nuc_idx : int
            Given nucleotide index (i.e. row in matrix).

        Returns
        -------
        int
            Corresponding entry in matrix.

        """
        return (
            self.matrix[nuc_idx, mass]
            if self.compression_rate == 1
            else self.matrix[nuc_idx, mass // self.compression_rate]
            >> 2 * (self.compression_rate - 1 - mass % self.compression_rate)
        )

    def is_unreachable(self, value: int) -> bool:
        """
        Check whether a given entry is unreachable.

        Parameters
        ----------
        value : int
            Given matrix entry.

        Returns
        -------
        bool
            Flag whether given entry is unreachable.

        """
        if self.compression_rate != 1 and value % self.compression_rate == 0.0:
            return True
        return False


def set_matrix_path(precision: float, compression_rate: int) -> str:
    """
    Set path to traceback matrix.

    Parameters
    ----------
    precision : float
        Precision used for (nucleotide) masses.
    compression_rate : int
        Compression per matrix cell.

    Returns
    -------
    path : str
        Path to traceback matrix.

    """
    # Set path for traceback matrix
    path = f"{MATRIX_DIR}/tol_{precision:.0E}.{compression_rate}_per_cell"

    # Create directory for traceback matrix if it does not already exist
    subdir = "/".join(path.split("/")[:-1])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    return path


def select_matrix_building_settings(compression_rate: int) -> dict:
    """
    Select parameters to build traceback matrix based on compression rate.

    Parameters
    ----------
    compression_rate : int
        Compression per matrix cell.

    Returns
    -------
    dict
        Dictionary containing building settings.

    """
    match compression_rate:
        case 4:
            return {
                "type": np.uint8,
                "init": 0xC0,
                "alt_first": 0xAA,
                "alt_sec": 0x55,
                "full": np.uint8(0xFF),
            }
        case 8:
            return {
                "type": np.uint16,
                "init": 0xC000,
                "alt_first": 0xAAAA,
                "alt_sec": 0x5555,
                "full": np.uint16(0xFFFF),
            }
        case 16:
            return {
                "type": np.uint32,
                "init": 0xC0000000,
                "alt_first": 0xAAAAAAAA,
                "alt_sec": 0x55555555,
                "full": np.uint32(0xFFFFFFFF),
            }
        case 32:
            return {
                "type": np.uint64,
                "init": 0xC000000000000000,
                "alt_first": 0xAAAAAAAAAAAAAAAA,
                "alt_sec": 0x5555555555555555,
                "full": np.uint64(0xFFFFFFFFFFFFFFFF),
            }
        case _:
            raise ValueError(
                f"The compression rate {compression_rate} is "
                f"not compatible with the matrix setup."
            )
