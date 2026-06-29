# -*- coding: utf-8 -*-
"""Module for commonly used functions."""

from pathlib import Path
from typing import Tuple

ERROR_METHOD = "l1_norm"


def set_output_path(input_path: Path, output_dir: Path) -> Tuple[Path, str]:
    """
    Set directory and prefix for output path.

    Parameters
    ----------
    input_path : Path
        Input path.
    output_dir : Path
        Output directory.

    Returns
    -------
    path_dir : Path
        Updated output directory.
    path_prefix : str
        Updated output prefix.

    """
    path = input_path.resolve()
    path_dir = path.parent if output_dir is None else output_dir
    path_prefix = path.stem

    return path_dir, path_prefix


def calculate_error_threshold(mass1: float, mass2: float, threshold: float) -> float:
    """
    Calculate maximum tolerated error to still consider two masses equal.

    Parameters
    ----------
    mass1 : float
        First mass.
    mass2 : float
        Second mass.
    threshold : float
        Relative tolerance.

    Returns
    -------
    float
        Error threshold.

    """
    match ERROR_METHOD:
        case "l1_norm":
            return threshold * (mass1 + mass2)
        case "l2_norm":
            return threshold * ((mass1**2 + mass2**2) ** 0.5)
        case _:
            raise NotImplementedError("This error method is not implemented.")
