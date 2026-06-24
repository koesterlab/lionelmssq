# -*- coding: utf-8 -*-
"""Module for commonly used functions."""

ERROR_METHOD = "l1_norm"


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
