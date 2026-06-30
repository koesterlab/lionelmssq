# -*- coding: utf-8 -*-
"""Module for error calculation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Self, Tuple

import numpy as np

from spectrseqtools.enums import ErrorMetric


@dataclass
class ErrorCalculator(ABC):
    """Class to calculate errors based on given precision and relative PPM tolerance."""

    tolerance: float = 10e-6
    decimal_places: int = 3

    @classmethod
    def with_metric(
        cls,
        metric: ErrorMetric = ErrorMetric.L1NORM,
        tolerance: float = 10e-6,
        decimal_places: int = 3,
    ) -> Self:
        """Initialize subclass based on given error metric."""
        match metric:
            case ErrorMetric.L1NORM:
                return ErrorUnderL1Norm(
                    tolerance=tolerance, decimal_places=decimal_places
                )
            case ErrorMetric.L2NORM:
                return ErrorUnderL2Norm(
                    tolerance=tolerance, decimal_places=decimal_places
                )
            case _:
                raise NotImplementedError(
                    f"Support for '{metric}' is currently not given."
                )

    @property
    def precision(self):
        """Return precision based on number of decimal places."""
        return 10 ** (-self.decimal_places)

    def set_target(
        self, su_mass: float, obs_masses: List[float] = None
    ) -> Tuple[int, int]:
        """
        Return precision-adjusted target and threshold (as integers).

        Parameters
        ----------
        su_mass : float
            Given target mass.
        obs_masses : List[float] | None
            Given masses used to compute threshold.

        Returns
        -------
        mass : int
            Precision-adapted target mass.
        threshold : int
            Precision-adapted error threshold.

        """
        # If observed masses are not defined, use SU mass for threshold calculation
        if obs_masses is None:
            obs_masses = [su_mass]

        # Set relative threshold and convert to integer
        threshold = int(
            np.ceil(self.get_threshold(mass_list=obs_masses) / self.precision)
        )

        # Convert the target to an integer for easy operations
        target = int(round(su_mass / self.precision, 0))

        return target, threshold

    @abstractmethod
    def get_threshold(self, mass_list: List[float]) -> float:
        """
        Calculate maximum tolerated error to still consider two masses equal.

        Parameters
        ----------
        mass_list : List[float]
            List of masses to consider.

        Returns
        -------
        float
            Error threshold.

        """


class ErrorUnderL1Norm(ErrorCalculator):
    """Class to calculate error thresholds based on the L1 norm."""

    def get_threshold(self, mass_list: List[float]) -> float:
        return self.tolerance * sum(mass_list)


class ErrorUnderL2Norm(ErrorCalculator):
    """Class to calculate error thresholds based on the L2 norm."""

    def get_threshold(self, mass_list: List[float]) -> float:
        return self.tolerance * sum(mass**2 for mass in mass_list) ** 0.5
