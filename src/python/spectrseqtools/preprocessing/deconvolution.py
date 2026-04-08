# -*- coding: utf-8 -*-
"""Deconvolution of raw mass spectrometry data."""

from dataclasses import dataclass
from typing import List, Self, Tuple

import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
from clr_loader import get_mono

rt = get_mono()


# TODO: Estimate the default value from sequence length (IMP: Should be
#  large enough to cover the charge states of the precursors!
DEFAULT_CHARGE_VALUE = 30
COL_TYPES_DEISOTOPED = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "neutral_mass": pl.Float64,
    "is_precursor_deisotoped": pl.Boolean,
    "mz": pl.Float64,
}


# METHOD: To deconvolute/deisotope (which we use interchangeable because both
# happen at the same time) a MS2 scan, we determine all its peaks with
# the ms_deisotope package. We then identify the precursor peak (if it exists)
# and perform a check whether it has been deisotoped (since this is the only
# one that we can reliably obtain as there exists a reference m/z).
# We call a precursor correctly deisotoped if the m/z of the peak is:
# (1) less than the m/z of the precursor, and
# (2) greater than its isotopic shift times a given factor (here: 10).


@dataclass
class DeconvolutionParameters:
    """Class for deconvolution parameters used by ms_deisotope."""

    min_precursor_charge: int
    isotopic_shift_factor: int
    charge_range: Tuple[int, int]
    minimum_intensity: float
    scorer: ms_ditp.MSDeconVFitter
    averagine: ms_ditp.Averagine
    max_missed_peaks: int
    scale_method: str
    error_tol: float
    truncate_after: float

    def to_scan_dependent_dict(self, scan: ms_ditp.data_source.Scan) -> dict:
        """Return dictionary of deconvolution parameters for the given scan.

        Parameters
        ----------
        scan : ms_deisotope.data_source.Scan
            ThermoFisher scan.

        Returns
        -------
        dict
            Dictionary containing deconvolution parameters for ms_deisotope call.

        """
        # Retrieve parameters from class
        output_dict = self.__dict__.copy()

        # Set scan-dependent parameters
        output_dict["peaklist"] = scan
        output_dict["charge_range"] = self.select_charge_range(scan=scan)
        output_dict["minimum_intensity"] = self.select_min_intensity(scan=scan)

        # Pop parameters not used by ms_deisotope
        output_dict.pop("min_precursor_charge")
        output_dict.pop("isotopic_shift_factor")

        return output_dict

    def select_charge_range(self, scan: ms_ditp.data_source.Scan) -> Tuple[int, int]:
        """
        Select range for accepted charge values.

        Parameters
        ----------
        scan : ms_deisotope.data_source.Scan
            ThermoFisher scan.

        Returns
        -------
        min_charge : int
            Minimum accepted charge value.
        max_charge : int
            Maximum accepted charge value.

        Notes
        -----
        This function is inspired by https://github.com/koesterlab/oliglow,
        originally implemented by Moshir Harsh (btemoshir@gmail.com).

        """
        # Return user-defined charge range (if given)
        if self.charge_range is not None:
            return self.charge_range

        # Select charge (or use default if not given)
        charge = scan.precursor_information.charge
        if not isinstance(charge, int):
            charge = DEFAULT_CHARGE_VALUE

        # Return charge with consideration to polarity
        return scan.polarity, charge * scan.polarity

    def select_min_intensity(self, scan: ms_ditp.data_source.Scan) -> float:
        """
        Select minimum intensity value below which peaks are ignored.

        Parameters
        ----------
        scan : ms_deisotope.data_source.Scan
            ThermoFisher scan.

        Returns
        -------
        float
            Minimum intensity value.

        Notes
        -----
        This function is inspired by https://github.com/koesterlab/oliglow,
        originally implemented by Moshir Harsh (btemoshir@gmail.com).

        """
        # If the user defined no minimum intensity, set it to -infinity
        if self.minimum_intensity is None:
            min_intensity = -np.inf
        else:
            min_intensity = self.minimum_intensity

        # Return maximum of intensity set by user and found in scan peak set
        return max(min_intensity, min(peak.intensity for peak in scan.peak_set))


@dataclass
class DeisotopedPeak:
    """Class for deisotoped peaks."""

    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    neutral_mass: float
    is_precursor_deisotoped: bool
    mz: float

    @classmethod
    def default(cls) -> Self:
        """Return default peak."""
        return DeisotopedPeak(
            scan_id=-1,
            scan_time=0.0,
            peak_idx=-1,
            intensity=0.0,
            neutral_mass=0.0,
            is_precursor_deisotoped=False,
            mz=0.0,
        )


@dataclass
class DeisotopedPeakList:
    """Class for list of deisotoped peaks from MS spectra."""

    peaks: List[DeisotopedPeak]

    def __add__(self, other) -> Self:
        """Add another peak list."""
        return DeisotopedPeakList(self.peaks + other.peaks)

    @classmethod
    def default(cls) -> Self:
        """Return empty peak list"""
        return DeisotopedPeakList(peaks=[])

    @classmethod
    def from_scan(
        cls, scan: ms_ditp.data_source.Scan, params: DeconvolutionParameters
    ) -> Self:
        """
        Obtain deconvoluted peaks from MS2 scan.

        Parameters
        ----------
        scan : ms_deisotope.data_source.Scan
            ThermoFisher scan.
        params : DeconvolutionParameters
            Deconvolution parameters (mainly used by ms_deisotope).

        Returns
        -------
        DeisotopedPeakList
            Object containing deconvoluted peak data.

        Notes
        -----
        This function is inspired by https://github.com/koesterlab/oliglow,
        originally implemented by Moshir Harsh (btemoshir@gmail.com).

        """
        # Return default if precursor charge is too low for consideration
        if (
            not isinstance(scan.precursor_information.charge, int)
            or scan.precursor_information.charge < params.min_precursor_charge
        ):
            return DeisotopedPeakList.default()

        # Convert scan to centroid data
        scan.pick_peaks()

        # Deconvolute/deisotope with ms_deisotope
        peak_set = ms_ditp.deconvolute_peaks(
            **params.to_scan_dependent_dict(scan)
        ).peak_set

        # Return default if scan does not contain any deisotoped peaks
        if len(peak_set) <= 0:
            return DeisotopedPeakList.default()

        # Obtain scan time and scan ID
        scan_time = scan.scan_time
        scan_id = int(scan.scan_id.split("scan=")[-1])

        # Calculate m/z of precursor and accepted m/z range
        precursor_mz = scan.precursor_information.mz
        min_mz = scan.isolation_window.target - (1 * scan.isolation_window.lower)
        max_mz = scan.isolation_window.target + (1 * scan.isolation_window.upper)

        # Iterate through the deisotoped scan
        peak_list = [DeisotopedPeak.default()] * len(peak_set)
        for idx in range(len(peak_set)):
            mz = peak_set.peaks[idx].mz
            is_precursor = min_mz <= mz <= max_mz
            peak_list[idx] = DeisotopedPeak(
                scan_id=scan_id,
                scan_time=scan_time,
                peak_idx=idx,
                intensity=peak_set.peaks[idx].intensity,
                neutral_mass=peak_set.peaks[idx].neutral_mass,
                is_precursor_deisotoped=(
                    False
                    if not is_precursor
                    else precursor_mz
                    - abs(
                        params.isotopic_shift_factor
                        * ms_ditp.averagine.isotopic_shift(peak_set.peaks[idx].charge)
                    )
                    <= mz
                    <= precursor_mz
                ),
                mz=mz,
            )
        return DeisotopedPeakList(peaks=peak_list)

    def to_fragments(self, tolerance: float) -> pl.DataFrame:
        """
        Aggregate deisotoped peaks into fragments by grouping based on similar mass.

        Build dataframe of deisotoped peaks, group the peaks by their mass
        (within PPM tolerance), and aggregate them by selecting the maximum
        observed mass and total observed intensity in each group as a fragment.

        Parameters
        ----------
        tolerance : float
            Error tolerance for individual masses.

        Returns
        -------
        peak_df : polars.DataFrame
            Dataframe containing fragment information.

        Notes
        -----
        This function is inspired by https://github.com/koesterlab/oliglow,
        originally implemented by Moshir Harsh (btemoshir@gmail.com).

        """
        # Build dataframe from peak list
        peak_df = pl.DataFrame(
            data=np.array(
                [
                    [peak.__dict__[key] for key in COL_TYPES_DEISOTOPED]
                    for peak in self.peaks
                ]
            ),
            schema=COL_TYPES_DEISOTOPED,
        )

        # Cluster peaks together when mass is within PPM tolerance of each other
        peak_df = peak_df.sort("neutral_mass").with_columns(
            (
                abs(pl.col("neutral_mass").shift(1) - pl.col("neutral_mass"))
                / pl.col("neutral_mass").shift(1)
            )
            .fill_null(0)
            .fill_nan(0)
            .gt(tolerance)
            .cum_sum()
            .alias("ppm_group")
        )

        # Aggregate by PPM group (assign maximum neutral mass and total intensity to each group)
        return (
            peak_df.group_by("ppm_group")
            .agg(
                neutral_mass=pl.col("neutral_mass").max(),
                intensity=pl.col("intensity").sum(),
                is_precursor_deisotoped=pl.col("is_precursor_deisotoped").max(),
            )
            .sort("neutral_mass")
        )
