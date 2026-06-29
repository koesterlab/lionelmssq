# -*- coding: utf-8 -*-
"""Singleton selection from raw mass spectrometry data."""

from dataclasses import dataclass
from typing import List, Self

import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
from clr_loader import get_mono
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import silhouette_score

from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet

rt = get_mono()


COL_TYPES_RAW = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "mz": pl.Float64,
}


@dataclass
class SingletonBoundaries:
    """Class for theoretical m/z boundaries for singleton identification."""

    min_mz: float
    max_mz: float

    @classmethod
    def from_alphabet(
        cls, alphabet: pl.DataFrame, tolerance: float, boundary_factor: float
    ) -> Self:
        """Set singleton boundaries based on nucleotide alphabet."""
        return SingletonBoundaries(
            min_mz=alphabet.min.singleton_mz * (1 - boundary_factor * tolerance),
            max_mz=alphabet.max.singleton_mz * (1 + boundary_factor * tolerance),
        )


@dataclass
class RawPeak:
    """Class for raw peaks."""

    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    mz: float


@dataclass
class RawPeakList:
    """Class for list of raw peaks from MS spectra."""

    peaks: List[RawPeak]

    def __add__(self, other) -> Self:
        """Add another peak list."""
        return RawPeakList(self.peaks + other.peaks)

    @classmethod
    def default(cls) -> Self:
        """Return empty peak list"""
        return RawPeakList(peaks=[])

    @classmethod
    def from_scan(
        cls, scan: ms_ditp.data_source.Scan, boundaries: SingletonBoundaries
    ) -> Self:
        """
        Extract raw peaks from MS2 scan.

        Parameters
        ----------
        scan : ms_deisotope.data_source.Scan
            ThermoFisher scan.
        boundaries : SingletonBoundaries
            Theoretical boundaries on singleton m/z imposed by alphabet.

        Returns
        -------
        RawPeakList
            Object containing raw peak data.

        """
        # Convert scan to centroid data
        scan.pick_peaks()

        # Return None if scan does not contain any peaks
        if len(scan.peaks) <= 0:
            return RawPeakList.default()

        # Obtain scan time and scan ID
        scan_time = scan.scan_time
        scan_id = int(scan.scan_id.split("scan=")[-1])

        peak_list = []
        for idx, _ in enumerate(scan.peaks):
            mz = scan.peaks[idx].mz

            # Only consider peaks with mass within theoretical bounds
            if boundaries.min_mz <= mz <= boundaries.max_mz:
                peak_list.append(
                    RawPeak(
                        scan_id=scan_id,
                        scan_time=scan_time,
                        peak_idx=idx,
                        intensity=scan.peaks[idx].intensity,
                        mz=mz,
                    )
                )
        return RawPeakList(peaks=peak_list)

    def to_singletons(
        self, alphabet: NucleotideAlphabet, tolerance: float
    ) -> pl.DataFrame:
        """
        Select candidate singletons based on raw peaks.

        Build dataframe of raw peaks, match theoretical and observed mz,
        cluster them, and filter the candidates based on their cluster score.

        Parameters
        ----------
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.
        tolerance : float
            Error tolerance for individual masses.

        Returns
        -------
        peak_df : polars.DataFrame
            Dataframe containing singleton candidates.

        """
        alphabet_df = alphabet.to_dataframe()
        alphabet_df = alphabet_df.with_columns(
            pl.col("names").first().alias("representative")
        )

        # Build dataframe from peak list
        peak_df = pl.DataFrame(
            data=np.array(
                [[peak.__dict__[key] for key in COL_TYPES_RAW] for peak in self.peaks]
            ),
            schema=COL_TYPES_RAW,
        )

        # Match observed m/z to singleton m/z from the reference table
        peak_df = peak_df.sort("mz").join_asof(
            alphabet_df.sort("singleton_mz"),
            left_on="mz",
            right_on="singleton_mz",
            strategy="nearest",
        )

        # Compute mass error between observed and singleton m/z
        peak_df = (
            peak_df.sort("mz")
            .with_columns(
                (abs(pl.col("mz") - pl.col("singleton_mz")) / pl.col("mz"))
                .fill_null(0)
                .fill_nan(0)
                .lt(tolerance)
                .alias("is_match")
            )
            .filter(pl.col("is_match"))
            .sort(["representative", "scan_time"])
        )

        # Map representative nucleotide, cluster score, and count to each nucleotide group
        peak_df = peak_df.group_by("names").map_groups(
            lambda x: pl.DataFrame(
                {
                    "id": x["names"][0],
                    "cluster_score": calculate_cluster_score(x["scan_time"]),
                    "count": len(x["names"]),
                }
            )
        )

        # Filter candidate singletons by cluster score
        return (
            peak_df.filter(pl.col("cluster_score") >= 0).select(
                ["id", "count", "cluster_score"]
            )
        ).sort("count", descending=True)


def calculate_cluster_score(scan_times: pl.Series) -> float:
    """
    Determine score measuring how clustered each scan peaks is.

    By scan time, use DBSCAN and Silhouette score to evaluate peak clustering.

    Parameters
    ----------
    scan_times : polars.Series
        Scan times.

    Returns
    -------
    score : float
        Silhouette score for the DBSCAN cluster of scan times.
    """
    # Transform series to numpy array
    scan_times = scan_times.sort().to_numpy()

    # Cluster scan times using 1D DBSCAN
    clusters = DBSCAN1D(eps=0.5, min_samples=10).fit_predict(scan_times)

    # Flatten array containing scan times
    scan_times = scan_times.reshape(-1, 1)

    # Raise error if no cluster was found
    if len(set(clusters)) == 0:
        raise NotImplementedError("No cluster was found. This should not be possible.")

    # Return silhouette score if multiple clusters were found
    if len(set(clusters)) > 1:
        return silhouette_score(scan_times, clusters)

    # Return minimum score if only noise was found, i.e. cluster == -1
    if list(set(clusters))[0] == -1:
        return -1.0

    # Return neutral score if only one (non-noisy) cluster was found
    return 0.0
