import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import silhouette_score
from typing import List

from spectrseqtools.common import initialize_raw_file_iterator
from spectrseqtools.masses import NUCLEOTIDE_DF

rt = get_mono()

PREPROCESS_TOL = 10e-6
THEORETICAL_BOUNDARY_FACTOR = 2
MIN_MZ = NUCLEOTIDE_DF["singleton_mz"].min() * (
    1 - THEORETICAL_BOUNDARY_FACTOR * PREPROCESS_TOL
)
MAX_MZ = NUCLEOTIDE_DF["singleton_mz"].max() * (
    1 + THEORETICAL_BOUNDARY_FACTOR * PREPROCESS_TOL
)
COL_TYPES_RAW = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "mz": pl.Float64,
}


def identify_singletons(file_path: str) -> pl.DataFrame:
    """
    Determine singleton candidates from MS2 scans in ThermoFisher RAW file.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    polars.DataFrame
        Dataframe containing singleton candidates obtained by matching m/z data.
    """
    # Initialize iterator for RAW file
    raw_file_read = initialize_raw_file_iterator(file_path=file_path)

    peak_list = []
    for _ in tqdm.tqdm(
        range(len(raw_file_read) - 1), desc="Extract m/z data from MS2 scans"
    ):
        # Select next scan
        scan = next(raw_file_read)

        # Skip scan if it is no MS2 scan
        if scan.ms_level != 2:
            continue

        # Extract raw peaks from scan (without deisotoping)
        peak_list += process_scan(scan)

    return select_singletons_from_peaks(peak_list=peak_list)


@dataclass
class RawPeak:
    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    mz: float


def process_scan(scan: ms_ditp.data_source.Scan) -> List[RawPeak]:
    """
    Extract raw peaks from MS2 scan.

    Parameters
    ----------
    scan : ms_deisotope.data_source.Scan
        ThermoFisher scan.

    Returns
    -------
    peak_list : List[RawPeak]
        List containing raw peak data.

    """
    # Convert scan to centroid data
    scan.pick_peaks()

    # Return None if scan does not contain any peaks
    if len(scan.peaks) <= 0:
        return []

    # Obtain scan time and scan ID
    scan_time = scan.scan_time
    scan_id = int(scan.scan_id.split("scan=")[-1])

    peak_list = []
    for idx in range(len(scan.peaks)):
        mz = scan.peaks[idx].mz

        # Only consider peaks with mass within theoretical bounds
        if MIN_MZ <= mz <= MAX_MZ:
            peak_list.append(
                RawPeak(
                    scan_id=scan_id,
                    scan_time=scan_time,
                    peak_idx=idx,
                    intensity=scan.peaks[idx].intensity,
                    mz=mz,
                )
            )
    return peak_list


def select_singletons_from_peaks(peak_list: List[RawPeak]) -> pl.DataFrame:
    """
    Select candidate singletons based on raw peaks.

    Build dataframe of raw peaks, match theoretical and observed mz,
    cluster them, and filter the candidates based on their cluster score.

    Parameters
    ----------
    peak_list : List[RawPeak]
        List containing raw peak data.

    Returns
    -------
    peak_df : polars.DataFrame
        Dataframe containing singleton candidates (name, score, and count).

    """
    # Build dataframe from peak list
    peak_df = pl.DataFrame(
        data=np.array(
            [[peak.__dict__[key] for key in COL_TYPES_RAW.keys()] for peak in peak_list]
        ),
        schema=COL_TYPES_RAW,
    )

    # Match observed m/z to singleton m/z from the reference table
    peak_df = peak_df.sort("mz").join_asof(
        NUCLEOTIDE_DF.sort("singleton_mz"),
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
            .lt(PREPROCESS_TOL)
            .alias("is_match")
        )
        .filter(pl.col("is_match"))
        .sort(["representative", "scan_time"])
    )

    # Map representative nucleoside, cluster score, and count to each nucleoside group
    peak_df = peak_df.group_by("id_list").map_groups(
        lambda x: pl.DataFrame(
            {
                "id": x["id_list"][0],
                "cluster_score": calculate_cluster_score(x["scan_time"]),
                "count": len(x["id_list"]),
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
