import importlib.resources
import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from typing import List, Tuple

from spectrseqtools.common import initialize_raw_file_iterator
from spectrseqtools.masses import ELEMENT_MASSES

rt = get_mono()

PREPROCESS_TOL = 10e-6
MIN_MS1_CHARGE_STATE = 3
# TODO: Estimate the default value from sequence length (IMP: Should be
#  large enough to cover the charge states of the precursors!
DEFAULT_CHARGE_VALUE = 30
ISOTOPIC_SHIFT_FACTOR = 10
COL_TYPES_DEISOTOPED = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "neutral_mass": pl.Float64,
    "is_precursor_deisotoped": pl.Boolean,
    "is_raw_peak": pl.Boolean,
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


class DeconvolutionParameters:
    def __init__(self, params: dict):
        # Set possibly scan-dependent parameters (if given)
        self.charge_range = params.pop("charge_range", None)
        self.minimum_intensity = params.pop("minimum_intensity", None)

        # Set scan-independent parameters
        self.scan_independent_params = set_scan_independent_params(params=params)


def set_scan_independent_params(params: dict) -> dict:
    """
    Set full dict for deconvolution parameters independent of the peak scan.

    Parameters
    ----------
    params : dict
        Dictionary containing some, not necessary all deconvolution parameters.

    Returns
    -------
    params : dict
        Dictionary containing all scan-independent deconvolution parameters.

    """
    # TODO: Take care of "min_score", there should be way to estimate it
    # Select minimum accepted score between theoretical and experimental spectra
    min_score = params.pop("min_score", 150.0)

    # Select error tolerance between theoretical and experimental m/z;
    # perhaps this should be < 1./max(charge) for a reasonable value
    mass_error_tol = params.pop("mass_error_tol", 0.02)

    # Calculate goodness-of-fit criterion for envelope scoring
    params.setdefault("scorer", ms_ditp.MSDeconVFitter(min_score, mass_error_tol))

    # Set average composition of considered bases
    params.setdefault(
        "averagine", ms_ditp.Averagine(set_averagine(backbone="phosphate"))
    )

    # Set maximum number of tolerated missed peaks (keep small, i.e. 0 or 1)
    params.setdefault("max_missed_peaks", 0)

    # Set name of method to scale intensity value
    params.setdefault("scale_method", "sum")

    # Set error tolerance for matching experimental and theoretical peaks
    params.setdefault("error_tol", 2e-5)

    # Set percentage of included isotopic pattern (very sensitive, see discussion in ms_deisotope docs)
    params.setdefault("truncate_after", 0.9)

    return params


def set_averagine(backbone: str) -> dict:
    """
    Calculate the average elemental composition of RNA.

    Parameters
    ----------
    backbone : ["no_backbone", "phosphate", "thiophosphate"]
        Backbone considered for the composition.

    Returns
    -------
    averagine_composition : dict
        Dictionary containing average elemental composition.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """
    # Build dict with elemental compositions from file
    bases = pl.read_csv(
        importlib.resources.files(__package__) / "assets" / "elemental_composition.tsv",
        separator="\t",
    )
    base_compositions = [
        {
            col: row[bases.get_column_index(col)]
            for col in bases.columns
            if col != "base"
        }
        for row in bases.iter_rows()
    ]

    # Calculate average elemental composition
    average_composition = {}
    for element in base_compositions[0].keys():
        average_composition[element] = sum(
            [float(base[element]) for base in base_compositions]
        ) / len(base_compositions)

    # Add backbone elements (if needed)
    match backbone:
        case "no_backbone":
            pass
        case "phosphate":
            # Add 1 phosphorus and 2 oxygen for the phosphate group
            average_composition["O"] += 2
            average_composition["P"] += 1
        case "thiophosphate":
            # Add 1 phosphorus, 1 sulfur, and 1 oxygen for the phosphate group
            average_composition["O"] += 1
            average_composition["S"] += 1
            average_composition["P"] += 1
        case _:
            raise NotImplementedError(
                f"Support for '{backbone}' is currently not given."
            )

    return average_composition


def deconvolute(file_path: str, params: dict, extract_all: bool = False) -> pl.DataFrame:
    """
    Deconvolute/deisotope peaks in MS2 scans from ThermoFisher RAW file.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    params : dict
        Dictionary containing deconvolution parameters.

    Returns
    -------
    polars.DataFrame
        Dataframe containing fragment monoisotopic masses and intensities.

    """
    # Load deconvolution parameter based on parameter dict
    params = DeconvolutionParameters(params)

    # Initialize iterator for RAW file
    raw_file_read = initialize_raw_file_iterator(file_path=file_path)

    peak_list = []
    for _ in tqdm.tqdm(range(len(raw_file_read) - 1), desc="Deisotoping MS2 scans"):
        # Select next scan
        scan = next(raw_file_read)

        # Skip scan if it is no MS2 scan
        if scan.ms_level != 2:
            continue
        # If it is an MS2 scan, skip it if the precursor charge is lower than MIN_MS1_CHARGE_STATE
        if (
            not isinstance(scan.precursor_information.charge, int)
            or scan.precursor_information.charge < MIN_MS1_CHARGE_STATE
        ):
            continue

        # Deconvolute scan to get list of deisotoped peaks
        peak_list += deconvolute_scan(scan=scan, params=params, extract_all=extract_all)

    return aggregate_peaks_into_fragments(peak_list)


@dataclass
class DeisotopedPeak:
    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    neutral_mass: float
    is_precursor_deisotoped: bool
    is_raw_peak: bool
    mz: float


def deconvolute_scan(
    scan: ms_ditp.data_source.Scan, params: dict, extract_all: bool = False
) -> List[DeisotopedPeak]:
    """
    Deconvolute peaks from MS2 scan.

    Parameters
    ----------
    scan : ms_deisotope.data_source.Scan
        ThermoFisher scan.
    params : dict
        Dictionary containing deconvolution parameters.

    Returns
    -------
    peak_list : List[DeisotopedPeak]
        List containing deconvoluted peak data.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """
    # Convert scan to centroid data
    scan.pick_peaks()

    # Deconvolute/deisotope with ms_deisotope
    peak_set = ms_ditp.deconvolute_peaks(
        peaklist=scan,
        charge_range=params.charge_range
        if params.charge_range is not None
        else select_charge_range(scan=scan),
        minimum_intensity=select_min_intensity(
            scan=scan, min_intensity=params.minimum_intensity
        ),
        **params.scan_independent_params,
    ).peak_set

    # Return None if scan does not contain any deisotoped peaks
    if len(peak_set) <= 0:
        return []

    # Obtain scan time and scan ID
    scan_time = scan.scan_time
    scan_id = int(scan.scan_id.split("scan=")[-1])

    # Calculate m/z of precursor and accepted m/z range
    precursor_mz = scan.precursor_information.mz
    min_mz = scan.isolation_window.target - (1 * scan.isolation_window.lower)
    max_mz = scan.isolation_window.target + (1 * scan.isolation_window.upper)

    # Iterate through the deisotoped scan
    peak_list = [0] * len(peak_set)
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
                    ISOTOPIC_SHIFT_FACTOR
                    * ms_ditp.averagine.isotopic_shift(peak_set.peaks[idx].charge)
                )
                <= mz
                <= precursor_mz
            ),
            is_raw_peak = False,
            mz=mz,
        )

    if extract_all:
        env_peak_idx = []
        for peak in peak_set.peaks:
            peak_envelope = peak.envelope
            for env_peak in peak_envelope:
                env_peak_mz = env_peak[0]
                raw_peak = scan.has_peak(env_peak_mz)
                if raw_peak is not None:
                    env_peak_idx.append(raw_peak.index)
        
        precursor_charge = scan.precursor_information.charge
        for peak in scan.peak_set:
            if peak.index in env_peak_idx:
                continue
            for z in range(1, precursor_charge+1):
                peak_list.append(
                    DeisotopedPeak(
                        scan_id=scan_id,
                        scan_time=scan_time,
                        peak_idx=idx*1000 + peak.index,
                        intensity=peak.intensity,
                        neutral_mass=(peak.mz*z) + (z*ELEMENT_MASSES["H+"]),
                        is_precursor_deisotoped=False,
                        is_raw_peak = True,
                        mz=peak.mz,
                    ))
    return peak_list


def select_charge_range(scan: ms_ditp.data_source.Scan) -> Tuple[int, int]:
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
    # Select charge (or use default if not given)
    charge = scan.precursor_information.charge
    if not isinstance(charge, int):
        charge = DEFAULT_CHARGE_VALUE

    # Return charge with consideration to polarity
    return scan.polarity, charge * scan.polarity


def select_min_intensity(scan: ms_ditp.data_source.Scan, min_intensity: float) -> float:
    """
    Select minimum intensity value below which peaks are ignored.

    Parameters
    ----------
    scan : ms_deisotope.data_source.Scan
        ThermoFisher scan.
    min_intensity : float
        Minimum intensity value set by user.

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
    if min_intensity is None:
        min_intensity = -np.inf

    # Return maximum of intensity set by user and found in scan peak set
    return max(min_intensity, min(peak.intensity for peak in scan.peak_set))


def aggregate_peaks_into_fragments(peak_list: List[DeisotopedPeak]) -> pl.DataFrame:
    """
    Aggregate deisotoped peaks into fragments by grouping based on similar mass.

    Build dataframe of deisotoped peaks, group the peaks by their mass
    (within PPM tolerance), and aggregate them by selecting the maximum
    observed mass and total observed intensity in each group as a fragment.

    Parameters
    ----------
    peak_list : List[DeisotopedPeak]
        List containing deconvoluted peak data.

    Returns
    -------
    peak_df : polars.DataFrame
        Dataframe containing fragment monoisotopic masses and intensities.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """
    # Build dataframe from peak list
    peak_df = pl.DataFrame(
        data=np.array(
            [
                [peak.__dict__[key] for key in COL_TYPES_DEISOTOPED.keys()]
                for peak in peak_list
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
        .gt(PREPROCESS_TOL)
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
