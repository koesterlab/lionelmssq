import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from typing import List

from spectrseqtools.masses import EXPLANATION_MASSES
from spectrseqtools.deconvolution import deconvolute_scan, DeisotopedPeak, select_min_intensity, MIN_MS1_CHARGE_STATE, PREPROCESS_TOL, DeconvolutionParameters, aggregate_peaks_into_fragments
from spectrseqtools.singleton_identification import process_scan, RawPeak, COL_TYPES_RAW, calculate_cluster_score

rt = get_mono()

@dataclass
class PrecursorPeak:
    scan_time: float
    mz: float
    intensity: float
    charge: int

@dataclass
class DeisotopedScanBunch:
    scan_time: float
    precursor_raw_mz: float
    precursor_raw_intensity: float
    precursor_neutral_mass: float
    fragments: List[DeisotopedPeak]
    singleton_raw_peaks: List[RawPeak]
    num_fragments : int

COL_TYPES_PRECURSOR = {
    "scan_time": pl.Float64,
    "mz": pl.Float64,
    "intensity": pl.Float64,
    "charge": pl.Int64,
}

COL_TYPES_DEISOTOPED_SCAN_BUNCH = {
    "scan_time": pl.Float64,
    "precursor_raw_mz": pl.Float64,
    "precursor_raw_intensity": pl.Float64,
    "precursor_neutral_mass": pl.Float64,
    "num_fragments": pl.Int64,
}


def initialize_raw_file_iterator_ungrouped(
    file_path: str,
) -> ms_ditp.data_source.thermo_raw_net.ThermoRawLoader:
    """
    Initialize iterator over scans in ThermoFisher RAW file format.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    raw_file : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Iterator over scans from RAW file.

    """
    # Read data from file
    raw_file = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    return raw_file

def get_precursor_peaks(raw_file):

    scan_processor = ms_ditp.ScanProcessor(raw_file)

    precursor_scans = []

    while True:
        try:
            scan_bunch = next(raw_file)

            ms1, precursors, _ = scan_processor.process_scan_group(scan_bunch.precursor, scan_bunch.products)

            scan_time = ms1.scan_time
            for p in precursors:
                precursor_scans.append(PrecursorPeak(
                    scan_time = scan_time,
                    mz = p.peak.mz,
                    intensity = p.peak.intensity,
                    charge = 0 if type(p.charge) is not int else p.charge,
                ))
        except StopIteration:
            break
    
    return precursor_scans

def get_representative_peaks(raw_file):

    precursor_scans = get_precursor_peaks(raw_file)

    df_precursors = pl.DataFrame(
        data=np.array(
            [[peak.__dict__[key] for key in COL_TYPES_PRECURSOR.keys()] for peak in precursor_scans]
        ),
        schema=COL_TYPES_PRECURSOR,
    )

    df_precursors = df_precursors.filter(pl.col("charge")>MIN_MS1_CHARGE_STATE).sort("mz").with_columns(
            (
                abs(pl.col("mz").shift(1) - pl.col("mz"))
                / pl.col("mz").shift(1)
            )
            .fill_null(0)
            .fill_nan(0)
            .gt(PREPROCESS_TOL)
            .cum_sum()
            .alias("cluster_group"))

    df_precursors_agg = df_precursors.group_by("cluster_group").agg(average_mz = pl.col("mz").mean(),
                                                                    intensity = pl.col("intensity").sum(),
                                                                    max_charge = pl.col("charge").max()).sort("cluster_group")
    return df_precursors_agg

def filter_precursor_charges(ms2prec, ms2s):
    ms2prec_final = []
    ms2prec_charge = []
    ms2_final = []
    for p in range(len(ms2prec)):
        pcharge = 0 if not isinstance(ms2prec[p].charge, int) else ms2prec[p].charge 
        if pcharge > MIN_MS1_CHARGE_STATE:
            ms2prec_final.append(ms2prec[p])
            ms2prec_charge.append(pcharge)
            ms2_final.append(ms2s[p])
    return ms2prec_final, ms2prec_charge, ms2_final
    
def process_and_deisotope_scan_bunch(ms1, ms2prec_final, ms2prec_charge, ms2_final, decon_params):

    max_prec_charge_state = max(ms2prec_charge)
    ms1_charge_range = (ms1.polarity, ms1.polarity*max_prec_charge_state)
    ms1_min_intensity = select_min_intensity(
                    scan=ms1, min_intensity=decon_params.minimum_intensity
                )
    
    #prec_peaks = [ms1.has_peak(prec_mz, 10) for prec_mz in df_precursor_agg_arr]
    decon_prec = ms_ditp.deconvolute_peaks(
                    peaklist=ms1, priority_list = ms2prec_final,
                    charge_range = ms1_charge_range,
                    minimum_intensity= ms1_min_intensity,
                    **decon_params.scan_independent_params,
                ).priorities#.neutral_mass
    deisotopedscanbunches = []
    for i in range(len(ms2prec_final)):
        decon_frags = deconvolute_scan(ms2_final[i], params = decon_params)
        deisotopedscanbunches.append(DeisotopedScanBunch(
        scan_time = ms1.scan_time,
        precursor_raw_mz = ms2prec_final[i].mz,
        precursor_raw_intensity = ms2prec_final[i].peak.intensity,
        precursor_neutral_mass = 0 if decon_prec[i] is None else decon_prec[i].neutral_mass,
        fragments = decon_frags,
        singleton_raw_peaks = process_scan(ms2_final[i]),
        num_fragments = len(decon_frags),))
    return deisotopedscanbunches

def select_singletons_from_peaks_raw(peak_list: List[RawPeak]) -> pl.DataFrame:
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

    # Match observed m/z to theoretical m/z from the reference table
    peak_df = peak_df.sort("mz").join_asof(
        EXPLANATION_MASSES.sort("theoretical_mz"),
        left_on="mz",
        right_on="theoretical_mz",
        strategy="nearest",
    )

    # Compute mass error between observed and theoretical m/z
    peak_df = (
        peak_df.sort("mz")
        .with_columns(
            (abs(pl.col("mz") - pl.col("theoretical_mz")) / pl.col("mz"))
            .fill_null(0)
            .fill_nan(0)
            .lt(PREPROCESS_TOL)
            .alias("is_match")
        )
        .filter(pl.col("is_match"))
        .sort(["nucleoside", "scan_time"])
    )

    # Map representative nucleoside, cluster score, and count to each nucleoside group
    peak_df = peak_df.group_by("nucleoside_list").map_groups(
        lambda x: pl.DataFrame(
            {
                "nucleoside": x["nucleoside_list"][0],
                "cluster_score": calculate_cluster_score(x["scan_time"]),
                "count": len(x["nucleoside_list"]),
            }
        )
    )

    # Filter candidate singletons by cluster score
    return (
        peak_df#.filter(pl.col("cluster_score") >= 0).select(
        #     ["nucleoside", "cluster_score"]
        # )
    ).sort("count", descending=True)

def deconvolute_scanbunch(file_path: str, params: dict, min_scan_time = 0, max_scan_time = np.inf):
    """
    Deconvolute/deisotope MS1 and MS2 scan bunches ThermoFisher RAW file.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    params : dict
        Dictionary containing deconvolution parameters.

    Returns
    -------
    List
        List containing deisotoped and deconvoluted scan bunches tagged to its precursor mass.

    """
    # Load deconvolution parameter based on parameter dict
    decon_params = DeconvolutionParameters(params)

    # Initialize iterator for RAW file
    raw_file_read = initialize_raw_file_iterator_ungrouped(file_path=file_path)
    raw_file_read = raw_file_read.start_from_scan(rt = min_scan_time)
    scan_processor = ms_ditp.ScanProcessor(raw_file_read)

    deisotopedscanbunches = []

    prev_scan_time = min_scan_time
    max_scan_time = min(max_scan_time, raw_file_read.get_scan_by_index(len(raw_file_read)-1).scan_time)

    with tqdm.tqdm(total = round(max_scan_time-min_scan_time,3)) as pbar:
        while True:
            try:
                scan_bunch = next(raw_file_read)

                if len(scan_bunch.products) < 1:
                    continue

                ms1, ms2prec, ms2s = scan_processor.process_scan_group(scan_bunch.precursor, scan_bunch.products)
                
                if ms1.scan_time > max_scan_time:
                    break
                dt = ms1.scan_time-prev_scan_time
                pbar.update(round(dt,3))
                prev_scan_time = ms1.scan_time
                
                ms2prec_final, ms2prec_charge, ms2_final = filter_precursor_charges(ms2prec, ms2s)

                if len(ms2prec_charge) <1 or  max(ms2prec_charge) < 0: 
                    continue
                deisotopedscanbunches += process_and_deisotope_scan_bunch(ms1, ms2prec_final, ms2prec_charge, ms2_final, decon_params)
                
                
            except StopIteration:
                break
    

    df_deisotoped_scan_bunch = pl.DataFrame(
            data=np.array(
                [[sbunch.__dict__[key] for key in COL_TYPES_DEISOTOPED_SCAN_BUNCH.keys()] for sbunch in deisotopedscanbunches]
            ),
            schema=COL_TYPES_DEISOTOPED_SCAN_BUNCH,
        ).with_row_index().sort("precursor_neutral_mass").with_columns(
                (
                    abs(pl.col("precursor_neutral_mass").shift(1) - pl.col("precursor_neutral_mass"))
                    / pl.col("precursor_neutral_mass").shift(1)
                )
                .fill_null(0)
                .fill_nan(0)
                .gt(PREPROCESS_TOL)
                .cum_sum()
                .alias("neutral_mass_grp")).rename({"precursor_neutral_mass": "neutral_mass", "precursor_raw_intensity": "intensity"})

    return df_deisotoped_scan_bunch, deisotopedscanbunches

def get_frags_sings(df_grp_filtered, deisotopedscanbunches):
    #df_filtered = df_deisotoped_scan_bunch.filter(pl.col("neutral_mass_grp") == grp_index).sort("scan_time")
    indices = df_grp_filtered["index"].to_numpy()
    fragment_peaks = []
    singleton_raw_peaks_filtered = []

    for idx in indices:
        fragment_peaks += deisotopedscanbunches[idx].fragments
        singleton_raw_peaks_filtered += deisotopedscanbunches[idx].singleton_raw_peaks
    try: 
        fragments = aggregate_peaks_into_fragments(fragment_peaks)
    except ValueError:
        fragments = pl.DataFrame(schema={"is_precursor_deisotoped":pl.Boolean, "intensity":pl.Float64, "neutral_mass":pl.Float64, "ppm_group":pl.Int32})
    
    try:
        singletons = select_singletons_from_peaks_raw(singleton_raw_peaks_filtered)
    except pl.exceptions.ComputeError:
        singletons = pl.DataFrame(schema={"nucleoside":pl.Utf8, "count":pl.Int32, "cluster_score":pl.Float64})
    return fragments, singletons
