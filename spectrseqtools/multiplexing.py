import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from typing import List

from spectrseqtools.masses import NUCLEOTIDE_DF
from spectrseqtools.deconvolution import select_min_intensity, PREPROCESS_TOL
from spectrseqtools.singleton_identification import RawPeak, COL_TYPES_RAW, calculate_cluster_score
from collections import defaultdict

rt = get_mono()

MIN_MS1_CHARGE_STATE = 1

@dataclass
class DeisotopedPrecursorPeak:
    min_average_scan_time: float
    max_average_scan_time: float
    precursor_neutral_mass: float
    precursor_intensity: float
    product_scan: ms_ditp.data_source.scan.scan.Scan

COL_TYPES_AVERAGED_PRECURSORS = {
    "precursor_neutral_mass": pl.Float64,
    "precursor_intensity": pl.Float64,
}

COL_TYPES_GROUPED_PRECURSORS = {
    "min_average_scan_time": pl.Float64,
    "max_average_scan_time": pl.Float64,
    "precursor_neutral_mass": pl.Float64,
    "precursor_intensity": pl.Float64,
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
        NUCLEOTIDE_DF.sort("theoretical_mz"),
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
    return peak_df.sort("count", descending=True)

def ms1_to_ms2_dict(raw_file_read):
    raw_file_read.reset()
    raw_file_read.make_iterator(grouped = False)
    
    ms2_to_ms1_idx = {}

    for _ in tqdm.tqdm(range(len(raw_file_read) - 1)):
        # Select next scan
        scan = next(raw_file_read)

        # Skip scan if it is no MS2 scan
        if scan.ms_level != 2:
            continue
        
        prod_idx = scan.index
        prec_idx = scan.precursor_information.precursor.index
        
        ms2_to_ms1_idx[prod_idx] = prec_idx

    ms1_to_ms2_idx = defaultdict(list)
    for prod_idx, prec_idx in ms2_to_ms1_idx.items():
        ms1_to_ms2_idx[prec_idx].append(prod_idx)
    
    return ms1_to_ms2_idx

def deconvolute_averaged_precursors(new_precursor, final_priority_peaks, priority_charges, final_products, decon_params):
    
    max_prec_charge_state = max(priority_charges)
    ms1_charge_range = (new_precursor.polarity, new_precursor.polarity*max_prec_charge_state)
    ms1_min_intensity = select_min_intensity(
                    scan=new_precursor, min_intensity=decon_params.minimum_intensity
                )

    decon_priority = ms_ditp.deconvolute_peaks(
                        peaklist=new_precursor, priority_list = final_priority_peaks,
                        charge_range = ms1_charge_range,
                        minimum_intensity= ms1_min_intensity,
                        **decon_params.scan_independent_params,
                    ).priorities

    ungrouped_precursors = []
    for i in range(len(decon_priority)):
        if decon_priority[i] is None:
            continue

        precursor_neutral_mass = decon_priority[i].neutral_mass
        precursor_intensity = decon_priority[i].intensity
        ungrouped_precursors.append(
            DeisotopedPrecursorPeak(
                min_average_scan_time=np.nan,
                max_average_scan_time=np.nan,
                precursor_neutral_mass=precursor_neutral_mass, 
                precursor_intensity= precursor_intensity,
                product_scan = final_products[i]
                ))
    if len(ungrouped_precursors)>0:
        df_averaged_precursors = pl.DataFrame(
                data=np.array(
                    [[frag.__dict__[key] for key in COL_TYPES_AVERAGED_PRECURSORS.keys()] for frag in ungrouped_precursors]
                ),
                schema=COL_TYPES_AVERAGED_PRECURSORS,
            ).with_row_index().sort("precursor_neutral_mass").with_columns(
                    (
                        abs(pl.col("precursor_neutral_mass").shift(1) - pl.col("precursor_neutral_mass"))
                        / pl.col("precursor_neutral_mass").shift(1)
                    )
                    .fill_null(0)
                    .fill_nan(0)
                    .gt(PREPROCESS_TOL)
                    .cum_sum()
                    .alias("average_grp"))
    else:
        df_averaged_precursors = pl.DataFrame(schema=COL_TYPES_AVERAGED_PRECURSORS,)
    return df_averaged_precursors, ungrouped_precursors

def averaged_precursors_products(raw_file_read, decon_params, min_scan_time, max_scan_time, dt_window):
    ms1_to_ms2_idx = ms1_to_ms2_dict(raw_file_read)

    raw_file_read.reset()
    raw_file_read.make_iterator(grouped = True)
    
    scan_processor = ms_ditp.ScanProcessor(raw_file_read)

    averaged_precursors = []

    for t in tqdm.tqdm(np.arange(min_scan_time, max_scan_time, dt_window/4)):#, max_scan_time, dt)):
        scan = raw_file_read.get_scan_by_time(t+dt_window)
        if scan.ms_level != 1:
            scan = raw_file_read.find_previous_ms1(scan.index)

        average_scan = scan.average(rt_interval = dt_window)
        average_scan.pick_peaks()

        average_scan_indices = average_scan.scan_indices
        min_average_scan_time = raw_file_read.get_scan_by_index(min(average_scan_indices)).scan_time
        max_average_scan_time = raw_file_read.get_scan_by_index(max(average_scan_indices)).scan_time

        product_scans = []
        for scan_idx in average_scan_indices:
            prod_idx = ms1_to_ms2_idx[scan_idx]
            for idx in prod_idx:
                product_scans.append(raw_file_read.get_scan_by_index(idx))

        new_precursor, priority_peaks, new_products = scan_processor.process_scan_group(average_scan, product_scans)
        final_priority_peaks, priority_charges, final_products = filter_precursor_charges(priority_peaks, new_products)
        if len(priority_charges) == 0:
            print("Time " + str(t) + " skipped because all charges were lower than threshold.")
            continue
        
        df_averaged_precursors, ungrouped_precursors = deconvolute_averaged_precursors(new_precursor, final_priority_peaks, priority_charges, final_products, decon_params)
        if len(ungrouped_precursors) == 0:
            continue
        average_grps = df_averaged_precursors["average_grp"].unique()

        for i in average_grps:
            df_filter = df_averaged_precursors.filter(pl.col("average_grp") == i)
            
            precursor_max_idx = df_filter["precursor_neutral_mass"].arg_max()
            
            avg_product_idx = df_filter["index"].to_numpy()
            
            avg_product_scans = []
            for idx in avg_product_idx:
                avg_product_scans.append(ungrouped_precursors[idx].product_scan)

            averaged_precursors.append(
                DeisotopedPrecursorPeak(
                    min_average_scan_time=min_average_scan_time,
                    max_average_scan_time=max_average_scan_time,
                    precursor_neutral_mass=df_filter["precursor_neutral_mass"][precursor_max_idx], 
                    precursor_intensity= df_filter["precursor_intensity"][precursor_max_idx],
                    product_scan = avg_product_scans
                    ))
        
    return averaged_precursors

def group_averaged_precursors_over_time(averaged_precursors):
    df_grouped_precursors = pl.DataFrame(
                data=np.array(
                    [[frag.__dict__[key] for key in COL_TYPES_GROUPED_PRECURSORS.keys()] for frag in averaged_precursors]
                ),
                schema=COL_TYPES_GROUPED_PRECURSORS,
            ).with_row_index().sort("precursor_neutral_mass").with_columns(
                    (
                        abs(pl.col("precursor_neutral_mass").shift(1) - pl.col("precursor_neutral_mass"))
                        / pl.col("precursor_neutral_mass").shift(1)
                    )
                    .fill_null(0)
                    .fill_nan(0)
                    .gt(PREPROCESS_TOL)
                    .cum_sum()
                    .alias("neutral_mass_grp")).sort(["neutral_mass_grp", "min_average_scan_time"])
    return df_grouped_precursors