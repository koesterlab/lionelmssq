import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from typing import List

from spectrseqtools.preprocessing import determine_intensity_percentiles
from spectrseqtools.masses import NUCLEOTIDE_DF, _COLS
from spectrseqtools.common import initialize_raw_file_iterator
from spectrseqtools.deconvolution import select_min_intensity, PREPROCESS_TOL, DeconvolutionParameters, deconvolute_scan, aggregate_peaks_into_fragments
from spectrseqtools.singleton_identification import RawPeak, COL_TYPES_RAW, calculate_cluster_score, identify_singletons, process_scan
from collections import defaultdict

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs
import importlib.resources
import altair as alt
rt = get_mono()

MIN_MS1_CHARGE_STATE = 1

masses = pl.read_csv(
        (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
        separator="\t",
    )
assert masses.columns == _COLS

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

    if peak_df.height == 0:
        return pl.DataFrame(schema = {"id": str, "cluster_score": float, "count": int})

    # Map representative nucleotide, cluster score, and count to each nucleotide group
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
    return peak_df.sort("count", descending=True)

def ms1_to_ms2_dict(raw_file_read):
    raw_file_read.reset()
    raw_file_read.make_iterator(grouped = False)
    
    ms2_to_ms1_idx = {}

    for _ in range(len(raw_file_read) - 1):
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

    for t in tqdm.tqdm(np.arange(min_scan_time, max_scan_time, dt_window/4), desc="Deisotoping average precursors and collecting products"):#, max_scan_time, dt)):
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
            #print("Time " + str(t) + " skipped because all charges were lower than threshold.")
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

@dataclass
class PreprocessedGroup:
    fragments: pl.DataFrame
    singletons: pl.DataFrame
    meta: dict

def average_and_deconvolute_product_scan(df_filter, averaged_precursors, ms2_decon_params):
    grp_product_idx = df_filter["index"].to_numpy()
        
    grp_product_scans = []
    gidx = set()

    for idx in grp_product_idx:
        for scan in averaged_precursors[idx].product_scan:
            if scan.index not in gidx:
                gidx.add(scan.index)
                grp_product_scans.append(scan)

    if len(grp_product_scans)>1:
        average_product_scan = grp_product_scans[0].average_with(grp_product_scans[1:])
    else:
        average_product_scan = grp_product_scans[0]

    decon_product_peaks = deconvolute_scan(average_product_scan, params = ms2_decon_params)

    return average_product_scan, decon_product_peaks

def pre_process_multiplexing(file_path, params, min_scan_time = 0, max_scan_time = np.inf, dt_window = 0.2, three_prime_tag = 728.2006, five_prime_tag = 170.9755):
    sample_name = file_path.stem
    ms1_decon_params = DeconvolutionParameters(params)
    ms2_decon_params = DeconvolutionParameters({"min_score": 0})
    raw_file_read = initialize_raw_file_iterator(str(file_path))

    max_scan_time = min(max_scan_time, raw_file_read.get_scan_by_index(len(raw_file_read)-1).scan_time)

    averaged_precursors = averaged_precursors_products(raw_file_read, ms1_decon_params, min_scan_time, max_scan_time, dt_window)
    df_grouped_precursors = group_averaged_precursors_over_time(averaged_precursors)

    default_singletons = identify_singletons(str(file_path))
    
    grp_indices = df_grouped_precursors["neutral_mass_grp"].unique()

    preprocessed_groups = []
    for g in tqdm.tqdm(grp_indices, desc="Deisotoping average product scans"):
        df_filter = df_grouped_precursors.filter(pl.col("neutral_mass_grp") == g)
        min_window_time = df_filter["min_average_scan_time"].min()
        max_window_time = df_filter["max_average_scan_time"].max()

        precursor_max_idx = df_filter["precursor_neutral_mass"].arg_max()
        intact_mass = df_filter["precursor_neutral_mass"][precursor_max_idx]
        
        average_product_scan, decon_product_peaks = average_and_deconvolute_product_scan(df_filter, averaged_precursors, ms2_decon_params)
        if len(decon_product_peaks) == 0:
            continue
        fragments = aggregate_peaks_into_fragments(decon_product_peaks)
        fragments = fragments.rename({"neutral_mass": "observed_mass"}).filter(pl.col("observed_mass")<intact_mass)

        singletons = select_singletons_from_peaks_raw(process_scan(average_product_scan))
        if singletons.height < 4:
            singletons = default_singletons.clone()
    
        intensity_cutoff = determine_intensity_percentiles(fragments).filter(pl.col("statistic") == "70%")["value"].to_list()[0]

        meta = {"identity": sample_name + "_" + str(g),
                "min_scan_time": min_window_time,
                "max_scan_time": max_window_time,
                "group_number": g,
                "intensity_cutoff": intensity_cutoff,
                "3_prime_tag": three_prime_tag, #728.2006,
                "5_prime_tag": five_prime_tag, #170.9755,
                "intact_mass": intact_mass,}
        
        if intact_mass < meta["3_prime_tag"]+meta["5_prime_tag"]:
            continue

        preprocessed_groups.append(PreprocessedGroup(fragments = fragments,
                                                     singletons = singletons,
                                                     meta = meta))
        
    return preprocessed_groups

#POST PROCESSING
def targ_pred_pairing(x):
    if x[0] in x[1]:
        return (x[0], x[0])
    else:
        return (x[0], x[1][0])

def compare_prediction_to_target(prediction_raw, target_sequence, is_backward = False):
    if is_backward:
        prediction_raw = prediction_raw[::-1]
    prediction_len = len(prediction_raw)
    prediction_string = "".join(masses.filter(pl.col("id") == n).select("encoding").item() for n in prediction_raw)

    target_strings = []

    for i in range(len(target_sequence)-prediction_len+1):
        target_sequence_window = target_sequence[i:i+prediction_len]
        targ_string = ""

        for p in zip(prediction_raw, target_sequence_window):
            pair = targ_pred_pairing(p)
            targ_string += masses.filter(pl.col("id") == pair[1])["encoding"].item()
        target_strings.append(targ_string)

    distances = normalized_damerau_levenshtein_distance_seqs(prediction_string, target_strings)
    min_distance_idx = np.argmin(distances)

    min_distance = distances[min_distance_idx]
    start_pos = min_distance_idx
    end_pos = min_distance_idx + prediction_len

    return (prediction_string, target_strings[min_distance_idx], min_distance, start_pos, end_pos, is_backward)

def align_prediction_results(prediction_vals, target_sequence):

    alignment_vals = []

    for i in prediction_vals:

        prediction_raw = i[2]

        if len(prediction_raw) == 0:
            continue
        
        forward_comparison = compare_prediction_to_target(prediction_raw, target_sequence, is_backward = False)
        backward_comparison = compare_prediction_to_target(prediction_raw, target_sequence, is_backward = True)

        final_comparison = forward_comparison if forward_comparison[2] <= backward_comparison[2] else backward_comparison
    
        alignment_vals.append(final_comparison+(i[0], i[1], i[3],))

    df_alignment = pl.DataFrame(alignment_vals, schema = ["predicted_string", "best_matching_target_string", "normalized_damerau_levenshtein_distance", "target_start_pos", "target_end_pos", "is_backward", "group", "intact_mass", "prediction_runtime"], orient = "row")

    return df_alignment

def interpret_alignment_results(df_alignment, target_sequence):

    match_rows = []

    for r in df_alignment.iter_rows(named = True):
        pred_string = r["predicted_string"]
        # if r["is_backward"]:
        #     pred_string = pred_string[::-1]
        target_string = r["best_matching_target_string"]
        start_pos = r["target_start_pos"]
        intact_mass = r["intact_mass"]

        for i, (p, t) in enumerate(zip(pred_string, target_string)):
            if p == t:
                status = "match"
            else:
                status = "mismatch"
            match_rows.append({"group": r["group"], 
                            "score": r["normalized_damerau_levenshtein_distance"],
                            "target_position": start_pos+i, 
                            "predicted_base": p, 
                            "target_base": t, 
                            "status": status,
                            "intact_mass": intact_mass})

    for b in range(len(target_sequence)):
        targ_base = masses.filter(pl.col("id") == target_sequence[b][0])["encoding"].item()
        match_rows.append({"group": -1, 
                            "score": 0,
                            "target_position": b, 
                            "predicted_base": targ_base, 
                            "target_base": targ_base, 
                            "status": "match"})

    df_expanded_alignment = pl.DataFrame(match_rows)

    df_expanded_alignment = df_expanded_alignment.sort(["group", "target_position"])

    df_expanded_alignment = df_expanded_alignment.with_columns([
        pl.col("status").shift(-1).over("group").alias("next_status"),
        pl.col("predicted_base").shift(-1).over("group").alias("next_pred"),
        pl.col("target_base").shift(-1).over("group").alias("next_target"),
    ])

    df_expanded_alignment = df_expanded_alignment.with_columns(
        (
            (pl.col("status") == "mismatch") &
            (pl.col("next_status") == "mismatch") &
            (pl.col("predicted_base") == pl.col("next_target")) &
            (pl.col("next_pred") == pl.col("target_base"))
        ).alias("is_swap_start")
    )

    df_expanded_alignment = df_expanded_alignment.with_columns(
        (
            pl.col("is_swap_start") |
            pl.col("is_swap_start").shift(1).over("group")
        ).alias("is_swap")
    )

    df_expanded_alignment = df_expanded_alignment.with_columns(
        pl.when(pl.col("is_swap"))
        .then(pl.lit("swap"))
        .otherwise(pl.col("status"))
        .alias("status")
    )

    df_expanded_alignment = df_expanded_alignment.drop([
        "next_status",
        "next_pred",
        "next_target",
        "is_swap_start",
        "is_swap"
    ])

    df_expanded_alignment = df_expanded_alignment.join(
        masses.select(["encoding", "canonical_name"]),
        left_on="predicted_base",
        right_on="encoding",
        how="left"
    )

    df_expanded_alignment = df_expanded_alignment.sort(["score", "group", "target_position"])
    group_to_pos = {group: i for i,group in enumerate(df_expanded_alignment["group"].unique(maintain_order = True).to_list())}
    df_expanded_alignment = df_expanded_alignment.with_columns(pos = pl.col("group").replace(group_to_pos))

    return df_expanded_alignment

def plot_interpreted_alignment(df_expanded_alignment):

    color_scale = alt.Scale(
        domain=["match", "mismatch", "swap"],
        range=["black", "red", "gold"]
    )

    chart = alt.Chart(df_expanded_alignment).mark_text(
        fontSize=14,
        font="monospace"
    ).encode(
        x=alt.X("target_position:Q", title="Base position", scale=alt.Scale(
                domain=[df_expanded_alignment["target_position"].min() - 1, df_expanded_alignment["target_position"].max() + 1],
                padding=0
            ),),
        y=alt.Y("pos:N"),
        text="predicted_base:N",
        color=alt.Color("status:N", scale=color_scale),
        tooltip=["group", "target_position", "score", "predicted_base", "target_base", "status", "canonical_name", "intact_mass"]
    ).properties(
        width=850,
        height=400
    )

    return chart

def post_processing_alignment(prediction_vals, sample_name, target_sequence, save_file = False):

    df_alignment = align_prediction_results(prediction_vals, target_sequence)
    df_expanded_alignment = interpret_alignment_results(df_alignment, target_sequence)

    chart = plot_interpreted_alignment(df_expanded_alignment)
    chart.show()
    if save_file:
        chart.save(sample_name + '_output.html')