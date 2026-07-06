import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass

from spectrseqtools.preprocessing.preprocessing import determine_intensity_percentiles, Preprocessor, set_averagine, initialize_raw_file_iterator
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.masses import ELEMENT_MASSES
from spectrseqtools.preprocessing.deconvolution import DeconvolutionParameters, DeisotopedPeakList
from spectrseqtools.preprocessing.singleton_identification import RawPeakList, SingletonBoundaries
from spectrseqtools.parsers import PreprocessingOptions


from collections import defaultdict

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs

import altair as alt

rt = get_mono()

SODIUM_MASS = 22.989769
POTASSIUM_MASS = 39.098300

@dataclass
class DeisotopedMS1Peak:
    min_scan_time: float
    max_scan_time: float
    ms1_neutral_mass: float
    ms1_intensity: float
    ms2_scan: ms_ditp.data_source.scan.scan.Scan
    max_charge_state: int

COL_TYPES_MS1_INFO = {
    "ms1_neutral_mass": pl.Float64,
    "ms1_intensity": pl.Float64,
}

COL_TYPES_MS1_CLUSTER = {
    "min_scan_time": pl.Float64,
    "max_scan_time": pl.Float64,
    "ms1_neutral_mass": pl.Float64,
    "ms1_intensity": pl.Float64,
}

def charge_filter(priority_ms1_peaks, ms2_scan_list, options):
    new_priority_ms1_peaks = []
    new_priority_ms1_charges = []
    new_ms2_scan_list = []
    for p in range(len(priority_ms1_peaks)):
        pcharge = 0 if not isinstance(priority_ms1_peaks[p].charge, int) else priority_ms1_peaks[p].charge 
        if pcharge >= options.min_precursor_charge:
            new_priority_ms1_peaks.append(priority_ms1_peaks[p])
            new_priority_ms1_charges.append(pcharge)
            new_ms2_scan_list.append(ms2_scan_list[p])
    return new_priority_ms1_peaks, new_priority_ms1_charges, new_ms2_scan_list
    
def ms1_to_ms2_dict(raw_file_read):
    raw_file_read.reset()
    raw_file_read.make_iterator(grouped = False)
    
    ms2_to_ms1_idx = {}

    while True:
        try:
            # Select next scan
            scan = next(raw_file_read)

            # Skip scan if it is no MS2 scan
            if scan.ms_level != 2:
                continue
            
            ms2_idx = scan.index
            ms1_idx = scan.precursor_information.precursor.index
            
            ms2_to_ms1_idx[ms2_idx] = ms1_idx
        except StopIteration:
            break

    ms1_to_ms2_idx = defaultdict(list)
    for ms2_idx, ms1_idx in ms2_to_ms1_idx.items():
        ms1_to_ms2_idx[ms1_idx].append(ms2_idx)
    
    return ms1_to_ms2_idx

def deconvolute_average_ms1_scan(average_ms1_scan, priority_ms1_peaks, priority_ms1_charges, ms2_scan_list, ms1_decon_params):
    max_abs_charge = max(abs(c) for c in priority_ms1_charges if c)
    if average_ms1_scan.polarity < 0:
        ms1_charge_range = (-max_abs_charge, -1)
    else:
        ms1_charge_range = (1, max_abs_charge)

    ms1_min_intensity = ms1_decon_params.select_min_intensity(
                    scan=average_ms1_scan
                )

    scan_independent_params = ms1_decon_params.__dict__.copy()
    scan_independent_params.pop("min_precursor_charge", None)
    scan_independent_params.pop("isotopic_shift_factor", None)
    scan_independent_params.pop("charge_range", None)
    scan_independent_params.pop("minimum_intensity", None)

    ms1_decon_priority_peaks = ms_ditp.deconvolute_peaks( 
                                    peaklist=average_ms1_scan, priority_list = priority_ms1_peaks,
                                    charge_range = ms1_charge_range,
                                    minimum_intensity= ms1_min_intensity,
                                    **scan_independent_params,
                                ).priorities

    average_ms1_mass_list = []
    for i in range(len(ms1_decon_priority_peaks)):
        if ms1_decon_priority_peaks[i] is None:
            continue

        ms1_neutral_mass = ms1_decon_priority_peaks[i].neutral_mass
        ms1_intensity = ms1_decon_priority_peaks[i].intensity
        average_ms1_mass_list.append(
            DeisotopedMS1Peak(
                min_scan_time=np.nan,
                max_scan_time=np.nan,
                ms1_neutral_mass=ms1_neutral_mass, 
                ms1_intensity= ms1_intensity,
                ms2_scan = ms2_scan_list[i],
                max_charge_state = 0
                ))
        
    return average_ms1_mass_list

def group_ms1_mass_collect_ms2_scan(average_ms1_mass_list, priority_charges, min_window_time, max_window_time, options):
    df_average_ms1_mass_info = pl.DataFrame(
            data=np.array(
                [[frag.__dict__[key] for key in COL_TYPES_MS1_INFO.keys()] for frag in average_ms1_mass_list]
            ),
            schema=COL_TYPES_MS1_INFO,
        ).with_row_index("ms1_window_scan_index").sort("ms1_neutral_mass").with_columns(
                (
                    abs(pl.col("ms1_neutral_mass").shift(1) - pl.col("ms1_neutral_mass"))
                    / pl.col("ms1_neutral_mass").shift(1)
                )
                .fill_null(0)
                .fill_nan(0)
                .gt(options.tolerance)
                .cum_sum()
                .alias("ms1_window_grp"))
    
    ms1_grp_list = df_average_ms1_mass_info["ms1_window_grp"].unique()
    grouped_ms1_mass_list = []

    for ms1_grp in ms1_grp_list:
        df_filter = df_average_ms1_mass_info.filter(pl.col("ms1_window_grp") == ms1_grp)
        
        ms1_min_idx = df_filter["ms1_neutral_mass"].arg_min()
        
        ms1_window_scan_index_list = df_filter["ms1_window_scan_index"].to_numpy()        
        ms2_window_scan_list = [average_ms1_mass_list[idx].ms2_scan for idx in ms1_window_scan_index_list]

        grouped_ms1_mass_list.append(
            DeisotopedMS1Peak(
                min_scan_time=min_window_time,
                max_scan_time=max_window_time,
                ms1_neutral_mass=df_filter["ms1_neutral_mass"][ms1_min_idx], 
                ms1_intensity= df_filter["ms1_intensity"][ms1_min_idx],
                ms2_scan = ms2_window_scan_list,
                max_charge_state = max(priority_charges)
                ))

    return grouped_ms1_mass_list

def average_deisotope_ms1_collect_ms2(raw_file_read, ms1_decon_params, min_scan_time, max_scan_time, dt, options):
    ms1_to_ms2_idx = ms1_to_ms2_dict(raw_file_read)

    raw_file_read.reset()
    raw_file_read.make_iterator(grouped = True)

    scan_processor = ms_ditp.ScanProcessor(raw_file_read)

    ms1_mass_ms2_scans_list = []

    for t in tqdm.tqdm(np.arange(min_scan_time, max_scan_time, dt/2), desc="Deisotoping average MS1 and collecting MS2"):
        scan = raw_file_read.get_scan_by_time(t)
        if scan.ms_level != 1:
            scan = raw_file_read.find_previous_ms1(scan.index)

        average_ms1_scan = scan.average(rt_interval = dt)
        average_ms1_scan.pick_peaks()

        ms1_index_list = average_ms1_scan.scan_indices
        min_window_time = raw_file_read.get_scan_by_index(min(ms1_index_list)).scan_time
        max_window_time = raw_file_read.get_scan_by_index(max(ms1_index_list)).scan_time

        ms2_scan_list = []
        for ms1_idx in ms1_index_list:
            ms2_index_list = ms1_to_ms2_idx[ms1_idx]
            for ms2_idx in ms2_index_list:
                ms2_scan_list.append(raw_file_read.get_scan_by_index(ms2_idx))

        average_ms1_scan, priority_ms1_peaks, ms2_scan_list = scan_processor.process_scan_group(average_ms1_scan, ms2_scan_list)
        priority_ms1_peaks, priority_ms1_charges, ms2_scan_list = charge_filter(priority_ms1_peaks, ms2_scan_list, options)
        if len(priority_ms1_charges) == 0:
            continue

        average_ms1_mass_list = deconvolute_average_ms1_scan(average_ms1_scan=average_ms1_scan,
                                                            priority_ms1_peaks=priority_ms1_peaks,
                                                            priority_ms1_charges=priority_ms1_charges,
                                                            ms2_scan_list=ms2_scan_list,
                                                            ms1_decon_params=ms1_decon_params,)

        if len(average_ms1_mass_list) == 0:
            continue
        grouped_ms1_mass_list = group_ms1_mass_collect_ms2_scan(average_ms1_mass_list, priority_ms1_charges, min_window_time, max_window_time, options)

        for ms1_mass_ms2_scans in grouped_ms1_mass_list:
            ms1_mass_ms2_scans_list.append(ms1_mass_ms2_scans)
    return ms1_mass_ms2_scans_list

def cluster_ms1_masses(ms1_mass_ms2_scans_list, options, correct_adducts = True):
    df_ms1_clusters = pl.DataFrame(
                    data=np.array(
                        [[frag.__dict__[key] for key in COL_TYPES_MS1_CLUSTER.keys()] for frag in ms1_mass_ms2_scans_list]
                    ),
                    schema=COL_TYPES_MS1_CLUSTER,
                ).with_row_index("ms1_cluster_index"
                ).sort(["min_scan_time", "max_scan_time", "ms1_neutral_mass"]
                ).with_columns(
                        ((
                            abs(pl.col("min_scan_time").shift(1) - pl.col("min_scan_time"))>1e-3
                        ) & 
                        (
                            pl.col("min_scan_time")>pl.col("max_scan_time").shift(1)
                        ))  
                        .cast(pl.Int8).fill_null(0).fill_nan(0).cum_sum().alias("time_grp"))

    time_grp_list = df_ms1_clusters["time_grp"].unique().to_numpy()

    last_grp_idx = 0

    list_df_timegrp = []

    if correct_adducts:
        salt_corrections = {
            "none": 0.0,
            "Na-H": SODIUM_MASS - ELEMENT_MASSES["H+"],
            "K-H": POTASSIUM_MASS - ELEMENT_MASSES["H+"],
            }
    else:
        salt_corrections = {
            "": 0.0,
            }
        
    for tgrp in time_grp_list:
        df_timegrp_filter = df_ms1_clusters.filter(pl.col("time_grp") == tgrp)

        expanded_rows = []

        for row in df_timegrp_filter.iter_rows(named=True):
            for adduct_type, correction in salt_corrections.items():
                new_row = dict(row)
                new_row["ms1_neutral_mass"] = row["ms1_neutral_mass"]
                new_row["corrected_neutral_mass"] = row["ms1_neutral_mass"] + correction
                new_row["adduct_type"] = adduct_type
                expanded_rows.append(new_row)

        df_timegrp_filter = pl.DataFrame(expanded_rows)
        
        df_timegrp_filter = df_timegrp_filter.sort(["corrected_neutral_mass"]).with_columns(
                        (
                            abs(pl.col("corrected_neutral_mass").shift(1) - pl.col("corrected_neutral_mass"))
                            / pl.col("corrected_neutral_mass").shift(1)
                        )
                        .fill_null(0)
                        .fill_nan(0)
                        .gt(options.tolerance)
                        .cum_sum()
                        .alias("add_mass_subgrp"))

        df_timegrp_filter = df_timegrp_filter.sort(["add_mass_subgrp", "adduct_type", "ms1_cluster_index"])
        
        df_timegrp_deduplicate = (
            df_timegrp_filter.group_by("add_mass_subgrp")
                            .agg(pl.col("ms1_cluster_index")
                                 .unique()
                                 .sort())
                            .unique("ms1_cluster_index", 
                                    keep='first', 
                                    maintain_order=True)
                                    )
        df_timegrp_filter = df_timegrp_filter.filter(
                                    pl.col("add_mass_subgrp")
                                    .is_in(df_timegrp_deduplicate["add_mass_subgrp"].to_list())
                                    )
        df_timegrp_filter = df_timegrp_filter.with_columns(
                                    (
                                        pl.col("add_mass_subgrp")
                                        .rank("dense")
                                        .cast(pl.Int64) - 1 + last_grp_idx
                                    ).alias("add_mass_subgrp")
                                )
        
        list_df_timegrp.append(df_timegrp_filter)
        last_grp_idx = (
                        df_timegrp_filter.select(
                            pl.col("add_mass_subgrp").max()
                        ).item() + 1
                    )
        
    df_ms1_clusters = pl.concat(list_df_timegrp).rename({"add_mass_subgrp": "neutral_mass_grp"}).drop(["corrected_neutral_mass"])

    return df_ms1_clusters

def average_and_deconvolute_ms2_scan(df_filter, ms1_mass_ms2_scans_list, ms2_decon_params):
    grp_ms2_idx = df_filter["ms1_cluster_index"].to_numpy()
    
    grp_ms2_scans = []
    gidx = set()
    grp_charge_states = []

    for idx in grp_ms2_idx:
        for scan in ms1_mass_ms2_scans_list[idx].ms2_scan:
            if scan.index not in gidx:
                gidx.add(scan.index)
                grp_ms2_scans.append(scan)
                grp_charge_states.append(ms1_mass_ms2_scans_list[idx].max_charge_state)

    if len(grp_ms2_scans)>1:
        average_ms2_scan = grp_ms2_scans[0].average_with(grp_ms2_scans[1:])
    else:
        average_ms2_scan = grp_ms2_scans[0]
    average_ms2_scan.pick_peaks()

    max_abs_charge = max(abs(c) for c in grp_charge_states if c)
    if average_ms2_scan.polarity < 0:
        ms2_charge_range = (-max_abs_charge, -1)
    else:
        ms2_charge_range = (1, max_abs_charge)

    ms2_decon_params.charge_range = ms2_charge_range

    decon_ms2_peaks = DeisotopedPeakList.from_scan(average_ms2_scan, ms2_decon_params)
    
    return average_ms2_scan, decon_ms2_peaks


@dataclass
class PreprocessedGroup:
    fragments: pl.DataFrame
    singletons: pl.DataFrame
    meta: dict

def pre_process_multiplexing(file_path, 
                             ms1_decon_params = None, 
                             ms2_decon_params = None,
                             options = None,
                             min_scan_time = 0, 
                             max_scan_time = np.inf, 
                             dt = 0.2, 
                             three_prime_tag = 728.2006, 
                             five_prime_tag = 170.9755):
    if options is None:    
        options = PreprocessingOptions(
            input=file_path,
            meta=None,
            alphabet=None,
            output_dir=None,
            charge_range=None,
            min_intensity=None,
        )

    alphabet = NucleotideAlphabet.from_file(input_path=None)
    preprocessor = Preprocessor(options)

    sample_name = options.input.stem

    if ms1_decon_params is None:
        ms1_decon_params = DeconvolutionParameters(
                scorer=ms_ditp.PenalizedMSDeconVFitter(minimum_score = options.envelope_min_score,
                                                    mass_error_tolerance=options.envelope_error_tol),
                max_missed_peaks=options.max_missed_peaks,
                scale_method=options.scale_method,
                error_tol=options.peak_error_tol,
                truncate_after=0.95,
                min_precursor_charge=options.min_precursor_charge,
                isotopic_shift_factor=options.isotopic_shift_factor,
                charge_range=options.charge_range,
                minimum_intensity=options.min_intensity,
                averagine = set_averagine(backbone=options.averagine_backbone)
            )

    if ms2_decon_params is None:
        ms2_decon_params = DeconvolutionParameters(
                scorer=ms_ditp.MSDeconVFitter(minimum_score = 0,
                                            mass_error_tolerance=options.envelope_error_tol),
                max_missed_peaks=options.max_missed_peaks,
                scale_method=options.scale_method,
                error_tol=options.peak_error_tol,
                truncate_after=options.truncate_after,
                min_precursor_charge=options.min_precursor_charge,
                isotopic_shift_factor=options.isotopic_shift_factor,
                charge_range=options.charge_range,
                minimum_intensity=options.min_intensity,
                averagine = set_averagine(backbone=options.averagine_backbone)
            )
    
    raw_file_read = initialize_raw_file_iterator(str(file_path))

    max_scan_time = min(max_scan_time, raw_file_read.get_scan_by_index(len(raw_file_read)-1).scan_time)

    ms1_mass_ms2_scans_list = average_deisotope_ms1_collect_ms2(raw_file_read, 
                                                                ms1_decon_params, 
                                                                min_scan_time,
                                                                max_scan_time, 
                                                                dt,
                                                                options)
    
    df_ms1_clusters = cluster_ms1_masses(ms1_mass_ms2_scans_list, options)
    
    default_singletons = preprocessor.identify_singletons()
    
    grp_indices = df_ms1_clusters["neutral_mass_grp"].unique()

    preprocessed_groups = []
    for g in tqdm.tqdm(grp_indices, desc="Deisotoping average MS2 scans"):
        df_filter = df_ms1_clusters.filter(pl.col("neutral_mass_grp") == g)
        
        min_cluster_time = df_filter["min_scan_time"].min()
        max_cluster_time = df_filter["max_scan_time"].max()
        adduct_types = df_filter["adduct_type"].unique().to_list()

        precursor_min_idx = df_filter["ms1_neutral_mass"].arg_min()
        intact_mass = df_filter["ms1_neutral_mass"][precursor_min_idx]

        if intact_mass < three_prime_tag+five_prime_tag:
            continue

        average_ms2_scan, decon_ms2_peaks = average_and_deconvolute_ms2_scan(df_filter, 
                                                                                ms1_mass_ms2_scans_list, 
                                                                                ms2_decon_params)

        fragments = decon_ms2_peaks.to_fragments(options.tolerance)
        if len(fragments) == 0:
            continue
        fragments = fragments.rename({"neutral_mass": "observed_mass"}).filter(pl.col("observed_mass")<intact_mass)

        singleton_boundaries = SingletonBoundaries.from_alphabet(alphabet = alphabet,
                                                                tolerance = options.tolerance,
                                                                boundary_factor = options.boundary_factor)
        singletons = RawPeakList.from_scan(average_ms2_scan, singleton_boundaries).to_singletons(alphabet = alphabet, tolerance = options.tolerance, filter_cluster_score = False)
        
        if singletons.height < 1:
            singletons = default_singletons.clone()

        intensity_cutoff = determine_intensity_percentiles(fragments).filter(pl.col("statistic") == "70%")["value"].to_list()[0]

        meta = {"identity": sample_name,
                "min_scan_time": min_cluster_time,
                "max_scan_time": max_cluster_time,
                "group_number": g,
                "intensity_cutoff": intensity_cutoff,
                "3_prime_tag": three_prime_tag, #728.2006,
                "5_prime_tag": five_prime_tag, #170.9755,
                "intact_mass": intact_mass,
                "adduct_types": adduct_types}

        preprocessed_groups.append(PreprocessedGroup(fragments = fragments,
                                                        singletons = singletons,
                                                        meta = meta))
        
    return preprocessed_groups

# #POST PROCESSING
# def targ_pred_pairing(x):
#     if x[0] in x[1]:
#         return (x[0], x[0])
#     else:
#         return (x[0], x[1][0])

# def compare_prediction_to_target(prediction_raw, target_sequence, is_backward = False):
#     if is_backward:
#         prediction_raw = prediction_raw[::-1]
#     prediction_len = len(prediction_raw)
#     prediction_string = "".join(masses.filter(pl.col("id") == n).select("encoding").item() for n in prediction_raw)

#     target_strings = []

#     for i in range(len(target_sequence)-prediction_len+1):
#         target_sequence_window = target_sequence[i:i+prediction_len]
#         targ_string = ""

#         for p in zip(prediction_raw, target_sequence_window):
#             pair = targ_pred_pairing(p)
#             targ_string += masses.filter(pl.col("id") == pair[1])["encoding"].item()
#         target_strings.append(targ_string)

#     distances = normalized_damerau_levenshtein_distance_seqs(prediction_string, target_strings)
#     min_distance_idx = np.argmin(distances)

#     min_distance = distances[min_distance_idx]
#     start_pos = min_distance_idx
#     end_pos = min_distance_idx + prediction_len

#     return (prediction_string, target_strings[min_distance_idx], min_distance, start_pos, end_pos, is_backward)

# def align_prediction_results(prediction_vals, target_sequence, alignment_score_threshold = 0):

#     alignment_vals = []

#     for i in tqdm.tqdm(prediction_vals, desc = "Calculating alignment scores"):

#         prediction_raw = i[2]

#         if len(prediction_raw) == 0:
#             continue
        
#         forward_comparison = compare_prediction_to_target(prediction_raw, target_sequence, is_backward = False)
#         backward_comparison = compare_prediction_to_target(prediction_raw, target_sequence, is_backward = True)

#         final_comparison = forward_comparison if forward_comparison[2] <= backward_comparison[2] else backward_comparison

#         if final_comparison[2] > alignment_score_threshold:
#             continue
    
#         alignment_vals.append(final_comparison+(i[0], i[1], i[3], i[4]))

#     df_alignment = pl.DataFrame(alignment_vals, schema = ["predicted_string", "best_matching_target_string", "normalized_damerau_levenshtein_distance", "target_start_pos", "target_end_pos", "is_backward", "group", "intact_mass", "min_scan_time", "max_scan_time"], orient = "row")

#     return df_alignment

# def interpret_alignment_results(df_alignment, target_sequence):

#     match_rows = []

#     for r in df_alignment.iter_rows(named = True):
#         pred_string = r["predicted_string"]
#         # if r["is_backward"]:
#         #     pred_string = pred_string[::-1]
#         target_string = r["best_matching_target_string"]
#         start_pos = r["target_start_pos"]
#         intact_mass = r["intact_mass"]
#         min_scan_time = r["min_scan_time"]
#         max_scan_time = r["max_scan_time"]

#         for i, (p, t) in enumerate(zip(pred_string, target_string)):
#             if p == t:
#                 status = "match"
#             else:
#                 status = "mismatch"
#             match_rows.append({"group": r["group"], 
#                             "score": r["normalized_damerau_levenshtein_distance"],
#                             "target_position": start_pos+i, 
#                             "predicted_base": p, 
#                             "target_base": t, 
#                             "status": status,
#                             "intact_mass": intact_mass,
#                             "min_scan_time": min_scan_time,
#                             "max_scan_time": max_scan_time})

#     for b in range(len(target_sequence)):
#         targ_base = masses.filter(pl.col("id") == target_sequence[b][0])["encoding"].item()
#         match_rows.append({"group": -1, 
#                             "score": 0,
#                             "target_position": b, 
#                             "predicted_base": targ_base, 
#                             "target_base": targ_base, 
#                             "status": "match",
#                             "intact_mass": np.nan,
#                             "min_scan_time": np.nan,
#                             "max_scan_time": np.nan})

#     df_expanded_alignment = pl.DataFrame(match_rows)

#     df_expanded_alignment = df_expanded_alignment.sort(["group", "target_position"])

#     df_expanded_alignment = df_expanded_alignment.with_columns([
#         pl.col("status").shift(-1).over("group").alias("next_status"),
#         pl.col("predicted_base").shift(-1).over("group").alias("next_pred"),
#         pl.col("target_base").shift(-1).over("group").alias("next_target"),
#     ])

#     df_expanded_alignment = df_expanded_alignment.with_columns(
#         (
#             (pl.col("status") == "mismatch") &
#             (pl.col("next_status") == "mismatch") &
#             (pl.col("predicted_base") == pl.col("next_target")) &
#             (pl.col("next_pred") == pl.col("target_base"))
#         ).alias("is_swap_start")
#     )

#     df_expanded_alignment = df_expanded_alignment.with_columns(
#         (
#             pl.col("is_swap_start") |
#             pl.col("is_swap_start").shift(1).over("group")
#         ).alias("is_swap")
#     )

#     df_expanded_alignment = df_expanded_alignment.with_columns(
#         pl.when(pl.col("is_swap"))
#         .then(pl.lit("swap"))
#         .otherwise(pl.col("status"))
#         .alias("status")
#     )

#     df_expanded_alignment = df_expanded_alignment.drop([
#         "next_status",
#         "next_pred",
#         "next_target",
#         "is_swap_start",
#         "is_swap"
#     ])

#     df_expanded_alignment = df_expanded_alignment.join(
#         masses.select(["encoding", "canonical_name"]),
#         left_on="predicted_base",
#         right_on="encoding",
#         how="left"
#     )

#     df_expanded_alignment = df_expanded_alignment.sort(["score", "group", "target_position"])
#     group_to_pos = {group: i for i,group in enumerate(df_expanded_alignment["group"].unique(maintain_order = True).to_list())}
#     df_expanded_alignment = df_expanded_alignment.with_columns(pos = pl.col("group").replace(group_to_pos))

#     return df_expanded_alignment

# def plot_interpreted_alignment(df_expanded_alignment):

#     color_scale = alt.Scale(
#         domain=["match", "mismatch", "swap"],
#         range=["black", "red", "gold"]
#     )

#     chart = alt.Chart(df_expanded_alignment).mark_text(
#         fontSize=14,
#         font="monospace"
#     ).encode(
#         x=alt.X("target_position:Q", title="Base position", scale=alt.Scale(
#                 domain=[df_expanded_alignment["target_position"].min() - 1, df_expanded_alignment["target_position"].max() + 1],
#                 padding=0
#             ),),
#         y=alt.Y("pos:N"),
#         text="predicted_base:N",
#         color=alt.Color("status:N", scale=color_scale),
#         tooltip=["group", "target_position", "score", "predicted_base", "target_base", "status", "canonical_name", "intact_mass", "min_scan_time", "max_scan_time"]
#     ).properties(
#         width=850,
#         height=400
#     )

#     return chart

# def post_processing_alignment(prediction_vals, sample_name, target_sequence, alignment_score_threshold = 0, save_file = False):

#     df_alignment = align_prediction_results(prediction_vals, target_sequence, alignment_score_threshold)
#     df_expanded_alignment = interpret_alignment_results(df_alignment, target_sequence)
#     chart = plot_interpreted_alignment(df_expanded_alignment)
#     chart.show()
#     if save_file:
#         chart.save(sample_name + '_output.html')

#     return df_expanded_alignment