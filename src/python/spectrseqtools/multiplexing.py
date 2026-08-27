import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
import yaml
from clr_loader import get_mono
from scipy.signal import find_peaks

from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.file_settings import PreprocessingFileSettings
from spectrseqtools.parsers import MixturePreprocessingOptions
from spectrseqtools.preprocessing.deconvolution import MS1Deconvoluter, MS2Deconvoluter
from spectrseqtools.preprocessing.preprocessing import (
    Preprocessor,
    initialize_raw_file_iterator,
    set_averagine,
)
from spectrseqtools.preprocessing.singleton_identification import (
    RawPeakList,
    SingletonBoundaries,
)

rt = get_mono()

# SODIUM_MASS = 22.989769
# POTASSIUM_MASS = 39.098300


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


def priority_list_charge_filter(priority_list, ms2_scans, min_precursor_charge):
    new_priority_list = []
    new_ms2_scans = []

    for p in range(len(priority_list)):
        priority_peak = priority_list[p]

        if (
            isinstance(priority_peak.charge, int)
            and priority_peak.charge >= min_precursor_charge
        ):
            new_priority_list.append(priority_list[p])
            new_ms2_scans.append(ms2_scans[p])

    return new_priority_list, new_ms2_scans


def ms1_to_ms2_dict(raw_file_read):
    raw_file_read.reset()
    raw_file_read.make_iterator(grouped=False)

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


def initialize_parsers(options):

    error = ErrorCalculator.with_metric(
        tolerance=options.tolerance,
        decimal_places=options.num_decimal_places,
    )

    averagine = ms_ditp.Averagine(
        base_composition=set_averagine(backbone=options.averagine_backbone)
    )

    ms1_deconvoluter = MS1Deconvoluter(
        minimum_intensity=options.min_intensity,
        averagine=averagine,
        max_missed_peaks=1,  # options.max_missed_peaks,
        scale_method=options.scale_method,
        error_tolerance=options.peak_error_tol,
        scorer=ms_ditp.PenalizedMSDeconVFitter(
            # minimum_score=options.envelope_min_score,
            # mass_error_tolerance=options.envelope_error_tol,
        ),
        charge_range=options.ms1_charge_range,
        truncate_after=options.ms1_truncate_after,
    )

    ms2_deconvoluter = MS2Deconvoluter(
        minimum_intensity=options.min_intensity,
        averagine=averagine,
        max_missed_peaks=options.max_missed_peaks,
        scale_method=options.scale_method,
        error_tolerance=options.peak_error_tol,
        scorer=ms_ditp.MSDeconVFitter(
            minimum_score=0,  # options.envelope_min_score,
            mass_error_tolerance=options.envelope_error_tol,
        ),
        charge_range=options.ms2_charge_range,
        truncate_after=options.ms2_truncate_after,
        isotopic_shift_factor=options.isotopic_shift_factor,
    )
    return options, error, ms1_deconvoluter, ms2_deconvoluter


def identify_tic_peaks(raw_file_read, start_time, end_time):
    raw_file_read.reset()
    raw_file_read.make_iterator(grouped=False)

    scan_times = []
    total_intensities = []

    for scan in raw_file_read:
        if scan.ms_level != 1:
            continue

        scan_times.append(scan.scan_time)
        total_intensities.append(np.sum(scan.arrays.intensity))

    scan_times = np.array(scan_times)
    total_intensities = np.array(total_intensities)

    # Local maxima
    max_idx, _ = find_peaks(total_intensities)

    peak_times = scan_times[max_idx]
    if start_time is not None and end_time is not None:
        return peak_times[(peak_times >= start_time) & (peak_times <= end_time)]
    else:
        return peak_times


def generate_ms1_windows(
    raw_file_read,
    options,
    ms1_deconvoluter,
    delta_time_window=0.4,
    start_time=None,
    end_time=None,
):

    ms1_to_ms2_idx = ms1_to_ms2_dict(raw_file_read)
    peak_times = identify_tic_peaks(raw_file_read, start_time, end_time)

    raw_file_read.reset()
    raw_file_read.make_iterator(grouped=True)

    scan_processor = ms_ditp.ScanProcessor(raw_file_read)

    window_mass_info = []

    seen_ms1_scan_sets = set()
    ms2_scan_cache = {}

    for idx, t in enumerate(
        tqdm.tqdm(
            peak_times,
            desc="Deisotoping average MS1 and collecting MS2",
        )
    ):
        scan = raw_file_read.get_scan_by_time(t)

        if scan.ms_level != 1:
            scan = raw_file_read.find_previous_ms1(scan.index)

        average_ms1_scan = scan.average(rt_interval=delta_time_window / 2)
        average_ms1_scan.pick_peaks()

        ms1_index_list = average_ms1_scan.scan_indices
        ms1_index_set = frozenset(ms1_index_list)

        if ms1_index_set in seen_ms1_scan_sets:
            continue

        seen_ms1_scan_sets.add(ms1_index_set)

        min_window_time = raw_file_read.get_scan_by_index(min(ms1_index_list)).scan_time

        max_window_time = raw_file_read.get_scan_by_index(max(ms1_index_list)).scan_time

        ms2_scans = []

        for ms1_idx in ms1_index_list:
            ms2_index_list = ms1_to_ms2_idx[ms1_idx]

            for ms2_idx in ms2_index_list:
                ms2_scans.append(raw_file_read.get_scan_by_index(ms2_idx))

        (
            average_ms1_scan,
            priority_list,
            ms2_scans,
        ) = scan_processor.process_scan_group(
            average_ms1_scan,
            ms2_scans,
        )

        (
            priority_list,
            ms2_scans,
        ) = priority_list_charge_filter(
            priority_list,
            ms2_scans,
            options.min_precursor_charge,
        )

        # Deconvolute MS1 scan to get list of deisotoped peaks
        ms1_masses = ms1_deconvoluter.deconvolute_scan(
            scan=average_ms1_scan,
            priority_list=priority_list,
        ).peaks

        ms2_scan_idx_list = []
        # Collect MS2 scans
        for ms2_scan in ms2_scans:
            ms2_scan_idx = int(ms2_scan.index)
            ms2_scan_idx_list.append(ms2_scan_idx)

            if ms2_scan_idx in ms2_scan_cache.keys():
                continue
            ms2_scan_cache[ms2_scan_idx] = ms2_scan

        for ms1_mass, ms2_scan_idx in zip(
            ms1_masses,
            ms2_scan_idx_list,
        ):
            window_mass_info.append(
                {
                    "min_window_time": min_window_time,
                    "max_window_time": max_window_time,
                    "ms1_mass": ms1_mass.neutral_mass,
                    "ms2_scan_idx": ms2_scan_idx,
                    "ms1_time_group": idx,
                }
            )

    df_window_info = pl.DataFrame(window_mass_info)

    return df_window_info, ms2_scan_cache


def adduct_detection(df_window_info, error, detect_adducts=True):
    df_window_info = df_window_info.with_row_index("_row_id")

    HYDROGEN_MONOISOTOPIC_MASS = 1.00782503223
    SODIUM_MONOISOTOPIC_MASS = 22.9897692820
    POTASSIUM_MONOISOTOPIC_MASS = 38.9637064864

    adducts = (
        {
            "none": 0.0,
            "Na-H": (SODIUM_MONOISOTOPIC_MASS - HYDROGEN_MONOISOTOPIC_MASS),
            "K-H": (POTASSIUM_MONOISOTOPIC_MASS - HYDROGEN_MONOISOTOPIC_MASS),
        }
        if detect_adducts
        else {
            "none": 0.0,
        }
    )

    df_adducts = pl.DataFrame(
        {
            "adduct_type": list(adducts.keys()),
            "adduct_mass": list(adducts.values()),
        }
    )

    df_adduct_candidates = df_window_info.join(
        df_adducts,
        how="cross",
    ).with_columns(
        (pl.col("ms1_mass") - pl.col("adduct_mass")).alias("_candidate_ms1_mass")
    )

    previous_candidate_mass = pl.col("_candidate_ms1_mass").shift(1)

    df_adduct_candidates = df_adduct_candidates.with_columns(
        (
            (pl.col("_candidate_ms1_mass") - previous_candidate_mass).abs()
            / previous_candidate_mass.abs()
        )
        .fill_null(0.0)
        .fill_nan(0.0)
        .gt(error.tolerance)
        .cum_sum()
        .over(
            "ms1_time_group",
            order_by="_candidate_ms1_mass",
        )
        .alias("_local_candidate_group")
    )

    candidate_group_support = df_adduct_candidates.group_by(
        [
            "ms1_time_group",
            "_local_candidate_group",
        ]
    ).agg(pl.col("_row_id").n_unique().alias("_group_support"))

    df_adduct_candidates = df_adduct_candidates.join(
        candidate_group_support,
        on=[
            "ms1_time_group",
            "_local_candidate_group",
        ],
        how="left",
    ).with_columns((pl.col("adduct_type") == "none").alias("_prefer_no_adduct"))

    selected_assignments = df_adduct_candidates.sort(
        by=[
            "_row_id",
            "_group_support",
            "_prefer_no_adduct",
            "_candidate_ms1_mass",
        ],
        descending=[
            False,
            True,
            True,
            False,
        ],
    ).unique(
        subset="_row_id",
        keep="first",
        maintain_order=True,
    )

    mass_group_mapping = (
        selected_assignments.select(
            [
                "ms1_time_group",
                "_local_candidate_group",
            ]
        )
        .unique()
        .sort(
            [
                "ms1_time_group",
                "_local_candidate_group",
            ]
        )
        .with_row_index("ms1_mass_group")
    )

    selected_assignments = selected_assignments.join(
        mass_group_mapping,
        on=[
            "ms1_time_group",
            "_local_candidate_group",
        ],
        how="left",
    ).select(
        "_row_id",
        "ms1_mass_group",
        pl.col("adduct_type").alias("inferred_adduct_type"),
        pl.col("_candidate_ms1_mass").alias("adduct_normalized_ms1_mass"),
    )

    df_window_info = df_window_info.join(
        selected_assignments,
        on="_row_id",
        how="left",
        validate="1:1",
    ).drop("_row_id")

    return df_window_info


def generate_global_singletons(options, error):
    # TODO: Remove this requirement.
    # We just need to get the singletons of the whole RAW file
    # without saving it or anything else
    preprocessor = Preprocessor.__new__(Preprocessor)

    # identify_singletons() and collect_raw_peaks() require these attributes
    preprocessor.file_settings = PreprocessingFileSettings(
        input_path=options.input,
        meta_path=Path("unused.yaml"),
        alphabet_path=options.alphabet,
        output_dir=Path("unused_output"),
    )
    preprocessor.error = error

    preprocessor.singleton_boundaries = SingletonBoundaries.from_alphabet_file(
        input_path=options.alphabet,
        boundary_factor=0.5,
        error=preprocessor.error,
    )

    default_singletons = preprocessor.identify_singletons()
    return default_singletons


def generate_singletons_and_fragments(
    grp_number,
    df_window_info,
    ms2_scan_cache,
    options,
    error,
    ms2_deconvoluter,
    default_singletons,
    meta,
):
    df_filter = df_window_info.filter(pl.col("ms1_mass_group") == grp_number)

    adduct_types = ", ".join(df_filter["inferred_adduct_type"].unique().to_list())
    min_window_time = df_filter["min_window_time"].min()
    max_window_time = df_filter["max_window_time"].max()
    intact_mass = df_filter["ms1_mass"].min()

    if intact_mass < meta["3_prime_tag"] + meta["5_prime_tag"]:
        return None, None

    ms2_scan_list = []

    ms2_index_list = df_filter["ms2_scan_idx"].unique().to_list()
    for ms2_idx in ms2_index_list:
        ms2_scan_list.append(ms2_scan_cache[ms2_idx])

    if len(ms2_scan_list) > 1:
        average_ms2_scan = ms2_scan_list[0].average_with(ms2_scan_list[1:])
    else:
        average_ms2_scan = ms2_scan_list[0]
    average_ms2_scan.pick_peaks()

    singletons = RawPeakList.from_scan(
        average_ms2_scan,
        SingletonBoundaries.from_alphabet_file(
            input_path=options.alphabet,
            boundary_factor=0.5,
            error=error,
        ),
    ).to_singletons(alphabet_path=options.alphabet, error=error, min_score=-np.inf)

    if singletons is None:
        singletons = default_singletons

    singletons = singletons.with_columns(pl.lit(grp_number).alias("ms1_mass_group"))

    ms2_peak_list = ms2_deconvoluter.deconvolute_scan(
        scan=average_ms2_scan,
    )

    if len(ms2_peak_list.peaks) == 0:
        return None, None

    fragments = ms2_peak_list.to_fragments(tolerance=error.tolerance)

    fragments = fragments.rename({"neutral_mass": "observed_mass"}).filter(
        pl.col("observed_mass") < intact_mass
    )

    fragments = fragments.with_columns(
        pl.lit(False).alias("is_ms1_mass"),
        pl.lit(grp_number).alias("ms1_mass_group"),
    )

    intact_mass_row = pl.DataFrame(
        {
            "observed_mass": [intact_mass],
            "is_ms1_mass": [True],
            "ms1_mass_group": [grp_number],
            "adduct_type": [adduct_types],
            "min_window_time": [min_window_time],
            "max_window_time": [max_window_time],
        }
    )

    if fragments is not None:
        fragments = pl.concat([fragments, intact_mass_row], how="diagonal_relaxed")
    else:
        return None, None

    return fragments, singletons


def pre_process_multiplexing(
    options: MixturePreprocessingOptions,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    start_time = options.min_time
    end_time = options.max_time
    delta_time_window = options.window_size
    file_path = options.input
    raw_file_read = initialize_raw_file_iterator(str(file_path))

    with open(options.meta, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
        if "true_sequence" not in meta:
            meta["true_sequence"] = "".join(meta["true_sequences"])

    (options, error, ms1_deconvoluter, ms2_deconvoluter) = initialize_parsers(options)

    (df_window_info, ms2_scan_cache) = generate_ms1_windows(
        raw_file_read,
        options,
        ms1_deconvoluter,
        delta_time_window,
        start_time,
        end_time,
    )
    df_window_info = adduct_detection(df_window_info, error, detect_adducts=True)

    default_singletons = generate_global_singletons(options, error)

    fragment_list = []
    singleton_list = []

    for grp_number in tqdm.tqdm(
        df_window_info["ms1_mass_group"].unique(),
        desc="Generating fragments and singletons",
    ):
        (fragments, singletons) = generate_singletons_and_fragments(
            grp_number,
            df_window_info,
            ms2_scan_cache,
            options,
            error,
            ms2_deconvoluter,
            default_singletons,
            meta,
        )
        if fragments is None or singletons is None:
            continue

        fragment_list.append(fragments)
        singleton_list.append(singletons)

    fragments = pl.concat(fragment_list)  # , how="diagonal_relaxed")
    singletons = pl.concat(singleton_list)  # , how="diagonal_relaxed")

    # Save all pre-processing files in preprocessing.output_dir
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    singletons.write_csv(
        options.output_dir / "fragments.singletons.tsv",
        separator="\t",
    )
    fragments.write_csv(options.output_dir / "fragments.tsv", separator="\t")
    with open(
        options.output_dir / "fragments.preprocessed.meta.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.dump(meta, f)

    return fragments, singletons, meta
