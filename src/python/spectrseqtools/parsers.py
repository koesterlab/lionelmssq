# -*- coding: utf-8 -*-
"""Module with ddargparse parser classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import ddargparse

from spectrseqtools.enums import (
    AveragineBackbone,
    ErrorMetric,
    LengthEstimatorMetric,
    SolverType,
)


@dataclass
class PreprocessingOptions(ddargparse.OptionsBase):
    """Preprocessing of raw data into fragments"""

    input: Path = field(
        metadata={"help": "Path to input file in RAW format"},
    )
    meta: Path = field(
        metadata={"help": "Path to YAML with meta information."},
    )
    alphabet: Path | None = field(
        default=None,
        metadata={"help": "Path to file containing nucleotide alphabet."},
    )
    output_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Output directory (default: input directory).",
        },
    )
    min_intensity: float | None = field(
        default=None,
        metadata={
            "help": "Minimum intensity required for peak consideration "
            "(used in ms_deisotope package as 'minimum_intensity')."
        },
    )
    tolerance: float = field(
        default=10e-6,
        metadata={"help": "Error tolerance to consider masses identical."},
    )
    num_decimal_places: int = field(
        default=3,
        metadata={"help": "Number of considered decimal places for precision."},
    )
    boundary_factor: int = field(
        default=2,
        metadata={"help": "Factor for scaling theoretical singleton boundaries."},
    )
    min_precursor_charge: int = field(
        default=3,
        metadata={"help": "Minimum MS1 charge to consider associated MS2 scans."},
    )
    isotopic_shift_factor: int = field(
        default=10,
        metadata={"help": "Factor for scaling isotopic shift for precursors."},
    )
    intact_mass_cutoff_factor: float = field(
        default=0.4,
        metadata={
            "help": "Factor for maximum mass to function as lower bound on intact mass."
        },
    )
    envelope_min_score: float = field(
        default=150.0,
        metadata={
            "help": "Minimum accepted score during envelope fitting "
            "(used in ms_deisotope package as 'minimum_score')."
        },
    )
    envelope_error_tol: float = field(
        default=0.02,
        metadata={
            "help": "Error tolerance for envelopes during fitting "
            "(used in ms_deisotope package as 'mass_error_tolerance')."
        },
    )
    averagine_backbone: AveragineBackbone = field(
        default=AveragineBackbone.PHOSPHATE,
        metadata={
            "help": "Backbone considered in Averagine model "
            "(used in ms_deisotope package)."
        },
    )
    max_missed_peaks: int = field(
        default=0,
        metadata={
            "help": "Maximum number of missed peaks tolerated in envelope fitting "
            "(used in ms_deisotope package)."
        },
    )
    scale_method: str = field(
        default="sum",
        metadata={
            "help": "Scale method for intensity values (used in ms_deisotope package)."
        },
    )
    peak_error_tol: float = field(
        default=2e-5,
        metadata={
            "help": "Error tolerance for each individual peak "
            "(used in ms_deisotope package as 'error_tolerance')."
        },
    )
    ms1_charge_range: Tuple[int, int] | None = field(
        default=None,
        metadata={
            "help": "Charge range considered for MS1 deconvolution "
            "(used in ms_deisotope package)."
        },
    )
    ms2_charge_range: Tuple[int, int] | None = field(
        default=None,
        metadata={
            "help": "Charge range considered for MS2 deconvolution "
            "(used in ms_deisotope package)."
        },
    )
    ms1_truncate_after: float = field(
        default=0.95,
        metadata={
            "help": "Percentage of included isotopic patterns for MS1 scans "
            "(used in ms_deisotope package)."
        },
    )
    ms2_truncate_after: float = field(
        default=0.9,
        metadata={
            "help": "Percentage of included isotopic patterns for MS2 scans "
            "(used in ms_deisotope package)."
        },
    )


@dataclass
class PredictionOptions(ddargparse.OptionsBase):
    """Prediction of sequence based on preprocessed fragments"""

    fragments: Path = field(
        metadata={"help": "Path to TSV table of observed fragments"},
    )
    meta: Path = field(
        metadata={"help": "Path to YAML with meta information."},
    )
    fragment_predictions: Path = field(
        metadata={
            "help": "Path to TSV table that shall contain the per fragment predictions."
        },
    )
    sequence_prediction: Path = field(
        metadata={
            "help": "Path to FASTA file that shall contain the predicted sequence."
        },
    )
    sequence_name: str = field(
        metadata={"help": "Header in FASTA output file."},
    )
    alphabet: Path | None = field(
        default=None,
        metadata={
            "help": "Path to file containing nucleotide alphabet. If preprocessing was "
            "used, this should correspond to the detected singletons."
        },
    )
    output_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Output directory (default: input directory).",
        },
    )
    tolerance: float = field(
        default=10e-6,
        metadata={"help": "Error tolerance to consider masses identical."},
    )
    num_decimal_places: int = field(
        default=3,
        metadata={"help": "Number of considered decimal places for precision."},
    )
    error_metric: ErrorMetric = field(
        default=ErrorMetric.L1NORM,
        metadata={
            "help": "Metric for used for error calculation over multiple values."
        },
    )
    max_intact_mass_variance: int = field(
        default=1,
        metadata={"help": "Maximum variance for intact mass."},
    )
    reduce_fragmentation_dict: bool = field(
        default=True,
        metadata={"help": "Flag whether only c/y-fragmentation should be considered."},
    )
    compression_rate: int = field(
        default=32,
        metadata={
            "help": "Number of binary-compressed masses per cell in traceback matrix."
        },
    )
    modification_rate: float = field(
        default=0.5,
        metadata={
            "help": "Maximum percentage of modification in sequence.",
        },
    )
    length_estimator_metric: LengthEstimatorMetric = field(
        default=LengthEstimatorMetric.JACCARD,
        metadata={"help": "Metric to use for sequence length estimation."},
    )
    solver: SolverType = field(
        default=SolverType.HIGHS,
        metadata={"help": "Solver to use for optimization problem."},
    )
    lp_timeout_short: int = field(
        default=5,
        metadata={"help": "Time-out for shorter solving of LP instances."},
    )
    lp_timeout_long: int = field(
        default=60,
        metadata={"help": "Time-out for longer solving of LP instances."},
    )
    threads: int = field(
        default=1,
        metadata={"help": "Number of threads to use for the optimization problem."},
    )
    intensity_cutoff_percentile: int = field(
        default=75,
        metadata={"help": "Intensity percentile used as cutoff."},
    )
    composition_filter_weight_factor: float = field(
        default=1.0,
        metadata={
            "help": "Nucleotide weight factor used during composition-based filtering."
        },
    )


@dataclass
class Options(ddargparse.OptionsBase):
    """
    De novo prediction of RNA sequences

    Usage:

    1. Preprocess raw data to gain fragments for prediction (grouped into
    files based on sequence).
    2. Predict sequence individually for each file output by preprocessing.

    """

    preprocessing: PreprocessingOptions | None
    prediction: PredictionOptions | None
