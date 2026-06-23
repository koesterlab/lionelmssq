# -*- coding: utf-8 -*-
"""Module with ddargparse parser classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import ddargparse

from spectrseqtools.enums import AveragineBackbone, SolverType


@dataclass
class PreprocessingOptions(ddargparse.OptionsBase):
    """Preprocessing of raw data into fragments"""

    input: Path = field(
        metadata={"help": "Path to input file in RAW format"},
    )
    meta: Path = field(metadata={"help": "Path to YAML with meta information"})
    alphabet: Path | None = field(
        metadata={"help": "Path to file containing nucleotide alphabet."}
    )
    output_dir: Path | None = field(
        metadata={
            "help": "Output directory (default: input directory)",
        }
    )
    charge_range: Tuple[int, int] | None = field(
        metadata={
            "help": "Charge range considered for deconvolution "
            "(used in ms_deisotope package)."
        }
    )
    min_intensity: float | None = field(
        metadata={
            "help": "Minimum intensity required for peak consideration "
            "(used in ms_deisotope package as 'minimum_intensity')."
        }
    )
    tolerance: float = field(
        default=10e-6,
        metadata={"help": "Error tolerance to consider masses identical."},
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
        default=1,
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
            "(used in ms_deisotope package as 'error_tol')."
        },
    )
    truncate_after: float = field(
        default=0.9,
        metadata={
            "help": "Percentage of included isotopic patterns "
            "(used in ms_deisotope package)."
        },
    )


@dataclass
class PredictionOptions(ddargparse.OptionsBase):
    """Prediction of sequence based on preprocessed fragments"""

    fragments: Path = field(
        metadata={"help": "Path to TSV table of observed fragments"},
    )
    meta: Path = field(metadata={"help": "Path to YAML with meta information"})
    singletons: Path = field(
        metadata={"help": "Path to TSV with singleton information"}
    )
    fragment_predictions: Path = field(
        metadata={
            "help": "Path to TSV table that shall contain the per fragment predictions"
        }
    )
    sequence_prediction: Path = field(
        metadata={
            "help": "Path to FASTA file that shall contain the predicted sequence"
        }
    )
    sequence_name: str = field(metadata={"help": "Header in FASTA output file"})
    output_dir: Path | None = field(
        metadata={
            "help": "Output directory (default: input directory)",
        }
    )
    modification_rate: float = field(
        default=0.5,
        metadata={
            "help": "Maximum percentage of modification in sequence",
        },
    )
    solver: SolverType = field(
        default=SolverType.HIGHS,
        metadata={"help": "Solver to use for optimization problem"},
    )
    lp_timeout_short: int = field(
        default=5, metadata={"help": "Time-out for shorter solving of LP instances"}
    )
    lp_timeout_long: int = field(
        default=60, metadata={"help": "Time-out for longer solving of LP instances"}
    )
    threads: int = field(
        default=1,
        metadata={"help": "Number of threads to use for the optimization problem"},
    )
    intensity_cutoff_percentile: int = field(
        default=75, metadata={"help": "Intensity percentile used as cutoff"}
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
