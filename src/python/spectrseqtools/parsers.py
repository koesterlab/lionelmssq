# -*- coding: utf-8 -*-
"""Module with ddargparse parser classes."""

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import ddargparse

from spectrseqtools.enums import (
    AveragineBackbone,
    ErrorMetric,
    LengthEstimatorMetric,
    SolverType,
)


@dataclass
class CustomMetadataSimulationOptions(ddargparse.OptionsBase):
    """Simulation of meta information for custom sequence."""

    sequence: str = field(
        metadata={"help": "Underlying custom sequence."},
    )
    output_dir: Path = field(
        metadata={
            "help": "Output directory.",
        },
    )
    start_tag: float = field(
        default=555.1294,
        metadata={"help": "Mass of 5'-tag of sequence."},
    )
    end_tag: float = field(
        default=455.1491,
        metadata={"help": "Mass of 3'-tag of sequence."},
    )


@dataclass
class RandomMetadataSimulationOptions(ddargparse.OptionsBase):
    """Simulation of meta information for random sequences."""

    num_sequences: int = field(
        metadata={"help": "Number of sequences for which to generate metadata."},
    )
    output_dir: Path = field(
        metadata={
            "help": "Output directory.",
        },
    )
    start_tag: float = field(
        default=555.1294,
        metadata={"help": "Mass of 5'-tag of sequence."},
    )
    end_tag: float = field(
        default=455.1491,
        metadata={"help": "Mass of 3'-tag of sequence."},
    )
    sequence_length: int = field(
        default=-1,
        metadata={"help": "Desired sequence length (random if set to -1)."},
    )
    modification_rate: float = field(
        default=0.1,
        metadata={"help": "Number of sequences for which to generate metadata."},
    )
    alphabet: Path | None = field(
        default=None,
        metadata={"help": "Path to file containing nucleotide alphabet."},
    )
    global_seed: int | None = field(
        default=None,
        metadata={"help": "Global random seed used to generate all random values."},
    )


@dataclass
class FragmentSimulationOptions(ddargparse.OptionsBase):
    """Simulation of random fragments."""

    elements: Path = field(
        metadata={"help": "Path to file with element-mass information in CSV format."},
    )
    input: Path = field(
        metadata={"help": "Path to YAML with meta information."},
    )
    fragments: Path = field(
        metadata={"help": "Path to file with simulated fragments in TSV format."},
    )
    singletons: Path = field(
        metadata={"help": "Path to file with simulated singleton information."},
    )
    meta: Path = field(
        metadata={"help": "Path to YAML with updated meta information."},
    )
    num_replicates: int = field(
        metadata={"help": "Number of copies of true sequence used for fragmentation."},
    )
    max_singletons: int = field(
        metadata={
            "help": "Maximum number of singletons reported (if true amount "
            "does not exceed it)."
        },
    )
    phantom_rate: float = field(
        metadata={"help": "Rate of noise fragments introduced during simulation."},
    )
    noise_rate: float = field(
        metadata={"help": "Noise (in ppm) induced on each simulated fragment."},
    )
    config: str = field(
        metadata={"help": "Configurations for parameters outside of comparison study."},
    )


@dataclass
class SimulationOptions(ddargparse.OptionsBase):
    """Simulation of random data imitating preprocessing results."""

    custom: CustomMetadataSimulationOptions | None
    random: RandomMetadataSimulationOptions | None
    fragments: FragmentSimulationOptions | None


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
    ms1_charge_range: tuple[int, int] | None = field(
        default=None,
        metadata={
            "help": "Charge range considered for MS1 deconvolution "
            "(used in ms_deisotope package)."
        },
    )
    ms2_charge_range: tuple[int, int] | None = field(
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

    fragments: Path | pl.DataFrame = field(
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
    alphabet: Path | pl.DataFrame | None = field(
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
        default=80,
        metadata={"help": "Intensity percentile used as cutoff."},
    )
    composition_filter_weight_factor: float = field(
        default=2.0,
        metadata={
            "help": "Nucleotide weight factor used during composition-based filtering."
        },
    )


@dataclass
class SingletonPlotOptions(ddargparse.OptionsBase):
    """Plotting of singletons detected during preprocessing."""

    input: Path = field(
        metadata={"help": "Path to input file in RAW format"},
    )
    meta: Path = field(
        metadata={"help": "Path to YAML with meta information."},
    )
    scan_dir: Path = field(
        metadata={
            "help": "Output directory for all individual scans.",
        },
    )
    output_path: Path = field(
        metadata={
            "help": "Output path for combined HTML of all scans.",
        },
    )
    alphabet: Path | None = field(
        default=None,
        metadata={"help": "Path to file containing nucleotide alphabet."},
    )


@dataclass
class FragmentPlotOptions(ddargparse.OptionsBase):
    """Plotting of fragments aligned to predicted sequence."""

    fragments: Path = field(
        metadata={"help": "Path to fragment file in TSV format"},
    )
    prediction: Path = field(
        metadata={"help": "Path to prediction file in FASTA format."},
    )
    meta: Path = field(
        metadata={"help": "Path to YAML with meta information."},
    )
    combined_plot: Path = field(
        metadata={
            "help": "Output path for combined HTML for fragment of each class.",
        },
    )
    start_fragment_plot: Path = field(
        metadata={
            "help": "Output path for HTML for START-fragments, i.e. 5'-fragments.",
        },
    )
    end_fragment_plot: Path = field(
        metadata={
            "help": "Output path for HTML for END-fragments, i.e. 3'-fragments.",
        },
    )
    internal_fragment_plot: Path = field(
        metadata={
            "help": "Output path for HTML for internal fragments.",
        },
    )
    mixed_fragment_plot: Path = field(
        metadata={
            "help": "Output path for HTML for fragments not divided by class.",
        },
    )
    simulation: Path | None = field(
        default=None,
        metadata={"help": "Path to file containing simulation data (if applicable)."},
    )
    alphabet: Path | None = field(
        default=None,
        metadata={"help": "Path to file containing nucleotide alphabet."},
    )


@dataclass
class SpectrumPlotOptions(ddargparse.OptionsBase):
    """Plotting of spectrum of preprocessed fragments based on usage in prediction."""

    raw_fragments: Path = field(
        metadata={"help": "Path to TSV file containing raw fragments."},
    )
    predicted_fragments: Path = field(
        metadata={"help": "Path to TSV file containing fragments used for prediction."},
    )
    output_path: Path = field(
        metadata={
            "help": "Output path for HTML with spectrum plot.",
        },
    )


@dataclass
class EvaluationPlotOptions(ddargparse.OptionsBase):
    """Plotting of overview of (evaluated) prediction results."""

    input: Path = field(
        metadata={"help": "Path to TSV file containing evaluation results."},
    )
    bar_path: Path = field(
        metadata={
            "help": "Output path for HTML with barplot with evaluation results.",
        },
    )
    donut_path: Path = field(
        metadata={
            "help": "Output path for HTML with donut plot with evaluation results.",
        },
    )
    evaluation_criterion: str = field(
        default="default",
        metadata={"help": "Criterion for differentiation between evaluation results."},
    )


@dataclass
class RunStatisticsPlotOptions(ddargparse.OptionsBase):
    """Plotting of run statistics."""

    simulation: Path = field(
        metadata={"help": "Path to TSV file containing statistics for simulations."},
    )
    experiment: Path = field(
        metadata={"help": "Path to TSV file containing statistics for experiments."},
    )
    output_path: Path = field(
        metadata={
            "help": "Output path for HTML with scatterplot.",
        },
    )
    statistic_criterion: str = field(
        metadata={"help": "Criterion for statistic to be plotted."},
    )


@dataclass
class PlottingOptions(ddargparse.OptionsBase):
    """Plotting of (intermediate) results of sequence prediction."""

    singletons: SingletonPlotOptions | None
    fragments: FragmentPlotOptions | None
    spectrum: SpectrumPlotOptions | None
    evaluation: EvaluationPlotOptions | None
    run_statistics: RunStatisticsPlotOptions | None


@dataclass
class PredictionPostprocessingOptions(ddargparse.OptionsBase):
    """Evaluation of prediction results."""

    prediction: list[Path] = field(
        metadata={"help": "List of paths to prediction files in FASTA format."},
    )
    meta: list[Path] = field(
        metadata={"help": "List of paths to YAML files with meta information."},
    )
    output_path: Path = field(
        metadata={
            "help": "Output path for evaluation results in TSV format.",
        },
    )
    evaluation_criterion: str = field(
        metadata={"help": "Criterion for differentiation between evaluation results."},
    )
    config: str | None = field(
        default=None,
        metadata={"help": "Configurations for parameter study (if applicable)."},
    )


@dataclass
class RunStatisticsPostprocessingOptions(ddargparse.OptionsBase):
    """Evaluation of run statistics for predictions."""

    benchmarks: list[Path] = field(
        metadata={"help": "List of paths to benchmark files in TSV format."},
    )
    fragments: list[Path] = field(
        metadata={"help": "List to paths to fragment files in TSV format"},
    )
    output_path: Path = field(
        metadata={
            "help": "Output path for run-statistic results in TSV format.",
        },
    )


@dataclass
class PostprocessingOptions(ddargparse.OptionsBase):
    """Postprocessing of prediction results."""

    prediction: PredictionPostprocessingOptions | None
    run_statistics: RunStatisticsPostprocessingOptions | None


@dataclass
class Options(ddargparse.OptionsBase):
    """
    De novo prediction of RNA sequences

    Usage:

    1. Preprocess raw data to gain fragments for prediction (grouped into
    files based on sequence).
    2. Predict sequence individually for each file output by preprocessing.

    """

    simulation: SimulationOptions | None
    preprocessing: PreprocessingOptions | None
    prediction: PredictionOptions | None
    postprocessing: PostprocessingOptions | None
    plotting: PlottingOptions | None
