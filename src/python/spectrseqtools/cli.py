# -*- coding: utf-8 -*-
"""Module for command-line interface."""

from spectrseqtools.parsers import Options
from spectrseqtools.plotting.plot_evaluation import plot_evaluation
from spectrseqtools.plotting.plot_fragments import plot_fragments
from spectrseqtools.plotting.plot_run_statistics import plot_run_statistics
from spectrseqtools.plotting.plot_singletons import plot_singletons
from spectrseqtools.plotting.plot_spectrum import plot_spectrum
from spectrseqtools.prediction.prediction import Predictor
from spectrseqtools.preprocessing.preprocessing import Preprocessor


def main():
    """Parse options to select and execute subcommands."""
    options = Options.from_cli_args()

    # Preprocess raw data
    if options.preprocessing is not None:
        Preprocessor(options=options.preprocessing).preprocess()

    # Predict sequence
    if options.prediction is not None:
        Predictor(options=options.prediction).predict()

    # Plot singletons
    if options.plot_singletons is not None:
        plot_singletons(options=options.plot_singletons)

    # Plot fragments
    if options.plot_fragments is not None:
        plot_fragments(options=options.plot_fragments)

    # Plot spectrum
    if options.plot_spectrum is not None:
        plot_spectrum(options=options.plot_spectrum)

    # Plot evaluation results
    if options.plot_evaluation is not None:
        plot_evaluation(options=options.plot_evaluation)

    # Plot run statistics
    if options.plot_run_statistics is not None:
        plot_run_statistics(options=options.plot_run_statistics)
