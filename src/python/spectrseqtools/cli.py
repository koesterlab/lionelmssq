# -*- coding: utf-8 -*-
"""Module for command-line interface."""

from spectrseqtools.parsers import Options
from spectrseqtools.plotting.plot_evaluation import plot_evaluation
from spectrseqtools.plotting.plot_fragments import plot_fragments
from spectrseqtools.plotting.plot_run_statistics import plot_run_statistics
from spectrseqtools.plotting.plot_singletons import plot_singletons
from spectrseqtools.plotting.plot_spectrum import plot_spectrum
from spectrseqtools.postprocessing.evaluate_prediction import evaluate_prediction
from spectrseqtools.postprocessing.evaluate_run_statistics import (
    evaluate_run_statistics,
)
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

    # Postprocess prediction results
    if options.postprocessing is not None:
        if options.postprocessing.prediction is not None:
            evaluate_prediction(options=options.postprocessing.prediction)

        if options.postprocessing.run_statistics is not None:
            evaluate_run_statistics(options=options.postprocessing.run_statistics)

    # Plot (intermediate) prediction results
    if options.plotting is not None:
        if options.plotting.singletons is not None:
            plot_singletons(options=options.plotting.singletons)

        if options.plotting.fragments is not None:
            plot_fragments(options=options.plotting.fragments)

        if options.plotting.spectrum is not None:
            plot_spectrum(options=options.plotting.spectrum)

        if options.plotting.evaluation is not None:
            plot_evaluation(options=options.plotting.evaluation)

        if options.plotting.run_statistics is not None:
            plot_run_statistics(options=options.plotting.run_statistics)
