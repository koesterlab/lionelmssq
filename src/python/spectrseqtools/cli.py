# -*- coding: utf-8 -*-
"""Module for command-line interface."""

from spectrseqtools.multiplexing import (
    evaluate_multiplexing,
    pre_process_multiplexing,
    predict_multiplexing,
)
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
from spectrseqtools.simulation.simulate_fragments import simulate_fragments
from spectrseqtools.simulation.simulate_metadata import (
    simulate_metadata_for_custom_sequence,
    simulate_metadata_for_random_sequences,
)


# TODO: Adapt to linting of Ruff v0.16
def main():
    """Parse options to select and execute subcommands."""
    options = Options.from_cli_args(args=None)

    # Simulate preprocessed data
    if options.simulation is not None:
        if options.simulation.custom is not None:
            simulate_metadata_for_custom_sequence(options=options.simulation.custom)
        if options.simulation.random is not None:
            simulate_metadata_for_random_sequences(options=options.simulation.random)
        if options.simulation.fragments is not None:
            simulate_fragments(options=options.simulation.fragments)

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

    # Analyze mixtures
    if options.mixture is not None:
        if options.mixture.preprocessing is not None:
            pre_process_multiplexing(options=options.mixture.preprocessing)
        if options.mixture.prediction is not None:
            predict_multiplexing(options=options.mixture.prediction)
        if options.mixture.postprocessing is not None:
            evaluate_multiplexing(options=options.mixture.postprocessing)
