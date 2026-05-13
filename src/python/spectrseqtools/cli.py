# -*- coding: utf-8 -*-
"""Module for command-line interface."""

from spectrseqtools.parsers import Options
from spectrseqtools.prediction.prediction import Predictor
from spectrseqtools.preprocessing.preprocessing import Preprocessor


def main():
    """Parse options to select and execute subcommands."""
    options = Options.parse_args()

    # Preprocess raw data
    if options.preprocessing is not None:
        Preprocessor(options=options.preprocessing).preprocess()

    # Predict sequence
    if options.prediction is not None:
        Predictor(options=options.prediction).predict()
