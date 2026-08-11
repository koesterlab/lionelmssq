# -*- coding: utf-8 -*-
"""Postprocessing of prediction results by evaluation thereof."""

from ast import literal_eval
from pathlib import Path
from typing import List

import polars as pl
import yaml

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.error_calculator import ErrorUnderL1Norm
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import PredictionPostprocessingOptions
from spectrseqtools.plotting.plot_evaluation import STATUS_ORDER

NUCLEOTIDE_DF = NucleotideAlphabet.from_file(error=ErrorUnderL1Norm()).to_dataframe()
NUC_REPS = {
    **{
        nuc: row[NUCLEOTIDE_DF.get_column_index("names")][0]
        for row in NUCLEOTIDE_DF.rows()
        for nuc in row[NUCLEOTIDE_DF.get_column_index("names")]
    }
}


def evaluate_prediction(options: PredictionPostprocessingOptions) -> None:
    """Evaluate prediction results.

    Parameters
    ----------
    options : PredictionPostprocessingOptions
        Options for prediction evaluation read by parser.

    """
    results = collect_results(
        prediction_files=options.prediction,
        meta_files=options.meta,
    )

    if options.config is not None:
        params = literal_eval(options.config)
        for key in params:
            if key == options.evaluation_criterion:
                continue
            results = results.with_columns(pl.lit(params[key][0]).alias(key))
        results = results.rename({"comp_val": options.evaluation_criterion})

    results.write_csv(options.output_path, separator="\t")


def collect_results(
    prediction_files: List[Path], meta_files: List[Path]
) -> pl.DataFrame:
    """Collect results from input files.

    Parameters
    ----------
    prediction_files : List[Path]
        List of paths for prediction results in FASTA format.
    meta_files : List[Path]
        List of paths for meta information in YAML format.

    Returns
    -------
    pl.DataFrame
        Polars dataframe containing collected results.

    """
    comp_values = []
    results = []
    true_sequences = []
    pred_sequences = []
    true_lengths = []
    pred_lengths = []
    for file_path, meta_path in zip(prediction_files, meta_files):
        # Read true sequence from meta file
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
            true_seq = Sequence.from_str(meta["true_sequence"]).sequence

        # Read predicted sequence from FASTA file
        with open(file_path, "r", encoding="utf-8") as f:
            f.readline()
            pred_seq = Sequence.from_str(f.readline().rstrip("\n")).sequence

        print(len(true_seq), len(pred_seq))
        print("true:", "".join(true_seq))
        print("pred:", "".join(pred_seq))
        result = compare_sequences(true_seq, pred_seq)
        print("result:", result)
        print()

        results.append(result)
        true_sequences.append("".join(true_seq))
        pred_sequences.append("".join(pred_seq))
        true_lengths.append(len(true_seq))
        pred_lengths.append(len(pred_seq))
        comp_values.append(str(file_path.parent.parent.name))

    return pl.DataFrame(
        {
            "true_sequence": true_sequences,
            "pred_sequence": pred_sequences,
            "true_len": true_lengths,
            "pred_len": pred_lengths,
            "comp_val": comp_values,
            "result": results,
            "order": [STATUS_ORDER.index(res) for res in results],
        }
    )


def compare_sequences(true_seq: List[str], pred_seq: List[str]) -> str:
    """Compare true and predicted sequences.

    Parameters
    ----------
    true_seq : List[str]
        True sequence.
    pred_seq : List[str]
        Predicted sequence.

    Returns
    -------
    str
        Evaluation result.

    """
    true_seq = [NUC_REPS[nuc] if nuc in NUC_REPS else nuc for nuc in true_seq]
    pred_seq = [NUC_REPS[nuc] if nuc in NUC_REPS else nuc for nuc in pred_seq]
    if len(pred_seq) < 1:
        return "no prediction"
    if len(true_seq) != len(pred_seq):
        return "wrong length"
    if true_seq == pred_seq:
        return "identical"
    if true_seq == ["G" if nuc == "55U" else nuc for nuc in pred_seq]:
        return "identical (minus 55U/G)"
    if sorted(true_seq) == sorted(pred_seq):
        return "correct composition"
    return "failed prediction"
