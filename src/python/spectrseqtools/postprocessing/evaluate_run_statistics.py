# -*- coding: utf-8 -*-
"""Postprocessing of prediction results by evaluation of run statistics."""

from pathlib import Path
from typing import List

import polars as pl

from spectrseqtools.parsers import RunStatisticsPostprocessingOptions


def evaluate_run_statistics(options: RunStatisticsPostprocessingOptions) -> None:
    """Evaluate run statistics.

    Parameters
    ----------
    options : RunStatisticsPostprocessingOptions
        Options for run-statistic evaluation read by parser.

    """
    results = collect_results(
        benchmark_files=options.benchmarks,
        fragment_files=options.fragments,
    )
    results.write_csv(options.output_path, separator="\t")


def collect_results(
    benchmark_files: List[Path], fragment_files: List[Path]
) -> pl.DataFrame:
    """Collect results from input files.

    Parameters
    ----------
    benchmark_files : List[Path]
        List of paths for benchmarking results in TSV format.
    fragment_files : List[Path]
        List of paths for fragment in TSV format.

    Returns
    -------
    pl.DataFrame
        Polars dataframe containing collected results.

    """
    results = []
    for benchmark, fragments in zip(benchmark_files, fragment_files):
        new_data = pl.read_csv(benchmark, separator="\t")
        for col in new_data.columns:
            if col == "h:m:s":
                continue
            new_data = new_data.with_columns(
                pl.when(pl.col(col).cast(str) == "NA")
                .then(None)
                .otherwise(pl.col(col))
                .name.keep()
                .cast(pl.Float64)
            )
            new_data = new_data.with_columns(
                pl.when(pl.col(col) < 5)
                .then(None)
                .otherwise(pl.col(col))
                .name.keep()
                .cast(pl.Float64)
            )
        with open(fragments, "r", encoding="utf-8") as f:
            num_entries = len(f.readlines())
        new_data.insert_column(0, pl.Series("num_frag", [num_entries]))
        results.append(new_data)

    return pl.concat(results)
