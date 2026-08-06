# from ast import literal_eval
from pathlib import Path
from typing import List

import polars as pl

from spectrseqtools.error_calculator import ErrorUnderL1Norm
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import RunStatisticsPostprocessingOptions

NUCLEOTIDE_DF = NucleotideAlphabet.from_file(error=ErrorUnderL1Norm()).to_dataframe()
NUC_REPS = {
    **{
        nuc: row[NUCLEOTIDE_DF.get_column_index("names")][0]
        for row in NUCLEOTIDE_DF.rows()
        for nuc in row[NUCLEOTIDE_DF.get_column_index("names")]
    }
}

STATUS_ORDER = [
    "identical",
    "identical (minus 55U/G)",
    "correct composition",
    "failed prediction",
    "wrong length",
    "no prediction",
]


def evaluate_run_statistics(options: RunStatisticsPostprocessingOptions) -> None:
    results = collect_results(
        benchmark_files=options.benchmarks,
        fragment_files=options.fragments,
    )

    # if options.config is not None:
    #     params = literal_eval(options.config)
    #     for key in params:
    #         if key == options.evaluation_criterion:
    #             continue
    #         results = results.with_columns(pl.lit(params[key][0]).alias(key))
    #     results = results.rename({"comp_val": options.evaluation_criterion})

    results.write_csv(options.output_path, separator="\t")


def collect_results(
    benchmark_files: List[Path], fragment_files: List[Path]
) -> pl.DataFrame:
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
        # frag = pl.read_csv(fragments, separator="\t")
        # if "ppm_group" in frag.columns:
        #      num_entries = len(frag.select("ppm_group").unique())
        # else:
        with open(fragments, "r", encoding="utf-8") as f:
            num_entries = len(f.readlines())
        new_data.insert_column(0, pl.Series("num_frag", [num_entries]))
        results.append(new_data)

    return pl.concat(results)
