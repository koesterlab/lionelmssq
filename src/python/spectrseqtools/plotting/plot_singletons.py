# -*- coding: utf-8 -*-
"""Plotting of singletons detected during preprocessing."""

from typing import Set

import altair as alt
import polars as pl
import yaml

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.file_settings import load_alphabet
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import PreprocessingOptions, SingletonPlotOptions
from spectrseqtools.plotting import HISTOGRAM_WIDTH
from spectrseqtools.preprocessing.preprocessing import Preprocessor


def plot_singletons(options: SingletonPlotOptions) -> None:
    """Plot singletons in all scans that contain all matches.

    Returns
    -------
    options : SingletonPlotOptions
        Options for singleton plot read by parser.

    """
    preprocessor = Preprocessor(
        options=PreprocessingOptions(
            input=options.input,
            meta=options.meta,
            alphabet=options.alphabet,
        )
    )
    alphabet = NucleotideAlphabet.from_file(
        error=preprocessor.error, input_path=preprocessor.file_settings.alphabet_path
    ).to_dataframe()
    nuc_reps = {
        **{
            nuc: row[alphabet.get_column_index("names")][0]
            for row in alphabet.rows()
            for nuc in row[alphabet.get_column_index("names")]
        }
    }
    with open(preprocessor.file_settings.meta_path, "r", encoding="utf-8") as f:
        meta_params = yaml.safe_load(f)
        true_nucleosides = set(
            nuc_reps[nuc]
            for nuc in Sequence.from_str(meta_params["true_sequence"]).sequence
        )

    id_list = [
        nuc.names[0]
        for nuc in NucleotideAlphabet.from_file(
            input_path=preprocessor.file_settings.alphabet_path,
            error=preprocessor.error,
        ).alphabet
    ]
    singletons = pl.read_csv(
        preprocessor.file_settings.updated_alphabet_path, separator="\t"
    ).filter((pl.col("cluster_score") > 0) & (pl.col("id").is_in(id_list)))
    # print(singletons)
    valid_scans = preprocessor.identify_scans_with_full_singleton_set(
        singletons=singletons, id_list=id_list
    )

    total_plot = alt.VConcatChart()
    masses = load_alphabet(input_path=preprocessor.file_settings.alphabet_path)
    for scan_idx, scan_data in enumerate(valid_scans):
        scan_plot = plot_scan(data=scan_data, true_nucs=true_nucleosides, masses=masses)
        total_plot &= scan_plot
        scan_plot.configure_view(strokeWidth=0).save(
            options.scan_dir / f"scan_{scan_idx}.html"
        )

    total_plot.configure_view(strokeWidth=0).save(options.output_path)


def plot_scan(
    data: pl.DataFrame, true_nucs: Set[str], masses: pl.DataFrame
) -> alt.Chart:
    """Plot scan containing all detected singletons.

    Parameters
    ----------
    data : pl.DataFrame
        Polars dataframe containing scan data.
    true_nucs : Set[str]
        Set of names of all nucleotides truly contained in underlying sequence.
    masses : pl.DataFrame
        Polars dataframe containing nucleotides.

    Returns
    -------
    alt.Chart
        Combined Altair plot of all histograms with singletons marked by category.

    """
    # Select highest measured intensity peak
    max_intensity = data["intensity"].max()

    # Adapt intensity for both dataframes
    data = data.with_columns(
        (100 * pl.col("intensity").cast(float) / max_intensity).alias("rel_intensity")
    )

    data_background = data.filter(pl.col("id").is_null())
    data_singletons = data.filter(pl.col("id").is_not_null())
    data_singletons = data_singletons.with_columns(
        pl.col("id")
        .map_elements(
            lambda x: masses.row(named=True, by_predicate=pl.col("id") == x)[
                "encoding"
            ],
            return_dtype=str,
        )
        .alias("nuc_id"),
        pl.col("id")
        .map_elements(
            lambda x: masses.row(named=True, by_predicate=pl.col("id") == x)[
                "canonical_name"
            ],
            return_dtype=str,
        )
        .alias("nuc_name"),
        pl.col("mz").map_elements(lambda x: f"{x:.4f}").alias("mass"),
        pl.lit("z=1").alias("charge"),
    )

    chart_background = plot_histogram(
        data=data_background, color="#808285", singleton=False
    )
    false_positive = [
        nuc for nuc in data_singletons["id"].to_list() if nuc not in true_nucs
    ]

    filtered_data = data_singletons.filter(pl.col("id").is_in(["A", "C", "G", "U"]))
    if len(filtered_data) != 0:
        chart_background += plot_histogram(
            data=filtered_data,
            color="black",
            singleton=True,
        )

    filtered_data = data_singletons.filter(
        pl.col("id").is_in(["A", "C", "G", "U"] + false_positive).not_()
    )
    if len(filtered_data) != 0:
        chart_background += plot_histogram(
            data=filtered_data,
            color="#35607A",
            singleton=True,
        )

    filtered_data = data_singletons.filter(pl.col("id").is_in(false_positive))

    if len(filtered_data) != 0:
        chart_background += plot_histogram(
            data=filtered_data,
            color="#990000",
            singleton=True,
        )

    # Return combined charts
    return chart_background.resolve_scale("shared")


def plot_histogram(data: pl.DataFrame, color: str, singleton: bool) -> alt.Chart:
    """Plot histogram with one category of singletons marked.

    Parameters
    ----------
    data : pl.DataFrame
        Polars dataframe containing scan data.
    color : str
        Name of color for marks.
    singleton : bool
        Flag whether current data contains singletons to mark.

    Returns
    -------
    alt.Chart
        Altair histogram with one singleton category marked.

    """
    # Create histogram for scan data
    chart = (
        alt.Chart(data)
        .mark_bar(size=1.5)
        .encode(
            x=alt.X(
                "mz:Q",
                title="m/z",
                scale=alt.Scale(
                    padding=0, domain=[data["mz"].min() - 10, data["mz"].max() + 10]
                ),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "rel_intensity:Q",
                title="Relative intensity",
                axis=alt.Axis(grid=False),
                scale=alt.Scale(domain=[0, 100]),
                stack=False,
            ),
            tooltip=["mz", "intensity", "rel_intensity"],
            color=alt.value(color),
        )
        .properties(width=HISTOGRAM_WIDTH)
    )

    if not singleton:
        return chart

    # Add singleton ID
    text_id = chart.mark_text(
        align="center", dy=-45, font="monospace", fontWeight="bold", fontSize=20
    ).encode(text="nuc_id:N", color=alt.value(color))

    # Add singleton name
    text_name = chart.mark_text(
        align="center", dy=-30, fontWeight="bold", fontSize=10
    ).encode(text="nuc_name:N", color=alt.value(color))

    # Add singleton mass
    text_mass = chart.mark_text(align="center", dy=-20, fontSize=10).encode(
        text="mass:N", color=alt.value(color)
    )

    # Add singleton charge state
    text_charge = chart.mark_text(align="center", dy=-10, fontSize=10).encode(
        text="charge:N", color=alt.value(color)
    )
    return chart + text_id + text_name + text_mass + text_charge
