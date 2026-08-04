# -*- coding: utf-8 -*-
"""Plotting of run statistics."""
from typing import List

import altair as alt
import polars as pl

from spectrseqtools.parsers import RunStatisticsPlotOptions

STATUS_COLORS = {
    "experiment": "#990000",
    # "true": "#2b73b5",
    "simulation": "#808285",
}

STATUS_ORDER = ["simulation", "experiment"]

LEGEND_PARAMS = {
    "padding": 10,
    "strokeColor": "black",
    "cornerRadius": 5,
    "fillColor": "white",
}

WIDTH = 850


def plot_run_statistics(options: RunStatisticsPlotOptions) -> None:
    """Plot run statistics.

    Parameters
    ----------
    options : RunStatisticsPlotOptions
        Options for run-statistics plot read by parser.

    """
    mode = options.statistic_criterion

    sim_results = pl.read_csv(options.simulation, separator="\t")
    sim_results = sim_results.with_columns(pl.lit("simulation").alias("type"))
    sim_chart = create_scatterplot(data=sim_results, mode=mode)

    exp_results = pl.read_csv(options.experiment, separator="\t")
    exp_results = exp_results.with_columns(pl.lit("experiment").alias("type"))
    exp_chart = create_scatterplot(data=exp_results, size=50, mode=mode)

    chart = (
        alt.layer(sim_chart, exp_chart)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(options.output_path)


def create_scatterplot(data: pl.DataFrame, mode: str, size: int = 20) -> alt.Chart:
    """Create scatterplot over given parameter.

    Parameters
    ----------
    data : pl.DataFrame
        Statistics data.
    mode : str
        Statistic to be plotted.
    size : int, optional
        Size of circles in scatterplot.

    Returns
    -------
    alt.Chart
        Scatterplot over given statistic.

    """
    base_chart = (
        alt.Chart(data)
        .encode(
            x=alt.X(
                "num_frag:Q",
                title="Number of fragments",
                scale=alt.Scale(padding=0),
            ),
            y=select_y_axis(mode=mode),
        )
        .properties(width=WIDTH)
    )

    scatterplot = base_chart.mark_circle(size=size).encode(
        # color=alt.value(STATUS_COLORS[kind]),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(
                domain=STATUS_ORDER,
                range=[
                    STATUS_COLORS[stat] if STATUS_COLORS.get(stat) else stat
                    for stat in STATUS_ORDER
                ],
            ),
            legend=alt.Legend(
                **LEGEND_PARAMS,
                orient="top-left",
                title="",
            ),
        ),
        tooltip=select_tooltip(mode=mode),
    )

    return scatterplot


def select_y_axis(mode: str) -> alt.Y:
    """Select Y-axis for Altair plot based on given statistic.

    Parameters
    ----------
    mode : str
        Statistic to be plotted.

    Returns
    -------
    alt.Y
        Y-axis for Altair plot.

    """
    match mode:
        case "runtime":
            return alt.Y(
                "s:Q",
                title="Runtime (in sec)",
                scale=alt.Scale(type="linear"),
            )
        case "memory":
            return alt.Y(
                "max_rss:Q",
                title="Memory",
                scale=alt.Scale(type="linear"),
            )
        case _:
            return alt.Y(
                "s:Q",
                title="Runtime (in sec)",
                scale=alt.Scale(type="linear"),
            )


def select_tooltip(mode: str) -> List[str]:
    """Select tooltip for Altair plot.

    Parameters
    ----------
    mode : str
        Statistic to be plotted.

    Returns
    -------
    List[str]
        List of columns to be used for tooltip in Altair plot.

    """
    match mode:
        case "runtime":
            return ["s", "num_frag"]
        case "memory":
            return ["max_rss", "num_frag"]
        case _:
            return ["s", "num_frag"]
