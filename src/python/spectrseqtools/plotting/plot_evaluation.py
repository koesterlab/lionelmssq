import altair as alt
import polars as pl

from spectrseqtools.parsers import EvaluationPlotOptions

STATUS_COLORS = {
    "identical": "#8cb110",
    "identical (minus 55U/G)": "yellow",
    "correct composition": "orange",
    # "failed prediction": "#be0a26",
    "failed prediction": "#990000",
    "wrong length": "gray",
    "no prediction": "black",
}

STATUS_COLORS = {
    "identical": "#35607A",
    # "identical": "#2B73B5",
    # "identical (minus 55U/G)": "#2B73B5",
    # "correct composition": "#639FD3",
    "identical (minus 55U/G)": "#639FD3",
    "correct composition": "#F9BD77",
    # "failed prediction": "#F9BD77",
    "failed prediction": "#E54A26",
    "wrong length": "#990000",
    "no prediction": "#808285",
}

STATUS_ORDER = [
    "identical",
    "identical (minus 55U/G)",
    "correct composition",
    "failed prediction",
    "wrong length",
    "no prediction",
]

LEGEND_PARAMS = {
    "padding": 10,
    "strokeColor": "black",
    "cornerRadius": 5,
    "fillColor": "white",
}


def plot_evaluation(options: EvaluationPlotOptions) -> None:
    results = pl.read_csv(options.input, separator="\t")

    donut_chart = create_donut_plot(data=results)
    donut_chart.save(options.donut_path)

    layer = create_stacked_barplot(
        data=results,
        param=options.evaluation_criterion
        if options.evaluation_criterion != "default"
        else "",
    )

    layer |= create_heatmap(
        data=results,
        param=options.evaluation_criterion
        if options.evaluation_criterion != "default"
        else "true_len",
    )

    layer.save(options.bar_path)


def create_donut_plot(data: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_arc(innerRadius=32, outerRadius=50)
        .encode(
            theta=alt.Theta("count(result):Q", sort=STATUS_ORDER),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=STATUS_ORDER,
                    range=[
                        STATUS_COLORS[stat] if STATUS_COLORS.get(stat) else stat
                        for stat in STATUS_ORDER
                    ],
                ),
            ),
            order=alt.Order("order:O", sort="descending"),
            tooltip=["result", "count(result)"],
        )
    )


def create_stacked_barplot(data: pl.DataFrame, param: str) -> alt.Chart:
    """Create stacked barplot over prediction status.

    Parameters
    ----------
    data : polars.Dataframe
        Dataframe containing prediction status data.
    param: str
        Column name for studied parameter.

    Returns
    -------
    altair.Chart
        Stacked barplot over prediction status.

    """
    chart = (
        alt.Chart(
            data,  # title="Status Assessment"
        )
        .mark_bar()
        .encode(
            x=select_x_axis(param=param),
            y=alt.Y("count(result):Q", sort=STATUS_ORDER, title="Number of sequences"),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=STATUS_ORDER,
                    range=[
                        STATUS_COLORS[stat] if STATUS_COLORS.get(stat) else stat
                        for stat in STATUS_ORDER
                    ],
                ),
                legend=alt.Legend(
                    **LEGEND_PARAMS,
                    orient="left",
                    title="",  # title="Prediction status"
                ),
                sort=STATUS_ORDER,
            ),
            order=alt.Order("order:O", sort="descending"),
            tooltip=["result", "count(result)"],
        )
    )
    return chart


def create_heatmap(data: pl.DataFrame, param: str) -> alt.Chart:
    """Create heatmaps over given column.

    Parameters
    ----------
    data : polars.Dataframe
        Dataframe containing prediction data.
    param : str
        Column name of chosen parameter.

    Returns
    -------
    altair.Chart
        Heatmap layer over chosen column.

    """
    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            alt.X("true_sequence:N", title="Sequence"),
            alt.Y(f"{param}:N", title=param, sort="descending"),
            alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=STATUS_ORDER,
                    range=[
                        STATUS_COLORS[stat] if STATUS_COLORS.get(stat) else stat
                        for stat in STATUS_ORDER
                    ],
                ),
            ),
            tooltip=["true_sequence", "pred_sequence", param, "result"],
        )
    )

    return heatmap


def select_x_axis(param: str):
    match param:
        case "modification_rate":
            return alt.X("modification_rate:N", title="Modification rate")
        case "num_replicates":
            return alt.X("num_replicates:N", title="Number of sequence replicates")
        case "max_singletons":
            return alt.X(
                "max_singletons:N", title="Maximum number of false positive singletons"
            )
        case "phantom_rate":
            return alt.X("phantom_rate:N", title="Phantom rate")
        case "noise_rate":
            return alt.X("noise_rate:N", title="Noise rate")
        case "intensity_cutoff":
            return alt.X("intensity_cutoff:N", title="Intensity cutoff percentile")
        case "lp_timeout_long":
            return alt.X("lp_timeout_long:N", title="Timeout (full LP) ")
        case "lp_timeout_short":
            return alt.X("lp_timeout_short:N", title="Timeout (reduced LP) ")
        case _:
            return alt.X("true_len:N", title="Sequence length")
