from typing import List
from lionelmssq.prediction import Prediction
import polars as pl
import altair as alt


def plot_prediction_with_truth(
    prediction: Prediction,
    true_sequence: List[str],
    simulation: pl.DataFrame,
) -> alt.Chart:
    seq_data = pl.DataFrame(
        {
            "nucleoside": true_sequence + prediction.sequence,
            "pos": list(range(len(true_sequence)))
            + list(range(len(prediction.sequence))),
            "type": ["truth"] * len(true_sequence)
            + ["predicted"] * len(prediction.sequence),
        }
    )

    def fmt_mass(cols):
        return pl.Series([f"{row[0]:.2f} ({row[1]:.2f})" for row in zip(*cols)])

    fragment_predictions = prediction.fragments.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.map_batches(
            ["observed_mass", "predicted_mass_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.lit("predicted").alias("type"),
    ).with_row_index()

    simulation = simulation.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.col("true_mass").map_elements(lambda mass: f"{mass:.2f}").alias("mass_info"),
        pl.lit("truth").alias("type"),
    ).with_row_index()

    data = pl.concat([fragment_predictions, simulation])

    base = alt.Chart(data)
    return (
        (
            base.mark_rule().encode(
                alt.X("left").axis(title=None, labels=False, ticks=False),
                alt.X2("right"),
                alt.Y("type").title(None),
            )
            + base.mark_text(align="left", dx=3).encode(
                alt.X("right").axis(title=None, labels=False, ticks=False),
                alt.Y("type").title(None),
                alt.Text("mass_info"),
            )
        ).facet(row=alt.Facet("index").title("fragments"))
        & alt.Chart(seq_data)
        .mark_text()
        .encode(
            alt.X("pos").title(None),
            alt.Y("type").title(None),
            alt.Text("nucleoside"),
            alt.Color("nucleoside", scale=alt.Scale(scheme="category10")).legend(None),
        )
    ).resolve_scale(x="shared")


def plot_prediction(
    prediction: Prediction,
    true_sequence: List[str],
    experiment: pl.DataFrame,
) -> alt.Chart:
    seq_data = pl.DataFrame(
        {
            "nucleoside": true_sequence + prediction.sequence,
            "pos": list(range(len(true_sequence)))
            + list(range(len(prediction.sequence))),
            "type": ["truth"] * len(true_sequence)
            + ["predicted"] * len(prediction.sequence),
        }
    )

    def fmt_mass(cols):
        return pl.Series([f"{row[0]:.2f} ({row[1]:.2f})" for row in zip(*cols)])

    fragment_predictions = prediction.fragments.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.map_batches(
            ["observed_mass", "predicted_mass_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.lit("predicted").alias("type"),
    ).with_row_index()

    # experiment = experiment.select(
    #    pl.col("left") - 0.5,
    #    pl.col("right") - 1 + 0.5,
    #    pl.col("true_mass").map_elements(lambda mass: f"{mass:.2f}").alias("mass_info"),
    #    pl.lit("truth").alias("type"),
    # ).with_row_index()

    # data = pl.concat([fragment_predictions, experiment])
    data = fragment_predictions

    base = alt.Chart(data)
    return (
        (
            base.mark_rule().encode(
                alt.X("left").axis(title=None, labels=False, ticks=False),
                alt.X2("right"),
                alt.Y("type").title(None),
            )
            + base.mark_text(align="left", dx=3).encode(
                alt.X("right").axis(title=None, labels=False, ticks=False),
                alt.Y("type").title(None),
                alt.Text("mass_info"),
            )
        ).facet(row=alt.Facet("index").title("fragments"))
        & alt.Chart(seq_data)
        .mark_text()
        .encode(
            alt.X("pos").title(None),
            alt.Y("type").title(None),
            alt.Text("nucleoside"),
            alt.Color("nucleoside", scale=alt.Scale(scheme="category10")).legend(None),
        )
    ).resolve_scale(x="shared")
