from typing import List
from lionelmssq.prediction import Prediction
from lionelmssq.common import parse_nucleosides
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

    def reject_none(str_list):
        # return ''.join([nuc for nuc in str_list if nuc is not None])
        return [nuc for nuc in str_list if nuc is not None]

    def create_range(left, right):
        return list(range(left, right))

    fragment_predictions = prediction.fragments.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.struct(["left", "right"])
        .map_elements(lambda x: create_range(x["left"], x["right"]))
        .alias("range"),
        pl.map_batches(
            ["observed_mass", "predicted_mass_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.col("predicted_fragment_seq")
        .map_elements(reject_none)
        .alias("fragment_seq"),
        pl.lit("predicted").alias("type"),
    ).with_row_index()

    simulation = simulation.select(
        pl.col("left") - 0.5,
        pl.col("right") + 0.5,
        pl.struct(["left", "right"])
        .map_elements(lambda x: create_range(x["left"], x["right"] + 1))
        .alias("range"),
        pl.col("true_mass").map_elements(lambda mass: f"{mass:.2f}").alias("mass_info"),
        # pl.col("sequence").alias("fragment_seq"),
        pl.col("sequence").map_elements(parse_nucleosides).alias("fragment_seq"),
        pl.lit("truth").alias("type"),
    ).with_row_index()

    data = pl.concat([fragment_predictions, simulation])

    data_seq = data.explode(["fragment_seq", "range"])

    # base = alt.Chart(data_seq)

    p1 = (
        alt.Chart()
        .mark_rule()
        .encode(
            alt.X("left").axis(title=None, labels=False, ticks=False),
            alt.X2("right"),
            alt.Y("type").title(None),
        )
    )
    p2 = (
        alt.Chart()
        .mark_text(align="left", dx=3)
        .encode(
            alt.X("right").axis(title=None, labels=False, ticks=False),
            alt.Y("type").title(None),
            alt.Text("mass_info"),
        )
    )

    p3 = (
        alt.Chart()
        .mark_text()
        .encode(
            alt.X("range").title(None),
            alt.Y("type").title(None),
            alt.Text("fragment_seq"),
            alt.Color("fragment_seq", scale=alt.Scale(scheme="category10")).legend(
                None
            ),
        )
    )
    p4 = (
        alt.Chart(seq_data)
        .mark_text()
        .encode(
            alt.X("pos").title(None),
            alt.Y("type").title(None),
            alt.Text("nucleoside"),
            alt.Color("nucleoside", scale=alt.Scale(scheme="category10")).legend(None),
        )
    )

    return (
        alt.layer(p1, p2, p3, data=data_seq).facet(
            row=alt.Facet("index").title("fragments")
        )
        & p4
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

    def create_range(left, right):
        return list(range(left, right))

    def reject_none(str_list):
        # return ''.join([nuc for nuc in str_list if nuc is not None])
        return [nuc for nuc in str_list if nuc is not None]

    fragment_predictions = prediction.fragments.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.struct(["left", "right"])
        .map_elements(lambda x: create_range(x["left"], x["right"]))
        .alias("range"),
        pl.map_batches(
            ["observed_mass", "predicted_mass_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.col("predicted_fragment_seq")
        .map_elements(reject_none)
        .alias("fragment_seq"),
        pl.lit("predicted").alias("type"),
    ).with_row_index()

    data = fragment_predictions.explode(["fragment_seq", "range"])

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
            + alt.Chart()
            .mark_text()
            .encode(
                alt.X("range").title(None),
                alt.Y("type").title(None),
                alt.Text("fragment_seq"),
                alt.Color("fragment_seq", scale=alt.Scale(scheme="category10")).legend(
                    None
                ),
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
