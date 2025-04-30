from typing import List
from lionelmssq.prediction import Prediction
from lionelmssq.common import parse_nucleosides
import polars as pl
import altair as alt


def plot_prediction(
    prediction: Prediction,
    true_sequence: List[str],
    simulation: pl.DataFrame = None,
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

    fragment_predictions = prediction.fragments.select(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.struct(["left", "right"])
        .map_elements(
            lambda x: create_range(x["left"], x["right"]),
            return_dtype=pl.List(pl.Int64),
        )
        .alias("range"),
        pl.map_batches(
            ["observed_mass", "predicted_mass_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.col("predicted_fragment_seq")
        # .map_elements(reject_none, return_dtype=pl.List(pl.Utf8))
        .alias("fragment_seq"),
        pl.lit("predicted").alias("type"),
    ).with_row_index()

    if simulation is not None:
        simulation = simulation.select(
            pl.col("left") - 0.5,
            pl.col("right") - 0.5,
            pl.struct(["left", "right"])
            .map_elements(
                lambda x: create_range(x["left"], x["right"]),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("range"),
            pl.col("true_mass")
            .map_elements(lambda mass: f"{mass:.2f}", return_dtype=pl.Utf8)
            .alias("mass_info"),
            # pl.col("sequence").alias("fragment_seq"),
            pl.col("sequence")
            .map_elements(parse_nucleosides, return_dtype=pl.List(pl.Utf8))
            .alias("fragment_seq"),
            pl.lit("truth").alias("type"),
        ).with_row_index()

        data = pl.concat([fragment_predictions, simulation])

    else:
        data = fragment_predictions

    # new = data.with_columns(pl.col("range").map_elements(lambda x: len(x)).alias("len_range")).with_columns(pl.col("fragment_seq").map_elements(lambda x: len(x)).alias("len_fragment_seq"))
    # with pl.Config(tbl_rows=-1):
    #    print(new)

    data_seq = data.filter(pl.col("fragment_seq").list.len() > 0).explode(
        ["fragment_seq", "range"]
    )
    # Remove the rows with empty sets for fragment_seq! This may happen when the LP_relaxation_threshold is too high and because of the LP relaxation, the pribability is low!

    def facet_plots(df_mass, df_seq, index):
        p1 = (
            alt.Chart(df_mass)
            .mark_rule()
            .encode(
                alt.X("left").axis(labels=False, ticks=False),
                alt.X2("right"),
                alt.Y("type"),  # .title("fragment"),
            )
        )
        p2 = (
            alt.Chart(df_mass)
            .mark_text(align="left", dx=3)
            .encode(
                alt.X("right").axis(labels=False, ticks=False),
                alt.Y("type"),
                alt.Text("mass_info"),
            )
        )

        p3 = (
            alt.Chart(df_seq)
            .mark_text()
            .encode(
                alt.X("range").title(None),
                alt.Y("type").title(str(index)),
                alt.Text("fragment_seq"),
                alt.Color("fragment_seq", scale=alt.Scale(scheme="category10")).legend(
                    None
                ),
            )
        )

        return alt.layer(p1 + p2 + p3)

    p_final_seq = (
        alt.Chart(seq_data)
        .mark_text()
        .encode(
            alt.X("pos").title(None),
            alt.Y("type").title("Final sequence"),
            alt.Text("nucleoside"),
            alt.Color("nucleoside", scale=alt.Scale(scheme="category10")).legend(None),
        )
    )

    layered_plots = alt.vconcat(
        *[
            facet_plots(
                data.filter(pl.col("index") == i),
                data_seq.filter(pl.col("index") == i),
                i,
            )
            for i in range(max(data["index"]) + 1)
        ],
        p_final_seq,
        title=alt.TitleParams(
            text="fragments", anchor="middle", orient="left", angle=-90, align="center"
        ),
    ).resolve_scale(x="shared")

    return layered_plots
