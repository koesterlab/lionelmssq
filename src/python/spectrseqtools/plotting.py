from typing import List

import altair as alt
import polars as pl

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.prediction.prediction import Prediction

STATUS_COLORS = {
    "False": "#808285",
    "True": "black",
}


def encode_seq(seq: str) -> List[str]:
    """Format sequence to use nucleotide encoding."""
    seq = Sequence.from_str(input_seq=seq)
    return seq.to_encoding()


def plot_prediction(
    prediction: Prediction,
    true_seq: Sequence,
    simulation: pl.DataFrame = None,
) -> alt.Chart:
    true_seq = true_seq.to_encoding()
    pred_seq = prediction.sequence.to_encoding()
    seq_data = pl.DataFrame(
        {
            "nuc": true_seq + pred_seq,
            "pos": list(range(len(true_seq))) + list(range(len(pred_seq))),
            "type": ["truth"] * len(true_seq) + ["predicted"] * len(pred_seq),
        }
    )

    def fmt_mass(cols):
        return pl.Series([f"{row[0]:.2f} ({row[1]:.2f})" for row in zip(*cols)])

    def fmt_ppm(cols):
        return pl.Series(
            [f"{row[0]:.2f} ({row[1]:.2f} = {row[2]:.3f} ppm)" for row in zip(*cols)]
        )

    def create_range(left, right):
        return list(range(left, right))

    fragment_predictions = prediction.fragments.fragments.with_columns(
        pl.col("observed_mass").round(2),
        pl.col("standard_unit_mass").round(2),
        pl.col("predicted_mass").round(2),
        pl.col("predicted_diff").round(2),
    )

    fragment_predictions = fragment_predictions.with_columns(
        pl.col("left") - 0.5,
        pl.col("right") - 1 + 0.5,
        pl.struct(["left", "right"])
        .map_elements(
            lambda x: create_range(x["left"], x["right"]),
            return_dtype=pl.List(pl.Int64),
        )
        .alias("range"),
        pl.map_batches(
            ["standard_unit_mass", "predicted_diff"],
            fmt_mass,
        ).alias("mass_info"),
        pl.struct(["observed_mass", "predicted_diff"])
        .map_elements(
            lambda x: (
                x["predicted_diff"] * 10**6 / (x["observed_mass"] - x["predicted_diff"])
            ),
            return_dtype=pl.Float64,
        )
        .alias("ppm_error"),
        pl.col("predicted_seq")
        .map_elements(encode_seq, return_dtype=pl.List(pl.Utf8))
        .alias("fragment_seq"),
        pl.lit("").cast(str).alias("type"),
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
            pl.col("true_mass_with_backbone")
            .map_elements(lambda mass: f"{mass:.2f}", return_dtype=pl.Utf8)
            .alias("mass_info"),
            pl.col("sequence")
            .map_elements(encode_seq, return_dtype=pl.List(pl.Utf8))
            .alias("fragment_seq"),
            pl.lit("truth").alias("type"),
        ).with_row_index()

        data = pl.concat([fragment_predictions, simulation])

    else:
        data = fragment_predictions

    data = data.with_columns(
        pl.when(pl.col("ppm_error").abs().lt(10))
        .then(pl.lit("True"))
        .otherwise(pl.lit("False"))
        .alias("within_tolerance"),
        pl.map_batches(
            ["observed_mass", "predicted_diff", "ppm_error"],
            fmt_ppm,
        ).alias("ppm_info"),
    )

    # new = data.with_columns(
    #     pl.col("range").map_elements(lambda x: len(x)).alias("len_range")
    # ).with_columns(
    #     pl.col("fragment_seq").map_elements(lambda x: len(x)).alias("len_fragment_seq")
    # )
    # with pl.Config(tbl_rows=-1):
    #     print(new)

    data_seq = data.filter(pl.col("fragment_seq").list.len() > 0).explode(
        ["fragment_seq", "range"]
    )
    # Remove the rows with empty sets for fragment_seq! This may happen when the
    # LP_relaxation_threshold is too high and because of the LP relaxation,
    # the probability is low!

    max_value = data_seq["right"].max()
    data = data.with_columns(pl.lit(2 * ((max_value + 2) // 2)).alias("max_value"))

    def facet_plots(df_mass, df_seq, index):
        p1 = (
            alt.Chart(df_mass)
            .mark_text(align="left", dx=5)
            .encode(
                x=alt.X("max_value").axis(labels=False, ticks=False),
                y=alt.Y("type", title=""),
                text=alt.Text("ppm_info"),
                color=alt.value(
                    STATUS_COLORS[df_mass.row(0, named=True)["within_tolerance"]]
                ),
            )
        )

        p2 = (
            alt.Chart(df_seq)
            .mark_text(fontWeight="bold")
            .encode(
                alt.X(
                    "range",
                    axis=alt.Axis(grid=False),
                    scale=alt.Scale(domain=[-0.5, max_value]),
                ).title(None),
                alt.Y("type").title(str(index)),
                alt.Text("fragment_seq"),
                alt.Color("fragment_seq", scale=alt.Scale(scheme="category10")).legend(
                    None
                ),
            )
        )

        return alt.layer(p1 + p2)

    p_final_seq = (
        alt.Chart(seq_data)
        .mark_text(fontWeight="bold")
        .encode(
            alt.X("pos", axis=alt.Axis(grid=False)).title(None),
            alt.Y("type").title("Final sequence"),
            alt.Text("nuc"),
            alt.Color("nuc", scale=alt.Scale(scheme="category10")).legend(None),
        )
    )

    def build_layer(df_data: pl.DataFrame) -> alt.Chart:
        return alt.vconcat(
            *[
                facet_plots(
                    df_data.filter(pl.col("orig_index") == i),
                    data_seq.filter(pl.col("orig_index") == i),
                    i,
                )
                for i in df_data["orig_index"].to_list()
            ],
            p_final_seq,
            title=alt.TitleParams(
                text="fragments",
                anchor="middle",
                orient="left",
                angle=-90,
                align="center",
            ),
        ).resolve_scale(x="shared")

    start_data = data.filter(pl.col("left") == -0.5)
    start_layer = build_layer(df_data=start_data)

    end_data = data.filter((pl.col("right") == max_value))
    end_layer = build_layer(df_data=end_data)

    internal_data = data.filter(
        (pl.col("right") != max_value) & (pl.col("left") != -0.5)
    )
    internal_layer = build_layer(df_data=internal_data)

    return start_layer, end_layer, internal_layer, build_layer(df_data=data)
