# -*- coding: utf-8 -*-
"""Plotting of preprocessing and prediction results."""

from typing import List

import altair as alt

LEGEND_PARAMS = {
    "padding": 10,
    "strokeColor": "black",
    "cornerRadius": 5,
    "fillColor": "white",
}
HISTOGRAM_WIDTH = 850


def select_scale(order: List[str], colors: dict) -> alt.Scale:
    """Select scale for colors in Altair plot.

    Parameters
    ----------
    order : List[str]
        Status order.
    colors : dict
        Dictionary assigning a color ot each status.

    Returns
    -------
    alt.Scale
        Scale for Altair plot.

    """
    return alt.Scale(
        domain=order,
        range=[colors[stat] if colors.get(stat) else stat for stat in order],
    )
