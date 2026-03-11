import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple

from spectrseqtools.deconvolution import deconvolute
from spectrseqtools.singleton_identification import identify_singletons


def preprocess(
    file_path: Path,
    deconvolution_params: dict,
    meta_params: dict,
    cutoff_percentile: int = 50,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Deconvolute MS2 scans and identify singletons.

    Main pipeline for deconvoluting MS2 scans and generating the metafile
    required for a SpectrSeqTools prediction as well as a list of candidate
    nucleotides from singletons (if desired).

    Parameters
    ----------
    file_path : Path
        Path of RAW file from ThermoFisher.
    deconvolution_params : dict
        Dictionary with parameters for deconvolution.
    meta_params : dict
        Dictionary with meta parameters.
    cutoff_percentile: int
        Intensity percentile used as cutoff. Default: 50.

    Returns
    -------
    fragments : polars.DataFrame
        Dataframe containing deconvoluted fragments.
    singletons : polars.DataFrame
        Dataframe containing singleton data.
    meta_params : dict
        Dictionary with updated meta parameters.

    """
    # Deconvolute raw data from file
    fragments = deconvolute(
        file_path=str(file_path),
        params=deconvolution_params,
    )

    # Identify singletons
    singletons = identify_singletons(file_path=str(file_path))

    # Update meta parameters (if needed)
    meta_params.setdefault("identity", file_path.stem)
    meta_params.setdefault(
        "sequence_mass", select_sequence_mass(fragments, meta_params)
    )
    meta_params.setdefault("true_sequence", None)

    # Set intensity cutoff
    meta_params["intensity_cutoff"] = (
        determine_intensity_percentiles(fragments)
        .filter(pl.col("statistic") == f"{cutoff_percentile}%")["value"]
        .to_list()[0]
    )

    return fragments, singletons, meta_params


def select_sequence_mass(
    fragments: pl.DataFrame,
    meta_params: dict,
) -> float:
    """
    Select intact sequence mass from deconvoluted fragments.

    Determine the aggregated neutral_mass with (1) a deisotoped precursor and
    (2) the largest aggregated intensity as estimated intact sequence mass.

    Parameters
    ----------
    fragments : polars.DataFrame
        Dataframe containing deconvoluted fragments.
    meta_params : dict
        Dictionary with meta parameters.

    Returns
    -------
    float
        Sequence mass estimation.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """

    return (
        fragments.filter(
            (pl.col("is_precursor_deisotoped"))
            & (
                pl.col("neutral_mass")
                > (meta_params["label_mass_5T"] + meta_params["label_mass_3T"])
            )
        )
        .filter((pl.col("intensity") == pl.col("intensity").max()))["neutral_mass"]
        .to_list()[0]
    )


def determine_intensity_percentiles(
    fragments: pl.DataFrame,
) -> pl.DataFrame:
    """
    Determine percentile values for intensities in given dataframe.

    Parameters
    ----------
    fragments : polars.DataFrame
        Dataframe containing deconvoluted fragments.

    Returns
    -------
    polars.DataFrame
        Dataframe containing intensity percentile values.
    """
    return fragments.get_column("intensity").describe(
        percentiles=np.linspace(0, 0.95, 20),
        interpolation="midpoint",
    )
