import numpy as np
import polars as pl
import yaml
from pathlib import Path

from spectrseqtools.common import set_output_path
from spectrseqtools.deconvolution import deconvolute
from spectrseqtools.singleton_identification import identify_singletons


def preprocess(options) -> None:
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
    print("RAW file found. Preprocessing raw data...")

    # Deconvolute raw data from file
    deconvolution_params = {}
    fragments = deconvolute(
        file_path=str(options.input),
        params=deconvolution_params,
    )

    output_dir, output_prefix = set_output_path(input_path=options.input,
                                         output_dir=options.output_dir)

    # Update meta parameters (if needed)
    meta_params = {}
    with open(options.meta, "r") as f:
        meta_params = yaml.safe_load(f)
    meta_params.setdefault("identity", output_prefix)
    meta_params.setdefault("intact_mass", select_intact_mass(fragments, meta_params))
    meta_params.setdefault("true_sequence", None)

    # Set intensity cutoff
    meta_params["intensity_cutoff"] = (
        determine_intensity_percentiles(fragments)
        .filter(pl.col("statistic") == f"{options.cutoff_percentile}%")[
            "value"]
        .to_list()[0]
    )

    # Save updated meta data
    with open(output_dir / f"{output_prefix}.preprocessed.meta.yaml", "w") as f:
        yaml.dump(meta_params, f)

    # Save preprocessed fragments
    fragments.write_csv(output_dir / f"{output_prefix}.tsv", separator="\t")

    # Identify singletons
    singletons = identify_singletons(file_path=str(options.input))

    # Save singletons detected from raw data
    singletons.write_csv(
        output_dir / f"{output_prefix}.singletons.tsv", separator="\t"
    )

    print("Preprocessing completed!\n")


def select_intact_mass(
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
                > (meta_params["5_prime_tag"] + meta_params["3_prime_tag"])
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
