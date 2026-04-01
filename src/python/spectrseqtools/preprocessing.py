import numpy as np
import polars as pl
import tqdm
import yaml

from spectrseqtools.common import set_output_path, initialize_raw_file_iterator
from spectrseqtools.deconvolution import DeconvolutionParameters, DeisotopedPeakList
from spectrseqtools.singleton_identification import identify_singletons


MIN_MS1_CHARGE_STATE = 3


class Preprocessor:
    """Class for preprocessing of raw MS data."""

    def __init__(self, options) -> None:
        self.deconvolution_params = {}
        self.input_path = options.input
        self.output_dir, self.output_id = set_output_path(
            input_path=options.input, output_dir=options.output_dir
        )
        self.output_prefix = self.output_dir / self.output_id
        with open(options.meta, "r", encoding="utf-8") as f:
            self.meta_params = yaml.safe_load(f)
        self.cutoff_percentile = options.cutoff_percentile

    def preprocess(self) -> None:
        """
        Deconvolute MS2 scans and identify singletons.

        Main pipeline for deconvoluting MS2 scans and generating the metafile
        required for a SpectrSeqTools prediction as well as a list of candidate
        nucleotides from singletons (if desired).

        """
        print("RAW file found. Preprocessing raw data...")

        # Deconvolute raw data from file
        fragments = self.deconvolute(file_path=str(self.input_path))

        # Update meta parameters (if needed)
        meta_params = self.meta_params
        meta_params.setdefault("identity", self.output_id)
        meta_params.setdefault("intact_mass", self.select_intact_mass(fragments))
        meta_params.setdefault("true_sequence", None)

        # Set intensity cutoff
        meta_params["intensity_cutoff"] = (
            determine_intensity_percentiles(fragments)
            .filter(pl.col("statistic") == f"{self.cutoff_percentile}%")["value"]
            .to_list()[0]
        )

        # Save updated meta data
        with open(
            f"{self.output_prefix}.preprocessed.meta.yaml", "w", encoding="utf-8"
        ) as f:
            yaml.dump(meta_params, f)

        # Save preprocessed fragments
        fragments.write_csv(f"{self.output_prefix}.tsv", separator="\t")

        # Identify singletons
        singletons = identify_singletons(file_path=str(self.input_path))

        # Save singletons detected from raw data
        singletons.write_csv(f"{self.output_prefix}.singletons.tsv", separator="\t")

        print("Preprocessing completed!\n")

    def deconvolute(self, file_path: str) -> pl.DataFrame:
        """
        Deconvolute/deisotope peaks in MS2 scans from ThermoFisher RAW file.

        Parameters
        ----------
        file_path : str
            Path of RAW file from ThermoFisher.

        Returns
        -------
        polars.DataFrame
            Dataframe containing fragment monoisotopic masses and intensities.

        """
        # Load deconvolution parameter based on parameter dict
        params = DeconvolutionParameters(self.deconvolution_params)

        # Initialize iterator for RAW file
        raw_file_read = initialize_raw_file_iterator(file_path=file_path)

        peak_list = DeisotopedPeakList.default()
        for _ in tqdm.tqdm(range(len(raw_file_read) - 1), desc="Deisotoping MS2 scans"):
            # Select next scan
            scan = next(raw_file_read)

            # Skip scan if it is no MS2 scan
            if scan.ms_level != 2:
                continue

            # Skip scan if the precursor charge is lower than MIN_MS1_CHARGE_STATE
            if (
                not isinstance(scan.precursor_information.charge, int)
                or scan.precursor_information.charge < MIN_MS1_CHARGE_STATE
            ):
                continue

            # Deconvolute scan to get list of deisotoped peaks
            peak_list += DeisotopedPeakList.from_scan(scan=scan, params=params)

        return peak_list.to_fragments()

    def select_intact_mass(self, fragments: pl.DataFrame) -> float:
        """
        Select intact sequence mass from deconvoluted fragments.

        Determine the aggregated neutral_mass with (1) a deisotoped precursor and
        (2) the largest aggregated intensity as estimated intact sequence mass.

        Parameters
        ----------
        fragments : polars.DataFrame
            Dataframe containing deconvoluted fragments.

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
                    > (
                        self.meta_params["5_prime_tag"]
                        + self.meta_params["3_prime_tag"]
                    )
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
