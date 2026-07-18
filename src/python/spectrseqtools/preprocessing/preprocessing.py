# -*- coding: utf-8 -*-
"""Preprocessing of raw mass spectrometry data."""

import importlib.resources
from typing import List, Tuple

import ms_deisotope as ms_ditp
import polars as pl
import tqdm
import yaml
from clr_loader import get_mono

from spectrseqtools.enums import AveragineBackbone
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.file_settings import PreprocessingFileSettings
from spectrseqtools.parsers import PreprocessingOptions
from spectrseqtools.preprocessing.deconvolution import (
    DeisotopedPeakList,
    MS1Deconvoluter,
    MS2Deconvoluter,
)
from spectrseqtools.preprocessing.singleton_identification import (
    RawPeakList,
    SingletonBoundaries,
)

rt = get_mono()


class Preprocessor:
    """Class for preprocessing of raw MS data."""

    def __init__(self, options: PreprocessingOptions) -> None:
        self.file_settings = PreprocessingFileSettings(
            input_path=options.input,
            meta_path=options.meta,
            alphabet_path=options.alphabet,
            output_dir=options.output_dir,
        )
        self.error = ErrorCalculator.with_metric(
            tolerance=options.tolerance,
            decimal_places=options.num_decimal_places,
        )
        self.singleton_boundaries = SingletonBoundaries.from_alphabet_file(
            input_path=self.file_settings.alphabet_path,
            boundary_factor=options.boundary_factor,
            error=self.error,
        )
        self.min_precursor_charge = options.min_precursor_charge
        self.intact_mass_cutoff_factor = options.intact_mass_cutoff_factor

        with open(self.file_settings.meta_path, "r", encoding="utf-8") as f:
            self.meta_params = yaml.safe_load(f)

        averagine = ms_ditp.Averagine(
            base_composition=set_averagine(backbone=options.averagine_backbone)
        )
        self.ms1_deconvoluter = MS1Deconvoluter(
            minimum_intensity=options.min_intensity,
            averagine=averagine,
            max_missed_peaks=options.max_missed_peaks,
            scale_method=options.scale_method,
            error_tolerance=options.peak_error_tol,
            scorer=ms_ditp.PenalizedMSDeconVFitter(
                minimum_score=options.envelope_min_score,
                mass_error_tolerance=options.envelope_error_tol,
            ),
            charge_range=options.ms1_charge_range,
            truncate_after=options.ms1_truncate_after,
        )
        self.ms2_deconvoluter = MS2Deconvoluter(
            minimum_intensity=options.min_intensity,
            averagine=averagine,
            max_missed_peaks=options.max_missed_peaks,
            scale_method=options.scale_method,
            error_tolerance=options.peak_error_tol,
            scorer=ms_ditp.MSDeconVFitter(
                minimum_score=options.envelope_min_score,
                mass_error_tolerance=options.envelope_error_tol,
            ),
            charge_range=options.ms2_charge_range,
            truncate_after=options.ms2_truncate_after,
            isotopic_shift_factor=options.isotopic_shift_factor,
        )

    def preprocess(self) -> None:
        """
        Deconvolute MS1 and MS2 scans, identify singletons, and update metadata.

        Main pipeline for deconvoluting MS1 and MS2 scans and generating the metafile
        required for a SpectrSeqTools prediction as well as a list of candidate
        nucleotides from singletons.

        """
        print("RAW file found. Preprocessing raw data...")

        # Deconvolute raw data from file
        ms1_fragments, ms2_fragments = self.deconvolute()

        # Update meta parameters (if needed)
        meta_params = self.meta_params
        meta_params.setdefault("identity", self.file_settings.file_prefix)
        meta_params.setdefault("intact_mass", self.select_intact_mass(ms1_fragments))
        meta_params.setdefault("true_sequence", None)

        # Save updated meta data
        with open(self.file_settings.updated_meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta_params, f)

        # Save preprocessed fragments
        ms2_fragments.write_csv(self.file_settings.fragment_path, separator="\t")

        # Save singletons detected from raw data as new nucleotide alphabet
        singletons = self.identify_singletons()
        singletons.write_csv(self.file_settings.updated_alphabet_path, separator="\t")

        print("Preprocessing completed!\n")

    def deconvolute(self) -> pl.DataFrame:
        """
        Deconvolute/deisotope peaks in MS1 and MS2 scans from ThermoFisher RAW file.

        Returns
        -------
        polars.DataFrame
            Dataframe containing fragment information.

        """
        # Initialize iterator for RAW file
        raw_file_read = initialize_raw_file_iterator(
            file_path=str(self.file_settings.input_path)
        )

        # Precount number of scan bunches and reset raw_file_read after
        num_scan_bunches = sum(scan.ms_level == 1 for scan in raw_file_read)
        raw_file_read.reset()
        raw_file_read.make_iterator(grouped=True)

        scan_processor = ms_ditp.ScanProcessor(raw_file_read)

        ms1_peak_list = DeisotopedPeakList.default()
        ms2_peak_list = DeisotopedPeakList.default()

        for scan_bunch in tqdm.tqdm(
            raw_file_read, desc="Deisotoping scan bunches", total=num_scan_bunches
        ):
            # Obtain MS1, MS2 and priority peaks
            # priority_list is a list of ms_deisotope 'PriorityTarget' objects,
            # which is similar to a 'Peak' object
            ms1_scan, priority_list, ms2_scans = scan_processor.process_scan_group(
                scan_bunch.precursor, scan_bunch.products
            )

            # Filter scans by requiring sufficient charge
            priority_list, ms2_scans = self.filter_ms2_scans_by_charge(
                ms2_scans=ms2_scans, priority_list=priority_list
            )

            # Deconvolute MS1 scan to get list of deisotoped peaks
            ms1_peak_list += self.ms1_deconvoluter.deconvolute_scan(
                scan=ms1_scan,
                priority_list=priority_list,
            )

            # Deconvolute MS2 scan to get list of deisotoped peaks
            for ms2_scan in ms2_scans:
                ms2_peak_list += self.ms2_deconvoluter.deconvolute_scan(
                    scan=ms2_scan,
                )
        ms1_fragments = ms1_peak_list.to_fragments(tolerance=self.error.tolerance)
        ms2_fragments = ms2_peak_list.to_fragments(tolerance=self.error.tolerance)

        return ms1_fragments, ms2_fragments

    def filter_ms2_scans_by_charge(
        self,
        priority_list: List[ms_ditp.processor.PriorityTarget],
        ms2_scans: List[ms_ditp.data_source.Scan],
    ) -> Tuple[List[ms_ditp.processor.PriorityTarget], List[ms_ditp.data_source.Scan]]:
        """
        Filter out MS2 scans (and their priority targets) if charge insufficient.

        Parameters
        ----------
        priority_list : List[ms_ditp.processor.PriorityTarget] | None
            List of priority targets for deconvolution.
        ms2_scans : List[ms_deisotope.data_source.Scan]
            List of ThermoFisher MS2 scan.

        Returns
        -------
        List[ms_ditp.processor.PriorityTarget] | None
            Updated list of priority targets for deconvolution.
        List[ms_deisotope.data_source.Scan]
            Updated list of ThermoFisher MS2 scan.

        """
        for target, scan in zip(priority_list, ms2_scans):
            if abs(target.mz - scan.precursor_information.mz) > 10e-3:
                print(target.mz, scan.precursor_information.mz)
                raise ValueError("Order not correct.")

        # Check whether target charge is too low for consideration (if given)
        target_mask = [
            isinstance(peak.charge, int) and peak.charge >= self.min_precursor_charge
            for peak in priority_list
        ]

        # Check whether precursor charge is too low for consideration (if given)
        scan_mask = [
            isinstance(peak.precursor_information.charge, int)
            and peak.precursor_information.charge >= self.min_precursor_charge
            for peak in ms2_scans
        ]

        # Return entries for which charges of both target and scan are sufficient
        mask = [target_mask[idx] and scan_mask[idx] for idx in range(len(ms2_scans))]
        return (
            [priority_list[idx] for idx in range(len(priority_list)) if mask[idx]],
            [ms2_scans[idx] for idx in range(len(ms2_scans)) if mask[idx]],
        )

    def collect_raw_peaks(self) -> RawPeakList:
        """
        Collect raw peaks from MS2 scans in ThermoFisher RAW file.

        Returns
        -------
        peak_list : RawPeakList
            List of raw peaks.

        """
        # Initialize iterator for RAW file
        raw_file_read = initialize_raw_file_iterator(
            file_path=str(self.file_settings.input_path)
        )

        peak_list = RawPeakList.default()
        for _ in tqdm.tqdm(
            range(len(raw_file_read) - 1), desc="Extract m/z data from MS2 scans"
        ):
            # Select next scan
            scan = next(raw_file_read)

            # Skip scan if it is no MS2 scan
            if scan.ms_level != 2:
                continue

            # Extract raw peaks from scan (without deisotoping)
            peak_list += RawPeakList.from_scan(
                scan=scan, boundaries=self.singleton_boundaries
            )
        return peak_list

    def identify_singletons(self) -> pl.DataFrame:
        """
        Determine singleton candidates from MS2 scans in ThermoFisher RAW file.

        Returns
        -------
        polars.DataFrame
            Dataframe containing singleton candidates.

        """
        peak_list = self.collect_raw_peaks()

        return peak_list.to_singletons(
            alphabet_path=self.file_settings.alphabet_path,
            error=self.error,
        )

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
                    self.meta_params["5_prime_tag"] + self.meta_params["3_prime_tag"]
                    <= pl.col("neutral_mass")
                )
                & (
                    self.intact_mass_cutoff_factor * pl.col("neutral_mass").max()
                    <= pl.col("neutral_mass")
                )
            )
            .filter((pl.col("intensity") == pl.col("intensity").max()))["neutral_mass"]
            .to_list()[0]
        )


def initialize_raw_file_iterator(
    file_path: str,
) -> ms_ditp.data_source.thermo_raw_net.ThermoRawLoader:
    """
    Initialize iterator over scans in ThermoFisher RAW file format.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    raw_file : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Iterator over scans from RAW file.

    """
    # Read data from file
    raw_file = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    # Initialize an iterator while ungrouping MS1 from MS2 scans
    raw_file.make_iterator(grouped=False)

    return raw_file


def set_averagine(backbone: AveragineBackbone) -> dict:
    """
    Calculate the average elemental composition of RNA.

    Parameters
    ----------
    backbone : AveragineBackbone
        Backbone considered for the composition.

    Returns
    -------
    average_composition : dict
        Dictionary containing average elemental composition.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """
    # Build dict with elemental compositions from file
    bases = pl.read_csv(
        importlib.resources.files(__package__)
        / "../assets"
        / "elemental_composition.tsv",
        separator="\t",
    )
    base_compositions = [
        {
            col: row[bases.get_column_index(col)]
            for col in bases.columns
            if col != "base"
        }
        for row in bases.iter_rows()
    ]

    # Calculate average elemental composition
    average_composition = {}
    for element in base_compositions[0]:
        average_composition[element] = sum(
            float(base[element]) for base in base_compositions
        ) / len(base_compositions)

    # Add backbone elements (if needed)
    match backbone:
        case AveragineBackbone.NONE:
            pass
        case AveragineBackbone.PHOSPHATE:
            # Add 1 phosphorus and 2 oxygen for the phosphate group
            average_composition["O"] += 2
            average_composition["P"] += 1
        case AveragineBackbone.THIOPHOSPHATE:
            # Add 1 phosphorus, 1 sulfur, and 1 oxygen for the phosphate group
            average_composition["O"] += 1
            average_composition["S"] += 1
            average_composition["P"] += 1
        case _:
            raise NotImplementedError(
                f"Support for '{backbone}' is currently not given."
            )

    return average_composition
