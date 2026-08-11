# -*- coding: utf-8 -*-
"""Simulation of random fragments."""

from ast import literal_eval
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import yaml

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.file_settings import load_alphabet
from spectrseqtools.parsers import FragmentSimulationOptions

PHANTOM_FRAGMENT_MAGNITUDE = 1000
NO_FRAGMENTATION_PROBABILITY = 0.05


def simulate_fragments(options: FragmentSimulationOptions) -> None:
    """Simulate fragments.

    Parameters
    ----------
    options : FragmentSimulationOptions
        Options for fragment simulation read by parser.

    """
    if options.output_dir is None:
        seed = 0
    else:
        with open(options.output_dir / "seed.txt", "r", encoding="utf-8") as seed_file:
            seed = int(seed_file.readline().rstrip("\n"))

    # Initialize random-number generator
    rng = np.random.default_rng(seed=seed)

    # Initialize config dict
    config = literal_eval(options.config)

    # Read metadata
    with open(options.input, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    true_sequence = Sequence.from_str(meta["true_sequence"]).sequence

    # Build dict with extra masses
    extra_mass_dict = build_extra_mass_dict(
        element_mass_path=options.elements,
        fragmentation_type=config["fragmentation_type"],
        start_tag=meta["5_prime_tag"],
        end_tag=meta["3_prime_tag"],
    )

    # Add sequence mass to meta dict
    nucleosides = load_alphabet()
    meta["intact_mass"] = (
        get_seq_weight(
            seq=true_sequence,
            masses=nucleosides,
        )
        + len(true_sequence) * extra_mass_dict["to_standard_unit"]
        + extra_mass_dict["3_prime_terminal"]
        + extra_mass_dict["5_prime_terminal"]
    )

    # Simulate fragments
    simulated_fragments = simulate(
        rng=rng,
        true_sequence=true_sequence,
        nucleoside_masses=nucleosides,
        num_replicates=int(options.num_replicates),
        phantom_rate=float(options.phantom_rate),
        noise_rate=float(options.noise_rate),
        noise_dist=config["noise_distribution"],
        extra_mass_dict=extra_mass_dict,
    )

    # Simulate singletons selection
    singletons = select_singletons(
        rng=rng,
        seq=true_sequence,
        nucleosides=nucleosides,
        max_singletons=int(options.max_singletons),
    )

    # Write simulation data to file
    simulated_fragments.write_csv(options.fragments, separator="\t")

    # Write metadata to file
    with open(options.meta, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)

    # Write singleton data to file
    singletons.write_csv(options.singletons, separator="\t")


def select_singletons(
    rng: np.random.Generator,
    seq: List[str],
    nucleosides: pl.DataFrame,
    max_singletons: int = 15,
) -> pl.DataFrame:
    """Select singletons.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    seq : List[str]
        True sequence.
    nucleosides : pl.DataFrame
        Nucleotide alphabet.
    max_singletons : int
        Maximum number of singletons.

    Returns
    -------
    pl.DataFrame
        Singleton alphabet.

    """
    # Select true nucleotides
    true_nucs = list(set(seq))

    num_additional_nucs = min(max_singletons, len(nucleosides)) - len(true_nucs)
    random_nucs = (
        []
        if num_additional_nucs < 1
        else rng.choice(
            [
                nuc
                for nuc in nucleosides.get_column("id").to_list()
                if nuc not in true_nucs
            ],
            size=num_additional_nucs,
            replace=False,
        ).tolist()
    )

    return nucleosides.filter(pl.col("id").is_in(true_nucs + random_nucs))


# METHOD: Consider each base in the form of a standard unit, which can be
# combined arbitrarily to build any sequence, and only adapt the masses of the
# fragment ends (either based on a tag or fragmentation).
def build_extra_mass_dict(
    fragmentation_type: str,
    element_mass_path: Path,
    start_tag: float,
    end_tag: float,
) -> dict:
    """Build dictionary with fragmentation-related masses.

    Parameters
    ----------
    fragmentation_type : str
        Fragmentation type.
    element_mass_path : Path
        Path to file containing element masses.
    start_tag : float
        Mass of 5'-tag.
    end_tag : float
        Mass of 3'-tag.

    Returns
    -------
    dict
        Dictionary of fragmentation-related masses.

    """
    # Build dict of elemental masses
    element_masses = pl.read_csv(element_mass_path, separator="\t")
    element_masses = {
        row[element_masses.get_column_index("symbol")]: row[
            element_masses.get_column_index("mass")
        ]
        for row in element_masses.iter_rows()
    }

    # Initialize dict with universal masses
    extra_mass_dict = {
        # Mass needed to turn a nucleoside to a standard unit (SU)
        "to_standard_unit": (
            element_masses["P"] + 2 * element_masses["O"] - element_masses["H+"]
        ),
        # Remove O from SU and add START tag (-H) for 5'-end of terminal fragments
        "5_prime_terminal": start_tag - element_masses["O"] - element_masses["H+"],
        # Remove PO3H from SU and add END tag (-H) for 3'-end of terminal fragments
        "3_prime_terminal": (
            end_tag
            - element_masses["P"]
            - 3 * element_masses["O"]
            - 2 * element_masses["H+"]
        ),
    }

    # Add fragmentation-specific masses for 5'- and 3'-ends of a fragment to dict
    match fragmentation_type:
        case "a/w":  # assuming double bond for 3'-end
            extra_mass_dict["5_prime_internal"] = (
                element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
            )
            extra_mass_dict["3_prime_internal"] = -(
                element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
            )
        case "b/x":
            extra_mass_dict["5_prime_internal"] = (
                element_masses["P"] + 2 * element_masses["O"]
            )
            extra_mass_dict["3_prime_internal"] = -(
                element_masses["P"] + 2 * element_masses["O"]
            )
        case "c/y":  # assuming cyclization for 3'-end
            extra_mass_dict["5_prime_internal"] = element_masses["H+"]
            extra_mass_dict["3_prime_internal"] = -element_masses["H+"]
        case "d/z":  # assuming double bond for 5'-end
            extra_mass_dict["5_prime_internal"] = -(
                element_masses["O"] + element_masses["H+"]
            )
            extra_mass_dict["3_prime_internal"] = (
                element_masses["O"] + element_masses["H+"]
            )
        case _:
            raise NotImplementedError(
                f"There is no fragmentation type called '{fragmentation_type}'."
            )

    return extra_mass_dict


def get_seq_weight(seq: List[str], masses: pl.DataFrame) -> float:
    """Get sequence weight.

    Parameters
    ----------
    seq : List[str]
        Sequence.
    masses : pl.DataFrame
        Nucleotide alphabet.

    Returns
    -------
    float
        Sequence mass.

    """
    seq_df = pl.DataFrame(data=seq, schema=["name"])
    seq_df = seq_df.with_columns(
        pl.col("name")
        .map_elements(
            lambda x: (
                masses.filter(pl.col("id") == x)
                .get_column("monoisotopic_mass")
                .to_list()[0]
            ),
            return_dtype=pl.Float64,
        )
        .alias("mass")
    )
    return seq_df.select("mass").sum().item()


def simulate(
    rng: np.random.Generator,
    true_sequence: List[str],
    nucleoside_masses: pl.DataFrame,
    num_replicates: int,
    phantom_rate: float,
    noise_rate: float,
    noise_dist: str,
    extra_mass_dict: dict,
) -> pl.DataFrame:
    """Simulate fragments.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    true_sequence : List[str]
        True sequence.
    nucleoside_masses : pl.DataFrame
        Nucleotide alphabet.
    num_replicates : int
        Number of replicates.
    phantom_rate : float
        Phantom rate.
    noise_rate : float
        Noise rate.
    noise_dist : str
        Noise distribution.
    extra_mass_dict : dict
        Dictionary with fragmentation-related masses.

    Returns
    -------
    pl.DataFrame
        Polars Dataframe containing simulated fragments.

    """
    # Sample random fragments from true sequence
    seq_len = len(true_sequence)
    frag_sites = [
        select_fragmentation_sites(
            num_sites=select_num_sites(seq_len=seq_len, rng=rng),
            seq_len=seq_len,
            rng=rng,
        )
        for _ in range(round(num_replicates * (1 + phantom_rate)))
    ]

    # Build fragment dataframe
    fragments = pl.from_records(
        compute_fragment_tuples(frag_sites, len(true_sequence)),
        schema=["left", "right"],
        orient="row",
    )

    # Add columns with boolean values to fragment dataframe
    fragments = fragments.with_columns(
        ((pl.col("left") == 0) & (~(pl.col("right") == len(true_sequence)))).alias(
            "is_start"
        ),
        ((~(pl.col("left") == 0)) & (pl.col("right") == len(true_sequence))).alias(
            "is_end"
        ),
        ((pl.col("left") == 0) & (pl.col("right") == len(true_sequence))).alias(
            "is_start_end"
        ),
        ((~(pl.col("left") == 0)) & (~(pl.col("right") == len(true_sequence)))).alias(
            "is_internal"
        ),
        (pl.col("right") == (pl.col("left") + 1)).alias("single_nucleoside"),
    )

    # Add column with true sequences to fragment dataframe
    fragments = fragments.with_columns(
        pl.struct("left", "right")
        .map_elements(
            lambda x: "".join(true_sequence[x["left"] : x["right"]]), return_dtype=str
        )
        .alias("sequence")
    )

    # Add column with exact nucleoside masses to fragment dataframe
    fragments = fragments.with_columns(
        pl.struct("left", "right")
        .map_elements(
            lambda x: sum(
                nucleoside_masses.filter(pl.col("id") == base)
                .select(pl.col("monoisotopic_mass"))
                .item()
                for base in true_sequence[x["left"] : x["right"]]
            ),
            return_dtype=float,
        )
        .alias("true_nucleoside_mass")
    )

    # Add column with noisy nucleoside masses to fragment dataframe
    fragments = fragments.with_columns(
        pl.struct("*")
        .map_elements(
            lambda x: induce_noise(
                rng=rng,
                distribution_method=noise_dist,
                noise_rate=noise_rate,
                mass=x["true_nucleoside_mass"],
            ),
            return_dtype=float,
        )
        .alias("observed_mass_without_backbone")
    )

    # Add column with exact nucleotide masses to fragment dataframe
    fragments = fragments.with_columns(
        pl.struct("*")
        .map_elements(
            lambda x: add_backbone_mass(
                fragment=x,
                mass=x["true_nucleoside_mass"],
                seq_len=len(true_sequence),
                mass_dict=extra_mass_dict,
            ),
            return_dtype=float,
        )
        .alias("true_mass_with_backbone")
    )

    # Add column with noisy nucleotide masses to fragment dataframe
    fragments = fragments.with_columns(
        pl.struct("*")
        .map_elements(
            lambda x: induce_noise(
                rng=rng,
                distribution_method=noise_dist,
                noise_rate=noise_rate,
                mass=x["true_mass_with_backbone"],
            ),
            return_dtype=float,
        )
        .alias("observed_mass")
    )

    # Select phantom (i.e. invalid) fragments
    fragments = fragments.with_columns(
        pl.struct("*")
        .map_elements(
            lambda x: rng.random() < phantom_rate,
            return_dtype=bool,
        )
        .alias("is_phantom_fragment")
    )

    # Update classification for phantom fragments by setting all to internal
    fragments = fragments.with_columns(
        (pl.col("is_start") & ~pl.col("is_phantom_fragment")).alias("is_start"),
        (pl.col("is_end") & ~pl.col("is_phantom_fragment")).alias("is_end"),
        (pl.col("is_start_end") & ~pl.col("is_phantom_fragment")).alias("is_start_end"),
        (pl.col("is_internal") | pl.col("is_phantom_fragment")).alias("is_internal"),
    )

    # Update observed mass for phantom fragments by adjusting it randomly
    fragments = fragments.with_columns(
        pl.struct("*")
        .map_elements(
            lambda x: (
                x["observed_mass"]
                + int(x["is_phantom_fragment"])
                * (
                    -PHANTOM_FRAGMENT_MAGNITUDE
                    + 2 * PHANTOM_FRAGMENT_MAGNITUDE * rng.random()
                )
            ),
            return_dtype=float,
        )
        .alias("observed_mass")
    )

    # Return final fragment dataframe
    return fragments


# METHOD: Consider fragments without any fragmentation, i.e. complete
# fragments, separately (randomly select based on given probability); if the
# sequence does fragment, use a geometric distribution to determine the
# number of sites while approximating the true distribution of fragment lengths
# observed in experimental data (exponential distribution with many small and
# few larger fragments, which gets sharper with increasing sequence length)
def select_num_sites(seq_len: int, rng: np.random.Generator) -> int:
    """Select number of fragmentation sites.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    int
        Number of fragmentation sites.

    """
    if rng.random() < NO_FRAGMENTATION_PROBABILITY:
        return 0
    # Note that p = factor/seq_len with factor = seq_len/alpha
    # thus using p = seq_len/alpha/seq_len = 1/alpha
    return min(rng.geometric(p=0.3), seq_len - 1)


# TODO: Implement that in some cases there is no base pair generated,
#  but only the backbone with sugar etc?
def select_fragmentation_sites(
    num_sites: int, seq_len: int, rng: np.random.Generator
) -> List[int]:
    """Select fragmentation sites.

    Parameters
    ----------
    num_sites : int
        Number of fragmentation sites.
    seq_len : int
        Sequence length.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    List[int]
        List of fragmentation sites (by index).

    """
    # Ensure there is a positive number of parts (i.e. number of sites + 1)
    if num_sites < 0:
        raise ValueError("The number of parts cannot be less than one!")

    # Ensure the number of parts is not greater than the sequence length
    if num_sites + 1 > seq_len:
        raise ValueError(
            "The number of parts cannot be greater than the sequence length!"
        )

    # If the sequence has zero sites, it remains intact
    if num_sites == 0:
        return [int(0)]

    # Return randomly sampled fragmentation sites in the sequence
    # Use beta distribution to avoid bias towards small terminal fragments (like e.g.
    # for uniform one)
    return sorted(
        set(round(val * seq_len) for val in rng.beta(a=2, b=2, size=num_sites))
    )


def compute_fragment_tuples(frag_sites: List[int], seq_len: int) -> Tuple[int, int]:
    """Compute tuples of start and end index for fragments.

    Parameters
    ----------
    frag_sites : List[int]
        Fragmentation sites.
    seq_len : int
        Sequence length.

    Returns
    -------
    Tuple[int, int]
        Fragment tuple.

    """
    tuples = []

    # Generate tuples of start and end index for each fragments
    for seq_replicate in frag_sites:
        if seq_replicate[0] != 0:
            seq_replicate.insert(0, 0)
        if seq_replicate[-1] != seq_len:
            seq_replicate.append(seq_len)

        tuples += list(zip(seq_replicate[:-1], seq_replicate[1:]))

    return tuples


def induce_noise(
    rng: np.random.Generator, distribution_method: str, noise_rate: float, mass: float
) -> float:
    """Induce noise in fragment.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    distribution_method : str
        Noise distribution method.
    noise_rate : float
        Noise rate.
    mass : float
        Fragment mass.

    Returns
    -------
    float
        Fragment mass with noise.

    """
    match distribution_method:
        case "normal":
            noise = rng.normal(scale=noise_rate)
        case "uniform":
            noise = -noise_rate + 2 * noise_rate * rng.random()
        case _:
            raise NotImplementedError(
                f"There is no option for the noise distribution called '{distribution_method}'."
            )

    return max(mass * (1 + noise), 0.0)


def add_backbone_mass(
    fragment: dict, mass: float, seq_len: int, mass_dict: dict
) -> float:
    """Add backbone mass to fragment mass.

    Parameters
    ----------
    fragment : dict
        Fragment information.
    mass : float
        Fragment mass.
    seq_len : int
        Sequence length.
    mass_dict : dict
        Dictionary with fragmentation-related masses.

    Returns
    -------
    float
        Fragment mass with added backbone.

    """
    # Turn nucleoside mass into the one of the corresponding standard units
    mass += (fragment["right"] - fragment["left"]) * mass_dict["to_standard_unit"]

    # Adapt 5'-end of fragment
    mass += (
        mass_dict["5_prime_terminal"]
        if fragment["left"] == 0
        else mass_dict["5_prime_internal"]
    )

    # Adapt 3'-end of fragment
    mass += (
        mass_dict["3_prime_terminal"]
        if fragment["right"] == seq_len
        else mass_dict["3_prime_internal"]
    )

    return mass
