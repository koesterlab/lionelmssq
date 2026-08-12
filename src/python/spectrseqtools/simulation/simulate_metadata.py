import random

import yaml
from pathlib import Path

from spectrseqtools.parsers import (
    CustomMetadataSimulationOptions,
    RandomMetadataSimulationOptions,
)


def simulate_metadata_for_custom_sequence(options: CustomMetadataSimulationOptions):
    # Initialize metadata
    seq = options.sequence
    meta = {
        "identity": f"custom_simulation_{seq}",
        "true_sequence": seq,
        "5_prime_tag": options.start_tag,
        "3_prime_tag": options.end_tag,
    }

    # Write metadata to file
    file_name = options.output_dir / "sample.meta.yaml"
    with open(file_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)


def simulate_metadata_for_random_sequences(options: RandomMetadataSimulationOptions):
    random.seed(options.global_seed)
    sequences = [
        generate_random_sequence_and_seed_pair(
            seq_len=random.choice(range(10, 21))
            if options.sequence_length == -1
            else options.sequence_length,
            modification_rate=options.modification_rate,
            modifications=options.alphabet,
        )
        for _ in range(options.num_sequences)
    ]

    for idx, seq in enumerate(sequences):
        # Initialize metadata
        meta = {
            "identity": f"random_simulation_{idx}",
            "true_sequence": seq[0],
            "seed": seq[1],
            "5_prime_tag": options.start_tag,
            "3_prime_tag": options.end_tag,
        }

        # Write metadata to file
        file_name = options.output_dir / f"sim_{idx + 1}/sample.meta.yaml"
        with open(file_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f)


def generate_random_sequence_and_seed_pair(
    seq_len: int, modification_rate: float, modifications: Path | None
):
    # If no modifications are given, simulate over unmodified bases only
    if modifications is None:
        return "".join(
            [random.choice(["A", "U", "G", "C"]) for _ in range(seq_len)]
        ), random.choice(range(10000))

    # Read modifications from file
    with open(modifications, "r") as file:
        lines = file.readlines()[1:]
    modified_nucleoside_names = [
        line.split("\t")[0]
        for line in lines
        if line.split("\t")[0] not in ["U", "A", "G", "C"]
    ]

    # Define probabilities for different bases (using modification rate)
    weights = [(1.0 - modification_rate) / 4] * 4 + [
        modification_rate / len(modified_nucleoside_names)
    ] * len(modified_nucleoside_names)

    return "".join(
        random.choices(
            ["A", "U", "G", "C"] + modified_nucleoside_names,
            weights=weights,
            k=seq_len,
        )
    ), random.choice(range(10000))
