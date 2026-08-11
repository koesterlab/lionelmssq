import yaml

from spectrseqtools.parsers import CustomMetadataSimulationOptions


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
    file_name = options.output_dir / f"sample.meta.yaml"
    with open(file_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)
