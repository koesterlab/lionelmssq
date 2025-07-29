import polars as pl

from lionelmssq.masses import UNIQUE_MASSES


def estimate_MS_error_matching_threshold(
    fragments, unique_masses=UNIQUE_MASSES, rejection_threshold=0.5, simulation=False
):
    """
    Using the mass of the single nucleosides, A, U, G, C, estimate the
    relative error that the MS makes, this is used to determine the
    MATCHING_THRESHOLD for the DP algorithm!

    """

    unique_natural_masses = (
        unique_masses.filter(pl.col("nucleoside").is_in(["A", "U", "G", "C"]))
        .select(pl.col("monoisotopic_mass"))
        .to_series()
        .to_list()
    )

    if simulation:
        singleton_masses = (
            fragments.filter(pl.col("single_nucleoside"))
            .select(pl.col("observed_mass"))
            .to_series()
            .to_list()
        )
    else:
        singleton_masses = (
            fragments.filter(
                pl.col("observed_mass").is_between(
                    min(unique_natural_masses) - rejection_threshold,
                    max(unique_natural_masses) + rejection_threshold,
                )
            )
            .select(pl.col("observed_mass"))
            .to_series()
            .to_list()
        )

    relative_errors = []
    for mass in singleton_masses:
        differences = [abs(unique_mass - mass) for unique_mass in unique_natural_masses]
        closest_mass = (
            min(differences) if min(differences) < rejection_threshold else None
        )
        if closest_mass:
            relative_errors.append(abs(closest_mass / mass))
            print(
                "Mass = ",
                mass,
                "Closest mass = ",
                closest_mass,
                "Relative error = ",
                abs(closest_mass / mass),
            )

    if relative_errors:
        average_error = sum(relative_errors) / len(relative_errors)
        std_deviation = (
            sum((x - average_error) ** 2 for x in relative_errors)
            / len(relative_errors)
        ) ** 0.5
        return max(relative_errors), average_error, std_deviation
    else:
        return None, None, None
